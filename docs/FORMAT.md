# File format

The on-disk layout of a sanadb store as written by 0.2.0, frozen for the
0.x series (see "Compatibility" at the end). All integers are little-endian.
Offsets are byte offsets from the start of the file they are in. `n` is the
number of records in an index, `dim` the vector dimension.

A store at `<path>` is these files:

| file | role |
|---|---|
| `<path>` | the log: header, then frames. The only file a writer appends to. |
| `<path>.idx.<id>` | the index named by the log's `Snapshot` frame; `<id>` is 16 lowercase hex digits. Written once, never modified; replaced by a new file with a new name at the next checkpoint. |
| `<path>.lock` | empty; the writer holds an exclusive advisory `flock` on it. |
| `<path>.tmp`, `<path>.vec.tmp`, `<path>.idx.*.tmp` | checkpoint temporaries; removed on success, failure or the next writer open. |

An index file no log names (left by a crash between the two renames, or by
a reader on Windows that still had the previous index mapped) is removed
at the next writer open.

## Log

### Header, 16 bytes

| offset | size | field |
|---|---|---|
| 0 | 8 | magic `SANADB02` |
| 8 | 4 | `dim` u32 |
| 12 | 1 | metric: 0 L2, 1 Dot, 2 Cosine |
| 13 | 1 | reader level the file needs, currently 3 |
| 14 | 2 | zero |

### Frames

Frames follow the header back to back until the end of the file:

| size | field |
|---|---|
| 4 | `len` u32: body length, a non-zero multiple of 4 |
| 4 | CRC-32 (IEEE, as `crc32fast`) of the body |
| `len` | body |

A reader stops at the first frame whose `len` is zero or not a multiple of
4, that runs past the end of the file, or whose CRC does not match: that is
the torn tail of a crash, and the writer truncates the file there on open.
Bytes after the last good frame are never trusted.

Body: `kind` u8, 3 zero bytes, then the fields of the kind, then zero
padding to a multiple of 4.

| kind | name | fields |
|---|---|---|
| 1 | Insert | `id` u64, vector `dim` × f32, metadata |
| 2 | Delete | `id` u64 |
| 3 | Meta | `id` u64, metadata. Written by 0.1.4 for records whose vector is in the index; a level-3 writer never writes one, a level-3 reader replays it. |
| 4 | Snapshot | `id` u64: the index file this log pairs with is `<path>.idx.<id as 16 hex digits>`. First frame of a compacted log. A log without one pairs with the unnumbered `<path>.idx` (0.1.x) or has no index. |
| 5 | Begin | none. Opens an atomic batch. |
| 6 | Commit | none. Closes it: the frames between take effect together. A `Begin` with no `Commit` before the end of the log is a crashed batch; its frames are dropped and the writer truncates them. |

Readers skip a frame whose kind they do not know (the CRC still has to
match), which is how a kind can be added without a level change.

Metadata: `count` u16, then `count` entries of

| size | field |
|---|---|
| 2 + k | key: length u16, then k bytes of UTF-8 |
| 1 | value kind: 0 Str, 1 Int, 2 Float, 3 Bool, 4 Sparse |
| | value: Str as length u16 + bytes; Int i64; Float f64; Bool u8 (0 or 1); Sparse as `count` u32 then `count` × (`index` u32, `weight` f32), ascending index, no duplicates, no zero weights |

A key has one kind for the life of the store; the first value written
under a key fixes it (`Error::TypeMismatch` otherwise). At most one key
holds Sparse values (`Error::SecondSparseKey`); it is the store's sparse
field, indexed in the sparse section below. Keys, string
values and metadata as a whole are at most 65,535 entries or bytes.

A record is the last Insert for its id not undone by a later Delete; an
Insert for an existing id replaces it. The log alone defines the contents
of a store; the index is a derived structure over the records present at
the checkpoint that built it, and the log's frames after the `Snapshot`
frame are the records written since (the "tail").

### Compacted log

A checkpoint rewrites the log as: header, one `Snapshot` frame, one
`Insert` frame per record written while the checkpoint was building, one
`Delete` frame per record deleted while it was building whose slot is in
the new index. Metadata of indexed records lives in the index's attribute
columns, so a quiet store's log is 36 bytes after a checkpoint.

### Reader level

Header byte 13 is the lowest reader level that can open the file. A
reader with a lower level fails with `Error::NeedsNewerVersion`. Additive
changes (new frame kinds) do not raise it. Levels so far:

| level | written by | difference |
|---|---|---|
| (`SANADB01`) | 0.1.1 | no padding after `kind`, bodies not padded to 4. Migrated on open. |
| 1 | 0.1.2, 0.1.3 | metadata values are bare strings (no kind byte). Migrated on open. |
| 2 | 0.1.4 | typed values; Meta frames for indexed records. Opened as is. |
| 3 | 0.2.0 | Snapshot, Begin, Commit frames; no Meta frames written. |

## Index

`<path>.idx.<id>`, memory-mapped and read in place. Sections in file
order; "pad to 8" means zero bytes to the next multiple of 8.

### Header, 36 bytes

| offset | size | field |
|---|---|---|
| 0 | 8 | magic `SANAIDX2` |
| 8 | 4 | `dim` u32 |
| 12 | 1 | metric, as in the log |
| 13 | 1 | index reader level, currently 1 |
| 14 | 1 | storage: 0 f32, 1 int8 |
| 15 | 1 | zero |
| 16 | 4 | `n` u32: records (slots), tombstoned ones included |
| 20 | 4 | `r` u32: maximum out-degree |
| 24 | 4 | `entry` u32: slot the walk starts from (medoid) |
| 28 | 4 | `block` u32: bytes per record block |
| 32 | 4 | `m` u32: PQ subspaces, 0 for none |

### Slot ids

`n` × u64 at offset 36: the record id of each slot, in slot order. Then
zero padding to a multiple of 64.

### Record blocks

`n` blocks of `block` bytes, 64-aligned, slot `s` at
`blocks_off + s × block`:

| field | f32 storage | int8 storage |
|---|---|---|
| vector | `dim` × f32 | `dim` × u8 codes, padded to a multiple of 4, then `min` f32, `scale` f32; value = `min + code × scale` |
| degree | u32 | u32 |
| neighbours | `degree` × u32 slots | same |
| | zero to `block` | |

`block = (vector_bytes + 4 + 4 r)` rounded up to a multiple of 64, where
`vector_bytes` is `4 dim` (f32) or `dim rounded up to 4, plus 8` (int8).

### Product quantisation (when `m` > 0)

Immediately after the blocks: codebook `m × 256 × (dim / m)` f32
(subspace-major, 256 centroids each), then codes `n × m` u8, slot-major.
`dim` is a multiple of `m`.

### Attribute columns

Pad to 8, then one region per metadata key in key order, then the
attribute table. The footer gives the table's offset. Per key, in this
order:

- presence: `ceil(n / 64)` u64 words, bit `s % 64` of word `s / 64` set
  when slot `s` has the key with the key's kind.
- column, by kind: Str `n` × u32 dictionary code; Int `n` × i64; Float
  `n` × f64; Bool `n` × u8 padded to a multiple of 8. Absent slots hold 0.
- dictionary (Str always; Int when at most 65,536 distinct values):
  `count` u32, 4 zero bytes, then for Str `count + 1` u64 offsets into the
  string bytes that follow, padded to 8; for Int `count` × i64 ascending.
- postings (when a dictionary of at most 65,536 entries exists, or Bool):
  `count + 1` u64 prefix sums, then the slots holding each value, ascending
  within a value, as u32, padded to an even count.

Attribute table: `n` u64, `count` u32, 4 zero bytes, then per key: name
length u16, name bytes, pad to 8, kind u8, flags u8 (bit 0 dictionary
present, bit 1 postings present), 2 zero bytes, `n_distinct` u32, then
five u64: presence offset, column offset, dictionary offset, dictionary
length, postings offset (0 when absent).

### Id table

Pad to 8, then `n` × u64 ids ascending, then `n` × u32 the slot of each.
`slot(id)` is a binary search over this table.

### Text section (when the store has a text field)

Pad to 8, then: lengths `n` × u32 (tokens per slot, 0 when none); pad to
8; term dictionary in the string-dictionary shape above (terms sorted);
`n_terms + 1` u64 posting offsets (in entries); postings, each entry
`slot` u32, `tf` u16, 2 zero bytes, ascending slot within a term; then the
table of nine u64: `n`, documents with tokens, total tokens, lengths
offset, dictionary offset, dictionary length, posting offsets offset,
postings offset, `n_terms`. Tokens are maximal runs of alphanumeric
characters, lowercased, at most 64 bytes. Scoring is BM25 with k1 = 1.2,
b = 0.75.

### Sparse section (when a key holds sparse vectors)

Pad to 8, then: field name as `length` u64, bytes, pad to 8; forward
offsets `n + 1` u64 (in pairs; slot `s` owns `[o[s], o[s+1])`); pairs
(`index` u32, `weight` f32) per record, ascending index; term dictionary
`count` u64 then `count` × u32 ascending, pad to 8; `count + 1` u64 posting
offsets (in postings); postings (`slot` u32, `weight` f32), ascending slot
within a term; then the table of ten u64: `n`, name offset, name length,
forward offsets offset, pairs offset, pairs count, dictionary offset, term
count, posting offsets offset, postings offset. A query scores a record by
the dot product of the two pair lists.

### Footer, 40 bytes, last in the file

| size | field |
|---|---|
| 8 | attribute table offset |
| 8 | id table offset |
| 8 | text table offset, 0 when there is no text section |
| 8 | sparse table offset, 0 when there is no sparse section |
| 8 | magic `SANAFTR2` |

Older index files: 0.1.4 ended in a 16-byte trailer `attribute table
offset u64 | ATTRTBL1` and had no id table (the reader builds the id map
in RAM); 0.1.3 and earlier ended after the PQ section. Both still open
and are replaced at the next checkpoint. An index whose header byte 13 is
above the reader's level fails with `Error::NeedsNewerVersion`.

### Validation on open

A reader checks the magic, level, `dim` and metric against the store's,
`block` against `dim`, `r` and storage, that the PQ section fits, that
`entry < n`, that the id table lies between the fixed sections and the
footer, and that every table offset it dereferences lies in the file. A
file that fails is `Error::Corrupt` with the reason. Neighbour slots and
dictionary offsets are bounds-checked on every read, so a damaged index
returns an error or an empty result, never reads out of bounds; the test
suite mutates and truncates index files at random to hold that.

## Compatibility

- Within 0.x, from 0.2.0: a store written by any 0.2.0 or later version
  opens in every later 0.x version. A later version may add frame kinds
  (older readers skip them) and index sections (named by a new footer
  slot or a longer footer; older readers ignore what they do not know).
  A change an older reader would misread raises the reader level in the
  header, is listed in `CHANGELOG.md`, and ships with conversion on open
  for every earlier layout. An older version opening a newer store either
  reads it or fails with `Error::NeedsNewerVersion`; it never misreads.
- Every published version's store is a fixture under `tests/fixtures/`,
  and the test suite opens, reads, writes to and checkpoints each one.
  A release is cut only when all of them pass.
- 1.0 freezes this layout further: no reader-level raise without a major
  version, so a 1.x reader opens every 0.2.0-or-later store and every 1.x
  store.
