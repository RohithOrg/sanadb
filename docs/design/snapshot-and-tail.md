# Snapshot and tail: metadata off the heap, readers in other processes

Status: design for 0.2.0. Covers two items from the gap list that turn out to
be one change: a reader that holds nothing in RAM but the tail of the log can
also be a reader in another process.

## The gap

Today a `Db` keeps, for every live record, an entry in `Inner::meta`
(`HashMap<u64, Arc<Metadata>>`) and, through `GraphIndex::slot_of`, an entry
in a `HashMap<u64, u32>`. Together about 140 bytes per record before the
metadata values themselves. At 5M records that is 700 MB of heap for a store
whose vectors need 160 MB of PQ codes; at 100M it does not fit. The attribute
columns written since 0.1.4 already hold every graph record's metadata in the
index file; the map is now a duplicate.

Separately, a store is open to one process at a time (exclusive `flock`).
SQLite's model, many readers and one writer across processes, is what "the
SQLite of vector stores" has to offer, and it is why the RAM question comes
first: a reader process cannot be asked to build a 100M-entry map on open.

## The model

A store is a **snapshot** plus a **tail**.

- The snapshot is the index file: graph, PQ codes, attribute columns, and
  (new) a sorted id table. Immutable once written, replaced whole by rename.
- The tail is the log since the snapshot: inserts, deletes and metadata
  changes, each a frame. Append-only within one snapshot's lifetime; a
  checkpoint writes a new log that begins with the frames the new snapshot
  does not cover.

Everything a reader needs is derivable from (snapshot, tail), and the tail is
small by construction (`auto_checkpoint_delta`). So the in-RAM state of any
`Db`, writer or reader, becomes: the tail's records (ids, vector offsets,
metadata, sequence numbers), the tombstone set (graph slots superseded by the
tail), the attribute type table, and two file mappings. Nothing proportional
to the snapshot.

### What moves out of RAM

| today | after |
|---|---|
| `Inner::meta` for every record | metadata of tail records only; graph records read columns |
| `GraphIndex::slot_of` hash map | sorted `(id, slot)` table in the index file, binary search on the mapping |
| `Inner::types` from replay of every record | attribute table from the index header plus tail records |
| `len()` from `meta.len()` | `graph.len() - tombstones.len() + delta.len()` |

`get(id)` for a graph record decodes its metadata from the columns (one
`Value` per attribute present), allocating a `Metadata` on demand. Closure
filters (`search_filtered`) do the same per visited node; that path was
already the slow one and is documented as such. `Filter` expressions do not
allocate.

Type checks on insert consult the index's attribute table, then the tail's
type table. A key that exists only in the tail is typed by the tail.

### Pairing the snapshot with its tail

A reader that opens the two files must know they belong together, since a
checkpoint replaces first the index, then the log. The compacted log's first
frame is a new kind, `KIND_SNAPSHOT = 4`, carrying a 16-byte snapshot id;
the index header carries the same id in its (padded, so additive) header. A
reader accepts a pair only when the ids match, and otherwise reopens after a
short pause: the window is the time between two renames. Old readers (0.1.4)
skip the unknown frame kind, which is what the frame-kind rule was for.

A log without a snapshot frame (written before 0.2.0) pairs with an index
whose id is zero, or with no index.

## Processes

### Locks

- Writer: exclusive `flock` on `<path>.lock`, as now. One writer per store.
- Reader: shared `flock` on the same file. Any number.
- `Db::open` is the writer; `Db::open_read_only` is a reader. A reader never
  writes any file, including the lock file's contents, and never
  checkpoints.

Readers do not block the writer's checkpoint on Unix: the renames replace
directory entries, and a reader's mappings keep the old inodes alive until
it reopens. On Windows a file that any process has mapped cannot be renamed
over; the writer retries the rename with backoff for a bounded time (say two
seconds) and then returns an error from `checkpoint()` that says a reader is
holding the files. This is the honest Windows story and it goes in the
README.

### Seeing new writes

The writer flushes the log after every write already (page cache, not
fsync), so frames are visible to other processes as soon as `insert` or
`delete` returns.

A reader refreshes at the start of each operation, cheaply:

1. `fstat` the open log. If its size grew, replay frames from the reader's
   last good offset; a torn frame at the end is simply not there yet. The
   log mapping is extended when the file grew past it.
2. If the log's inode changed (a checkpoint happened), reopen log and index
   and rebuild the tail state from the new log. Verify the pairing id.

Refresh cost in the common case is one `fstat`. `Options::refresh_interval`
(default: every operation) lets a reader that prefers throughput over
freshness check less often, and `Db::refresh()` forces one.

What a reader sees is a prefix of the writer's history: never a record
without its vector, never a metadata change without the record, because the
writer appends a record's frame whole and the reader stops at a torn tail.

### Growth and truncation

The writer extends the log in 64 MB steps and trims the slack on close. A
reader that maps the file sees the slack as zeros, which parse as a torn
tail. A reader whose mapping is shorter than the file remaps. Truncation
only happens on the writer's close, when the reader's next refresh sees a
smaller file and re-replays from the header; the frames it had are still
there.

### Snapshot id and the lock file

The lock file stays empty; it carries nothing readers depend on. The
snapshot id lives in the two data files so a reader that has both can
decide without the writer.

## Writer changes

- `Inner::meta` becomes `tail_meta`, only for delta records. `apply_insert`
  of a graph record moves it to the tail (as now, plus a tombstone).
- Checkpoint snapshot: the index writer gets metadata for graph records from
  the current index's columns (a `Metadata` per slot, read back) and for
  tail records from `tail_meta`. The `Arc` snapshot map goes away.
- Compaction writes the snapshot frame first, then the survivors.
- `len()`, `index_stats().tombstones` as above.

## Format

- Log: new frame kind 4; and, because a 0.2.0 log omits Meta frames for
  graph records (their metadata lives only in columns), a 0.1.4 reader would
  see records without metadata and misreport them. So the log's reader level
  goes to 3 and 0.1.4 refuses with `NeedsNewerVersion`. Downgrade is not
  supported, as before. 0.2.0 opens every 0.1.x store: the tail map is
  filled from the old log's Meta frames and the next checkpoint writes the
  new layout.
- Index: snapshot id in the header's padding; a sorted `(id, slot)` table
  after the attribute section, found through the trailer (which grows a
  second offset). Index reader level stays 1: a 0.2.0 reader opening a 0.1.4
  index has no id table and builds `slot_of` in RAM as today until the next
  checkpoint.

## Measuring it

- Resident memory of a `Db` over the 5M store: today about 1 GB for the maps
  alone; target under 50 MB plus the tail.
- Open time for the 5M store: today 1.8 s (building `slot_of`); target under
  100 ms.
- Cross-process test: a writer process inserts and checkpoints in a loop
  while a reader process queries and asserts every id it sees has a vector
  and metadata, and that `len()` never decreases below the writer's acked
  count at the time of the read. Runs in CI on all three platforms; on
  Windows it also exercises the rename retry.
- Query latency unchanged on the warm SIFT1M store (the walk does not touch
  the maps).

## Order of work

1. Sorted id table in the index; `slot_of` reads it; open time measured.
2. Columns authoritative for graph metadata; `tail_meta`; `len()` and types
   from the index; log level 3; checkpoint reads old columns to write new.
3. Snapshot frame and index id; pairing check on open.
4. `open_read_only`: shared lock, refresh, reopen on inode change.
5. Cross-process tests; Windows rename retry.
6. Python: `sanadb.Db(path, read_only=True)` and `refresh()`.
