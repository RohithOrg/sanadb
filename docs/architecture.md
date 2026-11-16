# How it works

A store is three files: `<path>` is the write-ahead log, `<path>.idx.<id>` the
graph index named by the log's first frame (a new file per checkpoint, so an
index is never overwritten while a reader maps it), `<path>.lock` an empty
file that is `flock`ed while a writer has the store open.

## Log

CRC-framed records: insert (id, vector, metadata), delete (id), and
meta (id, metadata only; the vector lives in the index). Opening replays the
log and truncates a torn tail. Writes go to the OS page cache at once and
survive a process crash; `sync()` or `sync_every_write` fsyncs for power loss.

## Graph index

A Vamana graph, the DiskANN layout: one fixed-size block per
node holding its vector, neighbour count and up to R neighbour slots. The file
is memory-mapped, so a query touches only the blocks it walks and the page
cache holds the working set. Per node, RAM holds the 8-byte id, the metadata,
and a 32-byte product-quantization code; the vectors and edges stay on disk.

## Product quantization

Indexes of 1024 nodes or more carry a codebook (256
centroids per subspace of about 4 floats) and one byte per subspace per node.
The walk scores neighbours from these codes, so expanding a node costs one
block read, its adjacency, rather than R reads of neighbour vectors. The best
candidates are re-ranked with exact distances from the mapped file.

## Beam-width expansion

The walk expands several candidates per step and
asks the kernel to prefetch all their blocks first (`madvise(WILLNEED)`), so a
cold query overlaps its disk reads instead of taking one page fault at a time.
`Options::io_width`, default 4.

## Delta and tombstones

Writes since the last checkpoint are searched
exactly. The delta keeps only ids and byte offsets into the memory-mapped log,
16 bytes per record; the vectors are read from the log in place, through the
page cache. Graph records that were since deleted or overwritten are hidden by
a tombstone set rebuilt from the log on open. A search walks the graph, scans
the delta, merges.

## Checkpoint

Writes the live vectors to a temp file under a brief exclusive
lock, memory-maps it, builds the new graph with no lock held, then swaps the
index in by rename, rewrites the log and swaps that in. Index first, so the
log never names a vector the index lacks. Writes that arrive during the build
stay in the delta, told apart by a per-record sequence number. Sets larger
than `shard_size` (default 1M) are split by k-means into overlapping shards,
each shard is built on its own and spilled to disk, and a final pass merges
and prunes each node's edges. Peak memory is one shard's adjacency, not the
whole graph's.

## Metadata and filters

Values are typed (`Str`, `Int`, `Float`, `Bool`);
the first value under a key fixes its type. At checkpoint each key becomes a
slot-ordered column in the index file (strings dictionary-coded, with a
presence bitmap and, up to 65,536 distinct values, posting lists). A
`Filter` expression runs on those columns: checking a visited node is an
array read. Per query one of two regimes is chosen from the posting lists.
If at most `exact_filter_scan` records (default 16,384) can match, they are
read and ranked exactly, so a rare filter gets recall 1.0 and never a short
answer. Otherwise the walk is steered by the filter: rejected nodes are
expanded for navigation but only passing nodes take result slots, so a 1%
filter costs a longer walk rather than repeated widened walks. Closure
filters cannot be inspected and keep the old path: applied to visited nodes,
beam doubled until `k` pass, up to `max_search_beam`.

## Concurrency

A `RwLock` guards the in-memory state: reads share it, writes
take it, checkpoints serialise on a separate mutex and hold neither during
the build. A read-only `Db` keeps the log tail in RAM instead of a mapping
and, before each read, checks the log's size and identity: appended frames
are replayed, a replaced log (checkpoint) means reload. A writer process
flushes every write, so readers see it as soon as `insert` returns. The
in-RAM state of any `Db` is the tail since the last checkpoint and the
tombstones; metadata of indexed records is read from the index columns.

## Keyword and sparse search

`Options::text_field` names a metadata key whose strings are tokenised
(lowercase alphanumeric runs) and written at checkpoint as an inverted
index section: a sorted term dictionary, posting lists of `(slot, term
frequency)` and per-record lengths. `search_text` scores BM25 (k1 = 1.2,
b = 0.75) term at a time over those lists plus the records written since
the checkpoint, with one set of corpus statistics. `Filter::contains` on
the text field answers from the same lists.

A `Value::Sparse` metadata value (one key per store) is written at
checkpoint as a sparse section with forward lists (so `get` returns the
pairs) and inverted lists of `(slot, weight)` per term id; `search_sparse`
accumulates dot products over the query's terms. `search_hybrid` and
`search_hybrid_sparse` fetch the dense and the keyword or sparse list at
`3k` and fuse them by reciprocal rank, `1 / (60 + rank)` summed.

The byte layout of every section is in [FORMAT.md](FORMAT.md); the design
notes under `design/` record why each piece looks the way it does.
