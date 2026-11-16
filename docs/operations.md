# Operating a store

What the files are, what survives what, and the knobs that matter.

## Files

A store at `<path>` is `<path>` (the log), `<path>.idx.<id>` (the index the
log names; a new file per checkpoint) and `<path>.lock` (empty, `flock`ed
by the writer). Copy or back up all of them together; `Db::backup` does
that consistently while writes continue. The layout is in
[FORMAT.md](FORMAT.md).

## Durability

- A write that returned `Ok` survives a process crash: it is in the OS
  page cache. It survives power loss only after `sync()` or with
  `Options::sync_every_write`, which fsyncs every write.
- A checkpoint either completes or leaves the previous files untouched.
  The index is written to a new file and the log is replaced by rename,
  index first, so the log never names an index that is missing. The test
  suite injects a full disk, a failed fsync, a failed rename and a failed
  append at those points and checks that the store keeps serving everything
  it acknowledged.
- Opening replays the log and truncates a torn tail (a frame whose length
  or CRC is wrong) and a batch whose `Commit` never made it.
- Readers never see a partially written record or index: frames are
  CRC-checked and the index is mapped only after its rename.

## Concurrency and processes

One writer per store, any number of readers, in any processes. A second
`Db::open` fails with `Error::Locked` (after a 200 ms retry);
`Db::open_read_only` takes no lock, follows the writer's inserts, deletes
and checkpoints, and always sees a prefix of the writer's history. Within
a process `Db` is `Send + Sync`: reads share a lock, writes take it, a
checkpoint builds the new graph holding neither.

The writer lock is an advisory `flock` on `<path>.lock`: reliable on local
filesystems, not on NFS, SMB or other network mounts, where two machines
can both believe they hold it. Keep stores on local disks.

On Windows a reader that still maps the previous index keeps the writer
from deleting it; the next writer open removes index files the log no
longer names.

## Backup and replication

`backup(dir)` writes a consistent point-in-time copy (the index file plus
the log's complete frames) that opens like any store, from a writer or a
read-only handle, while writes continue. Replication is file shipping:
back up, copy the directory, open it read-only or as a new writer. There is
no streaming replication and no second writer; one writer per store is the
model, as with SQLite.

## Encryption

None inside the files. The index is memory-mapped and read in place, which
is what makes queries cheap; decrypting per block would need a user-space
cache and give up most of that. Use the platform's volume or file
encryption (FileVault, LUKS, BitLocker, EBS encryption), which covers the
store files like any others. The project will not ship a weaker built-in
alternative.

## Compatibility

A newer sanadb opens any store an older one wrote, converting it on first
open if the layout changed. An older sanadb opens a newer store as long as
the layout is one it knows, skipping record kinds it does not understand;
if the layout is newer than it can read, it fails with
`Error::NeedsNewerVersion` rather than misreading. The layout is frozen
from 0.2.0 for the 0.x series; stores written by every published version
are fixtures in the test suite and each release opens all of them.
`CHANGELOG.md` lists every format change with upgrade notes.

## Observability

`Db::stats()` (Python `Db.stats()`) returns counters since open: searches
by regime (graph walk, filtered walk, exact scan, delta only), text and
sparse searches, inserts, deletes, batches, checkpoints (total,
incremental, failed, last duration in ms) and read-only refreshes
(appended, reloaded). `Db::index_stats()` describes the index: nodes, PQ
subspaces, tombstones, storage. The Cargo feature `tracing` emits events on
target `sanadb` for checkpoints, failed checkpoints, reader reloads and
orphan index files removed at open.

## Knobs

| option | default | effect |
|---|---|---|
| `search_beam` | 64 | candidates kept during the walk; recall and latency both rise with it |
| `max_search_beam` | 4096 | how far a closure-filtered search may widen the beam |
| `io_width` | 4 | nodes expanded per step with their blocks prefetched together; 1 is the classic walk, 4 to 8 overlap reads on a cold index |
| `exact_filter_scan` | 16,384 | a `Filter` with at most this many matches is answered by reading them all (recall 1.0) |
| `graph.max_degree` (R) | 32 | out-degree; 48 raises recall at a given beam at the cost of build time and index size |
| `graph.build_beam` (L) | 100 | candidate list during build |
| `graph.use_pq` | true | 32 bytes of codes per node in RAM; the walk scores from them and re-ranks exactly |
| `storage` | f32 | `Storage::I8` halves the index file; recall within a point on SIFT |
| `auto_checkpoint_delta` | 0 | checkpoint once the tail holds this many records and is at least as large as the index |
| `auto_checkpoint_garbage` | 0 | checkpoint once the log holds this many more entries than live records |
| `text_field` | none | metadata key indexed for keyword search |
| `sync_every_write` | false | fsync every write |

A checkpoint is the one heavy operation: it builds the graph over every
record (or updates it in place when the tail is smaller than the indexed
set) with all cores, and writes a new index file of about `n × block`
bytes. Run it on a schedule, on `auto_checkpoint_*`, or after a bulk load.
