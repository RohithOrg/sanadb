# Failure modes

What goes wrong, what the library says, and what to do. Everything on
this page was produced by running
[`python/examples/failure_modes.py`](https://github.com/RohithOrg/sanadb/blob/master/python/examples/failure_modes.py)
against a throwaway store on 16 November 2026 with sanadb 0.2.4; the
Rust errors are the same cases under their `Error` variant names. The
guarantees these cases rest on are in [operating a store](operations.md).

## The lock is held

One writer per store. A second `Db` on the same path fails at open; a
read-only handle opens regardless and follows the writer. When the
holding process dies, the operating system releases its `flock` and the
next writer opens. There is no stale-lock file to delete by hand: the
`.lock` file is always present and empty, and only the kernel knows
whether it is held.

```text
RuntimeError: store is open elsewhere (lock file held)      # Python
Error::Locked                                               # Rust
```

On NFS, SMB and other network mounts the lock is not reliable, two
machines can both believe they hold it, and the result is an
interleaved log. Keep stores on local disks.

## A torn log

A crash or power cut while a frame is being appended leaves the log
ending in a partial frame. Every frame carries its length and a CRC;
on open the reader replays frames until one does not check out, trims
the file there, and continues. The records before the tear are intact;
the one being written is gone, which is the write that had not returned
yet. A batch is wrapped in `Begin` and `Commit` frames and is dropped
whole if the `Commit` is missing. Nothing is reported: this is the
normal path, exercised by the test suite with a half frame of junk
appended and by the fault injection of a failed append.

Below, the log is trimmed by 7 bytes in the middle of the last Insert
frame; reopening finds 7 of 8 records and the file 52 bytes shorter, the
length of the torn frame.

## A file from a newer sanadb

Byte 13 of the log header is the reader level the file needs (3 since
0.2.0). A reader that supports a lower level stops with
`NeedsNewerVersion` and does not read on, so an old binary never
misinterprets a new layout. Additive changes, such as the Property frame
in 0.2.1, do not raise the level; a 0.2.0 reader skips frames it does
not know. The fix is to upgrade. An older file is never a problem: a
newer sanadb opens every store an older one wrote, converting on first
open when the layout changed, and the stores of every published version
are fixtures in the test suite.

## A damaged index

The index file's header, tables and footer are checked on open: magic,
dimension and metric against the store's, reader level, the geometry
that every offset is computed from, and the id table. Damage in those
parts fails the open with `Corrupt` naming the check, as below. Damage
in the body (a vector or an adjacency list) is not checksummed, since the
file is read in place through the mapping, so a flipped byte there
changes one record's vector or one node's neighbours and nothing else;
the test suite flips random bits across 400 copies of an index and
requires open, search and get to return a result or an error, never
panic. A damaged index is recovered from a `backup`, or, when the log
still holds every record (it does until the first checkpoint after the
damage), by deleting the index file and checkpointing again from the log.

## Wrong dimension, type or value

The dimension is fixed when the file is created; `Db::open` with another
`dim` or metric fails with `HeaderMismatch`, and a vector of another
length with `DimMismatch` (`ValueError` in Python). A metadata key's type
is fixed by its first value (`TypeMismatch`, `TypeError`). A vector with
NaN or infinity, or all zeros under cosine, is `InvalidVector`. Two keys
holding sparse vectors is `SecondSparseKey`. None of these write
anything.

## The program and its output

```python
import glob
import os
import shutil
import tempfile

import numpy as np
import sanadb

d = tempfile.mkdtemp()
p = os.path.join(d, "docs.sana")


def new_store():
    db = sanadb.Db(p, dim=4)
    for i in range(5):
        db.insert(i, np.full(4, i, dtype="f4"), {"i": i})
    db.checkpoint()                       # records 0..4 in the index
    for i in range(5, 8):
        db.insert(i, np.full(4, i, dtype="f4"), {"i": i})   # 5..7 in the tail
    db.sync()
    return db


db = new_store()
print("1. the lock is held: a second writer on the same path")
try:
    sanadb.Db(p, dim=4)
except Exception as e:
    print(f"   {type(e).__name__}: {e}")
reader = sanadb.Db(p, dim=4, read_only=True)
print(f"   read_only=True opens anyway: {len(reader)} records")
del db, reader
db = sanadb.Db(p, dim=4)
print(f"   after the first handle is dropped a new writer opens: {len(db)} records")
del db                                     # a clean close trims the preallocated log

size = os.path.getsize(p)
print(f"2. a torn log: {size} bytes on disk (16-byte header, Snapshot, 3 Insert frames); the crash cut 7 bytes off the last frame")
with open(p, "r+b") as f:
    f.truncate(size - 7)
db = sanadb.Db(p, dim=4)
print(f"   reopened: {len(db)} records, {db.delta_len} in the tail; the log was trimmed to {os.path.getsize(p)} bytes")
db.insert(7, np.full(4, 7, dtype="f4"), {"i": 7})
print(f"   the lost record written again: {len(db)} records")
del db

print("3. a file from a newer sanadb: header byte 13 is the reader level the file needs")
q = os.path.join(d, "future.sana")
shutil.copy(p, q)
with open(q, "r+b") as f:
    print(f"   header: {f.read(16)}")
    f.seek(13)
    f.write(bytes([9]))
try:
    sanadb.Db(q, dim=4)
except Exception as e:
    print(f"   {type(e).__name__}: {e}")

print("4. a damaged index: one byte of the index file's header flipped (the out-degree field)")
ip = glob.glob(os.path.join(d, "docs.sana.idx.*"))[0]
with open(ip, "r+b") as f:
    f.seek(20)
    b = f.read(1)
    f.seek(20)
    f.write(bytes([b[0] ^ 0xFF]))
try:
    db = sanadb.Db(p, dim=4)
    print(f"   opened: {len(db)} records")
except Exception as e:
    print(f"   {type(e).__name__}: {e}")

print("5. a vector of the wrong size")
try:
    sanadb.Db(os.path.join(d, "other.sana"), dim=4).insert(1, np.zeros(5, dtype="f4"))
except Exception as e:
    print(f"   {type(e).__name__}: {e}")
```

<!-- output: failure_modes.py -->
```text
1. the lock is held: a second writer on the same path
   RuntimeError: store is open elsewhere (lock file held)
   read_only=True opens anyway: 8 records
   after the first handle is dropped a new writer opens: 8 records
2. a torn log: 192 bytes on disk (16-byte header, Snapshot, 3 Insert frames); the crash cut 7 bytes off the last frame
   reopened: 7 records, 2 in the tail; the log was trimmed to 140 bytes
   the lost record written again: 8 records
3. a file from a newer sanadb: header byte 13 is the reader level the file needs
   header: b'SANADB02\x04\x00\x00\x00\x00\x03\x00\x00'
   RuntimeError: file needs reader level 9, this sanadb supports 3: upgrade sanadb
4. a damaged index: one byte of the index file's header flipped (the out-degree field)
   RuntimeError: corrupt file: index geometry inconsistent
5. a vector of the wrong size
   ValueError: vector has 5 dims, store expects 4
```

## Slow rather than broken

| symptom | cause | fix |
|---|---|---|
| searches get slower over hours of writes | the tail since the last checkpoint is scanned exactly on every query | `checkpoint()`, or `auto_checkpoint_delta` |
| the first queries after a restart take milliseconds | the index is memory-mapped and cold | expected; `io_width` 4 to 8 overlaps the reads, numbers in [benchmarks](benchmarks.md#cold-cache-5m-vectors-from-sift1b) |
| recall lower than the benchmarks | beam too small for the data, or a filter answered by a short walk | raise `search_beam`; for filters, a dict/`Filter` instead of a closure |
| a checkpoint takes minutes | it builds the graph over every record on all cores | expected at millions of records; run it off the request path |
| a read-only handle lags | it catches up at the start of each read, not continuously | call `refresh()` or just read |
