"""What the failure modes look like, done to a throwaway store: the lock
held by another handle, a log whose last frame was cut short (a crash
mid-write), a file written by a newer sanadb, a damaged index header, and
a vector of the wrong size. Runs in a temp directory; prints what the
library says in each case.

    python python/examples/failure_modes.py
"""
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
