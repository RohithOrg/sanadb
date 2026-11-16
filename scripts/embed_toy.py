"""A stand-in embedding command for studio screenshots and smoke tests:
a JSON array of strings on stdin, a JSON array of float vectors on stdout,
bag-of-words hashed into EMBED_DIM (default 128) buckets. Not a model."""
import hashlib
import json
import os
import sys

dim = int(os.environ.get("EMBED_DIM", "128"))
out = []
for text in json.load(sys.stdin):
    v = [0.0] * dim
    for w in text.lower().split():
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    out.append(v)
json.dump(out, sys.stdout)
