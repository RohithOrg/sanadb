"""A tour of the Python API on embeddings of any dimension: open a store,
write records with typed metadata, search plain, filtered, by keyword, by
sparse vector and hybrid, batch queries, read from a second handle, back
up, look at the counters.

    pip install sanadb numpy
    python python/examples/tour.py
"""
import tempfile
import os

import numpy as np
import sanadb

DIM = 384  # whatever your embedding model produces


def embed(text: str) -> np.ndarray:
    """Stand-in for an embedding model; deterministic so the output is stable."""
    v = np.zeros(DIM, dtype=np.float32)
    for i, b in enumerate(text.encode()):
        v[(i * 7 + b) % DIM] += 1.0
    return v


def show(what, hits):
    print(f"{what:<32} " + "  ".join(f"{i}({s:.3f})" for i, s in hits))


tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "docs.sana")

# 1. Open. Cosine normalises on insert and query; text_field names the key
#    indexed for BM25 keyword search.
db = sanadb.Db(path, dim=DIM, metric="cosine", text_field="body")

# 2. Records: id, vector, metadata dict with str/int/float/bool values, and
#    optionally one key holding a sparse vector as (indices, weights).
docs = [
    (1, "Rust borrow checker explained", "rust", 2021, 4.5, True),
    (2, "Python asyncio event loop internals", "python", 2020, 4.1, True),
    (3, "The borrow checker rejects aliasing", "rust", 2023, 4.8, False),
    (4, "Tokio select and cancellation safety", "rust", 2022, 4.6, True),
    (5, "Garbage collection in Go", "go", 2019, 3.9, True),
]
for id_, body, lang, year, score, public in docs:
    db.insert(id_, embed(body), {
        "body": body, "lang": lang, "year": year, "score": score, "public": public,
        "sparse": ([id_ * 10, 7], [1.0, 0.5]),
    })
# Many at once, atomically.
more = ["Lifetimes in Rust structs", "Pattern matching in Elixir"]
db.insert_many([6, 7], np.stack([embed(t) for t in more]),
               [{"body": t, "lang": l} for t, l in zip(more, ["rust", "elixir"])])
db.delete(5)
print(len(db), "records")

# 3. Nearest neighbours; scores are distances, lower is closer.
q = embed("borrow checker")
show("nearest", db.search(q, k=3))

# 4. Filters: a dict with operators, evaluated on the index columns; or any
#    callable on the metadata dict (slower, GIL held).
show("rust, 2022+, public", db.search(q, k=3, filter={
    "lang": "rust", "year": {"$gte": 2022}, "$not": {"public": False}}))
show("prefix", db.search(q, k=3, filter={"lang": {"$prefix": "ru"}}))
show("callable", db.search(q, k=3, filter=lambda m: m.get("score", 0) > 4.5))

# 5. Keyword (BM25) and hybrid of vector + keyword by reciprocal rank.
show("keyword 'borrow checker'", db.search_text("borrow checker", k=3))
show("hybrid", db.search(q, k=3, text="borrow checker", filter={"lang": "rust"}))

# 6. Sparse vectors: dot product over the sparse key, and hybrid with dense.
show("sparse", db.search_sparse(([30, 7], [1.0, 1.0]), k=3))
show("hybrid sparse", db.search(q, k=3, sparse=([30], [1.0])))

# 7. Many queries at once on all cores: (n, k) arrays of ids and scores.
ids, scores = db.search_many(np.stack([q, embed("asyncio")]), k=2)
print("search_many ids:", ids.tolist())

# 8. Read a record back.
vec, meta = db.get(3)
print("get(3): dim", vec.shape[0], "lang", meta["lang"], "year", meta["year"])

# 9. Checkpoint: build the graph index, compact the log. Searches work
#    before (exact scan of the tail) and after (graph walk).
db.checkpoint()
show("after checkpoint", db.search(q, k=3))

# 10. A read-only handle in this or any other process follows the writer.
reader = sanadb.Db(path, dim=DIM, metric="cosine", text_field="body", read_only=True)
db.insert(8, embed("Unsafe Rust and FFI"), {"body": "Unsafe Rust and FFI", "lang": "rust"})
print("reader sees", len(reader), "records")

# 11. Durability, backup, counters.
db.sync()
print("backup at", db.backup(os.path.join(tmp, "backup")))
s = db.stats()
print(f"stats: {s['searches']} searches, {s['checkpoints']} checkpoints, {s['inserts']} inserts")
print("index:", db.index_stats())
