# Getting started in Python

Every feature, in order, with the code and the output it printed. The
full program is
[`python/examples/tour.py`](https://github.com/RohithOrg/sanadb/blob/master/python/examples/tour.py);
the lines below are its. It uses 384-dimensional vectors from a stand-in
embedding function (a byte hash, so it runs with no model installed); any
dimension works, the store takes it from the open call and refuses
vectors of another length. For real text with a real model, go to
[examples](examples.md).

```bash
pip install sanadb
```

Wheels exist for Linux (glibc and musl, x86-64 and arm64), macOS (Intel
and Apple Silicon) and Windows, Python 3.9 and later. The only
dependency is numpy.

## 1. Open a store

```python
import numpy as np, sanadb
db = sanadb.Db("docs.sana", dim=384, metric="cosine", text_field="body")
```

`metric` is `"l2"`, `"dot"` or `"cosine"` (cosine normalises vectors on
insert and on query). `text_field` names the metadata key whose text is
indexed for keyword search; leave it out if you do not need that. Opening
creates `docs.sana` and `docs.sana.lock`; a second writer on the same
path raises `RuntimeError: store is open elsewhere (lock file held)`.

## 2. Write records

```python
db.insert(1, embed("Rust borrow checker explained"),
          {"body": "Rust borrow checker explained", "lang": "rust", "year": 2021,
           "score": 4.5, "public": True, "sparse": ([10, 7], [1.0, 0.5])})
more = ["Lifetimes in Rust structs", "Pattern matching in Elixir"]
db.insert_many([6, 7], np.stack([embed(t) for t in more]),
               [{"body": t, "lang": l} for t, l in zip(more, ["rust", "elixir"])])  # atomic
db.delete(5)
print(len(db), "records")
```

```text
6 records
```

Metadata values are strings, integers, floats, booleans, and one key per
store may hold a sparse vector as `(indices, weights)`. The first value
written under a key fixes its type. Inserting an existing id replaces the
record; `insert_many` lands entirely or not at all, and a reader never
sees part of it. Vectors are anything numpy can turn into a float32
array of the right length.

## 3. Search

```python
q = embed("borrow checker")
db.search(q, k=3)                                                       # nearest
db.search(q, k=3, filter={"lang": "rust", "year": {"$gte": 2022}, "$not": {"public": False}})
db.search(q, k=3, filter={"lang": {"$prefix": "ru"}})
db.search(q, k=3, filter=lambda m: m.get("score", 0) > 4.5)             # any callable, slower
db.search_text("borrow checker", k=3)                                   # BM25 on text_field
db.search(q, k=3, text="borrow checker", filter={"lang": "rust"})       # hybrid: vector + keyword
db.search_sparse(([30, 7], [1.0, 1.0]), k=3)                            # dot product on the sparse key
db.search(q, k=3, sparse=([30], [1.0]))                                 # hybrid: vector + sparse
ids, scores = db.search_many(np.stack([q, embed("asyncio")]), k=2)      # all cores, (n, k) arrays
```

```text
nearest                          4(-0.260)  2(-0.176)  7(-0.049)
rust, 2022+, public              4(-0.260)
prefix                           4(-0.260)  3(-0.000)  1(0.000)
callable                         4(-0.260)  3(-0.000)
keyword 'borrow checker'         1(-2.157)  3(-1.970)
hybrid                           1(-0.032)  3(-0.032)  4(-0.016)
sparse                           3(-1.500)  1(-0.500)  2(-0.500)
hybrid sparse                    3(-0.032)  4(-0.016)  2(-0.016)
search_many ids: [[4, 2], [6, 1]]
```

Each search returns a list of `(id, score)` tuples. Scores are
distances, lower is closer: for cosine the negative cosine similarity (−1
identical), for `l2` the squared distance, for `dot` the negative dot
product; keyword results carry negated BM25 and hybrid results the
negated fused rank score.

Filter operators: `$eq $ne $in $lt $le $gt $ge $prefix $contains
$exists`, combined with `$and`, `$or`, `$not`; several keys in one dict
are and-ed, and a bare value means `$eq`. A dict filter is evaluated on
the index's columns: few matching records are read exactly (recall 1.0),
many steer the graph walk. A callable cannot be inspected, so it is
applied to visited nodes with the GIL held, widening the beam until `k`
pass; use it for conditions a dict cannot say.

## 4. Read back

```python
vec, meta = db.get(3)
print("get(3): dim", vec.shape[0], "lang", meta["lang"], "year", meta["year"])
```

```text
get(3): dim 384 lang rust year 2023
```

`get` returns `None` for an id that is not there; `len(db)` counts live
records and `db.delta_len` those written since the last checkpoint.

## 5. Checkpoint

```python
db.checkpoint()
db.search(q, k=3)
```

```text
after checkpoint                 4(-0.260)  2(-0.176)  7(-0.049)
```

The checkpoint builds the graph index over every record and compacts the
log, on all cores with the GIL released, without blocking readers.
Everything works before the first checkpoint (records since the last one
are searched exactly), but the graph is what makes a million-record
search take microseconds, so checkpoint after a bulk load and then on a
schedule. The same query returns the same answer before and after.

## 6. Readers in other processes

```python
reader = sanadb.Db("docs.sana", dim=384, metric="cosine", text_field="body", read_only=True)
db.insert(8, embed("Unsafe Rust and FFI"), {"body": "Unsafe Rust and FFI", "lang": "rust"})
print("reader sees", len(reader), "records")
```

```text
reader sees 7 records
```

A read-only handle takes no lock, sees every write and checkpoint the
writer makes, from any process on the same machine, without being told.
Pass the same options as the writer. This is how a web worker pool reads
a store one indexing process writes.

## 7. Durability, backup, counters

```python
db.sync()                                   # fsync: acknowledged writes survive power loss
print("backup at", db.backup("backups/today"))
s = db.stats()
print(f"stats: {s['searches']} searches, {s['checkpoints']} checkpoints, {s['inserts']} inserts")
print("index:", db.index_stats())
```

```text
backup at /tmp/…/backup/docs.sana
stats: 9 searches, 1 checkpoints, 6 inserts
index: {'nodes': 6, 'pq_subspaces': 0, 'tombstones': 0, 'storage': 'f32'}
```

Without `sync()` a write survives a process crash but not power loss;
`sync_every_write=True` fsyncs every write. `backup` writes a consistent
copy while writes continue; it opens like any store.

## 8. Bring an embedding model

The store takes vectors; it does not make them. Attach the model you use
and work in text, the way Chroma's embedding functions work. The
runnable, real-model version with its output is the first
[example](examples.md).

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")                      # 384 dimensions
db = sanadb.Db("docs.sana", dim=384, metric="cosine", text_field="body",
               embedding_function=lambda texts: model.encode(texts, normalize_embeddings=True))
db.set_property("embedding_model", "all-MiniLM-L6-v2")              # the file remembers it
ids = db.add(["Rust borrow checker explained", "asyncio event loop internals"],
             metas=[{"lang": "rust"}, {"lang": "python"}])
hits = db.query(["why can't I have two mutable references"], k=5, filter={"lang": "rust"})[0]
```

`embedding_function` is any callable from a list of strings to an
`(n, dim)` array; `add` keeps each document's text under the text field
so `get` and `search_text` see it and returns the ids it assigned (or
takes `ids=`); `query` embeds and searches, one result list per text.
Properties are small string key–value pairs stored in the file itself:
the model's name there means a reader a year from now knows what to
embed queries with, and studio reads and checks it.

## 9. Look at it

```bash
pip install "sanadb[studio]"
sanadb-studio docs.sana
```

A local page with a map of the vectors, every search mode, the graph walk
of a search drawn on the map, and record editing: [studio](../studio.md).

## 10. Knobs worth knowing

`search_beam` (default 64) trades latency for recall; `storage="int8"`
halves the index file; `max_degree=48` raises recall per beam at the cost
of build time. All constructor arguments are in the [API](api.md) and
their effects in [operating a store](../operations.md).
