# sanadb

An embedded vector store for Python, written in Rust: a write-ahead log, a
memory-mapped graph index that keeps working when the data is larger than
RAM, typed metadata with filter-aware search, keyword and sparse search
beside the vectors. One file per store, no server, numpy the only
dependency. Wheels for Linux (glibc, musl), macOS and Windows, Python
3.9 and later.

```bash
pip install sanadb
pip install "sanadb[studio]"     # plus the local web UI
```

```python
import numpy as np, sanadb

db = sanadb.Db("vectors.sana", dim=768, metric="cosine", text_field="body")
db.insert_many(ids, np.asarray(vectors, dtype=np.float32), metas)
hits = db.search(query, k=10)                                                   # [(id, score), ...]
hits = db.search(query, k=10, filter={"year": {"$gte": 2020}, "lang": {"$in": ["en", "de"]}})
hits = db.search(query, k=10, text="tokio select cancellation")                 # hybrid: vector + BM25
db.checkpoint()                                                                 # build the graph index, compact the log
```

With your own embedding model, the way Chroma's embedding functions work:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
db = sanadb.Db("docs.sana", dim=384, metric="cosine", text_field="body",
               embedding_function=lambda t: model.encode(t, normalize_embeddings=True))
db.set_property("embedding_model", "all-MiniLM-L6-v2")        # the file remembers what made the vectors
db.add(["The borrow checker rejects aliasing", "asyncio runs coroutines on one thread"],
       metas=[{"lang": "rust"}, {"lang": "python"}])
db.query(["why can't I have two mutable references"], k=3, filter={"lang": "rust"})
```

Scores are distances, lower is closer. The package ships type stubs, so
an editor shows every signature.

## What it does

| | |
|---|---|
| Metrics | `"l2"`, `"dot"`, `"cosine"` |
| Index | Vamana graph (DiskANN layout), memory-mapped, product quantization in RAM, `storage="int8"` at half the file size |
| Writes | `insert`, `insert_many` (atomic), `delete`, `delete_many`; a write-ahead log replayed on open; `sync()` for power-loss safety |
| Metadata | `str`, `int`, `float`, `bool` and one sparse key, stored as columns in the index |
| Filters | `$eq $ne $in $lt $le $gt $ge $prefix $contains $exists` with `$and $or $not`, or any callable; exact scan or filter-steered walk chosen per query |
| Text | BM25 over an inverted index on `text_field` (`search_text`) |
| Sparse | SPLADE-style `(indices, weights)` vectors (`search_sparse`) |
| Hybrid | `search(..., text=)` and `search(..., sparse=)` fuse by reciprocal rank |
| Embeddings | `embedding_function=` with `add()` and `query()`; `set_property("embedding_model", ...)` kept in the file |
| Processes | one writer, any number of `read_only=True` handles in any process, following the writer live; the GIL is released on the hot paths |
| Operations | `backup()`, counters in `stats()`, `search_many` on all cores, `auto_checkpoint_delta=` |
| Format | frozen for 0.x and specified byte for byte; stores from every published version open and are tested |

## Numbers

One laptop (Apple Silicon, 16 GB), single query thread. SIFT1M warm:
recall@10 0.966 at 146 µs per query (beam 64), 0.995 at 405 µs (beam
256). Cold, 5M vectors from SIFT1B, 3.5 GB index evicted from the page
cache: 2.75 ms mean, 4.2 ms p99. Filtered search at 0.01% to 1%
selectivity: recall 1.0 at 9 µs to 1.3 ms. With a real model (BEIR
SciFact, all-MiniLM-L6-v2) the graph returns 99.6% of brute force's top
10 and the same nDCG@10 of 0.645. Method and every table:
[benchmarks](https://rohithorg.github.io/sanadb/benchmarks/).

## Guarantees

A write that returned survives a process crash; after `sync()` or with
`sync_every_write=True`, power loss too. A checkpoint either completes or
leaves the previous files untouched. Readers never see a partially written
record or index. A newer version opens any older store; an older version
opens a newer store or refuses, never misreads. Crash, disk-full, failed
fsync and failed rename are tested with fault injection. Local
filesystems only: the writer lock is an advisory `flock`.

## Documentation

- [Getting started in Python](https://rohithorg.github.io/sanadb/python/getting-started/): every feature in one walk, with output.
- [Python examples](https://rohithorg.github.io/sanadb/python/examples/): semantic search with sentence-transformers, the tour, SciFact with a real model, keyword search against a reference.
- [Python recipes](https://rohithorg.github.io/sanadb/python/recipes/): RAG indexing, checkpoint policy, processes, backup, moving from Chroma, failure modes.
- [Python API](https://rohithorg.github.io/sanadb/python/api/): every method, from the shipped stubs.
- [Studio](https://rohithorg.github.io/sanadb/studio/): `sanadb-studio docs.sana` maps, searches and edits a store in a local page.
- [How it works](https://rohithorg.github.io/sanadb/architecture/), [operating a store](https://rohithorg.github.io/sanadb/operations/), [file format](https://rohithorg.github.io/sanadb/FORMAT/).

Status: 0.x, nobody has run it in production yet; the design, the tests
and the numbers are real, the track record is not there. The Rust crate
is on crates.io as `sanadb`. Licensed under MIT or Apache-2.0, at your
option.
