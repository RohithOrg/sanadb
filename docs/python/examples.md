# Python examples

Complete programs with the output they printed, run on 16 November 2026
with sanadb 0.2.4. Every file is in the repository under
`python/examples/` and `benchmarks/`; copy one and change the data.

## Semantic search with a sentence-transformers model

What most people want first: their own text, searched by meaning. The
model is `all-MiniLM-L6-v2` (384 dimensions, 90 MB, runs on a laptop
CPU); sanadb stores the vectors with the text and metadata, searches by
meaning, filters on metadata, fuses with keyword matching, and records
which model produced the vectors.

```bash
pip install "sanadb[studio]" sentence-transformers
python python/examples/semantic_search.py
```

```python
import os
import tempfile

import sanadb
from sentence_transformers import SentenceTransformer

MODEL = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL)

path = os.environ.get("SANADB_PATH", os.path.join(tempfile.mkdtemp(), "docs.sana"))
db = sanadb.Db(
    path,
    dim=model.get_sentence_embedding_dimension(),
    metric="cosine",
    text_field="body",
    embedding_function=lambda texts: model.encode(texts, normalize_embeddings=True),
)
db.set_property("embedding_model", MODEL)

docs = [
    ("The borrow checker rejects aliasing of mutable references", {"lang": "rust", "year": 2021}),
    ("Lifetimes tie a reference to the scope that owns the data", {"lang": "rust", "year": 2022}),
    ("asyncio runs coroutines on a single-threaded event loop", {"lang": "python", "year": 2020}),
    ("Tokio's select! cancels the branches that did not complete", {"lang": "rust", "year": 2023}),
    ("Goroutines are multiplexed onto OS threads by the Go scheduler", {"lang": "go", "year": 2019}),
    ("Garbage collection pauses are tuned with GOGC", {"lang": "go", "year": 2020}),
    ("Pattern matching on tagged unions catches missing cases at compile time", {"lang": "rust", "year": 2022}),
    ("The GIL serialises bytecode execution across threads", {"lang": "python", "year": 2018}),
]
ids = db.add([t for t, _ in docs], metas=[m for _, m in docs])
print(f"{len(ids)} documents embedded with {MODEL} into {path}")

def show(title, hits):
    print(f"\n{title}")
    for id_, score in hits:
        _, meta = db.get(id_)
        print(f"  {score:+.3f}  [{meta['lang']}]  {meta['body']}")

# Search by meaning: the query shares no words with the sentence it finds.
show("why can't I have two mutable references", db.query(["why can't I have two mutable references"], k=3)[0])
show("exhaustive match on an enum", db.query(["exhaustive match on an enum"], k=3)[0])

# Meaning plus a metadata filter, evaluated on the index columns.
show("running many tasks on one thread, Python only", db.query(["running many tasks on one thread"], k=3, filter={"lang": "python"})[0])

# Hybrid: the vector result fused with BM25 on the text, for exact terms
# that embed poorly (identifiers, flags, names).
q = model.encode(["tuning the collector"], normalize_embeddings=True)[0]
show("hybrid: 'GOGC' as a keyword with a semantic query", db.search(q, k=3, text="GOGC"))

db.checkpoint()
print(f"\nindex built: {db.index_stats()}; properties: {db.properties()}")
print("scores are negative cosine similarity: -1 identical, lower is closer")
```

Output:

<!-- output: semantic_search.py -->
```text
8 documents embedded with all-MiniLM-L6-v2 into /tmp/…/docs.sana

why can't I have two mutable references
  -0.557  [rust]  The borrow checker rejects aliasing of mutable references
  -0.257  [rust]  Lifetimes tie a reference to the scope that owns the data
  -0.184  [go]  Garbage collection pauses are tuned with GOGC

exhaustive match on an enum
  -0.358  [rust]  Pattern matching on tagged unions catches missing cases at compile time
  -0.140  [rust]  The borrow checker rejects aliasing of mutable references
  -0.125  [python]  The GIL serialises bytecode execution across threads

running many tasks on one thread, Python only
  -0.350  [python]  asyncio runs coroutines on a single-threaded event loop
  -0.334  [python]  The GIL serialises bytecode execution across threads

hybrid: 'GOGC' as a keyword with a semantic query
  -0.033  [go]  Garbage collection pauses are tuned with GOGC
  -0.016  [rust]  Lifetimes tie a reference to the scope that owns the data
  -0.016  [rust]  The borrow checker rejects aliasing of mutable references

index built: {'nodes': 8, 'pq_subspaces': 0, 'tombstones': 0, 'storage': 'f32'}; properties: {'embedding_model': 'all-MiniLM-L6-v2'}
scores are negative cosine similarity: -1 identical, lower is closer
```

Reading it: the first query shares no word with the sentence it ranks
first, which is what embeddings buy; the third query shows a filter
narrowing the same search to one language, evaluated on the index's
columns rather than after the fact; the hybrid query puts the sentence
containing the exact token `GOGC` first because BM25 and the vector
result are fused by reciprocal rank. The store now carries
`embedding_model = all-MiniLM-L6-v2` in its properties, so a reader a
year from now knows what to embed queries with.

### The same store in studio

Studio needs the model as an endpoint to search by text. This serves the
sentence-transformers model as an OpenAI-compatible `/v1/embeddings`
endpoint on localhost, model loaded once:

```python
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from sentence_transformers import SentenceTransformer

MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
PORT = int(os.environ.get("EMBED_PORT", "8089"))
model = SentenceTransformer(MODEL)
print(f"{MODEL}: {model.get_sentence_embedding_dimension()} dimensions at http://127.0.0.1:{PORT}/v1/embeddings", file=sys.stderr)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/embeddings":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
        vecs = model.encode(texts, normalize_embeddings=True)
        out = {"model": MODEL, "data": [{"index": i, "embedding": v.tolist()} for i, v in enumerate(vecs)]}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass


HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
```

```bash
SANADB_PATH=docs.sana python python/examples/semantic_search.py   # keep the store
python python/examples/embed_server.py &
sanadb-studio docs.sana --embed-url http://127.0.0.1:8089/v1/embeddings --embed-model all-MiniLM-L6-v2
```

Studio reads `embedding_model` from the store, compares it with the
attached one, and refuses to pretend they match if they differ. See
[studio](../studio.md).

## A tour of every feature

`python/examples/tour.py` runs one scenario on 384-dimensional vectors
from a stand-in embedding function (so it runs with no model installed):
open, insert with typed metadata, batch, delete, nearest, filters,
keyword, hybrid, sparse, batch queries, get, checkpoint, a read-only
handle, sync, backup, counters. [Getting started](getting-started.md)
walks through it step by step.

```python
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
```

Output:

<!-- output: tour.py -->
```text
6 records
nearest                          4(-0.260)  2(-0.176)  7(-0.049)
rust, 2022+, public              4(-0.260)
prefix                           4(-0.260)  3(-0.000)  1(0.000)
callable                         4(-0.260)  3(-0.000)
keyword 'borrow checker'         1(-2.157)  3(-1.970)
hybrid                           1(-0.032)  3(-0.032)  4(-0.016)
sparse                           3(-1.500)  1(-0.500)  2(-0.500)
hybrid sparse                    3(-0.032)  4(-0.016)  2(-0.016)
search_many ids: [[4, 2], [6, 1]]
get(3): dim 384 lang rust year 2023
after checkpoint                 4(-0.260)  2(-0.176)  7(-0.049)
reader sees 7 records
backup at /tmp/…/backup/docs.sana
stats: 9 searches, 1 checkpoints, 6 inserts
index: {'nodes': 6, 'pq_subspaces': 0, 'tombstones': 0, 'storage': 'f32'}
```

The stand-in "embedding" is a byte hash, so the nearest neighbours are
not semantic, but every mechanism is exercised: the filtered search
returns the one record that passes all three conditions; the keyword
search ranks the two sentences containing both words; the hybrid list
puts records present in both the vector and the keyword result first;
the read-only handle sees the writer's new record without being told.

## Does the index lose what the model found? SciFact with a real model

`benchmarks/beir/scifact_dense.py` encodes BEIR SciFact (5,183 documents,
300 test queries) with all-MiniLM-L6-v2, stores the vectors in sanadb
with cosine, and measures two things: recall@10 of the graph search
against brute-force cosine over the same vectors (what the index could
lose), and nDCG@10 against the human relevance labels (what the whole
pipeline retrieves), next to the published figure for this model.

```bash
pip install sanadb sentence-transformers
python benchmarks/beir/scifact_dense.py data/beir/scifact
```

```text
5183 documents and 300 queries encoded with all-MiniLM-L6-v2 (384 dims) in 16s
sanadb: inserted and indexed in 0.6s; {'nodes': 5183, 'pq_subspaces': 96, 'tombstones': 0, 'storage': 'f32'}
```

| beam | recall@10 against brute force | nDCG@10 | µs/query |
|---|---|---|---|
| exact (brute force over the model's vectors) | 1.000 | 0.645 | |
| 16 | 0.945 | 0.641 | 106 |
| 32 | 0.986 | 0.642 | 103 |
| 64 (default) | 0.996 | 0.645 | 128 |
| 128 | 0.999 | 0.645 | 164 |

The sentence-transformers model card gives 0.645 nDCG@10 for this model
on SciFact; brute force over its vectors reproduces that exactly, and the
graph at the default beam returns 99.6% of what brute force would and the
same 0.645, at about 130 µs a query on a laptop. The index is not where
quality is lost; the model is where it is decided.

## Keyword search measured against a reference

`benchmarks/beir/scifact_bm25.py` runs `search_text` on BEIR SciFact
against `rank_bm25` with the same tokens:

| | nDCG@10 | Recall@100 | ms/query |
|---|---|---|---|
| sanadb `search_text` | 0.662 | 0.886 | 1.2 |
| rank_bm25 (Okapi) | 0.654 | 0.876 | 30.6 |

Anserini's published BM25 on SciFact is 0.665 nDCG@10; the small
differences are the idf variant. Method and every other number are in
[benchmarks](../benchmarks.md).

## Plugging into a benchmark harness

`benchmarks/vectordbbench/sanadb_client.py` is a VectorDBBench client:
`insert_embeddings` becomes `insert_many` with the id stored as an `Int`
column, `optimize` is `checkpoint()`, and the harness's `id >= N` filter
cases become `{"id": {"$gte": N}}` so they run on the exact-scan or
filtered-walk regime rather than post-filtering. The ann-benchmarks
adapter under `benchmarks/ann-benchmarks/` is the same idea for that
harness; its results are in [benchmarks](../benchmarks.md).
