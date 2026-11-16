# Getting started

Two runnable tours, one per language, walk through every feature on
384-dimensional vectors: [examples/tour.rs](https://github.com/RohithOrg/sanadb/blob/master/examples/tour.rs)
(`cargo run --release --example tour`) and
[python/examples/tour.py](https://github.com/RohithOrg/sanadb/blob/master/python/examples/tour.py)
(`python python/examples/tour.py`). This page is the same walk in prose.
Any dimension works; the store takes it from `Options::new(dim)` /
`Db(path, dim=...)` and refuses vectors of another length.

## 1. Open a store

```python
db = sanadb.Db("docs.sana", dim=384, metric="cosine", text_field="body")
```

`metric` is `"l2"`, `"dot"` or `"cosine"` (cosine normalises vectors on
insert and on query). `text_field` names the metadata key whose text is
indexed for keyword search; leave it out if you do not need that. Opening
creates the files; a second writer on the same path is refused.

## 2. Write records

```python
db.insert(1, embedding, {"body": "Rust borrow checker explained", "lang": "rust",
                         "year": 2021, "score": 4.5, "public": True,
                         "sparse": ([10, 7], [1.0, 0.5])})
db.insert_many(ids, vectors_2d, metas)   # atomic: all or none
db.delete(5)
```

Metadata values are `str`, `int`, `float`, `bool`, or one key per store
holding a sparse vector as `(indices, weights)`. The first value written
under a key fixes its type. Inserting an existing id replaces the record.

## 3. Search

```python
db.search(q, k=10)                                        # nearest, lower score is closer
db.search(q, k=10, filter={"lang": "rust", "year": {"$gte": 2022}, "$not": {"public": False}})
db.search(q, k=10, filter={"lang": {"$prefix": "ru"}})
db.search(q, k=10, filter=lambda m: m.get("score", 0) > 4.5)   # any callable, slower
db.search_text("borrow checker", k=10)                    # BM25 over text_field
db.search(q, k=10, text="borrow checker")                 # hybrid: vector + keyword
db.search_sparse(([30, 7], [1.0, 1.0]), k=10)             # dot product over the sparse key
db.search(q, k=10, sparse=([30], [1.0]))                  # hybrid: vector + sparse
ids, scores = db.search_many(queries_2d, k=10)            # all cores, (n, k) arrays
```

Filter operators: `$eq $ne $in $lt $le $gt $ge $prefix $contains $exists`,
combined with `$and`, `$or`, `$not`; several keys in one dict are and-ed.
A dict filter is evaluated on the index's columns: a filter matching few
records is answered by reading them all (recall 1.0), a common one steers
the graph walk. A callable cannot be inspected and is applied to visited
nodes, widening the beam until `k` pass.

In Rust the same calls are `search`, `search_where(&q, k, &Filter::and([...]))`,
`search_filtered(&q, k, |m| ...)`, `search_text`, `search_hybrid`,
`search_sparse`, `search_hybrid_sparse`, `search_many`.

## 3b. Bring an embedding model

The store takes vectors; it does not make them. Attach the model you use
and work in text:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")          # 384 dimensions

db = sanadb.Db("docs.sana", dim=384, metric="cosine", text_field="body",
               embedding_function=lambda texts: model.encode(texts))
ids = db.add(["Rust borrow checker explained", "asyncio event loop internals"],
             metas=[{"lang": "rust"}, {"lang": "python"}])
hits = db.query(["how does borrowing work"], k=5, filter={"lang": "rust"})[0]
```

`embedding_function` is any callable from a list of strings to an
`(n, dim)` array: a sentence-transformers model, an OpenAI or Ollama
client, your own. `add` keeps each document's text under the text field
so `get` and `search_text` see it; `query` embeds and searches. In Rust
the same is the `Embedder` trait (any closure or struct), `Db::add_text`
and `Db::search_text_embedded`, with an `OpenAiEmbedder` for any
OpenAI-compatible endpoint behind the `embed-http` feature. Studio takes
the model as a flag; see [studio.md](studio.md).

## 4. Read back

```python
vec, meta = db.get(3)     # None if absent; the vector as stored (normalised for cosine)
len(db)
```

## 5. Checkpoint

```python
db.checkpoint()
```

Builds the graph index over every record and compacts the log. Everything
works before the first checkpoint (records since the last one are searched
exactly), but the graph is what makes a million-record search take
microseconds, so checkpoint after a bulk load and then on a schedule, or
set `auto_checkpoint_delta` to have it happen once the tail is large. A
checkpoint runs on all cores and does not block readers or writers.

## 6. Readers in other processes

```python
reader = sanadb.Db("docs.sana", dim=384, metric="cosine", text_field="body", read_only=True)
```

Takes no lock, sees every write and checkpoint the writer makes, from any
process on the same machine. Pass the same options as the writer.

## 7. Durability, backup, counters

```python
db.sync()                       # fsync: acknowledged writes survive power loss
db.backup("backups/2026-11-16") # consistent copy while writes continue
db.stats()                      # searches by regime, writes, checkpoints, reader refreshes
db.index_stats()                # nodes, PQ subspaces, tombstones, storage
```

Without `sync()` a write survives a process crash but not power loss;
`sync_every_write=True` fsyncs every write.

## 8. Knobs worth knowing

`search_beam` (default 64) trades latency for recall; `storage="int8"`
halves the index file; `max_degree=48` raises recall per beam at the cost
of build time. The full list is in [operations.md](operations.md).
