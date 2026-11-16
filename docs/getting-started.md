# Getting started

Every feature, in order, with the code in both languages and the output
it printed. The full programs are
[`python/examples/tour.py`](https://github.com/RohithOrg/sanadb/blob/master/python/examples/tour.py)
and [`examples/tour.rs`](https://github.com/RohithOrg/sanadb/blob/master/examples/tour.rs);
the lines below are theirs. They use 384-dimensional vectors from a
stand-in embedding function (a byte hash, so they run with no model
installed); any dimension works, the store takes it from the open call
and refuses vectors of another length. For real text with a real model,
see [examples.md](examples.md).

```bash
pip install sanadb          # Python
cargo add sanadb            # Rust
```

## 1. Open a store

=== "Python"

    ```python
    import numpy as np, sanadb
    db = sanadb.Db("docs.sana", dim=384, metric="cosine", text_field="body")
    ```

=== "Rust"

    ```rust
    use sanadb::{Db, Filter, Metric, Options, Value};
    let db = Db::open("docs.sana", Options::new(384).metric(Metric::Cosine).text_field("body"))?;
    ```

`metric` is `l2`, `dot` or `cosine` (cosine normalises vectors on insert
and on query). `text_field` names the metadata key whose text is indexed
for keyword search; leave it out if you do not need that. Opening creates
`docs.sana` and `docs.sana.lock`; a second writer on the same path is
refused with `Locked`.

## 2. Write records

=== "Python"

    ```python
    db.insert(1, embed("Rust borrow checker explained"),
              {"body": "Rust borrow checker explained", "lang": "rust", "year": 2021, "score": 4.5, "public": True,
               "sparse": ([10, 7], [1.0, 0.5])})
    db.insert_many([6, 7], np.stack([embed(t) for t in more]), [{"body": t, "lang": l} for t, l in ...])  # atomic
    db.delete(5)
    print(len(db), "records")
    ```

=== "Rust"

    ```rust
    db.insert(1, &embed("Rust borrow checker explained", dim), HashMap::from([
        ("body".to_string(), Value::from("Rust borrow checker explained")),
        ("lang".to_string(), Value::from("rust")),
        ("year".to_string(), Value::from(2021i64)),
        ("score".to_string(), Value::from(4.5f64)),
        ("public".to_string(), Value::from(true)),
        ("sparse".to_string(), Value::Sparse(vec![(10, 1.0), (7, 0.5)])),
    ]))?;
    db.write_batch(Batch::new().insert(6, &v6, m6).insert(7, &v7, m7).delete(5))?;   // all or nothing
    println!("{} records", db.len());
    ```

```text
6 records
```

Metadata values are strings, integers, floats, booleans, and one key per
store may hold a sparse vector as `(indices, weights)`. The first value
written under a key fixes its type. Inserting an existing id replaces
the record; a batch lands entirely or not at all, and a reader never
sees part of it.

## 3. Search

=== "Python"

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

=== "Rust"

    ```rust
    let q = embed("borrow checker", dim);
    db.search(&q, 3)?;
    db.search_where(&q, 3, &Filter::and([Filter::eq("lang", "rust"), Filter::ge("year", 2022i64), Filter::not(Filter::eq("public", false))]))?;
    db.search_where(&q, 3, &Filter::prefix("lang", "ru"))?;
    db.search_filtered(&q, 3, |m| m.get("score").and_then(Value::as_float).is_some_and(|s| s > 4.5))?;
    db.search_text("borrow checker", 3, None)?;
    db.search_hybrid(&q, "borrow checker", 3, Some(&Filter::eq("lang", "rust")))?;
    db.search_sparse(&[(30, 1.0), (7, 1.0)], 3, None)?;
    db.search_hybrid_sparse(&q, &[(30, 1.0)], 3, None)?;
    db.search_many(&queries, 3)?;                       // row-major n × dim, all cores
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

Each line is `id(score)`. Scores are distances, lower is closer: for
cosine the negative cosine similarity (−1 identical), for `l2` the
squared distance, for `dot` the negative dot product; keyword results
carry negated BM25 and hybrid results the negated fused rank score.
Filter operators: `$eq $ne $in $lt $le $gt $ge $prefix $contains
$exists`, combined with `$and`, `$or`, `$not`; several keys in one dict
are and-ed. A dict filter is evaluated on the index's columns: few
matching records are read exactly (recall 1.0), many steer the graph
walk. A callable cannot be inspected and is applied to visited nodes,
widening the beam until `k` pass.

## 4. Read back

=== "Python"

    ```python
    vec, meta = db.get(3)
    print("get(3): dim", vec.shape[0], "lang", meta["lang"], "year", meta["year"])
    ```

=== "Rust"

    ```rust
    let (v, meta) = db.get(3).unwrap();
    println!("get(3): dim {}, lang {}, year {}", v.len(), meta["lang"], meta["year"]);
    ```

```text
get(3): dim 384 lang rust year 2023
```

## 5. Checkpoint

=== "Python"

    ```python
    db.checkpoint()
    db.search(q, k=3)
    ```

=== "Rust"

    ```rust
    db.checkpoint()?;
    db.search(&q, 3)?;
    ```

```text
after checkpoint                 4(-0.260)  2(-0.176)  7(-0.049)
```

The checkpoint builds the graph index over every record and compacts the
log, on all cores, without blocking readers or writers. Everything works
before the first checkpoint (records since the last one are searched
exactly), but the graph is what makes a million-record search take
microseconds, so checkpoint after a bulk load and then on a schedule, or
set `auto_checkpoint_delta` to have it happen once the tail is large. The
same query returns the same answer before and after.

## 6. Readers in other processes

=== "Python"

    ```python
    reader = sanadb.Db("docs.sana", dim=384, metric="cosine", text_field="body", read_only=True)
    db.insert(8, embed("Unsafe Rust and FFI"), {"body": "Unsafe Rust and FFI", "lang": "rust"})
    print("reader sees", len(reader), "records")
    ```

=== "Rust"

    ```rust
    let reader = Db::open_read_only("docs.sana", opts())?;
    db.insert(8, &embed("Unsafe Rust and FFI", dim), meta8)?;
    println!("reader sees {} records", reader.len());
    ```

```text
reader sees 7 records
```

A read-only handle takes no lock, sees every write and checkpoint the
writer makes, from any process on the same machine, without being told.
Pass the same options as the writer.

## 7. Durability, backup, counters

=== "Python"

    ```python
    db.sync()                                   # fsync: acknowledged writes survive power loss
    print("backup at", db.backup("backups/today"))
    s = db.stats()
    print(f"stats: {s['searches']} searches, {s['checkpoints']} checkpoints, {s['inserts']} inserts")
    print("index:", db.index_stats())
    ```

=== "Rust"

    ```rust
    db.sync()?;
    let copy = db.backup("backups/today")?;
    let s = db.stats();
    println!("stats: {} searches ({} exact scans, {} filtered walks), {} checkpoints, {} inserts", s.searches, s.exact_scans, s.filtered_walks, s.checkpoints, s.inserts);
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
and work in text. The runnable, real-model version is the first example
in [examples.md](examples.md), output included.

=== "Python"

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

=== "Rust"

    ```rust
    // Any closure or struct implementing `Embedder`; this one calls an
    // OpenAI-compatible endpoint (Ollama here) behind the `embed-http` feature.
    use sanadb::embed::OpenAiEmbedder;
    let model = OpenAiEmbedder::new("http://localhost:11434/v1/embeddings", "nomic-embed-text");
    db.set_property("embedding_model", "nomic-embed-text")?;
    db.add_text(1, "Rust borrow checker explained", meta, "body", &model)?;
    let hits = db.search_text_embedded("why can't I have two mutable references", 5, None, &model)?;
    ```

`embedding_function` is any callable from a list of strings to an
`(n, dim)` array; `add` keeps each document's text under the text field
so `get` and `search_text` see it; `query` embeds and searches. The
store records the model's name as a property so a reader a year from now
knows what to embed queries with; studio reads and checks it.

## 9. Look at it

```bash
pip install "sanadb[studio]"
sanadb-studio docs.sana
```

A local page with a map of the vectors, every search mode, the graph walk
of a search drawn on the map, and record editing: [studio.md](studio.md).

## 10. Knobs worth knowing

`search_beam` (default 64) trades latency for recall; `storage="int8"`
halves the index file; `max_degree=48` raises recall per beam at the cost
of build time. The full list is in [operations.md](operations.md).
