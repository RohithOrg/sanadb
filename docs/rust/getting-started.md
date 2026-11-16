# Getting started in Rust

Every feature, in order, with the code and the output it printed. The
full program is
[`examples/tour.rs`](https://github.com/RohithOrg/sanadb/blob/master/examples/tour.rs);
the lines below are its. It uses 384-dimensional vectors from a stand-in
embedding function (a byte hash, so it runs with no model installed); any
dimension works, the store takes it from the open call and refuses
vectors of another length. For real text with a real model, go to
[examples](examples.md).

```bash
cargo add sanadb
```

Rust 1.89 or later; the crate has no required system dependencies.
Optional features: `embed-http` for an OpenAI-compatible embedding
client, `tracing` for events on target `sanadb`.

## 1. Open a store

```rust
use std::collections::HashMap;
use sanadb::{Batch, Db, Filter, Metric, Options, Value};

let db = Db::open("docs.sana", Options::new(384).metric(Metric::Cosine).text_field("body"))?;
```

`Metric` is `L2`, `Dot` or `Cosine` (cosine normalises vectors on insert
and on query). `text_field` names the metadata key whose text is indexed
for keyword search; leave it out if you do not need that. Opening creates
`docs.sana` and `docs.sana.lock`; a second writer on the same path gets
`Error::Locked`. `Db` is `Send + Sync`: share one handle across threads
behind an `Arc`.

## 2. Write records

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

<!-- output: tour.rs lines=1-1 -->
```text
6 records
```

`Metadata` is a `HashMap<String, Value>`; `Value` is `Str`, `Int`,
`Float`, `Bool` or `Sparse`, with `From` impls for `&str`, `String`,
`i64`, `i32`, `u32`, `f64`, `f32`, `bool` and `Vec<(u32, f32)>`. One key
per store may hold sparse vectors. The first value written under a key
fixes its type (`Error::TypeMismatch` afterwards). Inserting an existing
id replaces the record; a `Batch` lands entirely or not at all, and a
reader never sees part of it.

## 3. Search

```rust
let q = embed("borrow checker", dim);
db.search(&q, 3)?;                                                               // nearest
db.search_where(&q, 3, &Filter::and([Filter::eq("lang", "rust"), Filter::ge("year", 2022i64), Filter::not(Filter::eq("public", false))]))?;
db.search_where(&q, 3, &Filter::prefix("lang", "ru"))?;
db.search_filtered(&q, 3, |m| m.get("score").and_then(Value::as_float).is_some_and(|s| s > 4.5))?;  // any closure, slower
db.search_text("borrow checker", 3, None)?;                                      // BM25 on text_field
db.search_hybrid(&q, "borrow checker", 3, Some(&Filter::eq("lang", "rust")))?;   // vector + keyword
db.search_sparse(&[(30, 1.0), (7, 1.0)], 3, None)?;                              // dot product on the sparse key
db.search_hybrid_sparse(&q, &[(30, 1.0)], 3, None)?;                             // vector + sparse
db.search_many(&queries, 3)?;                                                    // row-major n × dim, all cores
```

<!-- output: tour.rs lines=2-9 -->
```text
nearest                          4(-0.260)  2(-0.176)  7(-0.049)
rust, 2022 or later, public      4(-0.260)
prefix                           4(-0.260)  3(-0.000)  1(0.000)
closure                          4(-0.260)  3(-0.000)
keyword 'borrow checker'         1(-2.157)  3(-1.970)
hybrid                           1(-0.032)  3(-0.032)  4(-0.016)
sparse                           3(-1.500)  1(-0.500)  2(-0.500)
hybrid sparse                    3(-0.032)  4(-0.016)  2(-0.016)
```

Every search returns `Vec<Hit>` with `id: u64` and `score: f32`. Scores
are distances, lower is closer: for cosine the negative cosine similarity
(−1 identical), for `L2` the squared distance, for `Dot` the negative dot
product; keyword results carry negated BM25 and hybrid results the
negated fused rank score.

`Filter` builders: `eq ne is_in lt le gt ge prefix contains exists`,
combined with `and`, `or`, `not`. A `Filter` is evaluated on the index's
columns: few matching records are read exactly (recall 1.0), many steer
the graph walk. A closure cannot be inspected, so it is applied to
visited nodes, widening the beam until `k` pass; use it for conditions a
`Filter` cannot say.

## 4. Read back

```rust
let (v, meta) = db.get(3).unwrap();
println!("get(3): dim {}, lang {}, year {}", v.len(), meta["lang"], meta["year"]);
```

<!-- output: tour.rs lines=10-10 -->
```text
get(3): dim 384, lang rust, year 2023
```

`get` returns `None` for an id that is not there. `db.ids()` lists every
live id, `db.keys()` the metadata keys with their `Kind`, and
`Db::peek(path)` the dimension and metric of a file without opening it.

## 5. Checkpoint

```rust
db.checkpoint()?;
db.search(&q, 3)?;
```

<!-- output: tour.rs lines=11-11 -->
```text
after checkpoint                 4(-0.260)  2(-0.176)  7(-0.049)
```

The checkpoint builds the graph index over every record and compacts the
log, on all cores, holding neither the read nor the write lock while it
builds. Everything works before the first checkpoint (records since the
last one are searched exactly), but the graph is what makes a
million-record search take microseconds, so checkpoint after a bulk load
and then on a schedule, or set `Options::auto_checkpoint_delta` to have it
happen once the tail is large. The same query returns the same answer
before and after.

## 6. Readers in other processes

```rust
let reader = Db::open_read_only("docs.sana", opts())?;
db.insert(8, &embed("Unsafe Rust and FFI", dim), meta8)?;
println!("reader sees {} records", reader.len());
```

<!-- output: tour.rs lines=12-12 -->
```text
reader sees 7 records
```

A read-only handle takes no lock, sees every write and checkpoint the
writer makes, from any process on the same machine, without being told.
Pass the same options as the writer. Writes on it return
`Error::ReadOnly`.

## 7. Durability, backup, counters

```rust
db.sync()?;                                   // fsync: acknowledged writes survive power loss
let copy = db.backup("backups/today")?;
let s = db.stats();
println!("stats: {} searches ({} exact scans, {} filtered walks), {} checkpoints, {} inserts",
         s.searches, s.exact_scans, s.filtered_walks, s.checkpoints, s.inserts);
if let Some(ix) = db.index_stats() {
    println!("index: {} nodes, {} PQ subspaces, storage {:?}", ix.nodes, ix.pq_subspaces, ix.storage);
}
```

<!-- output: tour.rs lines=13-15 -->
```text
backup at /tmp/…/backup/docs.sana
stats: 7 searches (0 exact scans, 0 filtered walks), 1 checkpoints, 6 inserts
index: 6 nodes, 0 PQ subspaces, storage F32
```

Without `sync()` a write survives a process crash but not power loss;
`Options::sync_every_write(true)` fsyncs every write. `backup` writes a
consistent copy while writes continue; it opens like any store.

## 8. Bring an embedding model

The store takes vectors; it does not make them. Implement `Embedder` over
whatever model you run, or use `OpenAiEmbedder` (feature `embed-http`)
against any OpenAI-compatible endpoint: Ollama, LM Studio, vLLM,
text-embeddings-inference, OpenAI. The runnable version with its output
is the first [example](examples.md).

```rust
use sanadb::embed::OpenAiEmbedder;
let model = OpenAiEmbedder::new("http://localhost:11434/v1/embeddings", "nomic-embed-text");
db.set_property("embedding_model", "nomic-embed-text")?;             // the file remembers it
db.add_text(1, "Rust borrow checker explained", meta, "body", &model)?;
let hits = db.search_text_embedded("why can't I have two mutable references", 5, None, &model)?;
```

`Embedder` has one method, `embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>`,
and any `Fn(&[&str]) -> Result<Vec<Vec<f32>>>` implements it. `add_text`
embeds, stores the vector with the metadata and the text under the text
key, and returns the vector; `search_text_embedded` embeds the query and
searches, with an optional `Filter`. Properties are small string
key–value pairs stored in the file itself: the model's name there means a
reader a year from now knows what to embed queries with, and studio reads
and checks it.

## 9. Look at it

```bash
cargo install sanadb-studio
sanadb-studio docs.sana
```

A local page with a map of the vectors, every search mode, the graph walk
of a search drawn on the map, and record editing: [studio](../studio.md).

## 10. Knobs worth knowing

`Options::search_beam` (default 64) trades latency for recall;
`Options::storage(Storage::I8)` halves the index file;
`GraphParams { max_degree: 48, .. }` raises recall per beam at the cost
of build time. Every option is in the [API](api.md) and its effect in
[operating a store](../operations.md).
