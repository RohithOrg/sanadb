# sanadb

An embedded vector store in Rust: a write-ahead log, a memory-mapped graph
index that keeps working when the data is larger than RAM, typed metadata
with filter-aware search, keyword and sparse search beside the vectors.
One file per store, no server. `Send + Sync`; one writer, any number of
read-only handles in any process.

```bash
cargo add sanadb
```

```rust
use std::collections::HashMap;
use sanadb::{Db, Filter, Metric, Options, Value};

let db = Db::open("vectors.sana", Options::new(768).metric(Metric::Cosine).text_field("body"))?;
db.insert(1, &embedding, HashMap::from([
    ("body".to_string(), Value::from("Tokio select cancels the losing branches")),
    ("lang".to_string(), Value::from("rust")),
    ("year".to_string(), Value::from(2024i64)),
]))?;
let hits = db.search(&query, 10)?;                                                  // Vec<Hit { id, score }>
let hits = db.search_where(&query, 10, &Filter::and([Filter::eq("lang", "rust"), Filter::ge("year", 2020i64)]))?;
let hits = db.search_hybrid(&query, "tokio select cancellation", 10, None)?;        // vector + BM25
db.checkpoint()?;                                                                   // build the graph index, compact the log
```

With an embedding model behind any OpenAI-compatible endpoint (Ollama
here), feature `embed-http`:

```rust
use sanadb::embed::OpenAiEmbedder;
let model = OpenAiEmbedder::new("http://localhost:11434/v1/embeddings", "nomic-embed-text");
db.set_property("embedding_model", "nomic-embed-text")?;
db.add_text(1, "The borrow checker rejects aliasing", meta, "body", &model)?;
let hits = db.search_text_embedded("why can't I have two mutable references", 5, None, &model)?;
```

Scores are distances, lower is closer. Any `Fn(&[&str]) -> Result<Vec<Vec<f32>>>`
is an `Embedder`, so a local model plugs in the same way.

## What it does

| | |
|---|---|
| Metrics | L2, dot product, cosine |
| Index | Vamana graph (DiskANN layout), memory-mapped, product quantization in RAM, optional int8 storage at half the file size |
| Writes | `insert`, `delete`, atomic `Batch`; a write-ahead log replayed on open; `sync()` for power-loss safety |
| Metadata | `Value::{Str, Int, Float, Bool, Sparse}` stored as columns in the index |
| Filters | `Filter::{eq, ne, is_in, lt, le, gt, ge, prefix, contains, exists, and, or, not}` or any closure; exact scan or filter-steered walk chosen per query |
| Text | BM25 over an inverted index on one text field |
| Sparse | SPLADE-style `(index, weight)` vectors, dot-product search over inverted lists |
| Hybrid | reciprocal rank fusion of the dense result with the keyword or sparse result |
| Processes | one writer (`Error::Locked` otherwise), read-only handles anywhere, following the writer live |
| Operations | consistent `backup()`, counters in `stats()`, optional `tracing` events |
| Format | frozen for 0.x and specified byte for byte; stores from every published version open and are tested |

## Numbers

One laptop (Apple Silicon, 16 GB), single query thread. SIFT1M warm:
recall@10 0.966 at 146 µs per query (beam 64), 0.995 at 405 µs (beam
256). Cold, 5M vectors from SIFT1B, 3.5 GB index evicted from the page
cache: 2.75 ms mean, 4.2 ms p99. Filtered search at 0.01% to 1%
selectivity: recall 1.0 at 9 µs to 1.3 ms. In ann-benchmarks' own
harness, ahead of hnswlib and usearch at every recall on SIFT, holding
32 bytes per vector in RAM instead of the whole index. Method and every
table: [benchmarks](https://rohithorg.github.io/sanadb/benchmarks/).

## Guarantees

A write that returned `Ok` survives a process crash; after `sync()` or
with `sync_every_write`, power loss too. A checkpoint either completes or
leaves the previous files untouched. Readers never see a partially written
record or index. A newer version opens any older store; an older version
opens a newer store or refuses with `Error::NeedsNewerVersion`, never
misreads. Crash, disk-full, failed fsync and failed rename are tested with
fault injection. Local filesystems only: the writer lock is an advisory
`flock`.

## Documentation

- [Getting started in Rust](https://rohithorg.github.io/sanadb/rust/getting-started/): every feature in one walk, with output.
- [Rust examples](https://rohithorg.github.io/sanadb/rust/examples/): semantic search with a real model, the tour, SIFT1M, filtered search, cold cache.
- [Rust recipes](https://rohithorg.github.io/sanadb/rust/recipes/): `Embedder` implementations, RAG indexing, checkpoint policy, processes, backup, failure modes.
- [API map](https://rohithorg.github.io/sanadb/rust/api/) and the full reference on [docs.rs](https://docs.rs/sanadb).
- [How it works](https://rohithorg.github.io/sanadb/architecture/), [operating a store](https://rohithorg.github.io/sanadb/operations/), [file format](https://rohithorg.github.io/sanadb/FORMAT/).
- `sanadb-studio` (`cargo install sanadb-studio`): a local web UI that maps, searches and edits a store. [Studio](https://rohithorg.github.io/sanadb/studio/).

Status: 0.x, nobody has run it in production yet; the design, the tests
and the numbers are real, the track record is not there. Python bindings
are on PyPI as `sanadb`. Licensed under MIT or Apache-2.0, at your option.
