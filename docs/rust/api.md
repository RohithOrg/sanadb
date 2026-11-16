# Rust API

The complete reference with every doc comment is on
[docs.rs/sanadb](https://docs.rs/sanadb). This page is the map: what the
types are and which method does what.

```rust
use sanadb::{Batch, Db, Embedder, Error, Filter, GraphParams, Hit, IndexStats,
             Kind, Metadata, Metric, Options, Result, Stats, Storage, Value};
```

## Opening

| | |
|---|---|
| `Options::new(dim)` | builder; `.metric(Metric)` `.text_field(key)` `.search_beam(n)` `.max_search_beam(n)` `.io_width(n)` `.exact_filter_scan(n)` `.storage(Storage)` `.graph(GraphParams)` `.sync_every_write(bool)` `.auto_checkpoint_delta(n)` `.auto_checkpoint_garbage(n)` |
| `Metric` | `L2`, `Dot`, `Cosine` |
| `Storage` | `F32`, `I8` (indexed vectors at 8 bits, half the file) |
| `GraphParams` | `max_degree` 32, `build_beam` 100, `alpha` 1.2, `use_pq` true, `storage`; `..Default::default()` |
| `Db::open(path, opts)` | the writer; `Error::Locked` if another writer holds the path |
| `Db::open_read_only(path, opts)` | a follower in this or any process; never locks, never writes |
| `Db::peek(path)` | `(dim, Metric)` of a file without opening it |

## Writing

| | |
|---|---|
| `insert(id, &[f32], Metadata)` | insert or replace one record |
| `delete(id) -> bool` | whether it existed |
| `write_batch(Batch)` | `Batch::new().insert(..).delete(..)`, atomic |
| `add_text(id, text, meta, text_key, &impl Embedder) -> Vec<f32>` | embed and insert, text kept under `text_key` |
| `set_property(key, value)` / `property(key)` / `properties()` | strings kept with the store; empty value removes |
| `checkpoint()` | build the graph index, compact the log |
| `sync()` | fsync the log |
| `backup(dir) -> PathBuf` | consistent copy while writes continue |

`Metadata = HashMap<String, Value>`. `Value::{Str, Int, Float, Bool, Sparse}`
with `From` for `&str`, `String`, `i64`, `i32`, `u32`, `f64`, `f32`,
`bool`, `Vec<(u32, f32)>`; accessors `as_str`, `as_int`, `as_float`,
`as_bool`, `as_sparse`, `kind()`.

## Searching

All return `Result<Vec<Hit>>` (`Hit { id: u64, score: f32 }`, lower is
closer) unless noted.

| | |
|---|---|
| `search(&q, k)` | nearest neighbours |
| `search_where(&q, k, &Filter)` | filter evaluated on the index's columns: exact scan or filter-steered walk |
| `search_filtered(&q, k, Fn(&Metadata) -> bool)` | any closure; beam widens until `k` pass |
| `search_text(text, k, Option<&Filter>)` | BM25 over `text_field` |
| `search_hybrid(&q, text, k, Option<&Filter>)` | dense and keyword fused by reciprocal rank |
| `search_sparse(&[(u32, f32)], k, Option<&Filter>)` | dot product over the sparse key |
| `search_hybrid_sparse(&q, &[(u32, f32)], k, Option<&Filter>)` | dense and sparse fused |
| `search_text_embedded(text, k, Option<&Filter>, &impl Embedder)` | embed then `search` / `search_where` |
| `search_many(&queries, k)` / `search_many_where` / `search_many_filtered` | row-major `n × dim`, all cores, `Vec<Vec<Hit>>` |
| `search_traced(&q, k) -> (hits, visited)` | the walk order, for tooling |

`Filter::{eq, ne, is_in, lt, le, gt, ge, prefix, contains, exists, and, or, not}`;
`Filter::eval(&Metadata) -> bool` applies one in memory.

## Reading

| | |
|---|---|
| `get(id) -> Option<(Vec<f32>, Metadata)>` | the stored vector (normalised under cosine) and metadata |
| `len()`, `is_empty()`, `dim()`, `delta_len()` | counts; `delta_len` is the tail since the last checkpoint |
| `ids() -> Vec<u64>` | every live id |
| `keys() -> Vec<(String, Kind)>` | metadata keys and their types |
| `stats() -> Stats` | counters since open: searches by regime, text and sparse searches, inserts, deletes, batches, checkpoints, refreshes |
| `index_stats() -> Option<IndexStats>` | `nodes`, `pq_subspaces`, `tombstones`, `storage` |
| `index_path() -> Option<PathBuf>` | the current index file |
| `refresh()` | a read-only handle catches up now (reads do it anyway) |

## Embedding

```rust
pub trait Embedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}
```

Any `Fn(&[&str]) -> Result<Vec<Vec<f32>>>` implements it. With feature
`embed-http`, `sanadb::embed::OpenAiEmbedder::new(url, model).api_key(key)`
calls any OpenAI-compatible `/v1/embeddings` endpoint.

## Errors

`Error::{Io, Corrupt, HeaderMismatch, DimMismatch, MetaTooLarge,
TypeMismatch, SecondSparseKey, InvalidVector, Locked, ReadOnly,
NeedsNewerVersion}`; `Result<T>` is the crate alias. `Corrupt` carries
the check that failed; `NeedsNewerVersion` carries the file's reader
level and the one this build supports.

## Features

| feature | adds |
|---|---|
| `embed-http` | `OpenAiEmbedder` (pulls `ureq` and `serde_json`) |
| `tracing` | events on target `sanadb` for opens, checkpoints, failed checkpoints, reader reloads, orphan index files removed |
