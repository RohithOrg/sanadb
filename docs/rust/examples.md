# Rust examples

Complete programs with the output they printed, run on 16 November 2026
with sanadb 0.2.7. Every file is in the repository under `examples/`;
`cargo run --release --example <name>` runs one. Copy one and change the
data.

## Semantic search with a real embedding model

What most people want first: their own text, searched by meaning. The
crate never ships a model; it ships the `Embedder` trait and, behind the
`embed-http` feature, a client for any OpenAI-compatible `/v1/embeddings`
endpoint (Ollama, LM Studio, vLLM, text-embeddings-inference, OpenAI).
Here the endpoint is the 40-line
[`python/examples/embed_server.py`](https://github.com/RohithOrg/sanadb/blob/master/python/examples/embed_server.py)
serving `all-MiniLM-L6-v2` (384 dimensions) on localhost, so the result
is comparable with the Python example to the third decimal.

```bash
pip install sentence-transformers && python python/examples/embed_server.py &
cargo run --release --features embed-http --example semantic_search
# or against Ollama:
cargo run --release --features embed-http --example semantic_search -- http://localhost:11434/v1/embeddings nomic-embed-text 768
```

```rust
use sanadb::embed::OpenAiEmbedder;
use sanadb::{Db, Embedder, Filter, Metric, Options, Value};

fn main() -> Result<(), sanadb::Error> {
    let args: Vec<String> = std::env::args().collect();
    let url = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "http://127.0.0.1:8089/v1/embeddings".into());
    let model_name = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "all-MiniLM-L6-v2".into());
    let dim: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(384);
    let model = OpenAiEmbedder::new(&url, &model_name);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("docs.sana");
    let db = Db::open(
        &path,
        Options::new(dim).metric(Metric::Cosine).text_field("body"),
    )?;
    db.set_property("embedding_model", &model_name)?;

    let docs = [
        ("The borrow checker rejects aliasing of mutable references", "rust", 2021),
        ("Lifetimes tie a reference to the scope that owns the data", "rust", 2022),
        ("asyncio runs coroutines on a single-threaded event loop", "python", 2020),
        ("Tokio's select! cancels the branches that did not complete", "rust", 2023),
        ("Goroutines are multiplexed onto OS threads by the Go scheduler", "go", 2019),
        ("Garbage collection pauses are tuned with GOGC", "go", 2020),
        ("Pattern matching on tagged unions catches missing cases at compile time", "rust", 2022),
        ("The GIL serialises bytecode execution across threads", "python", 2018),
    ];
    for (i, (text, lang, year)) in docs.iter().enumerate() {
        // Embeds the text, stores the vector with the metadata and the text
        // itself under "body" (so `search_text` and `get` see it).
        db.add_text(
            i as u64 + 1,
            text,
            HashMap::from([
                ("lang".to_string(), Value::from(*lang)),
                ("year".to_string(), Value::from(*year as i64)),
            ]),
            "body",
            &model,
        )?;
    }
    println!("{} documents embedded with {model_name} into {}", docs.len(), path.display());

    let show = |title: &str, hits: Vec<sanadb::Hit>| {
        println!("\n{title}");
        for h in hits {
            let (_, meta) = db.get(h.id).unwrap();
            println!(
                "  {:+.3}  [{}]  {}",
                h.score,
                meta["lang"].as_str().unwrap(),
                meta["body"].as_str().unwrap()
            );
        }
    };

    // Search by meaning: the query shares no words with the sentence it finds.
    show(
        "why can't I have two mutable references",
        db.search_text_embedded("why can't I have two mutable references", 3, None, &model)?,
    );
    show(
        "exhaustive match on an enum",
        db.search_text_embedded("exhaustive match on an enum", 3, None, &model)?,
    );
    // Meaning plus a metadata filter, evaluated on the index columns.
    show(
        "running many tasks on one thread, Python only",
        db.search_text_embedded(
            "running many tasks on one thread",
            3,
            Some(&Filter::eq("lang", "python")),
            &model,
        )?,
    );
    // Hybrid: the vector result fused with BM25 on the text, for exact terms
    // that embed poorly (identifiers, flags, names).
    let q = model.embed(&["tuning the collector"])?.remove(0);
    show(
        "hybrid: 'GOGC' as a keyword with a semantic query",
        db.search_hybrid(&q, "GOGC", 3, None)?,
    );

    db.checkpoint()?;
    let ix = db.index_stats().unwrap();
    println!(
        "\nindex built: {} nodes, storage {:?}; properties: {:?}",
        ix.nodes,
        ix.storage,
        db.properties()
    );
    println!("scores are negative cosine similarity: -1 identical, lower is closer");
    Ok(())
}
```

(with `use std::collections::HashMap;` at the top.) Output:

<!-- output: semantic_search.rs -->
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

index built: 8 nodes, storage F32; properties: {"embedding_model": "all-MiniLM-L6-v2"}
scores are negative cosine similarity: -1 identical, lower is closer
```

Reading it: the first query shares no word with the sentence it ranks
first, which is what embeddings buy; the third query shows a `Filter`
narrowing the same search to one language, evaluated on the index's
columns rather than after the fact; the hybrid query puts the sentence
containing the exact token `GOGC` first because BM25 and the vector
result are fused by reciprocal rank. The scores are identical to the
Python example's because the same model produced the same vectors and
the same engine searched them. The store carries
`embedding_model = all-MiniLM-L6-v2` in its properties, so a reader a
year from now knows what to embed queries with, and studio checks it.

## A tour of every feature

`examples/tour.rs` runs one scenario on 384-dimensional vectors from a
stand-in embedding function (so it runs with no model installed): open,
insert with typed metadata, batch, delete, nearest, filters, keyword,
hybrid, sparse, get, checkpoint, a read-only handle, sync, backup,
counters. [Getting started](getting-started.md) walks through it step by
step.

```rust
use std::collections::HashMap;

use sanadb::{Batch, Db, Filter, Metric, Options, Value};

/// Stand-in for an embedding model: any `dim` works, this just has to be
/// deterministic so the searches below have known answers.
fn embed(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0f32; dim];
    for (i, b) in text.bytes().enumerate() {
        v[(i * 7 + b as usize) % dim] += 1.0;
    }
    v
}

fn main() -> Result<(), sanadb::Error> {
    let dim = 384; // whatever your model produces
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("docs.sana");

    // 1. Open (creates the files). Cosine normalises vectors on insert and
    //    on query; `text_field` names the metadata key to index for BM25.
    let opts = || {
        Options::new(dim as u32)
            .metric(Metric::Cosine)
            .text_field("body")
    };
    let db = Db::open(&path, opts())?;

    // 2. Records: an id, a vector, typed metadata. The first value under a
    //    key fixes its type for the store.
    let docs = [
        (
            1u64,
            "Rust borrow checker explained",
            "rust",
            2021i64,
            4.5f64,
            true,
        ),
        (
            2,
            "Python asyncio event loop internals",
            "python",
            2020,
            4.1,
            true,
        ),
        (
            3,
            "The borrow checker rejects aliasing",
            "rust",
            2023,
            4.8,
            false,
        ),
        (
            4,
            "Tokio select and cancellation safety",
            "rust",
            2022,
            4.6,
            true,
        ),
        (5, "Garbage collection in Go", "go", 2019, 3.9, true),
    ];
    for (id, body, lang, year, score, public) in docs {
        db.insert(
            id,
            &embed(body, dim),
            HashMap::from([
                ("body".to_string(), Value::from(body)),
                ("lang".to_string(), Value::from(lang)),
                ("year".to_string(), Value::from(year)),
                ("score".to_string(), Value::from(score)),
                ("public".to_string(), Value::from(public)),
                // A learned sparse vector (SPLADE-style) under one key.
                (
                    "sparse".to_string(),
                    Value::Sparse(vec![(id as u32 * 10, 1.0), (7, 0.5)]),
                ),
            ]),
        )?;
    }
    // Many at once, atomically: all of them or none.
    db.write_batch(
        Batch::new()
            .insert(
                6,
                &embed("Lifetimes in Rust structs", dim),
                HashMap::from([
                    ("body".to_string(), Value::from("Lifetimes in Rust structs")),
                    ("lang".to_string(), Value::from("rust")),
                ]),
            )
            .insert(
                7,
                &embed("Pattern matching in Elixir", dim),
                HashMap::from([
                    (
                        "body".to_string(),
                        Value::from("Pattern matching in Elixir"),
                    ),
                    ("lang".to_string(), Value::from("elixir")),
                ]),
            )
            .delete(5),
    )?;
    println!("{} records", db.len());

    // 3. Plain nearest neighbours. Scores are distances: lower is closer.
    let q = embed("borrow checker", dim);
    show("nearest", &db.search(&q, 3)?);

    // 4. Filters run on the index's columns and never return a short list
    //    for a rare filter. Any closure also works, more slowly.
    let f = Filter::and([
        Filter::eq("lang", "rust"),
        Filter::ge("year", 2022i64),
        Filter::not(Filter::eq("public", false)),
    ]);
    show("rust, 2022 or later, public", &db.search_where(&q, 3, &f)?);
    show(
        "prefix",
        &db.search_where(&q, 3, &Filter::prefix("lang", "ru"))?,
    );
    show(
        "closure",
        &db.search_filtered(&q, 3, |m| {
            m.get("score")
                .and_then(Value::as_float)
                .is_some_and(|s| s > 4.5)
        })?,
    );

    // 5. Keyword (BM25) and hybrid (vector + keyword by reciprocal rank).
    show(
        "keyword 'borrow checker'",
        &db.search_text("borrow checker", 3, None)?,
    );
    show(
        "hybrid",
        &db.search_hybrid(&q, "borrow checker", 3, Some(&Filter::eq("lang", "rust")))?,
    );

    // 6. Sparse vectors: dot product over the sparse key, and hybrid with dense.
    show(
        "sparse",
        &db.search_sparse(&[(30, 1.0), (7, 1.0)], 3, None)?,
    );
    show(
        "hybrid sparse",
        &db.search_hybrid_sparse(&q, &[(30, 1.0)], 3, None)?,
    );

    // 7. Read a record back: the vector (normalised, since cosine) and metadata.
    let (v, meta) = db.get(3).unwrap();
    println!(
        "get(3): dim {}, lang {}, year {}",
        v.len(),
        meta["lang"],
        meta["year"]
    );

    // 8. Checkpoint: build the graph index over everything written so far
    //    and compact the log. Searches work before it (exact scan of the
    //    tail) and after (graph walk); do it after bulk loads or on a timer.
    db.checkpoint()?;
    show("after checkpoint", &db.search(&q, 3)?);

    // 9. A second handle, in this or any other process, follows the writer.
    let reader = Db::open_read_only(&path, opts())?;
    db.insert(
        8,
        &embed("Unsafe Rust and FFI", dim),
        HashMap::from([
            ("body".to_string(), Value::from("Unsafe Rust and FFI")),
            ("lang".to_string(), Value::from("rust")),
        ]),
    )?;
    println!("reader sees {} records", reader.len());

    // 10. Durability and backup.
    db.sync()?; // fsync: survives power loss from here
    let copy = db.backup(dir.path().join("backup"))?;
    println!("backup at {}", copy.display());

    // 11. What the store has been doing.
    let s = db.stats();
    println!(
        "stats: {} searches ({} exact scans, {} filtered walks), {} checkpoints, {} inserts",
        s.searches, s.exact_scans, s.filtered_walks, s.checkpoints, s.inserts
    );
    if let Some(ix) = db.index_stats() {
        println!(
            "index: {} nodes, {} PQ subspaces, storage {:?}",
            ix.nodes, ix.pq_subspaces, ix.storage
        );
    }
    Ok(())
}

fn show(what: &str, hits: &[sanadb::Hit]) {
    let ids: Vec<String> = hits
        .iter()
        .map(|h| format!("{}({:.3})", h.id, h.score))
        .collect();
    println!("{what:<32} {}", ids.join("  "));
}
```

Output:

<!-- output: tour.rs -->
```text
6 records
nearest                          4(-0.260)  2(-0.176)  7(-0.049)
rust, 2022 or later, public      4(-0.260)
prefix                           4(-0.260)  3(-0.000)  1(0.000)
closure                          4(-0.260)  3(-0.000)
keyword 'borrow checker'         1(-2.157)  3(-1.970)
hybrid                           1(-0.032)  3(-0.032)  4(-0.016)
sparse                           3(-1.500)  1(-0.500)  2(-0.500)
hybrid sparse                    3(-0.032)  4(-0.016)  2(-0.016)
get(3): dim 384, lang rust, year 2023
after checkpoint                 4(-0.260)  2(-0.176)  7(-0.049)
reader sees 7 records
backup at /tmp/…/backup/docs.sana
stats: 7 searches (0 exact scans, 0 filtered walks), 1 checkpoints, 6 inserts
index: 6 nodes, 0 PQ subspaces, storage F32
```

The stand-in "embedding" is a byte hash, so the nearest neighbours are
not semantic, but every mechanism is exercised: the filtered search
returns the one record that passes all three conditions; the keyword
search ranks the two sentences containing both words; the hybrid list
puts records present in both the vector and the keyword result first;
the read-only handle sees the writer's new record without being told.

## Recall and latency at a million vectors

`examples/sift1m.rs` loads the public SIFT1M corpus (1,000,000 vectors,
dim 128, L2), checkpoints, and runs the 10,000 published queries against
the published ground truth at several beams:

```bash
cargo run --release --example sift1m -- data/sift [n] [queries] [pq:on|off]
```

| | PQ on | PQ off |
|---|---|---|
| build | 75 s | 71 s |
| index file | 744 MB | 712 MB |
| beam 64: recall@10, mean, p99 | 0.966, 146 µs, 213 µs | 0.971, 225 µs, 319 µs |
| beam 128 | 0.987, 240 µs, 328 µs | 0.989, 371 µs, 490 µs |
| beam 256 | 0.995, 405 µs, 555 µs | 0.996, 623 µs, 832 µs |

The example reuses a store it already built, so the second run measures
queries only.

## Filtered search at a million records

`examples/filtered.rs` labels SIFT1M with four string keys whose equality
filters pass 10%, 1%, 0.1% and 0.01% of the set, and compares
`search_where` (the column path) with `search_filtered` and an equivalent
closure (the post-filter path), against brute-force truth over the
passing records:

```bash
SIFT_STORE=data/sift/store cargo run --release --example filtered -- data/sift [queries] [beam] [exact_filter_scan]
```

| selectivity | regime | recall@10 | mean | closure recall@10 | closure mean |
|---|---|---|---|---|---|
| 10% | filtered walk | 0.995 | 1.04 ms | 0.792 | 0.42 ms |
| 1% | exact scan | 1.000 | 1.34 ms | 0.900 | 3.96 ms |
| 0.1% | exact scan | 1.000 | 110 µs | 0.399 | 10.4 ms |
| 0.01% | exact scan | 1.000 | 9 µs | 0.044 | 10.5 ms |

A rare filter is answered by reading every matching record, so it gets
recall 1.0 and is the cheapest query in the table. `examples/yfcc.rs`
does the same on the big-ann-benchmarks filter track, where each record
carries a set of labels stored as a space-separated string in the text
field and a query is `Filter::contains` terms and-ed together, answered
from the inverted index: recall 1.0 where post-filtering gets 0.17 to
0.83.

## Larger than RAM

`examples/bigann.rs` builds a 5,000,000-vector store from SIFT1B (3.5 GB
index), evicts it from the page cache and measures cold queries; it is
the source of the cold-cache table in [benchmarks](../benchmarks.md):
2.75 ms mean and 4.2 ms p99 cold at the default `io_width` of 4, against
8.6 ms and 39 ms with `io_width(1)`.

```bash
cargo run --release --example bigann -- build data/bigann 5000000 pq:on
cargo run --release --example bigann -- gt    data/bigann 5000000 1000
cargo run --release --example bigann -- query data/bigann 5000000 1000 pq:on 64
```

## Write rates

`examples/throughput.rs` inserts 1,000,000 random vectors one by one and
in atomic batches, checkpoints, and times single and batched searches
from a writer and a read-only handle; the table is in
[operating a store](../operations.md#throughput).

```bash
cargo run --release --example throughput -- [n] [dim]
```
