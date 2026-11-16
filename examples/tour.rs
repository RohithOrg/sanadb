//! A tour of the API on vectors of any dimension: open a store, write
//! records with typed metadata, search plain, filtered, by keyword, by
//! sparse vector and hybrid, read from a second handle, back up, look at
//! the counters. Runs in a temp directory and prints what it finds.
//!
//! `cargo run --release --example tour`

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
