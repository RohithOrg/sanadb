//! Semantic search over text with a real embedding model, from Rust.
//!
//! The model runs behind any OpenAI-compatible `/v1/embeddings` endpoint:
//! Ollama (`ollama pull nomic-embed-text`, http://localhost:11434/v1/embeddings),
//! LM Studio, vLLM, text-embeddings-inference, OpenAI itself, or the
//! 40-line server in `python/examples/embed_server.py` that serves a
//! sentence-transformers model. The store records which model produced the
//! vectors as a property, so a later reader knows what to embed queries with.
//!
//! `cargo run --release --features embed-http --example semantic_search -- [url] [model] [dim]`
//! defaults to the sentence-transformers server: all-MiniLM-L6-v2, 384 dims,
//! http://127.0.0.1:8089/v1/embeddings.

use std::collections::HashMap;

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
        (
            "The borrow checker rejects aliasing of mutable references",
            "rust",
            2021,
        ),
        (
            "Lifetimes tie a reference to the scope that owns the data",
            "rust",
            2022,
        ),
        (
            "asyncio runs coroutines on a single-threaded event loop",
            "python",
            2020,
        ),
        (
            "Tokio's select! cancels the branches that did not complete",
            "rust",
            2023,
        ),
        (
            "Goroutines are multiplexed onto OS threads by the Go scheduler",
            "go",
            2019,
        ),
        ("Garbage collection pauses are tuned with GOGC", "go", 2020),
        (
            "Pattern matching on tagged unions catches missing cases at compile time",
            "rust",
            2022,
        ),
        (
            "The GIL serialises bytecode execution across threads",
            "python",
            2018,
        ),
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
    println!(
        "{} documents embedded with {model_name} into {}",
        docs.len(),
        path.display()
    );

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
