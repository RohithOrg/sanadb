//! Open a store, report how long it took and the process's resident memory,
//! run a few queries, then optionally checkpoint (to rewrite the index in the
//! current layout) and repeat. `cargo run --release --example opentime -- <store> [checkpoint]`
use std::time::Instant;

use sanadb::{Db, Options};

fn rss_mb() -> f64 {
    let out = std::process::Command::new("ps").args(["-o", "rss=", "-p", &std::process::id().to_string()]).output().unwrap();
    String::from_utf8_lossy(&out.stdout).trim().parse::<f64>().unwrap_or(0.0) / 1024.0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let dim: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);
    let checkpoint = args.iter().any(|a| a == "checkpoint");
    let before = rss_mb();
    let t = Instant::now();
    let db = Db::open(path, Options::new(dim).auto_checkpoint_delta(0)).unwrap();
    let open = t.elapsed();
    println!("open: {open:.2?}  len={}  stats={:?}  rss {before:.0} MB -> {:.0} MB", db.len(), db.index_stats(), rss_mb());
    let q = db.get(0).map(|(v, _)| v).unwrap_or_else(|| vec![0.0; dim as usize]);
    let t = Instant::now();
    for _ in 0..1000 {
        db.search(&q, 10).unwrap();
    }
    println!("1000 queries: {:.2?}", t.elapsed());
    if checkpoint {
        let t = Instant::now();
        db.checkpoint().unwrap();
        println!("checkpoint: {:.1?}", t.elapsed());
    }
}
