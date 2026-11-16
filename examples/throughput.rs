//! Write and read throughput on random vectors, for the numbers in the README:
//! single inserts against one atomic batch, `search` against `search_many`,
//! and what a read-only handle's per-read refresh costs beside a writer.
//! `cargo run --release --example throughput -- [n] [dim]`

use std::collections::HashMap;
use std::time::Instant;

use sanadb::{Batch, Db, Options, Value};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let dim: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);
    let mut x = 0x5EEDu64;
    let mut rnd = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 40) as f32 / 16777216.0
    };
    let vecs: Vec<f32> = (0..n * dim).map(|_| rnd()).collect();
    let meta = |i: usize| HashMap::from([("g".to_string(), Value::from((i % 100) as i64))]);
    let dir = tempfile::tempdir().unwrap();

    // Single inserts: one frame, one flush, one lock per record.
    let db = Db::open(dir.path().join("single.sana"), Options::new(dim as u32).auto_checkpoint_delta(0)).unwrap();
    let t = Instant::now();
    for i in 0..n {
        db.insert(i as u64, &vecs[i * dim..(i + 1) * dim], meta(i)).unwrap();
    }
    let single = t.elapsed();
    println!("insert one by one: {n} records in {single:.2?} = {:.0} /s", n as f64 / single.as_secs_f64());

    // One batch per 10,000 records.
    let dbb = Db::open(dir.path().join("batch.sana"), Options::new(dim as u32).auto_checkpoint_delta(0)).unwrap();
    let t = Instant::now();
    for start in (0..n).step_by(10_000) {
        let mut b = Batch::new();
        for i in start..(start + 10_000).min(n) {
            b = b.insert(i as u64, &vecs[i * dim..(i + 1) * dim], meta(i));
        }
        dbb.write_batch(b).unwrap();
    }
    let batched = t.elapsed();
    println!("insert in batches of 10,000: {batched:.2?} = {:.0} /s", n as f64 / batched.as_secs_f64());

    let t = Instant::now();
    db.checkpoint().unwrap();
    println!("checkpoint (build) of {n}: {:.2?}", t.elapsed());

    let queries: Vec<f32> = (0..1000 * dim).map(|_| rnd()).collect();
    let t = Instant::now();
    for q in queries.chunks(dim) {
        db.search(q, 10).unwrap();
    }
    let one = t.elapsed();
    println!("search, one query at a time: {:.0} µs/query", one.as_secs_f64() * 1e6 / 1000.0);
    let t = Instant::now();
    db.search_many(&queries, 10).unwrap();
    let many = t.elapsed();
    println!("search_many, 1000 queries on all cores: {:.2?} = {:.0} queries/s", many, 1000.0 / many.as_secs_f64());

    // Reader beside the writer: each read checks the log first.
    let reader = Db::open_read_only(dir.path().join("single.sana"), Options::new(dim as u32)).unwrap();
    let t = Instant::now();
    for q in queries.chunks(dim) {
        reader.search(q, 10).unwrap();
    }
    let rd = t.elapsed();
    println!("read-only handle, search: {:.0} µs/query (refresh check included)", rd.as_secs_f64() * 1e6 / 1000.0);
    let t = Instant::now();
    for i in 0..10_000u64 {
        reader.get(i);
    }
    println!("read-only handle, get: {:.2} µs each", t.elapsed().as_secs_f64() * 1e6 / 10_000.0);
    let t = Instant::now();
    for i in 0..10_000u64 {
        db.get(i);
    }
    println!("writer handle, get: {:.2} µs each", t.elapsed().as_secs_f64() * 1e6 / 10_000.0);
}
