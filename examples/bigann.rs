//! Cold- vs warm-cache measurement on a prefix of SIFT1B (big-ann-benchmarks
//! "bigann", uint8 vectors, dim 128, L2).
//!
//! Files in `<dir>`: `base.5M.u8` (first N rows of base.1B.u8bin with the
//! 8-byte header stripped) and `query.public.10K.u8bin`.
//!
//! ```bash
//! cargo run --release --example bigann -- build data/bigann 5000000 pq:on
//! cargo run --release --example bigann -- gt    data/bigann 5000000 1000
//! sudo purge   # macOS: drop the page cache so the next pass reads from disk
//! cargo run --release --example bigann -- query data/bigann 5000000 1000 pq:on 64
//! ```
//!
//! `query` runs the same queries twice and reports each pass: the first is
//! cold if the cache was purged, the second is warm.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;
use sanadb::{Db, GraphParams, Options, Storage};

/// `SANADB_I8=1` stores the index with 8-bit codes (and names the store `.i8`).
fn storage() -> Storage {
    if std::env::var_os("SANADB_I8").is_some() { Storage::I8 } else { Storage::F32 }
}

const DIM: usize = 128;
const K: usize = 10;

fn base(dir: &Path, n: usize) -> Vec<u8> {
    let mut b = fs::read(dir.join("base.5M.u8")).expect("base.5M.u8");
    assert!(
        b.len() >= n * DIM,
        "base file has {} rows, asked for {n}",
        b.len() / DIM
    );
    b.truncate(n * DIM);
    b
}

fn queries(dir: &Path, q: usize) -> Vec<f32> {
    let b = fs::read(dir.join("query.public.10K.u8bin")).expect("query.public.10K.u8bin");
    let nq = u32::from_le_bytes(b[..4].try_into().unwrap()) as usize;
    assert_eq!(
        u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize,
        DIM
    );
    b[8..8 + q.min(nq) * DIM]
        .iter()
        .map(|&x| x as f32)
        .collect()
}

fn row(base: &[u8], i: usize) -> impl Iterator<Item = f32> + '_ {
    base[i * DIM..(i + 1) * DIM].iter().map(|&x| x as f32)
}

fn store_path(dir: &Path, n: usize, pq: bool) -> std::path::PathBuf {
    dir.join(format!(
        "sift{}k.{}{}.sana",
        n / 1000,
        if pq { "pq" } else { "nopq" },
        if storage() == Storage::I8 { ".i8" } else { "" }
    ))
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let cmd = a.get(1).map(String::as_str).unwrap_or("");
    let dir = Path::new(a.get(2).map(String::as_str).unwrap_or("data/bigann"));
    let n: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(5_000_000);
    match cmd {
        "build" => {
            let pq = a.get(4).is_none_or(|s| s != "pq:off");
            let path = store_path(dir, n, pq);
            let _ = fs::remove_file(&path);
            for e in fs::read_dir(dir).into_iter().flatten().flatten() {
                if e.file_name().to_string_lossy().starts_with(&format!(
                    "{}.idx",
                    path.file_name().unwrap().to_string_lossy()
                )) {
                    let _ = fs::remove_file(e.path());
                }
            }
            let b = base(dir, n);
            let db = Db::open(
                &path,
                Options::new(DIM as u32).graph(GraphParams {
                    use_pq: pq,
                    storage: storage(),
                    ..GraphParams::default()
                }),
            )
            .unwrap();
            let t = Instant::now();
            let mut v = vec![0f32; DIM];
            for i in 0..n {
                v.iter_mut().zip(row(&b, i)).for_each(|(d, s)| *d = s);
                db.insert(i as u64, &v, HashMap::new()).unwrap();
            }
            println!("insert {n}: {:.1?}", t.elapsed());
            drop(b);
            let t = Instant::now();
            db.checkpoint().unwrap();
            let s = db.index_stats().unwrap();
            let idx = fs::metadata(db.index_path().unwrap()).unwrap().len();
            let log = fs::metadata(&path).unwrap().len();
            println!(
                "build: {:.1?} nodes={} pq_subspaces={} index={:.2} GB log={:.0} MB",
                t.elapsed(),
                s.nodes,
                s.pq_subspaces,
                idx as f64 / 1e9,
                log as f64 / 1e6
            );
        }
        "gt" => {
            let q: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(1000);
            let b = base(dir, n);
            let qs = queries(dir, q);
            let t = Instant::now();
            let gt: Vec<u32> = (0..q)
                .into_par_iter()
                .flat_map_iter(|qi| {
                    let qv = &qs[qi * DIM..(qi + 1) * DIM];
                    let mut best: Vec<(f32, u32)> = Vec::with_capacity(K + 1);
                    for i in 0..n {
                        let d: f32 = qv
                            .iter()
                            .zip(row(&b, i))
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum();
                        if best.len() < K || d < best[K - 1].0 {
                            let pos = best.partition_point(|e| e.0 <= d);
                            best.insert(pos, (d, i as u32));
                            best.truncate(K);
                        }
                    }
                    best.into_iter().map(|e| e.1)
                })
                .collect();
            let out: Vec<u8> = gt.iter().flat_map(|x| x.to_le_bytes()).collect();
            fs::write(dir.join(format!("gt.{n}.{q}.u32")), out).unwrap();
            println!(
                "ground truth for {q} queries over {n} rows: {:.1?}",
                t.elapsed()
            );
        }
        "query" => {
            let q: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(1000);
            let pq = a.get(5).is_none_or(|s| s != "pq:off");
            let beam: u32 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(64);
            let width: u32 = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(4);
            let gt_bytes = fs::read(dir.join(format!("gt.{n}.{q}.u32"))).expect("run `gt` first");
            let gt: Vec<u32> = gt_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect();
            let qs = queries(dir, q);
            let t = Instant::now();
            let opts = Options::new(DIM as u32)
                .graph(GraphParams {
                    use_pq: pq,
                    storage: storage(),
                    ..GraphParams::default()
                })
                .search_beam(beam)
                .max_search_beam(beam)
                .io_width(width);
            let db = Db::open(store_path(dir, n, pq), opts).unwrap();
            println!(
                "open: {:.2?}  n={} pq={pq} beam={beam} io_width={width} queries={q}",
                t.elapsed(),
                db.len()
            );
            println!(
                "{:>6} {:>9} {:>9} {:>9} {:>9} {:>9}",
                "pass", "recall@10", "mean", "p50", "p99", "total"
            );
            for pass in ["cold", "warm"] {
                let mut lat = Vec::with_capacity(q);
                let mut hit = 0usize;
                let t0 = Instant::now();
                for qi in 0..q {
                    let t = Instant::now();
                    let hits = db.search(&qs[qi * DIM..(qi + 1) * DIM], K).unwrap();
                    lat.push(t.elapsed().as_secs_f64() * 1e6);
                    let truth = &gt[qi * K..(qi + 1) * K];
                    hit += hits
                        .iter()
                        .filter(|h| truth.contains(&(h.id as u32)))
                        .count();
                }
                let total = t0.elapsed();
                lat.sort_by(|a, b| a.total_cmp(b));
                let mean = lat.iter().sum::<f64>() / q as f64;
                println!(
                    "{pass:>6} {:>9.4} {:>7.0}us {:>7.0}us {:>7.0}us {:>8.2?}",
                    hit as f64 / (q * K) as f64,
                    mean,
                    lat[q / 2],
                    lat[q * 99 / 100],
                    total
                );
            }
        }
        _ => eprintln!("usage: bigann build|gt|query <dir> <n> ..."),
    }
}
