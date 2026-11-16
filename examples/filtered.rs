//! Filtered search on SIFT1M with synthetic labels, at four selectivities.
//!
//! Each record gets `lang = i % 10` (10% per value), `cat = i % 100` (1%),
//! `tag = i % 1000` (0.1%) and `uid = i % 10000` (0.01%), all as strings.
//! For each selectivity the first `q` published queries are run with the
//! filter `key = <query index % cardinality>` through `search_where` (the
//! column path: exact scan or filtered walk) and through `search_filtered`
//! with an equivalent closure (the post-filter path with beam widening), and
//! compared against brute-force ground truth over the records that pass.
//!
//! Usage: `SIFT_STORE=data/sift/store cargo run --release --example filtered -- data/sift [queries] [beam] [exact_filter_scan]`
//! The store is built once (about 80 s) and reused.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;
use sanadb::{Db, Filter, GraphParams, Options, Value};

fn read_vecs(path: &Path) -> (usize, Vec<f32>) {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let dim = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let stride = 4 + dim * 4;
    let mut out = Vec::with_capacity(bytes.len() / stride * dim);
    for rec in bytes.chunks_exact(stride) {
        out.extend(
            rec[4..]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c)),
        );
    }
    (dim, out)
}

const LABELS: [(&str, u64); 4] = [("lang", 10), ("cat", 100), ("tag", 1000), ("uid", 10000)];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).map(String::as_str).unwrap_or("data/sift"));
    let n_queries: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let beam: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let scan: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let (dim, base) = read_vecs(&dir.join("sift_base.fvecs"));
    let n = base.len() / dim;
    let (_, queries) = read_vecs(&dir.join("sift_query.fvecs"));
    let n_queries = n_queries.min(queries.len() / dim);

    let tmp = tempfile::tempdir().unwrap();
    let keep = std::env::var_os("SIFT_STORE").map(std::path::PathBuf::from);
    let path = keep
        .as_deref()
        .unwrap_or(tmp.path())
        .join("sift1000.labels.sana");
    let opts = Options::new(dim as u32)
        .graph(GraphParams::default())
        .auto_checkpoint_delta(0)
        .search_beam(beam)
        .max_search_beam(beam.max(4096))
        .exact_filter_scan(scan);
    let db = Db::open(&path, opts).unwrap();
    if db.index_stats().is_some_and(|s| s.nodes == n) {
        println!("reusing {}", path.display());
    } else {
        let t = Instant::now();
        for i in 0..n as u64 {
            let m: HashMap<String, Value> = LABELS
                .iter()
                .map(|(k, card)| (k.to_string(), Value::from((i % card).to_string())))
                .collect();
            db.insert(i, &base[i as usize * dim..(i as usize + 1) * dim], m)
                .unwrap();
        }
        println!("insert {n} with 4 labels: {:.1?}", t.elapsed());
        let t = Instant::now();
        db.checkpoint().unwrap();
        println!(
            "build + attribute columns: {:.1?}, index {:.0} MB",
            t.elapsed(),
            fs::metadata(format!("{}.idx", path.display()))
                .unwrap()
                .len() as f64
                / 1e6
        );
    }
    let k = 10;
    // Warm the mapping (vectors, codes and columns) so the first label
    // measured does not pay first-touch page faults for everyone.
    for qi in 0..n_queries.min(200) {
        let q = &queries[qi * dim..(qi + 1) * dim];
        for (key, card) in LABELS {
            let _ = db
                .search_where(
                    q,
                    k,
                    &Filter::eq(key, ((qi as u64) % card).to_string().as_str()),
                )
                .unwrap();
        }
    }
    println!("{n_queries} queries, k={k}, beam {beam}, exact_filter_scan {scan}");
    println!(
        "{:<6} {:>7} | {:>9} {:>9} {:>9} | {:>9} {:>9} {:>9}",
        "filter", "select.", "col rec", "col mean", "col p99", "clo rec", "clo mean", "clo p99"
    );
    for (key, card) in LABELS {
        // Ground truth: brute force among records with the query's label.
        let truth: Vec<Vec<u64>> = (0..n_queries)
            .into_par_iter()
            .map(|qi| {
                let q = &queries[qi * dim..(qi + 1) * dim];
                let want = (qi as u64) % card;
                let mut d: Vec<(f32, u64)> = (want..n as u64)
                    .step_by(card as usize)
                    .map(|i| {
                        (
                            q.iter()
                                .zip(&base[i as usize * dim..(i as usize + 1) * dim])
                                .map(|(a, b)| (a - b) * (a - b))
                                .sum(),
                            i,
                        )
                    })
                    .collect();
                d.sort_by(|a, b| a.0.total_cmp(&b.0));
                d.iter().take(k).map(|e| e.1).collect()
            })
            .collect();
        let run = |column: bool| -> (f64, f64, f64) {
            let mut hit = 0usize;
            let mut lat = Vec::with_capacity(n_queries);
            for qi in 0..n_queries {
                let q = &queries[qi * dim..(qi + 1) * dim];
                let want = ((qi as u64) % card).to_string();
                let t = Instant::now();
                let hits = if column {
                    db.search_where(q, k, &Filter::eq(key, want.as_str()))
                        .unwrap()
                } else {
                    db.search_filtered(q, k, |m| m.get(key).is_some_and(|v| v == want.as_str()))
                        .unwrap()
                };
                lat.push(t.elapsed().as_secs_f64() * 1e6);
                hit += hits.iter().filter(|h| truth[qi].contains(&h.id)).count();
            }
            lat.sort_by(|a, b| a.total_cmp(b));
            (
                hit as f64 / (n_queries * k) as f64,
                lat.iter().sum::<f64>() / lat.len() as f64,
                lat[lat.len() * 99 / 100],
            )
        };
        let (cr, cm, cp) = run(true);
        let (fr, fm, fp) = run(false);
        println!(
            "{key:<6} {:>6.2}% | {cr:>9.4} {cm:>7.0}us {cp:>7.0}us | {fr:>9.4} {fm:>7.0}us {fp:>7.0}us",
            100.0 / card as f64
        );
    }
}
