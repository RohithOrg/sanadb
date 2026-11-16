//! Recall and latency on the public SIFT1M corpus (irisa.fr texmex).
//!
//! Download and extract `sift.tar.gz` into `data/`, then:
//!
//! ```bash
//! cargo run --release --example sift1m -- data/sift [n] [queries] [pq:on|off]
//! ```
//!
//! `n` limits the base set (default all 1,000,000). With the full set the
//! published ground truth is used; with a subset, brute force on that subset
//! is computed for the first `queries` queries.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use sanadb::{Db, GraphParams, Options, Storage};

fn read_vecs<T: Copy, const W: usize>(path: &Path, from: impl Fn([u8; W]) -> T) -> (usize, Vec<T>) {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let dim = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let stride = 4 + dim * W;
    assert_eq!(
        bytes.len() % stride,
        0,
        "{} is not a whole number of records",
        path.display()
    );
    let mut out = Vec::with_capacity(bytes.len() / stride * dim);
    for rec in bytes.chunks_exact(stride) {
        assert_eq!(
            u32::from_le_bytes(rec[..4].try_into().unwrap()) as usize,
            dim
        );
        out.extend(rec[4..].as_chunks::<W>().0.iter().map(|c| from(*c)));
    }
    (dim, out)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).map(String::as_str).unwrap_or("data/sift"));
    let n_limit: Option<usize> = args.get(2).and_then(|s| s.parse().ok());
    let n_queries: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let use_pq = args.get(4).is_none_or(|s| s != "pq:off");
    let prune_slack: f32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1.3);
    let storage = if args.get(6).is_some_and(|s| s == "i8") { Storage::I8 } else { Storage::F32 };

    let (dim, base) = read_vecs::<f32, 4>(&dir.join("sift_base.fvecs"), f32::from_le_bytes);
    let n_total = base.len() / dim;
    let n = n_limit.map_or(n_total, |l| l.min(n_total));
    let (_, queries) = read_vecs::<f32, 4>(&dir.join("sift_query.fvecs"), f32::from_le_bytes);
    let (gt_k, gt) = read_vecs::<u32, 4>(&dir.join("sift_groundtruth.ivecs"), u32::from_le_bytes);
    let n_queries = n_queries.min(queries.len() / dim);
    println!("SIFT1M: dim={dim} base={n_total} using={n} queries={n_queries} pq={use_pq}");

    // SIFT_STORE=<dir> keeps the store between runs and skips the build when
    // an index over the same `n` already exists there (for profiling queries).
    let tmp = tempfile::tempdir().unwrap();
    let keep = std::env::var_os("SIFT_STORE").map(std::path::PathBuf::from);
    let path = keep.as_deref().unwrap_or(tmp.path()).join(format!(
        "sift{}.{}{}.sana",
        n / 1000,
        if use_pq { "pq" } else { "nopq" },
        if storage == Storage::I8 { ".i8" } else { "" }
    ));
    let params = GraphParams {
        use_pq,
        prune_slack,
        storage,
        ..GraphParams::default()
    };
    // Bulk load: one explicit build at the end, no automatic ones in between.
    let mut db = Db::open(
        &path,
        Options::new(dim as u32)
            .graph(params)
            .auto_checkpoint_delta(0),
    )
    .unwrap();
    println!("prune_slack={prune_slack}");

    if db.index_stats().is_some_and(|s| s.nodes == n) {
        println!("reusing index at {}", path.display());
    } else {
        let t = Instant::now();
        for i in 0..n {
            db.insert(i as u64, &base[i * dim..(i + 1) * dim], HashMap::new())
                .unwrap();
        }
        println!("insert {n}: {:.1?}", t.elapsed());
        let t = Instant::now();
        db.checkpoint().unwrap();
        let build = t.elapsed();
        let stats = db.index_stats().unwrap();
        let idx_bytes = fs::metadata(db.index_path().unwrap()).unwrap().len();
        println!(
            "build: {build:.1?}  nodes={} pq_subspaces={} index={:.0} MB",
            stats.nodes,
            stats.pq_subspaces,
            idx_bytes as f64 / 1e6
        );
    }

    // Ground truth: published top-100 for the full set, brute force otherwise.
    let k = 10;
    let truth: Vec<Vec<u64>> = if n == n_total {
        (0..n_queries)
            .map(|q| {
                gt[q * gt_k..q * gt_k + k]
                    .iter()
                    .map(|&i| i as u64)
                    .collect()
            })
            .collect()
    } else {
        println!("subset: computing brute-force ground truth for {n_queries} queries");
        (0..n_queries)
            .map(|qi| {
                let q = &queries[qi * dim..(qi + 1) * dim];
                let mut d: Vec<(f32, u64)> = (0..n)
                    .map(|i| {
                        (
                            q.iter()
                                .zip(&base[i * dim..(i + 1) * dim])
                                .map(|(a, b)| (a - b) * (a - b))
                                .sum(),
                            i as u64,
                        )
                    })
                    .collect();
                d.sort_by(|a, b| a.0.total_cmp(&b.0));
                d[..k].iter().map(|e| e.1).collect()
            })
            .collect()
    };

    println!(
        "{:>6} {:>9} {:>9} {:>9} {:>9}",
        "beam", "recall@10", "mean", "p50", "p99"
    );
    for beam in [32u32, 64, 128, 256] {
        drop(db);
        db = Db::open(
            &path,
            Options::new(dim as u32)
                .graph(params)
                .search_beam(beam)
                .max_search_beam(beam)
                .auto_checkpoint_delta(0),
        )
        .unwrap();
        let mut hit = 0usize;
        let mut lat: Vec<f64> = Vec::with_capacity(n_queries);
        for qi in 0..n_queries {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let t = Instant::now();
            let hits = db.search(q, k).unwrap();
            lat.push(t.elapsed().as_secs_f64() * 1e6);
            hit += hits.iter().filter(|h| truth[qi].contains(&h.id)).count();
        }
        lat.sort_by(|a, b| a.total_cmp(b));
        let mean = lat.iter().sum::<f64>() / lat.len() as f64;
        println!(
            "{beam:>6} {:>9.4} {:>7.0}us {:>7.0}us {:>7.0}us",
            hit as f64 / (n_queries * k) as f64,
            mean,
            lat[lat.len() / 2],
            lat[lat.len() * 99 / 100]
        );
    }
}
