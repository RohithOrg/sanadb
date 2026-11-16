//! Filtered search on the big-ann-benchmarks filter track (YFCC100M, 192-dim
//! CLIP embeddings as uint8, every record tagged with a set of labels, every
//! query asking for one or two labels that must all be present).
//!
//! Files from https://big-ann-benchmarks.com (the `yfcc100M` filter track):
//! `base.10M.u8bin`, `base.metadata.10M.spmat`, `query.public.100K.u8bin`,
//! `query.metadata.public.100K.spmat`. The published ground truth covers the
//! whole 10M; this example takes the first `n` records and recomputes the
//! truth by brute force over the records that carry the query's labels, so
//! it runs on a laptop.
//!
//! Each record's labels are stored as the string "t12 t345 ..." in the text
//! field, and a query's labels become `Filter::contains` terms and-ed
//! together, which the engine answers from the inverted index.
//!
//! Usage: `cargo run --release --example yfcc -- data/yfcc <store dir> [n=1000000] [queries=500] [beam=64]`

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;
use sanadb::{Db, Filter, Options, Value};

fn read_u8bin(path: &Path, n: usize) -> (usize, Vec<u8>) {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let rows = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let n = n.min(rows);
    (dim, bytes[8..8 + n * dim].to_vec())
}

/// CSR label matrix: the labels of row `i` are `indices[indptr[i]..indptr[i+1]]`.
fn read_spmat(path: &Path, n: usize) -> Vec<Vec<u32>> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let i64_at = |o: usize| i64::from_le_bytes(bytes[o..o + 8].try_into().unwrap()) as usize;
    let (nrow, _ncol, nnz) = (i64_at(0), i64_at(8), i64_at(16));
    let n = n.min(nrow);
    let indptr = 24;
    let indices = indptr + (nrow + 1) * 8;
    let _data = indices + nnz * 4;
    (0..n)
        .map(|i| {
            let (lo, hi) = (i64_at(indptr + i * 8), i64_at(indptr + (i + 1) * 8));
            (lo..hi)
                .map(|j| {
                    u32::from_le_bytes(
                        bytes[indices + j * 4..indices + j * 4 + 4]
                            .try_into()
                            .unwrap(),
                    )
                })
                .collect()
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = Path::new(args.get(1).map(String::as_str).unwrap_or("data/yfcc"));
    let store = Path::new(args.get(2).map(String::as_str).unwrap_or("data/yfcc/store"));
    let n: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let nq: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(500);
    let beam: u32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(64);
    let k = 10;

    let (dim, base) = read_u8bin(&data.join("base.10M.u8bin"), n);
    let n = base.len() / dim;
    let labels = read_spmat(&data.join("base.metadata.10M.spmat"), n);
    let (_, queries) = read_u8bin(&data.join("query.public.100K.u8bin"), 100_000);
    let qlabels = read_spmat(&data.join("query.metadata.public.100K.spmat"), 100_000);
    println!(
        "{n} records, dim {dim}, {:.1} labels per record",
        labels.iter().map(Vec::len).sum::<usize>() as f64 / n as f64
    );

    // Inverted label lists over the subset, for ground truth and selectivity.
    let mut by_label: HashMap<u32, Vec<u32>> = HashMap::new();
    for (i, ls) in labels.iter().enumerate() {
        for &l in ls {
            by_label.entry(l).or_default().push(i as u32);
        }
    }
    // Queries whose labels all occur in the subset and that have at least k matches.
    let matching = |ql: &[u32]| -> Vec<u32> {
        let mut lists: Vec<&Vec<u32>> = ql.iter().filter_map(|l| by_label.get(l)).collect();
        if lists.len() != ql.len() || lists.is_empty() {
            return Vec::new();
        }
        lists.sort_by_key(|l| l.len());
        lists[0]
            .iter()
            .copied()
            .filter(|&r| lists[1..].iter().all(|l| l.binary_search(&r).is_ok()))
            .collect()
    };
    let picked: Vec<(usize, Vec<u32>)> = (0..100_000)
        .filter_map(|qi| {
            let m = matching(&qlabels[qi]);
            (m.len() >= k).then_some((qi, m))
        })
        .take(nq)
        .collect();
    println!(
        "{} usable queries (labels present, at least {k} matches)",
        picked.len()
    );

    fs::create_dir_all(store).unwrap();
    let path = store.join(format!("yfcc{}k.sana", n / 1000));
    let opts = || {
        Options::new(dim as u32)
            .text_field("tags")
            .auto_checkpoint_delta(0)
            .search_beam(beam)
            .max_search_beam(beam.max(4096))
    };
    let db = Db::open(&path, opts()).unwrap();
    if db.index_stats().is_some_and(|s| s.nodes == n) {
        println!("reusing {}", path.display());
    } else {
        let t = Instant::now();
        let mut v = vec![0f32; dim];
        for i in 0..n {
            for (d, &x) in v.iter_mut().zip(&base[i * dim..(i + 1) * dim]) {
                *d = x as f32;
            }
            let tags: String = labels[i]
                .iter()
                .map(|l| format!("t{l}"))
                .collect::<Vec<_>>()
                .join(" ");
            db.insert(
                i as u64,
                &v,
                HashMap::from([("tags".to_string(), Value::from(tags))]),
            )
            .unwrap();
        }
        println!("insert: {:.1?}", t.elapsed());
        let t = Instant::now();
        db.checkpoint().unwrap();
        println!(
            "build with attribute columns and text index: {:.1?}, index {:.0} MB",
            t.elapsed(),
            fs::metadata(db.index_path().unwrap()).unwrap().len() as f64 / 1e6
        );
    }

    // Ground truth by brute force within the matching set (L2 on the u8 values).
    let truth: Vec<Vec<u64>> = picked
        .par_iter()
        .map(|(qi, m)| {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let mut d: Vec<(u32, u64)> = m
                .iter()
                .map(|&r| {
                    let v = &base[r as usize * dim..(r as usize + 1) * dim];
                    (
                        q.iter()
                            .zip(v)
                            .map(|(&a, &b)| (a as i32 - b as i32).pow(2) as u32)
                            .sum(),
                        r as u64,
                    )
                })
                .collect();
            d.sort_unstable();
            d.iter().take(k).map(|e| e.1).collect()
        })
        .collect();

    // Buckets by selectivity.
    let buckets = [
        (0usize, 100usize),
        (100, 1_000),
        (1_000, 10_000),
        (10_000, usize::MAX),
    ];
    let mut stats: Vec<[f64; 6]> = vec![[0.0; 6]; buckets.len()]; // n, col hits, col time, clo hits, clo time, matches
    let mut qv = vec![0f32; dim];
    for ((qi, m), truth) in picked.iter().zip(&truth) {
        for (d, &x) in qv.iter_mut().zip(&queries[qi * dim..(qi + 1) * dim]) {
            *d = x as f32;
        }
        let f = Filter::and(
            qlabels[*qi]
                .iter()
                .map(|l| Filter::contains("tags", format!("t{l}"))),
        );
        let t = Instant::now();
        let hits = db.search_where(&qv, k, &f).unwrap();
        let col_t = t.elapsed().as_secs_f64();
        let col_hits = hits.iter().filter(|h| truth.contains(&h.id)).count();
        let want: Vec<String> = qlabels[*qi].iter().map(|l| format!("t{l}")).collect();
        let t = Instant::now();
        let hits = db
            .search_filtered(&qv, k, |meta| {
                meta.get("tags").and_then(|v| v.as_str()).is_some_and(|s| {
                    let toks: Vec<&str> = s.split(' ').collect();
                    want.iter().all(|w| toks.contains(&w.as_str()))
                })
            })
            .unwrap();
        let clo_t = t.elapsed().as_secs_f64();
        let clo_hits = hits.iter().filter(|h| truth.contains(&h.id)).count();
        let b = buckets
            .iter()
            .position(|(lo, hi)| m.len() >= *lo && m.len() < *hi)
            .unwrap();
        let s = &mut stats[b];
        s[0] += 1.0;
        s[1] += col_hits as f64;
        s[2] += col_t;
        s[3] += clo_hits as f64;
        s[4] += clo_t;
        s[5] += m.len() as f64;
    }
    println!(
        "beam {beam}, k={k}; recall@10 and mean latency, filter via text index (col) against the closure post-filter (clo)"
    );
    println!(
        "{:<18} {:>5} {:>10} | {:>8} {:>9} | {:>8} {:>9}",
        "matches in subset", "qs", "avg match", "col rec", "col mean", "clo rec", "clo mean"
    );
    for ((lo, hi), s) in buckets.iter().zip(&stats) {
        if s[0] == 0.0 {
            continue;
        }
        let range = if *hi == usize::MAX {
            format!("{lo}+")
        } else {
            format!("{lo}..{hi}")
        };
        println!(
            "{:<18} {:>5} {:>10.0} | {:>8.4} {:>7.2}ms | {:>8.4} {:>7.2}ms",
            range,
            s[0],
            s[5] / s[0],
            s[1] / (s[0] * k as f64),
            1000.0 * s[2] / s[0],
            s[3] / (s[0] * k as f64),
            1000.0 * s[4] / s[0]
        );
    }
}
