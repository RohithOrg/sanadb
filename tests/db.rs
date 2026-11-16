use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;

use sanadb::{Db, Metadata, Metric, Options, Value};

/// Process start, for timestamping diagnostics across tests.
static START: std::sync::LazyLock<std::time::Instant> =
    std::sync::LazyLock::new(std::time::Instant::now);

fn meta(pairs: &[(&str, &str)]) -> Metadata {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), (*v).into()))
        .collect()
}

fn vec_of(dim: usize, seed: u64) -> Vec<f32> {
    // Cheap deterministic pseudo-random vector.
    let mut x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..dim)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

#[test]
fn insert_search_delete() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path().join("v.sana"), Options::new(4)).unwrap();
    db.insert(1, &[0.0, 0.0, 0.0, 0.0], meta(&[("kind", "a")]))
        .unwrap();
    db.insert(2, &[1.0, 0.0, 0.0, 0.0], meta(&[("kind", "b")]))
        .unwrap();
    db.insert(3, &[5.0, 5.0, 5.0, 5.0], meta(&[("kind", "a")]))
        .unwrap();

    let hits = db.search(&[0.9, 0.0, 0.0, 0.0], 2).unwrap();
    assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![2, 1]);

    let hits = db
        .search_filtered(&[0.9, 0.0, 0.0, 0.0], 2, |m| m["kind"] == "a")
        .unwrap();
    assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![1, 3]);

    assert!(db.delete(2).unwrap());
    assert!(!db.delete(2).unwrap());
    let hits = db.search(&[0.9, 0.0, 0.0, 0.0], 5).unwrap();
    assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![1, 3]);
    assert_eq!(db.len(), 2);
}

#[test]
fn upsert_replaces_vector_and_meta() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path().join("v.sana"), Options::new(2)).unwrap();
    db.insert(7, &[0.0, 0.0], meta(&[("v", "1")])).unwrap();
    db.insert(7, &[9.0, 9.0], meta(&[("v", "2")])).unwrap();
    assert_eq!(db.len(), 1);
    let (v, m) = db.get(7).unwrap();
    assert_eq!(v, &[9.0, 9.0]);
    assert_eq!(m["v"], "2");
}

#[test]
fn reopen_replays_log() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.sana");
    {
        let db = Db::open(&path, Options::new(3)).unwrap();
        db.insert(1, &[1.0, 2.0, 3.0], meta(&[("a", "b")])).unwrap();
        db.insert(2, &[4.0, 5.0, 6.0], HashMap::new()).unwrap();
        db.delete(1).unwrap();
        db.insert(3, &[7.0, 8.0, 9.0], meta(&[("x", "y")])).unwrap();
    } // dropped without sync or checkpoint
    let db = Db::open(&path, Options::new(3)).unwrap();
    assert_eq!(db.len(), 2);
    assert!(db.get(1).is_none());
    assert_eq!(db.get(2).unwrap().0, &[4.0, 5.0, 6.0]);
    assert_eq!(db.get(3).unwrap().1["x"], "y");
}

#[test]
fn torn_tail_is_truncated_not_fatal() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.sana");
    {
        let db = Db::open(&path, Options::new(2)).unwrap();
        db.insert(1, &[1.0, 1.0], HashMap::new()).unwrap();
        db.insert(2, &[2.0, 2.0], HashMap::new()).unwrap();
        db.sync().unwrap();
    }
    // Simulate a crash mid-write: append half a frame of junk.
    OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(&[0x20, 0, 0, 0, 1, 2, 3])
        .unwrap();
    let before = std::fs::metadata(&path).unwrap().len();
    let db = Db::open(&path, Options::new(2)).unwrap();
    assert_eq!(db.len(), 2);
    assert_eq!(std::fs::metadata(&path).unwrap().len(), before - 7);
    // Store still writable after repair.
    db.insert(3, &[3.0, 3.0], HashMap::new()).unwrap();
    drop(db);
    assert_eq!(Db::open(&path, Options::new(2)).unwrap().len(), 3);
}

#[test]
fn checkpoint_compacts_and_preserves_data() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.sana");
    let db = Db::open(&path, Options::new(8)).unwrap();
    for i in 0..200u64 {
        db.insert(i, &vec_of(8, i), meta(&[("i", &i.to_string())]))
            .unwrap();
    }
    for i in 0..150u64 {
        db.delete(i).unwrap();
    }
    let big = std::fs::metadata(&path).unwrap().len();
    db.checkpoint().unwrap();
    let small = std::fs::metadata(&path).unwrap().len();
    assert!(small < big / 2, "{small} vs {big}");
    db.insert(999, &vec_of(8, 999), HashMap::new()).unwrap();
    drop(db);
    let db = Db::open(&path, Options::new(8)).unwrap();
    assert_eq!(db.len(), 51);
    assert_eq!(db.get(199).unwrap().1["i"], "199");
    assert!(db.get(999).is_some());
}

#[test]
fn header_mismatch_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.sana");
    Db::open(&path, Options::new(4)).unwrap();
    assert!(Db::open(&path, Options::new(8)).is_err());
    assert!(Db::open(&path, Options::new(4).metric(Metric::Dot)).is_err());
    let db = Db::open(&path, Options::new(4)).unwrap();
    assert!(db.search(&[0.0; 3], 1).is_err());
}

fn brute_force_top(vectors: &[Vec<f32>], q: &[f32], k: usize) -> Vec<u64> {
    let mut all: Vec<(f32, u64)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            (
                v.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum::<f32>(),
                i as u64,
            )
        })
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0));
    all[..k].iter().map(|e| e.1).collect()
}

#[test]
fn graph_recall_vs_brute_force() {
    let (dim, n, k) = (32usize, 5000u64, 10usize);
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path().join("v.sana"), Options::new(dim as u32)).unwrap();
    let vectors: Vec<Vec<f32>> = (0..n).map(|i| vec_of(dim, i)).collect();
    for (i, v) in vectors.iter().enumerate() {
        db.insert(i as u64, v, HashMap::new()).unwrap();
    }
    db.checkpoint().unwrap();
    assert_eq!(db.delta_len(), 0);
    assert_eq!(db.len(), n as usize);
    let stats = db.index_stats().unwrap();
    assert_eq!(stats.nodes, n as usize);
    assert_eq!(
        stats.pq_subspaces, 8,
        "dim 32 should quantize to 8 subspaces"
    );

    let mut hit = 0;
    let queries = 50;
    for qi in 0..queries {
        let q = vec_of(dim, 1_000_000 + qi);
        let truth = brute_force_top(&vectors, &q, k);
        let got = db.search(&q, k).unwrap();
        assert_eq!(got.len(), k);
        hit += got.iter().filter(|h| truth.contains(&h.id)).count();
    }
    let recall = hit as f32 / (queries * k as u64) as f32;
    assert!(recall >= 0.9, "recall@10 = {recall}");
    eprintln!("recall@10 = {recall}");
}

#[test]
fn graph_survives_reopen_upsert_delete_and_recheckpoint() {
    let (dim, n) = (16usize, 300u64);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.sana");
    let db = Db::open(&path, Options::new(dim as u32)).unwrap();
    for i in 0..n {
        db.insert(i, &vec_of(dim, i), meta(&[("i", &i.to_string())]))
            .unwrap();
    }
    db.checkpoint().unwrap();
    drop(db);

    // Log holds metadata only now (about 32 bytes per record here), smaller
    // than the 64-byte vectors it describes.
    let log_bytes = std::fs::metadata(&path).unwrap().len();
    assert!(
        log_bytes < n * dim as u64 * 4 * 3 / 5,
        "log is {log_bytes} bytes"
    );
    {
        let db = Db::open(&path, Options::new(dim as u32)).unwrap();
        let idx = db.index_path().expect("index exists after checkpoint");
        assert!(
            idx.exists()
                && idx
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .starts_with("v.sana.idx."),
            "{idx:?}"
        );
    }

    let db = Db::open(&path, Options::new(dim as u32)).unwrap();
    assert_eq!(db.len(), n as usize);
    assert_eq!(db.delta_len(), 0);
    // Too few nodes for k-means: walk on full vectors.
    assert_eq!(db.index_stats().unwrap().pq_subspaces, 0);
    let (v, m) = db.get(42).unwrap();
    assert_eq!(v, &vec_of(dim, 42)[..]);
    assert_eq!(m["i"], "42");

    // Delete hides a graph record; upsert moves one.
    assert!(db.delete(5).unwrap());
    let far = vec![100.0; dim];
    db.insert(7, &far, meta(&[("i", "moved")])).unwrap();
    assert_eq!(db.delta_len(), 1);

    let q5 = vec_of(dim, 5);
    assert!(db.search(&q5, 10).unwrap().iter().all(|h| h.id != 5));
    let q7 = vec_of(dim, 7);
    assert!(
        db.search(&q7, 10).unwrap().iter().all(|h| h.id != 7),
        "old position of 7 still visible"
    );
    let near_far = db.search(&far, 1).unwrap();
    assert_eq!(near_far[0].id, 7);
    assert_eq!(near_far[0].score, 0.0);
    drop(db);

    // Replay must re-derive the tombstones.
    let db = Db::open(&path, Options::new(dim as u32)).unwrap();
    assert_eq!(db.len(), n as usize - 1);
    assert!(db.get(5).is_none());
    assert_eq!(db.get(7).unwrap().0, &far[..]);
    assert!(db.search(&q7, 10).unwrap().iter().all(|h| h.id != 7));

    // Second checkpoint folds delta and tombstones into a fresh graph.
    db.checkpoint().unwrap();
    assert_eq!(db.delta_len(), 0);
    assert_eq!(db.len(), n as usize - 1);
    assert_eq!(db.search(&far, 1).unwrap()[0].id, 7);
    assert!(db.get(5).is_none());
    for i in (0..n).filter(|&i| i != 5 && i != 7) {
        assert_eq!(db.get(i).unwrap().0, &vec_of(dim, i)[..], "id {i}");
    }
}

#[test]
fn filtered_graph_search_respects_filter() {
    let dim = 8usize;
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(
        dir.path().join("v.sana"),
        Options::new(dim as u32).search_beam(256),
    )
    .unwrap();
    for i in 0..1000u64 {
        db.insert(
            i,
            &vec_of(dim, i),
            meta(&[("parity", if i % 2 == 0 { "even" } else { "odd" })]),
        )
        .unwrap();
    }
    db.checkpoint().unwrap();
    let q = vec_of(dim, 777_777);
    let hits = db.search_filtered(&q, 5, |m| m["parity"] == "odd").unwrap();
    assert_eq!(hits.len(), 5);
    assert!(hits.iter().all(|h| h.id % 2 == 1));
}

#[test]
fn pq_off_matches_pq_on_recall() {
    use sanadb::GraphParams;
    let (dim, n, k) = (32usize, 3000u64, 10usize);
    let vectors: Vec<Vec<f32>> = (0..n).map(|i| vec_of(dim, i)).collect();
    let mut recalls = Vec::new();
    for use_pq in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let params = GraphParams {
            use_pq,
            ..GraphParams::default()
        };
        let db = Db::open(
            dir.path().join("v.sana"),
            Options::new(dim as u32).graph(params),
        )
        .unwrap();
        for (i, v) in vectors.iter().enumerate() {
            db.insert(i as u64, v, HashMap::new()).unwrap();
        }
        db.checkpoint().unwrap();
        assert_eq!(
            db.index_stats().unwrap().pq_subspaces,
            if use_pq { 8 } else { 0 }
        );
        let mut hit = 0;
        for qi in 0..30u64 {
            let q = vec_of(dim, 3_000_000 + qi);
            let truth = brute_force_top(&vectors, &q, k);
            hit += db
                .search(&q, k)
                .unwrap()
                .iter()
                .filter(|h| truth.contains(&h.id))
                .count();
        }
        recalls.push(hit as f32 / 300.0);
    }
    eprintln!("recall@10 exact-walk={} pq-walk={}", recalls[0], recalls[1]);
    assert!(recalls[0] >= 0.9 && recalls[1] >= 0.9, "{recalls:?}");
}

#[test]
fn writes_and_reads_during_checkpoint_are_kept() {
    // Large enough that the build takes long enough for the writes below to
    // land while it runs (either ordering must still be correct).
    let (dim, n) = (16usize, 30_000u64);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.sana");
    let db = Db::open(&path, Options::new(dim as u32)).unwrap();
    for i in 0..n {
        db.insert(i, &vec_of(dim, i), meta(&[("i", &i.to_string())]))
            .unwrap();
    }
    db.checkpoint().unwrap();
    // Dirty the graph so the next checkpoint has tombstones and delta to fold.
    for i in 0..50u64 {
        db.delete(i).unwrap();
    }

    std::thread::scope(|s| {
        s.spawn(|| db.checkpoint().unwrap());
        // Concurrent writer + reader while the build runs.
        for i in n..n + 500 {
            db.insert(i, &vec_of(dim, i), meta(&[("late", "yes")]))
                .unwrap();
        }
        for i in 100..150u64 {
            db.delete(i).unwrap();
        }
        db.insert(200, &vec![7.0; dim], meta(&[("moved", "yes")]))
            .unwrap();
        for qi in 0..20u64 {
            assert_eq!(
                db.search(&vec_of(dim, 9_000_000 + qi), 10).unwrap().len(),
                10
            );
        }
    });

    let check = |db: &Db| {
        assert_eq!(db.len(), (n + 500 - 100) as usize);
        for i in 0..50u64 {
            assert!(db.get(i).is_none(), "deleted before checkpoint: {i}");
        }
        for i in 100..150u64 {
            assert!(db.get(i).is_none(), "deleted during checkpoint: {i}");
        }
        for i in n..n + 500 {
            assert_eq!(
                db.get(i).unwrap().1["late"],
                "yes",
                "inserted during checkpoint: {i}"
            );
        }
        assert_eq!(db.get(200).unwrap().0, vec![7.0; dim]);
        // Filtered so the adaptive beam makes the lookup exhaustive: an
        // outlier vector may be unreachable by a plain approximate walk.
        let moved = db
            .search_filtered(&vec![7.0; dim], 1, |m| m.contains_key("moved"))
            .unwrap();
        assert_eq!(moved[0].id, 200);
        assert_eq!(moved[0].score, 0.0);
        assert!(
            db.search(&vec_of(dim, 120), 10)
                .unwrap()
                .iter()
                .all(|h| h.id != 120)
        );
    };
    check(&db);
    drop(db);
    let db = Db::open(&path, Options::new(dim as u32)).unwrap();
    check(&db);
    db.checkpoint().unwrap();
    assert_eq!(db.delta_len(), 0);
    assert_eq!(db.index_stats().unwrap().tombstones, 0);
    check(&db);
}

#[test]
fn selective_filter_fills_k_with_small_beam() {
    // 1% of records pass the filter; a beam of 16 visits far fewer than the
    // ~1000 nodes needed to see 10 of them. The adaptive retry must widen it.
    let (dim, n, k) = (16usize, 5000u64, 10usize);
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(
        dir.path().join("v.sana"),
        Options::new(dim as u32).search_beam(16),
    )
    .unwrap();
    let vectors: Vec<Vec<f32>> = (0..n).map(|i| vec_of(dim, i)).collect();
    for (i, v) in vectors.iter().enumerate() {
        db.insert(
            i as u64,
            v,
            meta(&[("tag", if i % 100 == 0 { "rare" } else { "common" })]),
        )
        .unwrap();
    }
    db.checkpoint().unwrap();

    let mut hit = 0;
    let queries = 20;
    for qi in 0..queries {
        let q = vec_of(dim, 2_000_000 + qi);
        let rare: Vec<Vec<f32>> = vectors.iter().step_by(100).cloned().collect();
        let truth: Vec<u64> = brute_force_top(&rare, &q, k)
            .iter()
            .map(|i| i * 100)
            .collect();
        let got = db.search_filtered(&q, k, |m| m["tag"] == "rare").unwrap();
        assert_eq!(got.len(), k, "query {qi} returned {} hits", got.len());
        assert!(got.iter().all(|h| h.id % 100 == 0));
        hit += got.iter().filter(|h| truth.contains(&h.id)).count();
    }
    let recall = hit as f32 / (queries * k as u64) as f32;
    eprintln!("filtered recall@10 = {recall}");
    assert!(recall >= 0.8, "filtered recall@10 = {recall}");

    // With the retry disabled the same query comes back short.
    drop(db);
    let db = Db::open(
        dir.path().join("v.sana"),
        Options::new(dim as u32).search_beam(16).max_search_beam(16),
    )
    .unwrap();
    let got = db
        .search_filtered(&vec_of(dim, 2_000_000), k, |m| m["tag"] == "rare")
        .unwrap();
    assert!(got.len() < k);
}

#[test]
fn sharded_build_matches_single_shard_recall() {
    use sanadb::GraphParams;
    let (dim, n, k) = (32usize, 6000u64, 10usize);
    let vectors: Vec<Vec<f32>> = (0..n).map(|i| vec_of(dim, i)).collect();
    let mut recalls = Vec::new();
    for shard_size in [u32::MAX, 1000] {
        let dir = tempfile::tempdir().unwrap();
        let params = GraphParams {
            shard_size,
            ..GraphParams::default()
        };
        let db = Db::open(
            dir.path().join("v.sana"),
            Options::new(dim as u32).graph(params),
        )
        .unwrap();
        for (i, v) in vectors.iter().enumerate() {
            db.insert(i as u64, v, meta(&[("i", &i.to_string())]))
                .unwrap();
        }
        db.checkpoint().unwrap();
        assert_eq!(db.len(), n as usize);
        assert_eq!(db.index_stats().unwrap().nodes, n as usize);
        // No temp files left behind.
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .filter(|f| f.contains("shard") || f.contains("tmp"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
        for i in [0u64, 2999, 5999] {
            assert_eq!(db.get(i).unwrap().0, vectors[i as usize]);
        }
        let mut hit = 0;
        for qi in 0..40u64 {
            let q = vec_of(dim, 4_000_000 + qi);
            let truth = brute_force_top(&vectors, &q, k);
            hit += db
                .search(&q, k)
                .unwrap()
                .iter()
                .filter(|h| truth.contains(&h.id))
                .count();
        }
        recalls.push(hit as f32 / 400.0);
    }
    eprintln!(
        "recall@10 single={} sharded(6 shards)={}",
        recalls[0], recalls[1]
    );
    assert!(recalls[1] >= 0.9, "{recalls:?}");
}

#[test]
fn cosine_normalises_and_rejects_non_finite() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.sana");
    let db = Db::open(&path, Options::new(3).metric(Metric::Cosine)).unwrap();
    db.insert(1, &[2.0, 0.0, 0.0], HashMap::new()).unwrap(); // same direction as query
    db.insert(2, &[0.0, 5.0, 0.0], HashMap::new()).unwrap(); // orthogonal
    db.insert(3, &[-1.0, 0.0, 0.0], HashMap::new()).unwrap(); // opposite
    db.insert(4, &[1.0, 1.0, 0.0], HashMap::new()).unwrap(); // 45 degrees
    let hits = db.search(&[10.0, 0.0, 0.0], 4).unwrap();
    assert_eq!(
        hits.iter().map(|h| h.id).collect::<Vec<_>>(),
        vec![1, 4, 2, 3]
    );
    assert!(
        (hits[0].score + 1.0).abs() < 1e-6,
        "score {}",
        hits[0].score
    );
    assert!(
        (hits[1].score + std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5,
        "score {}",
        hits[1].score
    );
    assert!(hits[2].score.abs() < 1e-6);
    assert!((hits[3].score - 1.0).abs() < 1e-6);
    assert_eq!(db.get(1).unwrap().0, vec![1.0, 0.0, 0.0]);

    assert!(matches!(
        db.insert(5, &[f32::NAN, 0.0, 0.0], HashMap::new()),
        Err(sanadb::Error::InvalidVector)
    ));
    assert!(
        matches!(
            db.insert(6, &[0.0, 0.0, 0.0], HashMap::new()),
            Err(sanadb::Error::InvalidVector)
        ),
        "zero vector has no cosine"
    );
    assert!(matches!(
        db.search(&[f32::INFINITY, 0.0, 0.0], 1),
        Err(sanadb::Error::InvalidVector)
    ));
    assert_eq!(db.len(), 4);

    // Metric survives reopen and a checkpoint keeps normalised vectors.
    db.checkpoint().unwrap();
    drop(db);
    assert!(Db::open(&path, Options::new(3)).is_err());
    let db = Db::open(&path, Options::new(3).metric(Metric::Cosine)).unwrap();
    assert_eq!(db.search(&[1.0, 0.0, 0.0], 1).unwrap()[0].id, 1);
}

#[test]
fn delta_survives_log_growth_and_remap() {
    // 150K x 128 floats is ~80 MB of log: past the 64 MB growth step, so the
    // mapping is rebuilt at least once while the delta is live.
    let (dim, n) = (128usize, 150_000u64);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.sana");
    let db = Db::open(&path, Options::new(dim as u32)).unwrap();
    for i in 0..n {
        db.insert(i, &vec_of(dim, i), HashMap::new()).unwrap();
    }
    assert_eq!(db.delta_len(), n as usize);
    let log = std::fs::metadata(&path).unwrap().len();
    assert!(log > 64 << 20, "log is {log} bytes");

    let check = |db: &Db| {
        for &i in &[0u64, 1, 77_777, n - 1] {
            assert_eq!(db.get(i).unwrap().0, vec_of(dim, i), "id {i}");
            let hits = db.search(&vec_of(dim, i), 1).unwrap();
            assert_eq!(hits[0].id, i);
            assert_eq!(hits[0].score, 0.0);
        }
    };
    check(&db);
    drop(db);
    // Reopen trims growth slack and must still serve every vector.
    let db = Db::open(&path, Options::new(dim as u32)).unwrap();
    assert!(std::fs::metadata(&path).unwrap().len() < log);
    assert_eq!(db.len(), n as usize);
    check(&db);
}

type V1Record<'a> = (u64, Vec<f32>, &'a [(&'a str, &'a str)]);

/// Write a log in the 0.1.1 format (`SANADB01`, unpadded frames) by hand.
fn write_v1_log(path: &std::path::Path, dim: u32, records: &[V1Record], deletes: &[u64]) {
    let mut out = Vec::new();
    out.extend_from_slice(b"SANADB01");
    out.extend_from_slice(&dim.to_le_bytes());
    out.extend_from_slice(&[0, 0, 0, 0]); // metric L2, pad
    let mut frame = |body: &[u8]| {
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&crc32fast::hash(body).to_le_bytes());
        out.extend_from_slice(body);
    };
    for (id, v, m) in records {
        let mut b = vec![1u8];
        b.extend_from_slice(&id.to_le_bytes());
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b.extend_from_slice(&(m.len() as u16).to_le_bytes());
        for (k, val) in m.iter() {
            b.extend_from_slice(&(k.len() as u16).to_le_bytes());
            b.extend_from_slice(k.as_bytes());
            b.extend_from_slice(&(val.len() as u16).to_le_bytes());
            b.extend_from_slice(val.as_bytes());
        }
        frame(&b);
    }
    for id in deletes {
        let mut b = vec![2u8];
        b.extend_from_slice(&id.to_le_bytes());
        frame(&b);
    }
    std::fs::write(path, out).unwrap();
}

#[test]
fn opens_and_migrates_a_0_1_1_log() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("old.sana");
    write_v1_log(
        &path,
        4,
        &[
            (1, vec![1.0, 0.0, 0.0, 0.0], &[("k", "a")]),
            (2, vec![0.0, 1.0, 0.0, 0.0], &[]),
            (3, vec![0.0, 0.0, 1.0, 0.0], &[("k", "c")]),
        ],
        &[2],
    );
    let db = Db::open(&path, Options::new(4)).unwrap();
    assert_eq!(db.len(), 2);
    assert_eq!(
        db.get(1).unwrap(),
        (vec![1.0, 0.0, 0.0, 0.0], meta(&[("k", "a")]))
    );
    assert!(db.get(2).is_none());
    assert_eq!(db.search(&[0.0, 0.0, 0.9, 0.0], 1).unwrap()[0].id, 3);
    db.insert(4, &[0.0, 0.0, 0.0, 1.0], HashMap::new()).unwrap();
    drop(db);
    // Now a current-format file; opens again without migration.
    let head = &std::fs::read(&path).unwrap()[..8];
    assert_eq!(head, b"SANADB02");
    let db = Db::open(&path, Options::new(4)).unwrap();
    assert_eq!(db.len(), 3);
    // Dimension mismatch is reported before anything is rewritten.
    let other = dir.path().join("old2.sana");
    write_v1_log(&other, 4, &[(1, vec![1.0; 4], &[])], &[]);
    assert!(matches!(
        Db::open(&other, Options::new(8)),
        Err(sanadb::Error::HeaderMismatch { .. })
    ));
    assert_eq!(&std::fs::read(&other).unwrap()[..8], b"SANADB01");
}

#[test]
fn unknown_frame_kinds_are_skipped_and_newer_levels_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.sana");
    {
        let db = Db::open(&path, Options::new(2)).unwrap();
        db.insert(1, &[1.0, 0.0], HashMap::new()).unwrap();
        db.sync().unwrap();
    }
    // Append a well-formed frame of kind 200, as a future version might.
    let body = [200u8, 0, 0, 0, 7, 7, 7, 7];
    let mut f = OpenOptions::new().append(true).open(&path).unwrap();
    f.write_all(&(body.len() as u32).to_le_bytes()).unwrap();
    f.write_all(&crc32fast::hash(&body).to_le_bytes()).unwrap();
    f.write_all(&body).unwrap();
    // And a record after it, which a reader must still reach.
    let mut rec = vec![1u8, 0, 0, 0];
    rec.extend_from_slice(&2u64.to_le_bytes());
    rec.extend_from_slice(&bytemuck_free_f32s(&[0.0, 1.0]));
    rec.extend_from_slice(&0u16.to_le_bytes());
    while rec.len() % 4 != 0 {
        rec.push(0);
    }
    f.write_all(&(rec.len() as u32).to_le_bytes()).unwrap();
    f.write_all(&crc32fast::hash(&rec).to_le_bytes()).unwrap();
    f.write_all(&rec).unwrap();
    drop(f);
    let size_before = std::fs::metadata(&path).unwrap().len();
    let db = Db::open(&path, Options::new(2)).unwrap();
    assert_eq!(db.len(), 2, "record after the unknown frame must survive");
    assert_eq!(db.get(2).unwrap().0, vec![0.0, 1.0]);
    drop(db);
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        size_before,
        "nothing truncated"
    );

    // A header that demands a newer reader is refused, not misread.
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[13] = 99;
    std::fs::write(&path, bytes).unwrap();
    assert!(matches!(
        Db::open(&path, Options::new(2)),
        Err(sanadb::Error::NeedsNewerVersion { file_level: 99, .. })
    ));
}

fn bytemuck_free_f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn incremental_checkpoint_matches_full_rebuild() {
    use sanadb::GraphParams;
    let (dim, n, k) = (24usize, 6000u64, 10usize);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.sana");
    let db = Db::open(&path, Options::new(dim as u32).auto_checkpoint_delta(0)).unwrap();
    let mut live: std::collections::BTreeMap<u64, Vec<f32>> =
        (0..n).map(|i| (i, vec_of(dim, i))).collect();
    for (i, v) in &live {
        db.insert(*i, v, meta(&[("i", &i.to_string())])).unwrap();
    }
    db.checkpoint().unwrap(); // full build
    let t_full = {
        let t = std::time::Instant::now();
        db.insert(999_999, &vec_of(dim, 999_999), HashMap::new())
            .unwrap();
        db.delete(999_999).unwrap();
        t.elapsed()
    };
    let _ = t_full;

    // Small change: 300 inserts, 200 deletes, 50 upserts, then an in-place update.
    for i in n..n + 300 {
        db.insert(i, &vec_of(dim, i), HashMap::new()).unwrap();
        live.insert(i, vec_of(dim, i));
    }
    for i in 0..200u64 {
        db.delete(i).unwrap();
        live.remove(&i);
    }
    for i in 1000..1050u64 {
        let v = vec_of(dim, i + 7_000_000);
        db.insert(i, &v, meta(&[("moved", "1")])).unwrap();
        live.insert(i, v);
    }
    let t = std::time::Instant::now();
    db.checkpoint().unwrap();
    let t_incr = t.elapsed();
    let stats = db.index_stats().unwrap();
    assert_eq!(stats.nodes, live.len(), "graph holds exactly the live set");
    assert_eq!(stats.tombstones, 0);
    assert_eq!(db.delta_len(), 0);
    for i in 0..200u64 {
        assert!(db.get(i).is_none());
    }
    assert_eq!(db.get(1000).unwrap().0, live[&1000]);

    // Recall against brute force over the live set, compared with a from-scratch build.
    let ids: Vec<u64> = live.keys().copied().collect();
    let vecs: Vec<Vec<f32>> = live.values().cloned().collect();
    let recall = |db: &Db| {
        let mut hit = 0;
        for qi in 0..40u64 {
            let q = vec_of(dim, 5_000_000 + qi);
            let truth: Vec<u64> = brute_force_top(&vecs, &q, k)
                .into_iter()
                .map(|j| ids[j as usize])
                .collect();
            hit += db
                .search(&q, k)
                .unwrap()
                .iter()
                .filter(|h| truth.contains(&h.id))
                .count();
        }
        hit as f32 / 400.0
    };
    let r_incr = recall(&db);
    drop(db);
    let dir2 = tempfile::tempdir().unwrap();
    let full = Db::open(
        dir2.path().join("f.sana"),
        Options::new(dim as u32).graph(GraphParams::default()),
    )
    .unwrap();
    for (i, v) in &live {
        full.insert(*i, v, HashMap::new()).unwrap();
    }
    let t = std::time::Instant::now();
    full.checkpoint().unwrap();
    let t_rebuild = t.elapsed();
    let r_full = recall(&full);
    eprintln!(
        "recall incremental={r_incr} rebuild={r_full}; time incremental={t_incr:?} rebuild={t_rebuild:?}"
    );
    assert!(
        r_incr >= 0.9 && r_incr >= r_full - 0.03,
        "incremental {r_incr} vs rebuild {r_full}"
    );
    // Timing is printed, not asserted: CI runners make it flaky.
    let _ = (t_incr, t_rebuild);

    // Reopen the incrementally updated store and check it again.
    let db = Db::open(&path, Options::new(dim as u32)).unwrap();
    assert_eq!(db.len(), live.len());
    assert!(recall(&db) >= 0.9);
}

/// Child half of the crash test: write forever until killed. Only runs when
/// the parent sets the env var; as a plain test it returns at once.
#[test]
fn crash_child_writes_until_killed() {
    let Ok(path) = std::env::var("SANADB_CRASH_CHILD") else {
        return;
    };
    let db = Db::open(&path, Options::new(8).auto_checkpoint_delta(400)).unwrap();
    let mut i = db.len() as u64;
    loop {
        db.insert(i, &vec_of(8, i), meta(&[("i", &i.to_string())]))
            .unwrap();
        if i > 0 && i.is_multiple_of(7) {
            db.delete(i - 1).unwrap();
        }
        i += 1;
    }
}

#[test]
fn survives_being_killed_mid_write() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("crash.sana");
    let exe = std::env::current_exe().unwrap();
    let mut expected_min = 0usize;
    for round in 0..6 {
        eprintln!(
            "crash test: spawning child round {round} at {:?}",
            START.elapsed()
        );
        let mut child = std::process::Command::new(&exe)
            .args([
                "crash_child_writes_until_killed",
                "--exact",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("SANADB_CRASH_CHILD", &path)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        // Let it write (and, every ~400 records, checkpoint) for a while, then SIGKILL.
        std::thread::sleep(std::time::Duration::from_millis(150 + 90 * round));
        child.kill().unwrap();
        child.wait().unwrap();
        eprintln!(
            "crash test: child round {round} gone at {:?}",
            START.elapsed()
        );

        let db = Db::open(&path, Options::new(8))
            .unwrap_or_else(|e| panic!("round {round}: reopen failed: {e}"));
        let n = db.len();
        assert!(
            n >= expected_min,
            "round {round}: records went backwards: {n} < {expected_min}"
        );
        // Every surviving record must be fully readable and searchable.
        let mut seen = 0;
        for id in 0..(n as u64 * 2 + 10) {
            if let Some((v, m)) = db.get(id) {
                assert_eq!(v, vec_of(8, id), "round {round}: vector of {id}");
                assert_eq!(m["i"], id.to_string());
                let top = db.search(&v, 3).unwrap();
                assert_eq!(
                    top[0].id,
                    id,
                    "round {round}: own-vector search for {id} of {n} records returned {:?}; index {:?}, delta {}",
                    top.iter().map(|h| (h.id, h.score)).collect::<Vec<_>>(),
                    db.index_stats(),
                    db.delta_len()
                );
                seen += 1;
            }
        }
        assert_eq!(
            seen, n,
            "round {round}: len disagrees with readable records"
        );
        expected_min = n;
        eprintln!(
            "round {round}: {n} records survived, stats {:?}",
            db.index_stats()
        );
    }
    assert!(expected_min > 0, "the child never got to write anything");
}

#[test]
fn corrupted_logs_never_panic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fuzz.sana");
    {
        let db = Db::open(&path, Options::new(4)).unwrap();
        for i in 0..50u64 {
            db.insert(i, &vec_of(4, i), meta(&[("k", "v")])).unwrap();
        }
        db.sync().unwrap();
    }
    let clean = std::fs::read(&path).unwrap();
    let mut rng = 0x5EEDu64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut outcomes = [0usize; 3]; // ok, corrupt error, other error
    for iter in 0..300 {
        let mut bytes = clean.clone();
        for _ in 0..1 + next() % 4 {
            let pos = (next() % bytes.len() as u64) as usize;
            bytes[pos] ^= 1 << (next() % 8);
        }
        if next().is_multiple_of(5) {
            let cut = (next() % bytes.len() as u64) as usize;
            bytes.truncate(cut);
        }
        std::fs::write(&path, &bytes).unwrap();
        for e in std::fs::read_dir(dir.path()).unwrap().flatten() {
            if e.file_name().to_str().unwrap().starts_with("fuzz.sana.idx") {
                let _ = std::fs::remove_file(e.path());
            }
        }
        match Db::open(&path, Options::new(4)) {
            Ok(db) => {
                // Whatever survived must be internally consistent.
                for id in 0..50u64 {
                    if let Some((v, _)) = db.get(id) {
                        assert_eq!(v, vec_of(4, id));
                    }
                }
                let _ = db.search(&[0.0; 4], 5).unwrap();
                outcomes[0] += 1;
            }
            Err(sanadb::Error::Corrupt(_))
            | Err(sanadb::Error::HeaderMismatch { .. })
            | Err(sanadb::Error::NeedsNewerVersion { .. }) => outcomes[1] += 1,
            Err(e) => {
                // How long does it stay held? Retry every millisecond and
                // time it, then ask lsof; a holder that outlives the retry
                // window shows up there, a release lag does not.
                let t0 = std::time::Instant::now();
                let mut held_for = None;
                for _ in 0..500 {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    if Db::open(&path, Options::new(4)).is_ok() {
                        held_for = Some(t0.elapsed());
                        break;
                    }
                }
                eprintln!(
                    "lock held for {held_for:?} after the failure at {:?} since test start",
                    START.elapsed()
                );
                let lock = format!("{}.lock", path.display());
                let who = std::process::Command::new("lsof")
                    .arg(&lock)
                    .output()
                    .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                    .unwrap_or_default();
                let again = Db::open(&path, Options::new(4)).map(|_| ());
                eprintln!(
                    "unexpected error class at iteration {iter} (bytes {}): {e}; this pid {}; lsof:\n{who}retry after 50 ms: {:?}",
                    bytes.len(),
                    std::process::id(),
                    again.map_err(|e| e.to_string())
                );
                outcomes[2] += 1;
            }
        }
    }
    eprintln!(
        "fuzz outcomes: ok={} corrupt={} other={}",
        outcomes[0], outcomes[1], outcomes[2]
    );
    assert_eq!(
        outcomes[2], 0,
        "only Ok or a corruption error is acceptable"
    );
}

#[test]
fn checkpoint_of_empty_store_is_fine() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path().join("v.sana"), Options::new(4)).unwrap();
    db.checkpoint().unwrap();
    assert_eq!(db.index_stats().unwrap().nodes, 0);
    assert!(db.search(&[0.0; 4], 5).unwrap().is_empty());
    db.insert(1, &[1.0; 4], HashMap::new()).unwrap();
    assert_eq!(db.search(&[1.0; 4], 5).unwrap()[0].id, 1);
    db.checkpoint().unwrap();
    assert_eq!(db.search(&[1.0; 4], 5).unwrap()[0].id, 1);
}

#[test]
fn second_open_is_refused_until_first_is_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.sana");
    let first = Db::open(&path, Options::new(4)).unwrap();
    match Db::open(&path, Options::new(4)) {
        Err(sanadb::Error::Locked) => {}
        other => panic!("expected Locked, got {:?}", other.map(|_| ())),
    }
    // The lock survives a checkpoint, which replaces the data files.
    first.insert(1, &[0.0; 4], HashMap::new()).unwrap();
    first.checkpoint().unwrap();
    assert!(matches!(
        Db::open(&path, Options::new(4)),
        Err(sanadb::Error::Locked)
    ));
    drop(first);
    let again = Db::open(&path, Options::new(4)).unwrap();
    assert_eq!(again.len(), 1);
}

#[test]
fn exact_search_matches_brute_force() {
    let dim = 32;
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(
        dir.path().join("v.sana"),
        Options::new(dim as u32).metric(Metric::Dot),
    )
    .unwrap();
    let vectors: Vec<Vec<f32>> = (0..500).map(|i| vec_of(dim, i)).collect();
    for (i, v) in vectors.iter().enumerate() {
        db.insert(i as u64, v, HashMap::new()).unwrap();
    }
    let q = vec_of(dim, 12345);
    let mut expect: Vec<(f32, u64)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (-v.iter().zip(&q).map(|(a, b)| a * b).sum::<f32>(), i as u64))
        .collect();
    expect.sort_by(|a, b| a.0.total_cmp(&b.0));
    let hits = db.search(&q, 10).unwrap();
    assert_eq!(
        hits.iter().map(|h| h.id).collect::<Vec<_>>(),
        expect[..10].iter().map(|e| e.1).collect::<Vec<_>>()
    );
}

#[test]
fn search_many_matches_single_searches() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("many.sana");
    let dim = 16usize;
    let db = Db::open(&path, Options::new(dim as u32)).unwrap();
    let n = 3000u64;
    for i in 0..n {
        let m = if i % 2 == 0 {
            meta(&[("even", "yes")])
        } else {
            HashMap::new()
        };
        db.insert(i, &vec_of(dim, i), m).unwrap();
    }
    db.checkpoint().unwrap();
    for i in n..n + 100 {
        db.insert(i, &vec_of(dim, i), HashMap::new()).unwrap();
    }
    let queries: Vec<f32> = (0..64u64)
        .flat_map(|q| vec_of(dim, 5_000_000 + q * 7))
        .collect();
    let many = db.search_many(&queries, 10).unwrap();
    assert_eq!(many.len(), 64);
    for (row, q) in queries.chunks(dim).enumerate() {
        assert_eq!(many[row], db.search(q, 10).unwrap(), "row {row}");
    }
    let filtered = db
        .search_many_filtered(&queries, 5, |m| m.contains_key("even"))
        .unwrap();
    for (row, q) in queries.chunks(dim).enumerate() {
        assert_eq!(
            filtered[row],
            db.search_filtered(q, 5, |m| m.contains_key("even"))
                .unwrap(),
            "row {row}"
        );
        assert!(filtered[row].iter().all(|h| h.id % 2 == 0));
    }
    assert!(matches!(
        db.search_many(&queries[..dim + 3], 10),
        Err(sanadb::Error::DimMismatch { .. })
    ));
}

#[test]
fn typed_values_round_trip_and_keys_keep_their_type() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("typed.sana");
    let dim = 4usize;
    let m: Metadata = HashMap::from([
        ("doc".to_string(), Value::from("intro.md")),
        ("year".to_string(), Value::from(2024i64)),
        ("score".to_string(), Value::from(0.25f64)),
        ("public".to_string(), Value::from(true)),
    ]);
    let db = Db::open(&path, Options::new(dim as u32)).unwrap();
    db.insert(1, &[1.0, 0.0, 0.0, 0.0], m.clone()).unwrap();
    db.insert(
        2,
        &[0.0, 1.0, 0.0, 0.0],
        HashMap::from([("year".to_string(), Value::from(1999i64))]),
    )
    .unwrap();
    assert_eq!(db.get(1).unwrap().1, m);
    // A key's type is fixed by its first value.
    let bad = db.insert(
        3,
        &[0.0, 0.0, 1.0, 0.0],
        HashMap::from([("year".to_string(), Value::from("2024"))]),
    );
    assert!(
        matches!(bad, Err(sanadb::Error::TypeMismatch { ref key, expected: sanadb::Kind::Int, got: sanadb::Kind::Str }) if key == "year"),
        "{bad:?}"
    );
    assert!(db.get(3).is_none(), "a rejected insert writes nothing");
    // Typed filter through the closure API.
    let hits = db
        .search_filtered(&[0.0, 0.9, 0.0, 0.0], 1, |m| {
            m["year"].as_int() == Some(2024)
        })
        .unwrap();
    assert_eq!(hits[0].id, 1);
    drop(db);
    // Survives replay, checkpoint (values ride through the compacted log) and reopen.
    let db = Db::open(&path, Options::new(dim as u32)).unwrap();
    assert_eq!(db.get(1).unwrap().1, m);
    db.checkpoint().unwrap();
    drop(db);
    let db = Db::open(&path, Options::new(dim as u32)).unwrap();
    if db.get(1).is_none() {
        let log = std::fs::read(&path).unwrap();
        panic!(
            "record 1 vanished after checkpoint + reopen: len={} stats={:?} log bytes={} level={} tail={:?}",
            db.len(),
            db.index_stats(),
            log.len(),
            log[13],
            &log[log.len().saturating_sub(48)..]
        );
    }
    assert_eq!(db.get(1).unwrap().1, m);
    assert_eq!(db.get(2).unwrap().1["year"], Value::Int(1999));
    assert!(matches!(
        db.insert(
            4,
            &[0.0; 4],
            HashMap::from([("public".to_string(), Value::from(1i64))])
        ),
        Err(sanadb::Error::TypeMismatch { .. })
    ));
}

/// Write a reader-level-1 `SANADB02` log by hand: padded frames, bare string values.
fn write_level1_log(path: &std::path::Path, dim: u32, records: &[V1Record], deletes: &[u64]) {
    let mut out = Vec::new();
    out.extend_from_slice(b"SANADB02");
    out.extend_from_slice(&dim.to_le_bytes());
    out.extend_from_slice(&[0, 1, 0, 0]); // metric L2, reader level 1, pad
    let mut frame = |mut body: Vec<u8>| {
        while !body.len().is_multiple_of(4) {
            body.push(0);
        }
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&crc32fast::hash(&body).to_le_bytes());
        out.extend_from_slice(&body);
    };
    for (id, v, m) in records {
        let mut b = vec![1u8, 0, 0, 0];
        b.extend_from_slice(&id.to_le_bytes());
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b.extend_from_slice(&(m.len() as u16).to_le_bytes());
        for (k, val) in m.iter() {
            b.extend_from_slice(&(k.len() as u16).to_le_bytes());
            b.extend_from_slice(k.as_bytes());
            b.extend_from_slice(&(val.len() as u16).to_le_bytes());
            b.extend_from_slice(val.as_bytes());
        }
        frame(b);
    }
    for id in deletes {
        let mut b = vec![2u8, 0, 0, 0];
        b.extend_from_slice(&id.to_le_bytes());
        frame(b);
    }
    std::fs::write(path, out).unwrap();
}

#[test]
fn opens_and_migrates_a_level_1_log() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("l1.sana");
    write_level1_log(
        &path,
        4,
        &[
            (1, vec![1.0, 0.0, 0.0, 0.0], &[("k", "a"), ("n", "7")]),
            (2, vec![0.0, 1.0, 0.0, 0.0], &[]),
            (3, vec![0.0, 0.0, 1.0, 0.0], &[("k", "c")]),
        ],
        &[2],
    );
    let db = Db::open(&path, Options::new(4)).unwrap();
    assert_eq!(db.len(), 2);
    // Old values come back as strings; nothing is guessed from their spelling.
    assert_eq!(db.get(1).unwrap().1, meta(&[("k", "a"), ("n", "7")]));
    assert_eq!(db.get(1).unwrap().1["n"], Value::Str("7".into()));
    assert!(db.get(2).is_none());
    assert_eq!(db.search(&[0.0, 0.0, 0.9, 0.0], 1).unwrap()[0].id, 3);
    // The key keeps the string type it was migrated with.
    assert!(matches!(
        db.insert(
            9,
            &[0.0; 4],
            HashMap::from([("n".to_string(), Value::from(7i64))])
        ),
        Err(sanadb::Error::TypeMismatch { .. })
    ));
    drop(db);
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], b"SANADB02");
    assert!(
        bytes[13] >= 2,
        "header now at the current reader level, got {}",
        bytes[13]
    );
    let db = Db::open(&path, Options::new(4)).unwrap();
    assert_eq!(db.len(), 2);
}

/// Brute-force k-NN over the records that pass `keep`, by id.
fn brute<F: Fn(u64) -> bool>(dim: usize, n: u64, q: &[f32], k: usize, keep: F) -> Vec<u64> {
    let mut d: Vec<(f32, u64)> = (0..n)
        .filter(|&i| keep(i))
        .map(|i| {
            (
                q.iter()
                    .zip(&vec_of(dim, i))
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum(),
                i,
            )
        })
        .collect();
    d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    d.into_iter().take(k).map(|e| e.1).collect()
}

#[test]
fn search_where_uses_columns_scan_and_filtered_walk() {
    use sanadb::Filter;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("where.sana");
    let (dim, n) = (16usize, 20_000u64);
    let db = Db::open(&path, Options::new(dim as u32).exact_filter_scan(500)).unwrap();
    let lang = |i: u64| ["en", "de", "fr", "it", "es"][(i % 5) as usize];
    let rare = |i: u64| i % 400 == 7; // 50 records: scanned exactly
    for i in 0..n {
        let mut m = Metadata::new();
        m.insert("lang".into(), Value::from(lang(i)));
        m.insert("year".into(), Value::from(2000 + (i % 30) as i64));
        m.insert("score".into(), Value::from((i % 1000) as f64 / 1000.0));
        if rare(i) {
            m.insert("tag".into(), Value::from("rare"));
        }
        db.insert(i, &vec_of(dim, i), m).unwrap();
    }
    db.checkpoint().unwrap();
    // A few records written after the checkpoint live in the delta and must
    // still be found by the same filters; one graph record is deleted.
    for i in n..n + 20 {
        let mut m = Metadata::new();
        m.insert("lang".into(), Value::from("en"));
        m.insert("year".into(), Value::from(2029i64));
        m.insert("score".into(), Value::from(0.0f64));
        m.insert("tag".into(), Value::from("rare"));
        db.insert(i, &vec_of(dim, i), m).unwrap();
    }
    db.delete(7).unwrap();
    let total = n + 20;
    let live = |i: u64| i != 7;
    let k = 10;
    let mut walk_recall = 0usize;
    let mut walk_total = 0usize;
    for qi in 0..40u64 {
        let q = vec_of(dim, 9_000_000 + qi);
        // Exact scan path: the rare tag has 50 + 20 matches, under the 500 cap.
        let got: Vec<u64> = db
            .search_where(&q, k, &Filter::eq("tag", "rare"))
            .unwrap()
            .iter()
            .map(|h| h.id)
            .collect();
        let want = brute(dim, total, &q, k, |i| live(i) && (rare(i) || i >= n));
        assert_eq!(got, want, "scan regime must be exact, query {qi}");
        // Scan with a conjunction: posting list seeds, column checks the rest.
        let f = Filter::and([
            Filter::eq("tag", "rare"),
            Filter::ge("year", 2010i64),
            Filter::lt("score", 0.5f64),
        ]);
        let got: Vec<u64> = db
            .search_where(&q, k, &f)
            .unwrap()
            .iter()
            .map(|h| h.id)
            .collect();
        let want = brute(dim, total, &q, k, |i| {
            live(i)
                && (rare(i) || i >= n)
                && (if i >= n { 2029 } else { 2000 + i % 30 }) >= 2010
                && (if i >= n {
                    0.0
                } else {
                    (i % 1000) as f64 / 1000.0
                }) < 0.5
        });
        assert_eq!(got, want, "conjunction scan must be exact, query {qi}");
        // Filtered walk: 20% of records pass, far above the scan cap.
        let got: Vec<u64> = db
            .search_where(&q, k, &Filter::eq("lang", "de"))
            .unwrap()
            .iter()
            .map(|h| h.id)
            .collect();
        assert_eq!(got.len(), k);
        assert!(
            got.iter().all(|&i| live(i) && lang(i) == "de"),
            "every hit passes the filter"
        );
        let want = brute(dim, total, &q, k, |i| live(i) && i < n && lang(i) == "de");
        walk_recall += got.iter().filter(|i| want.contains(i)).count();
        walk_total += k;
        // Range over a float column (no postings) with an Or: walk, all hits pass.
        let f = Filter::or([Filter::lt("score", 0.05f64), Filter::eq("lang", "it")]);
        let got = db.search_where(&q, k, &f).unwrap();
        assert_eq!(got.len(), k);
        assert!(got.iter().all(|h| {
            let (_, m) = db.get(h.id).unwrap();
            f.eval(&m)
        }));
        // Nothing matches: empty, not an error.
        assert!(
            db.search_where(&q, k, &Filter::eq("lang", "zz"))
                .unwrap()
                .is_empty()
        );
        assert!(
            db.search_where(&q, k, &Filter::eq("year", "2001"))
                .unwrap()
                .is_empty(),
            "type mismatch matches nothing"
        );
    }
    let recall = walk_recall as f64 / walk_total as f64;
    eprintln!("filtered walk recall@10 at 20% selectivity: {recall:.3}");
    assert!(recall > 0.9, "filtered walk recall {recall}");
    // A callable filter still works. It is the old post-filter path, so on a
    // 0.35% filter it is approximate where the column path is exact.
    let q = vec_of(dim, 9_000_000);
    let a: Vec<u64> = db
        .search_filtered(&q, k, |m| m.get("tag").is_some_and(|v| v == "rare"))
        .unwrap()
        .iter()
        .map(|h| h.id)
        .collect();
    let b: Vec<u64> = db
        .search_where(&q, k, &Filter::eq("tag", "rare"))
        .unwrap()
        .iter()
        .map(|h| h.id)
        .collect();
    assert_eq!(a.len(), k);
    assert!(a.iter().all(|&i| rare(i) || i >= n));
    let overlap = a.iter().filter(|i| b.contains(i)).count();
    assert!(
        overlap >= k / 2,
        "closure path found {overlap}/{k} of the exact answer"
    );
    // Batch form.
    let qs: Vec<f32> = (0..5u64).flat_map(|i| vec_of(dim, 9_000_000 + i)).collect();
    let many = db
        .search_many_where(&qs, k, &Filter::eq("tag", "rare"))
        .unwrap();
    for (row, q) in qs.chunks(dim).enumerate() {
        assert_eq!(
            many[row],
            db.search_where(q, k, &Filter::eq("tag", "rare")).unwrap()
        );
    }
}

#[test]
fn metadata_lives_in_columns_after_checkpoint() {
    use sanadb::Filter;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cols.sana");
    let (dim, n) = (8usize, 3000u64);
    let db = Db::open(&path, Options::new(dim as u32)).unwrap();
    for i in 0..n {
        let mut m = Metadata::new();
        m.insert("i".into(), Value::from(i as i64));
        m.insert("lang".into(), Value::from(["en", "de"][(i % 2) as usize]));
        if i % 3 == 0 {
            m.insert("flag".into(), Value::from(true));
        }
        db.insert(i, &vec_of(dim, i), m).unwrap();
    }
    db.checkpoint().unwrap();
    drop(db);
    // The compacted log holds the header and the frame naming its index,
    // nothing else: every record's metadata is in the index columns now.
    let log_bytes = std::fs::metadata(&path).unwrap().len();
    assert_eq!(log_bytes, 16 + 8 + 12, "log is {log_bytes} bytes");
    let db = Db::open(&path, Options::new(dim as u32)).unwrap();
    assert_eq!(db.len(), n as usize);
    for i in [0u64, 1, 2, 1500, n - 1] {
        let (v, m) = db.get(i).unwrap();
        assert_eq!(v, vec_of(dim, i));
        assert_eq!(m["i"], Value::Int(i as i64));
        assert_eq!(m["lang"], ["en", "de"][(i % 2) as usize]);
        assert_eq!(m.contains_key("flag"), i % 3 == 0, "{i}");
    }
    // Types are known from the columns alone: no log entry mentions them.
    assert!(matches!(
        db.insert(
            n,
            &vec_of(dim, n),
            HashMap::from([("i".to_string(), Value::from("x"))])
        ),
        Err(sanadb::Error::TypeMismatch { .. })
    ));
    // Deleting and overwriting graph records, then reading back.
    assert!(db.delete(2).unwrap());
    assert!(!db.delete(2).unwrap());
    assert!(db.get(2).is_none());
    db.insert(
        3,
        &vec_of(dim, 3),
        HashMap::from([("i".to_string(), Value::from(-3i64))]),
    )
    .unwrap();
    assert_eq!(db.get(3).unwrap().1["i"], Value::Int(-3));
    assert_eq!(db.len(), n as usize - 1);
    // Closure filters read the columns; expression filters use them directly.
    let q = vec_of(dim, 7_000_000);
    let a: Vec<u64> = db
        .search_filtered(&q, 5, |m| m.get("flag").is_some())
        .unwrap()
        .iter()
        .map(|h| h.id)
        .collect();
    assert!(a.iter().all(|&i| i % 3 == 0));
    let b: Vec<u64> = db
        .search_where(&q, 5, &Filter::eq("flag", true))
        .unwrap()
        .iter()
        .map(|h| h.id)
        .collect();
    assert!(b.iter().all(|&i| i % 3 == 0) && b.len() == 5);
    assert!(db.search(&q, 10).unwrap().iter().all(|h| h.id != 2));
    drop(db);
    // Reopen: the delete and the overwrite come from the small log tail.
    let db = Db::open(&path, Options::new(dim as u32)).unwrap();
    assert_eq!(db.len(), n as usize - 1);
    assert!(db.get(2).is_none());
    assert_eq!(db.get(3).unwrap().1["i"], Value::Int(-3));
    assert_eq!(db.get(5).unwrap().1["i"], Value::Int(5));
    db.checkpoint().unwrap();
    assert_eq!(db.len(), n as usize - 1);
    assert!(db.get(2).is_none());
    assert_eq!(db.get(3).unwrap().1["i"], Value::Int(-3));
    assert_eq!(db.index_stats().unwrap().tombstones, 0);
}

#[test]
fn reader_in_the_same_process_follows_the_writer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shared.sana");
    let dim = 8usize;
    let w = Db::open(&path, Options::new(dim as u32)).unwrap();
    for i in 0..500u64 {
        w.insert(i, &vec_of(dim, i), meta(&[("k", "a")])).unwrap();
    }
    // A reader opens beside the writer: no lock conflict either way.
    let r = Db::open_read_only(&path, Options::new(dim as u32)).unwrap();
    assert_eq!(r.len(), 500);
    assert!(matches!(
        r.insert(9, &vec_of(dim, 9), HashMap::new()),
        Err(sanadb::Error::ReadOnly)
    ));
    assert!(matches!(r.delete(9), Err(sanadb::Error::ReadOnly)));
    assert!(matches!(r.checkpoint(), Err(sanadb::Error::ReadOnly)));
    // Appended frames are visible on the reader's next read.
    w.insert(500, &vec_of(dim, 500), meta(&[("k", "b")]))
        .unwrap();
    w.delete(3).unwrap();
    assert_eq!(r.len(), 500);
    assert_eq!(r.get(500).unwrap().1["k"], "b");
    assert!(r.get(3).is_none());
    // A checkpoint replaces both files under the reader; it reloads and
    // keeps answering from the new index.
    w.checkpoint().unwrap();
    assert_eq!(r.len(), 500);
    assert_eq!(r.get(500).unwrap().1["k"], "b");
    assert!(r.get(3).is_none());
    assert_eq!(r.search(&vec_of(dim, 77), 1).unwrap()[0].id, 77);
    assert_eq!(r.index_stats().unwrap().nodes, 500);
    // Writes after the checkpoint, then a second checkpoint while the reader
    // holds the first new index mapped.
    for i in 1000..1100u64 {
        w.insert(i, &vec_of(dim, i), HashMap::new()).unwrap();
    }
    assert_eq!(r.len(), 600);
    w.checkpoint().unwrap();
    assert_eq!(r.len(), 600);
    assert_eq!(r.search(&vec_of(dim, 1050), 1).unwrap()[0].id, 1050);
    // Only the current index file remains.
    let idx_files: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .flatten()
        .filter(|e| {
            e.file_name()
                .to_str()
                .unwrap()
                .starts_with("shared.sana.idx")
        })
        .collect();
    assert_eq!(idx_files.len(), 1, "{idx_files:?}");
    assert_eq!(idx_files[0].path(), w.index_path().unwrap());
    // The writer closes; the reader goes on.
    drop(w);
    assert_eq!(r.len(), 600);
    assert_eq!(r.get(500).unwrap().1["k"], "b");
    // A second writer opens while the reader is attached.
    let w2 = Db::open(&path, Options::new(dim as u32)).unwrap();
    w2.insert(7, &vec_of(dim, 7), meta(&[("k", "z")])).unwrap();
    assert_eq!(r.get(7).unwrap().1["k"], "z");
}

#[test]
fn orphan_index_files_are_collected_on_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gc.sana");
    let dim = 4usize;
    {
        let db = Db::open(&path, Options::new(dim as u32)).unwrap();
        db.insert(1, &vec_of(dim, 1), HashMap::new()).unwrap();
        db.checkpoint().unwrap();
    }
    // Leftovers a crash could produce: an index nobody names, a snapshot temp file.
    std::fs::write(dir.path().join("gc.sana.idx.00000000deadbeef"), b"junk").unwrap();
    std::fs::write(dir.path().join("gc.sana.idx"), b"junk").unwrap();
    std::fs::write(dir.path().join("gc.sana.vec.tmp"), b"junk").unwrap();
    let db = Db::open(&path, Options::new(dim as u32)).unwrap();
    assert_eq!(db.get(1).unwrap().0, vec_of(dim, 1));
    let names: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_str().unwrap().to_string())
        .collect();
    assert!(
        !names
            .iter()
            .any(|n| n.ends_with("deadbeef") || n == "gc.sana.idx" || n.ends_with(".vec.tmp")),
        "{names:?}"
    );
    assert!(
        names.contains(
            &db.index_path()
                .unwrap()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string()
        )
    );
}

/// Child half of the cross-process reader test: read until the parent's
/// stop file appears, checking every answer for internal consistency.
#[test]
fn reader_child_reads_until_told_to_stop() {
    let Ok(path) = std::env::var("SANADB_READER_CHILD") else {
        return;
    };
    let path = std::path::PathBuf::from(path);
    let stop = path.with_extension("stop");
    let dim = 8usize;
    let r = loop {
        match Db::open_read_only(&path, Options::new(dim as u32)) {
            Ok(r) => break r,
            Err(_) => std::thread::sleep(std::time::Duration::from_millis(5)),
        }
    };
    let mut rounds = 0u64;
    let mut max_len = 0usize;
    while !stop.exists() {
        let n = r.len();
        assert!(n >= max_len, "len went backwards: {n} < {max_len}");
        max_len = n;
        let hits = r.search(&vec_of(dim, rounds % 300), 5).unwrap();
        for h in &hits {
            // A record may be deleted between the search and the get; what
            // is there must be the right vector with the right metadata.
            if let Some((v, m)) = r.get(h.id) {
                assert_eq!(v, vec_of(dim, h.id));
                assert_eq!(m["i"], Value::Int(h.id as i64));
            }
        }
        rounds += 1;
        if rounds.is_multiple_of(5) {
            let _ = std::fs::write(path.with_extension("progress"), rounds.to_string());
        }
    }
    eprintln!("reader child: {rounds} rounds, final len {max_len}");
    assert!(rounds > 0, "the reader never completed a round");
    std::fs::write(path.with_extension("rounds"), rounds.to_string()).unwrap();
}

#[test]
fn reader_in_another_process_survives_writes_and_checkpoints() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("xp.sana");
    let dim = 8usize;
    let w = Db::open(&path, Options::new(dim as u32)).unwrap();
    for i in 0..300u64 {
        w.insert(
            i,
            &vec_of(dim, i),
            HashMap::from([("i".to_string(), Value::from(i as i64))]),
        )
        .unwrap();
    }
    w.checkpoint().unwrap();
    let exe = std::env::current_exe().unwrap();
    let mut child = std::process::Command::new(&exe)
        .args([
            "reader_child_reads_until_told_to_stop",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("SANADB_READER_CHILD", &path)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .unwrap();
    // Write, delete and checkpoint under the reader for a while.
    let t = std::time::Instant::now();
    let mut next = 300u64;
    let mut round = 0;
    let progress = || -> u64 {
        std::fs::read_to_string(path.with_extension("progress"))
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    };
    while t.elapsed() < std::time::Duration::from_millis(1500)
        || (progress() < 20 && t.elapsed() < std::time::Duration::from_secs(60))
    {
        for _ in 0..50 {
            w.insert(
                next,
                &vec_of(dim, next),
                HashMap::from([("i".to_string(), Value::from(next as i64))]),
            )
            .unwrap();
            next += 1;
        }
        w.delete(round * 7 % 300).unwrap_or(false);
        if round % 3 == 2 {
            w.checkpoint().unwrap();
        }
        round += 1;
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    std::fs::write(path.with_extension("stop"), b"").unwrap();
    let status = child.wait().unwrap();
    assert!(status.success(), "reader child failed: {status}");
    let rounds: u64 = std::fs::read_to_string(path.with_extension("rounds"))
        .unwrap()
        .parse()
        .unwrap();
    eprintln!(
        "writer: {round} rounds, {} records; reader: {rounds} rounds",
        w.len()
    );
    assert!(rounds >= 20, "reader did only {rounds} rounds");
    // While the reader mapped an old index, the writer could not delete it on
    // Windows (Unix deletes it at once). Either way, once the reader is gone,
    // the next writer open leaves exactly the current index behind.
    drop(w);
    let w = Db::open(&path, Options::new(dim as u32)).unwrap();
    let idx_files: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_str().unwrap().to_string())
        .filter(|n| n.starts_with("xp.sana.idx"))
        .collect();
    assert_eq!(idx_files.len(), 1, "{idx_files:?}");
    assert_eq!(dir.path().join(&idx_files[0]), w.index_path().unwrap());
}

/// Flip bits in the index file at random; every open must be `Ok` or a
/// corruption error, never a panic, and a store that opens must answer reads
/// without panicking. Reads of a damaged-but-accepted index may return wrong
/// neighbours; they may not read out of bounds.
#[test]
fn corrupted_indexes_never_panic() {
    use sanadb::Filter;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ifuzz.sana");
    let dim = 4usize;
    {
        let db = Db::open(&path, Options::new(dim as u32)).unwrap();
        for i in 0..1200u64 {
            let mut m = Metadata::new();
            m.insert("k".into(), Value::from(["a", "b", "c"][(i % 3) as usize]));
            m.insert("n".into(), Value::from(i as i64));
            m.insert("f".into(), Value::from(i as f64 * 0.5));
            m.insert("b".into(), Value::from(i % 2 == 0));
            db.insert(i, &vec_of(dim, i), m).unwrap();
        }
        db.checkpoint().unwrap();
    }
    let idx_path = {
        let db = Db::open(&path, Options::new(dim as u32)).unwrap();
        db.index_path().unwrap()
    };
    let clean_log = std::fs::read(&path).unwrap();
    let clean_idx = std::fs::read(&idx_path).unwrap();
    let mut rng = 0xC0FFEEu64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut outcomes = [0usize; 3];
    for iter in 0..400 {
        let mut bytes = clean_idx.clone();
        // Mostly the header, tables and footer: the parts that steer reads.
        let hot = [(0usize, 64usize), (bytes.len() - 4096, bytes.len())];
        for _ in 0..1 + next() % 4 {
            let (lo, hi) = if next() % 4 == 0 {
                (0, bytes.len())
            } else {
                hot[(next() % 2) as usize]
            };
            let pos = lo + (next() % (hi - lo) as u64) as usize;
            bytes[pos] ^= 1 << (next() % 8);
        }
        if next().is_multiple_of(7) {
            let cut = (next() % bytes.len() as u64) as usize;
            bytes.truncate(cut.max(64));
        }
        std::fs::write(&idx_path, &bytes).unwrap();
        std::fs::write(&path, &clean_log).unwrap();
        match Db::open(&path, Options::new(dim as u32)) {
            Ok(db) => {
                for id in (0..1200u64).step_by(97) {
                    let _ = db.get(id);
                }
                let _ = db.search(&vec_of(dim, 3), 5);
                let _ = db.search_where(&vec_of(dim, 3), 5, &Filter::eq("k", "b"));
                let _ = db.search_where(
                    &vec_of(dim, 3),
                    5,
                    &Filter::and([
                        Filter::gt("n", 100i64),
                        Filter::lt("f", 400.0f64),
                        Filter::eq("b", true),
                    ]),
                );
                let _ = db.search_filtered(&vec_of(dim, 3), 5, |m| m.contains_key("k"));
                let _ = db.len();
                outcomes[0] += 1;
            }
            Err(sanadb::Error::Corrupt(_))
            | Err(sanadb::Error::NeedsNewerVersion { .. })
            | Err(sanadb::Error::HeaderMismatch { .. }) => outcomes[1] += 1,
            Err(e) => {
                eprintln!("iteration {iter}: unexpected error class: {e}");
                outcomes[2] += 1;
            }
        }
    }
    eprintln!(
        "index fuzz outcomes: ok={} corrupt={} other={}",
        outcomes[0], outcomes[1], outcomes[2]
    );
    assert_eq!(outcomes[2], 0);
}

/// Stores written by the published 0.1.1, 0.1.3 and 0.1.4 crates
/// (`tests/fixtures/`, 68 records of dim 4 with two string keys, one delete
/// and one overwrite before the checkpoint, ten inserts and a delete after)
/// and by 0.2.0 (67 records, typed keys and a text field; the recipe is
/// `current_layout_matches_the_format_document`).
/// Each must open, read back exactly, accept new writes, checkpoint into the
/// current layout and reopen.
#[test]
fn opens_stores_written_by_every_published_version() {
    let vec_of4 = |i: u64| -> Vec<f32> {
        (0..4u64)
            .map(|d| ((i * 7 + d * 13) % 101) as f32 / 101.0)
            .collect()
    };
    for version in ["0.1.1", "0.1.3", "0.1.4", "0.2.0"] {
        let dir = tempfile::tempdir().unwrap();
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(format!("v{version}"));
        for e in std::fs::read_dir(&src).unwrap().flatten() {
            std::fs::copy(e.path(), dir.path().join(e.file_name())).unwrap();
        }
        let path = dir.path().join("fixture.sana");
        let typed = matches!(version, "0.1.4" | "0.2.0");
        let records = if version == "0.2.0" { 67 } else { 68 };
        let check = |db: &Db, stage: &str, len: usize| {
            assert_eq!(db.len(), len, "{version} {stage}: len");
            assert!(
                db.get(5).is_none() && db.get(7).is_none(),
                "{version} {stage}: deletes"
            );
            let (v, m) = db.get(1).unwrap();
            assert_eq!(v, vec_of4(1), "{version} {stage}: vector");
            assert_eq!(m["k"], "v1", "{version} {stage}: metadata");
            // 0.1.4 wrote typed ints; the older two wrote strings, kept as strings.
            if version == "0.2.0" {
                assert_eq!(
                    m["sp"],
                    Value::Sparse(vec![(1, 1.0), (101, 0.5)]),
                    "{version} {stage}"
                );
            }
            if typed {
                assert_eq!(m["i"], Value::Int(1), "{version} {stage}");
            } else {
                assert_eq!(m["i"], "1", "{version} {stage}");
            }
            assert_eq!(
                db.get(6).unwrap().1["k"],
                "over",
                "{version} {stage}: overwrite"
            );
            assert_eq!(
                db.get(65).unwrap().1["k"],
                "tail",
                "{version} {stage}: log tail"
            );
            assert_eq!(
                db.search(&vec_of4(40), 1).unwrap()[0].id,
                40,
                "{version} {stage}: search"
            );
            let hits = db
                .search_filtered(&vec_of4(40), 3, |m| m["k"] == "v2")
                .unwrap();
            assert!(
                !hits.is_empty() && hits.iter().all(|h| db.get(h.id).unwrap().1["k"] == "v2"),
                "{version} {stage}: filter"
            );
        };
        let db =
            Db::open(&path, Options::new(4)).unwrap_or_else(|e| panic!("{version}: open: {e}"));
        check(&db, "opened", records);
        // Write with the current version, then move everything into the new layout.
        db.insert(
            100,
            &vec_of4(100),
            HashMap::from([("k".to_string(), Value::from("new"))]),
        )
        .unwrap();
        db.checkpoint().unwrap();
        assert_eq!(db.get(100).unwrap().1["k"], "new");
        assert!(
            db.index_path().unwrap().to_str().unwrap().contains(".idx."),
            "{version}: new-style index name"
        );
        drop(db);
        let db = Db::open(&path, Options::new(4)).unwrap();
        check(&db, "after checkpoint and reopen", records + 1);
        assert_eq!(db.get(100).unwrap().1["k"], "new");
        let r = Db::open_read_only(&path, Options::new(4)).unwrap();
        assert_eq!(r.get(100).unwrap().1["k"], "new");
    }
}

#[test]
fn write_batch_is_all_or_nothing() {
    use sanadb::Batch;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("batch.sana");
    let dim = 4usize;
    let db = Db::open(&path, Options::new(dim as u32)).unwrap();
    for i in 0..10u64 {
        db.insert(i, &vec_of(dim, i), meta(&[("k", "old")]))
            .unwrap();
    }
    // A bad vector anywhere leaves the store untouched.
    let bad = Batch::new()
        .insert(100, &vec_of(dim, 100), HashMap::new())
        .insert(101, &[f32::NAN; 4], HashMap::new());
    assert!(matches!(
        db.write_batch(bad),
        Err(sanadb::Error::InvalidVector)
    ));
    assert!(db.get(100).is_none());
    // A type conflict within the batch too.
    let bad = Batch::new()
        .insert(
            100,
            &vec_of(dim, 100),
            HashMap::from([("n".to_string(), Value::from(1i64))]),
        )
        .insert(
            101,
            &vec_of(dim, 101),
            HashMap::from([("n".to_string(), Value::from("1"))]),
        );
    assert!(matches!(
        db.write_batch(bad),
        Err(sanadb::Error::TypeMismatch { .. })
    ));
    assert!(db.get(100).is_none() && db.len() == 10);
    // A good batch: inserts, an overwrite and deletes, visible together.
    let r = Db::open_read_only(&path, Options::new(dim as u32)).unwrap();
    assert_eq!(r.len(), 10);
    let good = Batch::new()
        .insert(100, &vec_of(dim, 100), meta(&[("k", "new")]))
        .insert(3, &vec_of(dim, 3), meta(&[("k", "over")]))
        .delete(4)
        .delete(999)
        .insert(101, &vec_of(dim, 101), meta(&[("k", "new")]));
    db.write_batch(good).unwrap();
    assert_eq!(db.len(), 11);
    assert_eq!(db.get(3).unwrap().1["k"], "over");
    assert!(db.get(4).is_none());
    assert_eq!(r.len(), 11, "reader sees the whole batch at once");
    assert_eq!(r.get(101).unwrap().1["k"], "new");
    drop(db);
    // Crash mid-batch: cut the log just before the Commit frame (12 bytes:
    // 8-byte frame header, 4-byte body) and reopen. The batch must vanish
    // entirely, the earlier state must be intact, and new writes must work.
    let bytes = std::fs::read(&path).unwrap();
    let before = bytes.len();
    std::fs::write(&path, &bytes[..before - 12]).unwrap();
    let db = Db::open(&path, Options::new(dim as u32)).unwrap();
    assert_eq!(db.len(), 10, "torn batch must not be applied");
    assert_eq!(db.get(3).unwrap().1["k"], "old");
    assert!(db.get(4).is_some() && db.get(100).is_none() && db.get(101).is_none());
    db.insert(200, &vec_of(dim, 200), HashMap::new()).unwrap();
    drop(db);
    let db = Db::open(&path, Options::new(dim as u32)).unwrap();
    assert_eq!(db.len(), 11);
    assert!(
        db.get(200).is_some() && db.get(100).is_none(),
        "the cut-off batch stays gone after more writes"
    );
    assert_eq!(r.len(), 11);
    assert!(r.get(200).is_some() && r.get(100).is_none());
    // Batches survive checkpoint and are part of the compacted log's tail.
    db.write_batch(
        Batch::new()
            .insert(300, &vec_of(dim, 300), HashMap::new())
            .delete(0),
    )
    .unwrap();
    db.checkpoint().unwrap();
    db.write_batch(Batch::new().insert(301, &vec_of(dim, 301), HashMap::new()))
        .unwrap();
    drop(db);
    let db = Db::open(&path, Options::new(dim as u32)).unwrap();
    assert_eq!(db.len(), 12);
    assert!(db.get(0).is_none() && db.get(300).is_some() && db.get(301).is_some());
}

#[test]
fn int8_storage_keeps_recall_and_shrinks_the_index() {
    use sanadb::Storage;
    let (dim, n) = (32usize, 20_000u64);
    let dir = tempfile::tempdir().unwrap();
    let build = |name: &str, storage: Storage| -> (Db, u64) {
        let path = dir.path().join(name);
        let db = Db::open(&path, Options::new(dim as u32).storage(storage)).unwrap();
        for i in 0..n {
            db.insert(i, &vec_of(dim, i), meta(&[("k", "v")])).unwrap();
        }
        db.checkpoint().unwrap();
        let size = std::fs::metadata(db.index_path().unwrap()).unwrap().len();
        (db, size)
    };
    let (f32db, f32_size) = build("f32.sana", Storage::F32);
    let (i8db, i8_size) = build("i8.sana", Storage::I8);
    assert_eq!(i8db.index_stats().unwrap().storage, Storage::I8);
    assert!(
        i8_size < f32_size * 3 / 4,
        "int8 index {i8_size} vs f32 {f32_size}"
    );
    // get() returns the dequantised vector: close, not exact.
    let (v, m) = i8db.get(123).unwrap();
    assert_eq!(m["k"], "v");
    let truth = vec_of(dim, 123);
    let err = v
        .iter()
        .zip(&truth)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(err > 0.0 && err < 0.01, "max abs error {err}");
    // Recall against brute force, both storages.
    let recall = |db: &Db| {
        let mut hit = 0;
        for qi in 0..100u64 {
            let q = vec_of(dim, 7_000_000 + qi);
            let want = brute(dim, n, &q, 10, |_| true);
            hit += db
                .search(&q, 10)
                .unwrap()
                .iter()
                .filter(|h| want.contains(&h.id))
                .count();
        }
        hit as f64 / 1000.0
    };
    let (rf, ri) = (recall(&f32db), recall(&i8db));
    eprintln!("recall@10 f32={rf} int8={ri}; index bytes f32={f32_size} int8={i8_size}");
    assert!(ri >= rf - 0.03, "int8 recall {ri} vs f32 {rf}");
    // Incremental update (expands the old int8 vectors), then reopen.
    for i in n..n + 500 {
        i8db.insert(i, &vec_of(dim, i), HashMap::new()).unwrap();
    }
    i8db.delete(5).unwrap();
    i8db.checkpoint().unwrap();
    assert_eq!(i8db.index_stats().unwrap().nodes, n as usize + 499);
    assert_eq!(i8db.index_stats().unwrap().storage, Storage::I8);
    let path = dir.path().join("i8.sana");
    drop(i8db);
    let db = Db::open(&path, Options::new(dim as u32).storage(Storage::I8)).unwrap();
    assert_eq!(db.len(), n as usize + 499);
    assert!(db.get(5).is_none());
    assert_eq!(db.search(&vec_of(dim, n + 100), 1).unwrap()[0].id, n + 100);
    assert!(recall(&db) >= rf - 0.03);
    // A second checkpoint of the same data reproduces the same codes.
    let before = db.get(123).unwrap().0;
    db.insert(n + 1000, &vec_of(dim, n + 1000), HashMap::new())
        .unwrap();
    db.checkpoint().unwrap();
    assert_eq!(
        db.get(123).unwrap().0,
        before,
        "requantisation must not drift"
    );
    assert!(
        !dir.path().join("i8.sana.f32.tmp").exists()
            && std::fs::read_dir(dir.path()).unwrap().flatten().all(|e| !e
                .file_name()
                .to_str()
                .unwrap()
                .ends_with(".f32.tmp"))
    );
}

/// Random operations against an in-memory model: inserts, overwrites,
/// deletes, atomic batches, checkpoints, reopens, and a read-only reader
/// that must agree with the writer after every step. Exact reads (`get`,
/// `len`, filtered exact scans) must match the model exactly; the
/// approximate search must never return a record the model does not hold.
#[test]
fn random_operations_agree_with_a_model() {
    use sanadb::{Batch, Filter};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.sana");
    let dim = 6usize;
    let mut model: HashMap<u64, (Vec<f32>, Metadata)> = HashMap::new();
    let mut rng = 0xA11CEu64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut db = Db::open(&path, Options::new(dim as u32).exact_filter_scan(100_000)).unwrap();
    let reader =
        Db::open_read_only(&path, Options::new(dim as u32).exact_filter_scan(100_000)).unwrap();
    let mk_meta = |i: u64| -> Metadata {
        let mut m = Metadata::new();
        m.insert("g".into(), Value::from((i % 7) as i64));
        m.insert("s".into(), Value::from(["x", "y", "z"][(i % 3) as usize]));
        if i.is_multiple_of(5) {
            m.insert("flag".into(), Value::from(true));
        }
        m
    };
    let check = |db: &Db, model: &HashMap<u64, (Vec<f32>, Metadata)>, step: usize, who: &str| {
        assert_eq!(db.len(), model.len(), "{who} step {step}: len");
        for (&id, (v, m)) in model.iter().take(40) {
            let (gv, gm) = db
                .get(id)
                .unwrap_or_else(|| panic!("{who} step {step}: {id} missing"));
            assert_eq!(&gv, v, "{who} step {step}: vector of {id}");
            assert_eq!(&gm, m, "{who} step {step}: metadata of {id}");
        }
        for id in [1u64, 7, 99, 100_000, 5_000_000] {
            assert_eq!(
                db.get(id).is_some(),
                model.contains_key(&id),
                "{who} step {step}: presence of {id}"
            );
        }
        // Exact filtered search over a group: the result set must equal the
        // model's k nearest within that group (scan regime, cap raised).
        let q = vec_of(dim, 42 + step as u64);
        let g = (step % 7) as i64;
        let want: Vec<u64> = {
            let mut d: Vec<(f32, u64)> = model
                .iter()
                .filter(|(_, (_, m))| m["g"] == Value::Int(g))
                .map(|(&id, (v, _))| (q.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum(), id))
                .collect();
            d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            d.into_iter().take(5).map(|e| e.1).collect()
        };
        let got: Vec<u64> = db
            .search_where(&q, 5, &Filter::eq("g", g))
            .unwrap()
            .iter()
            .map(|h| h.id)
            .collect();
        if db.index_stats().is_some() {
            assert_eq!(
                got, want,
                "{who} step {step}: exact filtered search for g={g}"
            );
        }
        for h in db.search(&q, 10).unwrap() {
            assert!(
                model.contains_key(&h.id),
                "{who} step {step}: search returned deleted {}",
                h.id
            );
        }
    };
    let mut next_id = 0u64;
    for step in 0..120 {
        match next() % 10 {
            0..=3 => {
                // Fresh inserts.
                for _ in 0..1 + next() % 40 {
                    let id = next_id;
                    next_id += 1;
                    let v = vec_of(dim, id);
                    db.insert(id, &v, mk_meta(id)).unwrap();
                    model.insert(id, (v, mk_meta(id)));
                }
            }
            4 => {
                // Overwrite existing ids with new vectors and metadata.
                let ids: Vec<u64> = model.keys().copied().take(10).collect();
                for id in ids {
                    let v = vec_of(dim, id + 1_000_000);
                    let mut m = mk_meta(id);
                    m.insert("s".into(), Value::from("w"));
                    db.insert(id, &v, m.clone()).unwrap();
                    model.insert(id, (v, m));
                }
            }
            5 => {
                let ids: Vec<u64> = model.keys().copied().take(7).collect();
                for id in ids {
                    assert!(db.delete(id).unwrap());
                    model.remove(&id);
                }
                assert!(!db.delete(777_777).unwrap());
            }
            6 => {
                // One atomic batch mixing all three.
                let mut b = Batch::new();
                let del: Vec<u64> = model.keys().copied().take(3).collect();
                for &id in &del {
                    b = b.delete(id);
                }
                for _ in 0..5 {
                    let id = next_id;
                    next_id += 1;
                    b = b.insert(id, &vec_of(dim, id), mk_meta(id));
                }
                db.write_batch(b).unwrap();
                for id in del {
                    model.remove(&id);
                }
                for id in next_id - 5..next_id {
                    model.insert(id, (vec_of(dim, id), mk_meta(id)));
                }
            }
            7 => {
                db.checkpoint().unwrap();
            }
            8 => {
                // Reopen the writer; the reader stays attached throughout.
                drop(db);
                db = Db::open(&path, Options::new(dim as u32).exact_filter_scan(100_000)).unwrap();
            }
            _ => {
                // A batch that must fail leaves everything untouched.
                let id = next_id + 10;
                let r = db.write_batch(
                    Batch::new()
                        .insert(id, &vec_of(dim, id), mk_meta(id))
                        .insert(
                            id + 1,
                            &[1.0; 6],
                            HashMap::from([("g".to_string(), Value::from("bad type"))]),
                        ),
                );
                assert!(matches!(r, Err(sanadb::Error::TypeMismatch { .. })));
                assert!(db.get(id).is_none());
            }
        }
        check(&db, &model, step, "writer");
        check(&reader, &model, step, "reader");
    }
    eprintln!(
        "model test: {} live records after 120 steps, {} ids allocated",
        model.len(),
        next_id
    );
    assert!(model.len() > 200);
}

#[test]
fn backup_is_a_consistent_point_in_time_copy() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("live.sana");
    let dim = 4usize;
    let db = Db::open(&path, Options::new(dim as u32)).unwrap();
    for i in 0..3000u64 {
        db.insert(i, &vec_of(dim, i), meta(&[("k", "a")])).unwrap();
    }
    db.checkpoint().unwrap();
    for i in 3000..3100u64 {
        db.insert(i, &vec_of(dim, i), meta(&[("k", "tail")]))
            .unwrap();
    }
    db.delete(1).unwrap();
    // Back up while the writer keeps going afterwards.
    let copy = db.backup(dir.path().join("backup")).unwrap();
    db.insert(9999, &vec_of(dim, 9999), HashMap::new()).unwrap();
    db.delete(2).unwrap();
    db.checkpoint().unwrap();
    // The copy is the state at backup time, opens on its own, and is
    // independent of what happened to the original since.
    let b = Db::open(&copy, Options::new(dim as u32)).unwrap();
    assert_eq!(b.len(), 3099);
    assert!(b.get(1).is_none() && b.get(2).is_some() && b.get(9999).is_none());
    assert_eq!(b.get(3050).unwrap().1["k"], "tail");
    assert_eq!(b.search(&vec_of(dim, 77), 1).unwrap()[0].id, 77);
    b.insert(5, &vec_of(dim, 5), meta(&[("k", "copy")]))
        .unwrap();
    b.checkpoint().unwrap();
    assert_eq!(
        db.get(5).unwrap().1["k"],
        "a",
        "the original is untouched by writes to the copy"
    );
    assert_eq!(db.len(), 3099);
    // A read-only handle can back up too.
    let r = Db::open_read_only(&path, Options::new(dim as u32)).unwrap();
    let copy2 = r.backup(dir.path().join("backup2")).unwrap();
    let b2 = Db::open_read_only(&copy2, Options::new(dim as u32)).unwrap();
    assert_eq!(b2.len(), db.len());
    assert!(b2.get(9999).is_some() && b2.get(2).is_none());
    // Only the referenced index file travels with the copy.
    let files: Vec<String> = std::fs::read_dir(dir.path().join("backup"))
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_str().unwrap().to_string())
        .collect();
    assert!(
        files.iter().filter(|f| f.contains(".idx.")).count() <= 2,
        "{files:?}"
    );
}

#[test]
fn keyword_and_hybrid_search_over_a_text_field() {
    use sanadb::Filter;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("text.sana");
    let dim = 4usize;
    let opts = || Options::new(dim as u32).text_field("body");
    let db = Db::open(&path, opts()).unwrap();
    let docs: [(&str, &str); 6] = [
        (
            "rust",
            "Rust ownership and borrowing keep memory safe without a garbage collector",
        ),
        (
            "rust",
            "Tokio select cancels the other branches when one future completes",
        ),
        (
            "py",
            "Python asyncio gather runs coroutines concurrently on one thread",
        ),
        (
            "py",
            "Garbage collection in CPython uses reference counting plus a cycle collector",
        ),
        (
            "db",
            "SQLite keeps one writer and many readers with a write ahead log",
        ),
        (
            "db",
            "A write ahead log makes a crash safe database: replay the log on open",
        ),
    ];
    for (i, (lang, body)) in docs.iter().enumerate() {
        let mut m = Metadata::new();
        m.insert("lang".into(), Value::from(*lang));
        m.insert("body".into(), Value::from(*body));
        db.insert(i as u64, &vec_of(dim, i as u64), m).unwrap();
    }
    // Before any checkpoint: everything is tail, tokenised on the fly.
    let top = |db: &Db, q: &str| {
        db.search_text(q, 3, None)
            .unwrap()
            .iter()
            .map(|h| h.id)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        top(&db, "garbage collector")[0],
        0,
        "rare + rare terms: {:?}",
        top(&db, "garbage collector")
    );
    assert_eq!(
        top(&db, "write ahead log")[..2]
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>(),
        [4u64, 5].into_iter().collect()
    );
    assert_eq!(top(&db, "tokio select"), [1]);
    assert!(db.search_text("zzz qqq", 3, None).unwrap().is_empty());
    assert!(db.search_text("", 3, None).unwrap().is_empty());
    // Case and punctuation do not matter.
    assert_eq!(top(&db, "TOKIO, Select!"), [1]);
    // Scores are negated BM25: best first, strictly ordered.
    let hits = db.search_text("log", 5, None).unwrap();
    assert!(hits.len() == 2 && hits[0].score <= hits[1].score && hits[0].score < 0.0);
    // Filters apply.
    assert_eq!(top(&db, "garbage"), [0, 3]);
    assert_eq!(
        db.search_text("garbage", 3, Some(&Filter::eq("lang", "py")))
            .unwrap()
            .iter()
            .map(|h| h.id)
            .collect::<Vec<_>>(),
        [3]
    );

    // Checkpoint: same answers from the inverted index.
    db.checkpoint().unwrap();
    assert_eq!(top(&db, "garbage collector")[0], 0);
    assert_eq!(top(&db, "tokio select"), [1]);
    assert_eq!(top(&db, "garbage"), [0, 3]);
    // Mixed: one indexed document, one new in the tail, one deleted.
    let mut m = Metadata::new();
    m.insert("lang".into(), Value::from("db"));
    m.insert(
        "body".into(),
        Value::from("the log is replayed; a log is append only"),
    );
    db.insert(6, &vec_of(dim, 6), m).unwrap();
    db.delete(4).unwrap();
    let hits = top(&db, "log");
    assert!(
        hits.contains(&6) && hits.contains(&5) && !hits.contains(&4),
        "{hits:?}"
    );
    // Hybrid: a vector near document 2 and the words of document 1 should put both on top.
    let hybrid: Vec<u64> = db
        .search_hybrid(&vec_of(dim, 2), "tokio select", 2, None)
        .unwrap()
        .iter()
        .map(|h| h.id)
        .collect();
    assert!(hybrid.contains(&2) && hybrid.contains(&1), "{hybrid:?}");
    let hybrid_py: Vec<u64> = db
        .search_hybrid(
            &vec_of(dim, 2),
            "tokio select",
            2,
            Some(&Filter::eq("lang", "py")),
        )
        .unwrap()
        .iter()
        .map(|h| h.id)
        .collect();
    assert!(hybrid_py.iter().all(|&i| i == 2 || i == 3), "{hybrid_py:?}");
    // A reader and a reopened writer agree.
    let r = Db::open_read_only(&path, opts()).unwrap();
    assert_eq!(top(&r, "tokio select"), [1]);
    assert!(top(&r, "log").contains(&6));
    drop(db);
    let db = Db::open(&path, opts()).unwrap();
    assert_eq!(top(&db, "garbage collector")[0], 0);
    db.checkpoint().unwrap();
    let hits = top(&db, "log");
    assert!(
        hits.contains(&6) && hits.contains(&5) && !hits.contains(&4),
        "{hits:?}"
    );
    // Without a text field the call is an error, not a silent empty result.
    let plain = Db::open(dir.path().join("plain.sana"), Options::new(dim as u32)).unwrap();
    assert!(plain.search_text("x", 3, None).is_err());
}

#[test]
fn contains_filter_runs_on_the_text_index() {
    use sanadb::Filter;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tags.sana");
    let dim = 4usize;
    let opts = || {
        Options::new(dim as u32)
            .text_field("tags")
            .exact_filter_scan(100_000)
    };
    let db = Db::open(&path, opts()).unwrap();
    let n = 5000u64;
    // Tag sets as a space-separated string, like YFCC's label lists.
    let tags = |i: u64| {
        format!(
            "t{} t{} {}",
            i % 50,
            100 + i % 7,
            if i.is_multiple_of(500) {
                "rare"
            } else {
                "common"
            }
        )
    };
    for i in 0..n {
        let mut m = Metadata::new();
        m.insert("tags".into(), Value::from(tags(i)));
        m.insert("note".into(), Value::from(format!("note {}", i % 3)));
        db.insert(i, &vec_of(dim, i), m).unwrap();
    }
    // Tail: evaluated from metadata by tokenising.
    let q = vec_of(dim, 123);
    let got: Vec<u64> = db
        .search_where(&q, 5, &Filter::contains("tags", "rare"))
        .unwrap()
        .iter()
        .map(|h| h.id)
        .collect();
    assert!(
        got.iter().all(|i| i.is_multiple_of(500)) && got.len() == 5,
        "{got:?}"
    );
    db.checkpoint().unwrap();
    // Indexed: posting-list candidates, exact answer.
    let want = brute(dim, n, &q, 5, |i| i.is_multiple_of(500));
    let got: Vec<u64> = db
        .search_where(&q, 5, &Filter::contains("tags", "RARE"))
        .unwrap()
        .iter()
        .map(|h| h.id)
        .collect();
    assert_eq!(got, want);
    // Two tags and-ed: both posting lists, intersection exact.
    let f = Filter::and([
        Filter::contains("tags", "t7"),
        Filter::contains("tags", "t103"),
    ]);
    let want = brute(dim, n, &q, 5, |i| i % 50 == 7 && i % 7 == 3);
    let got: Vec<u64> = db
        .search_where(&q, 5, &f)
        .unwrap()
        .iter()
        .map(|h| h.id)
        .collect();
    assert_eq!(got, want);
    // Contains on a plain Str column (not the text field) still works, by decoding.
    let got: Vec<u64> = db
        .search_where(&q, 5, &Filter::contains("note", "2"))
        .unwrap()
        .iter()
        .map(|h| h.id)
        .collect();
    assert!(got.len() == 5 && got.iter().all(|i| i % 3 == 2), "{got:?}");
    // A token nobody has, and a common token (walk regime).
    assert!(
        db.search_where(&q, 5, &Filter::contains("tags", "nothing"))
            .unwrap()
            .is_empty()
    );
    let got = db
        .search_where(&q, 5, &Filter::contains("tags", "common"))
        .unwrap();
    assert!(got.len() == 5 && got.iter().all(|h| !h.id.is_multiple_of(500)));
}

/// Each injected failure must leave the store usable: the failed operation
/// reports an error, nothing already acknowledged is lost, and the next
/// attempt succeeds. Faults are process-global, so this test runs its
/// scenarios one after another in one function.
#[test]
fn injected_io_failures_leave_the_store_consistent() {
    use sanadb::fault::{Fault, set_fault};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("faults.sana");
    let dim = 4usize;
    let db = Db::open(&path, Options::new(dim as u32)).unwrap();
    for i in 0..2000u64 {
        db.insert(i, &vec_of(dim, i), meta(&[("k", "a")])).unwrap();
    }
    db.checkpoint().unwrap();
    let first_index = db.index_path().unwrap();
    for i in 2000..2100u64 {
        db.insert(i, &vec_of(dim, i), meta(&[("k", "b")])).unwrap();
    }
    let check = |db: &Db, stage: &str| {
        assert_eq!(db.len(), 2100, "{stage}");
        assert_eq!(db.get(5).unwrap().1["k"], "a", "{stage}");
        assert_eq!(db.get(2050).unwrap().1["k"], "b", "{stage}");
        assert_eq!(
            db.search(&vec_of(dim, 2050), 1).unwrap()[0].id,
            2050,
            "{stage}"
        );
    };

    // 1. Disk fills while the new index is being written.
    set_fault(&path, Some(Fault::IndexWriteAfter(100_000)));
    let r = db.checkpoint();
    assert!(matches!(r, Err(sanadb::Error::Io(_))), "{r:?}");
    check(&db, "after failed index write");
    assert_eq!(
        db.index_path().unwrap(),
        first_index,
        "old index still current"
    );
    let leftovers: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_str().unwrap().to_string())
        .filter(|n| n.contains(".idx.") || n.ends_with(".tmp"))
        .collect();
    assert_eq!(
        leftovers.len(),
        1,
        "no partial index or temp file left: {leftovers:?}"
    );

    // 2. The index fsync fails.
    set_fault(&path, Some(Fault::IndexSync));
    assert!(db.checkpoint().is_err());
    check(&db, "after failed index sync");
    assert_eq!(db.index_path().unwrap(), first_index);

    // 3. The log rename at the end of the swap fails: the new index is
    //    discarded and the old log stays, so nothing acknowledged is lost.
    set_fault(&path, Some(Fault::LogRename));
    assert!(db.checkpoint().is_err());
    check(&db, "after failed log rename");
    assert_eq!(db.index_path().unwrap(), first_index);
    db.insert(2100, &vec_of(dim, 2100), meta(&[("k", "c")]))
        .unwrap();
    assert_eq!(db.len(), 2101);

    // 4. A log append fails: the insert reports it and writes nothing.
    set_fault(&path, Some(Fault::LogAppend));
    assert!(matches!(
        db.insert(3000, &vec_of(dim, 3000), HashMap::new()),
        Err(sanadb::Error::Io(_))
    ));
    assert!(db.get(3000).is_none());
    assert_eq!(db.len(), 2101);
    set_fault(&path, Some(Fault::LogAppend));
    assert!(db.delete(0).is_err());
    assert!(db.get(0).is_some());
    set_fault(&path, Some(Fault::LogAppend));
    assert!(
        db.write_batch(sanadb::Batch::new().insert(3001, &vec_of(dim, 3001), HashMap::new()))
            .is_err()
    );
    assert!(db.get(3001).is_none());

    // 5. With the faults gone, the checkpoint succeeds and a reopen agrees.
    set_fault(&path, None);
    db.checkpoint().unwrap();
    assert_ne!(db.index_path().unwrap(), first_index);
    assert_eq!(db.len(), 2101);
    assert_eq!(db.get(2100).unwrap().1["k"], "c");
    drop(db);
    let db = Db::open(&path, Options::new(dim as u32)).unwrap();
    assert_eq!(db.len(), 2101);
    assert_eq!(db.get(2050).unwrap().1["k"], "b");
    let idx_files: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_str().unwrap().to_string())
        .filter(|n| n.contains(".idx."))
        .collect();
    assert_eq!(idx_files.len(), 1, "{idx_files:?}");
}

#[test]
fn stats_count_operations_by_regime() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stats.sana");
    let dim = 4usize;
    let db = Db::open(&path, Options::new(dim as u32).exact_filter_scan(100)).unwrap();
    assert_eq!(db.stats(), sanadb::Stats::default());
    for i in 0..3000u64 {
        let k = if i % 1000 == 0 { "rare" } else { "common" };
        db.insert(i, &vec_of(dim, i), meta(&[("k", k)])).unwrap();
    }
    db.search(&vec_of(dim, 1), 5).unwrap();
    let s = db.stats();
    assert_eq!(
        (s.inserts, s.searches, s.delta_only, s.checkpoints),
        (3000, 1, 1, 0)
    );
    db.checkpoint().unwrap();
    db.insert(5000, &vec_of(dim, 5000), meta(&[("k", "rare")]))
        .unwrap();
    db.delete(1).unwrap();
    db.write_batch(sanadb::Batch::new().delete(2).delete(3))
        .unwrap();
    db.checkpoint().unwrap();
    db.search(&vec_of(dim, 1), 5).unwrap();
    db.search_filtered(&vec_of(dim, 1), 5, |_| true).unwrap();
    db.search_where(&vec_of(dim, 1), 5, &sanadb::Filter::eq("k", "rare"))
        .unwrap();
    db.search_where(&vec_of(dim, 1), 5, &sanadb::Filter::eq("k", "common"))
        .unwrap();
    let s = db.stats();
    assert_eq!(s.inserts, 3001);
    assert_eq!((s.deletes, s.batches), (1, 1));
    assert_eq!(
        (
            s.checkpoints,
            s.incremental_checkpoints,
            s.failed_checkpoints
        ),
        (2, 1, 0)
    );
    assert_eq!(
        (s.searches, s.graph_walks, s.exact_scans, s.filtered_walks),
        (5, 2, 1, 1)
    );
    assert_eq!(s.delta_only, 1);

    let reader = Db::open_read_only(&path, Options::new(dim as u32)).unwrap();
    db.insert(6000, &vec_of(dim, 6000), HashMap::new()).unwrap();
    assert_eq!(reader.len(), 2999);
    db.checkpoint().unwrap();
    assert_eq!(reader.len(), 2999);
    let r = reader.stats();
    assert_eq!((r.refresh_appended, r.refresh_reloaded), (1, 1), "{r:?}");
}

/// The store the current version writes, checked byte for byte against
/// `docs/FORMAT.md`: the layout is frozen from 0.2.0 on, so the constants
/// here change only with a reader level and a changelog entry. Run with
/// `SANADB_WRITE_FIXTURE=1` to write `tests/fixtures/v<version>/` for the
/// upgrade test above (done once per release, after the version bump).
#[test]
fn current_layout_matches_the_format_document() {
    let vec_of4 = |i: u64| -> Vec<f32> {
        (0..4u64)
            .map(|d| ((i * 7 + d * 13) % 101) as f32 / 101.0)
            .collect()
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fixture.sana");
    let opts = || Options::new(4).text_field("body").exact_filter_scan(8);
    let db = Db::open(&path, opts()).unwrap();
    for i in 0..60u64 {
        db.insert(
            i,
            &vec_of4(i),
            HashMap::from([
                ("k".to_string(), Value::from(format!("v{}", i % 3))),
                ("i".to_string(), Value::Int(i as i64)),
                ("f".to_string(), Value::Float(i as f64 / 4.0)),
                ("b".to_string(), Value::Bool(i % 2 == 0)),
                (
                    "body".to_string(),
                    Value::from(format!("record {i} tag{}", i % 5)),
                ),
                (
                    "sp".to_string(),
                    Value::Sparse(vec![(i as u32 % 7, 1.0), (100 + i as u32, 0.5)]),
                ),
            ]),
        )
        .unwrap();
    }
    db.delete(5).unwrap();
    db.delete(7).unwrap();
    db.insert(6, &vec_of4(6), meta(&[("k", "over")])).unwrap();
    db.checkpoint().unwrap();
    db.write_batch(
        (60..70u64)
            .fold(sanadb::Batch::new(), |b, i| {
                b.insert(
                    i,
                    &vec_of4(i),
                    meta(&[("k", if i == 65 { "tail" } else { "v9" })]),
                )
            })
            .delete(8),
    )
    .unwrap();
    db.sync().unwrap();
    assert_eq!(db.len(), 67);

    // Log bytes.
    let log = std::fs::read(&path).unwrap();
    assert_eq!(&log[..8], b"SANADB02");
    assert_eq!(u32::from_le_bytes(log[8..12].try_into().unwrap()), 4, "dim");
    assert_eq!(log[12], 0, "metric L2");
    assert_eq!(log[13], 3, "log reader level");
    assert_eq!(&log[14..16], &[0, 0]);
    // Frame 1 is the Snapshot naming the index.
    let len = u32::from_le_bytes(log[16..20].try_into().unwrap()) as usize;
    let crc = u32::from_le_bytes(log[20..24].try_into().unwrap());
    let body = &log[24..24 + len];
    assert_eq!(len, 12, "snapshot body: kind, pad, id");
    assert_eq!(crc32fast::hash(body), crc);
    assert_eq!(&body[..4], &[4, 0, 0, 0], "kind Snapshot");
    let snap = u64::from_le_bytes(body[4..12].try_into().unwrap());
    let idx = dir.path().join(format!("fixture.sana.idx.{snap:016x}"));
    assert_eq!(db.index_path().unwrap(), idx);
    // Then Begin, eight Inserts, a Delete, Commit.
    let mut pos = 24 + len;
    let mut kinds = Vec::new();
    while pos < log.len() {
        let len = u32::from_le_bytes(log[pos..pos + 4].try_into().unwrap()) as usize;
        if len == 0 {
            break; // growth slack: the file is extended ahead of the frames
        }
        assert!(len.is_multiple_of(4));
        kinds.push(log[pos + 8]);
        if log[pos + 8] == 1 {
            // Insert: id, 4 f32, metadata count u16, key "k" (len 1), kind Str (0), value.
            let b = &log[pos + 8..pos + 8 + len];
            assert!(b.len() >= 4 + 8 + 16 + 2 + 3 + 1 + 2);
            assert_eq!(
                u16::from_le_bytes(b[28..30].try_into().unwrap()),
                1,
                "one key"
            );
            assert_eq!(&b[30..33], &[1, 0, b'k'], "key length and bytes");
            assert_eq!(b[33], 0, "value kind Str");
        }
        pos += 8 + len;
    }
    assert_eq!(
        kinds,
        [5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 6],
        "frame kinds"
    );

    // Index bytes.
    let ix = std::fs::read(&idx).unwrap();
    assert_eq!(&ix[..8], b"SANAIDX2");
    assert_eq!(u32::from_le_bytes(ix[8..12].try_into().unwrap()), 4);
    assert_eq!(
        &ix[12..16],
        &[0, 1, 0, 0],
        "metric, index level 1, f32 storage, pad"
    );
    let u32_at = |o: usize| u32::from_le_bytes(ix[o..o + 4].try_into().unwrap()) as usize;
    let u64_at = |o: usize| u64::from_le_bytes(ix[o..o + 8].try_into().unwrap()) as usize;
    let (n, r, entry, block, m) = (u32_at(16), u32_at(20), u32_at(24), u32_at(28), u32_at(32));
    assert_eq!(n, 58, "60 inserted, two deleted before the checkpoint");
    assert_eq!(r, 32);
    assert!(entry < n);
    assert_eq!(block, (4 * 4 + 4 + 4 * r).next_multiple_of(64));
    assert_eq!(m, 0, "PQ is skipped below pq::MIN_N");
    let ids: Vec<u64> = (0..n).map(|s| u64_at(36 + s * 8) as u64).collect();
    assert!(!ids.contains(&5) && !ids.contains(&7) && ids.contains(&6));
    let blocks_off = (36 + n * 8).next_multiple_of(64);
    for s in 0..n {
        let b = &ix[blocks_off + s * block..blocks_off + (s + 1) * block];
        let v: Vec<f32> = b[..16]
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let want = if ids[s] == 6 {
            vec_of4(6)
        } else {
            vec_of4(ids[s])
        };
        assert_eq!(v, want, "vector of slot {s}");
        let degree = u32::from_le_bytes(b[16..20].try_into().unwrap()) as usize;
        assert!(degree <= r);
        for e in 0..degree {
            assert!(
                (u32::from_le_bytes(b[20 + e * 4..24 + e * 4].try_into().unwrap()) as usize) < n
            );
        }
        assert!(
            b[20 + degree * 4..].iter().all(|&x| x == 0),
            "zero after the edges"
        );
    }
    assert_eq!(&ix[ix.len() - 8..], b"SANAFTR2");
    let f = ix.len() - 40;
    let (attrs_table, idtab, text_table, sparse_table) =
        (u64_at(f), u64_at(f + 8), u64_at(f + 16), u64_at(f + 24));
    assert_eq!(u64_at(attrs_table), n, "attribute table starts with n");
    assert_eq!(
        u32_at(attrs_table + 8),
        5,
        "five columns: b, body, f, i, k; the sparse key is not a column"
    );
    assert_eq!(u64_at(sparse_table), n, "sparse table starts with n");
    assert_eq!(u64_at(sparse_table + 16), 2, "field name length");
    assert_eq!(&ix[u64_at(sparse_table + 8)..][..2], b"sp");
    assert_eq!(
        u64_at(sparse_table + 40),
        (n - 1) * 2,
        "two pairs per record with a body"
    );
    // Id table: sorted ids then their slots.
    let sorted: Vec<u64> = (0..n).map(|s| u64_at(idtab + s * 8) as u64).collect();
    assert!(sorted.windows(2).all(|w| w[0] < w[1]));
    for (s, id) in sorted.iter().enumerate() {
        let slot = u32_at(idtab + n * 8 + s * 4);
        assert_eq!(ids[slot], *id);
    }
    assert_eq!(u64_at(text_table), n, "text table starts with n");
    assert_eq!(
        u64_at(text_table + 8),
        n - 1,
        "record 6 was overwritten without a body"
    );
    assert_eq!(
        u64_at(text_table + 64),
        1 + 5 + 57,
        "terms: 'record', five tags, 57 numbers"
    );
    drop(db);

    if std::env::var_os("SANADB_WRITE_FIXTURE").is_some() {
        let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(format!("v{}", env!("CARGO_PKG_VERSION")));
        std::fs::create_dir_all(&out).unwrap();
        for e in std::fs::read_dir(dir.path()).unwrap().flatten() {
            let name = e.file_name();
            if name.to_str().unwrap().ends_with(".lock") {
                continue;
            }
            std::fs::copy(e.path(), out.join(name)).unwrap();
        }
    }
}

/// Soak: one writer (inserts, overwrites, deletes, batches), a checkpoint
/// loop, two read-only handles in this process and one in a child process,
/// for `SANADB_SOAK_SECS` seconds (default 30). Every reader checks what it
/// sees for internal consistency; at the end the store must agree with the
/// writer's model exactly, before and after a reopen. Ignored by default:
/// `cargo test --release -- --ignored soak --nocapture`, or `scripts/soak.sh`.
#[test]
#[ignore]
fn soak_readers_writer_and_checkpoints() {
    use std::collections::HashSet;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};
    let secs: u64 = std::env::var("SANADB_SOAK_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("soak.sana");
    let dim = 8usize;
    let opts = || Options::new(dim as u32).exact_filter_scan(1000);
    let w = Db::open(&path, opts()).unwrap();
    // Model of what the store holds: id -> present. Vectors and metadata are
    // functions of the id (`vec_of`, `i == id`), as the child reader expects.
    let model: Mutex<HashSet<u64>> = Mutex::new(HashSet::new());
    for i in 0..1000u64 {
        w.insert(
            i,
            &vec_of(dim, i),
            HashMap::from([("i".to_string(), Value::from(i as i64))]),
        )
        .unwrap();
        model.lock().unwrap().insert(i);
    }
    w.checkpoint().unwrap();
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "reader_child_reads_until_told_to_stop",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("SANADB_READER_CHILD", &path)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .unwrap();
    let stop = AtomicBool::new(false);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(secs);
    let counts = Mutex::new([0u64; 4]); // inserts, deletes, batches, checkpoints
    std::thread::scope(|s| {
        // Writer.
        s.spawn(|| {
            let mut next = 1000u64;
            let mut rng = 0x5EEDu64;
            let mut rnd = || {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng
            };
            let mk = |i: u64| HashMap::from([("i".to_string(), Value::from(i as i64))]);
            while std::time::Instant::now() < deadline {
                let mut m = model.lock().unwrap();
                let mut c = counts.lock().unwrap();
                match rnd() % 10 {
                    0..=5 => {
                        w.insert(next, &vec_of(dim, next), mk(next)).unwrap();
                        m.insert(next);
                        next += 1;
                        c[0] += 1;
                    }
                    6 => {
                        // Overwrite an existing id with the same content.
                        let id = rnd() % next;
                        if m.contains(&id) {
                            w.insert(id, &vec_of(dim, id), mk(id)).unwrap();
                            c[0] += 1;
                        }
                    }
                    7 => {
                        let id = rnd() % next;
                        // Keep the count growing so the child's monotonic check holds.
                        if m.len() > 1000 && m.remove(&id) {
                            assert!(w.delete(id).unwrap());
                            c[1] += 1;
                        }
                    }
                    _ => {
                        let mut b = sanadb::Batch::new();
                        for _ in 0..20 {
                            b = b.insert(next, &vec_of(dim, next), mk(next));
                            m.insert(next);
                            next += 1;
                        }
                        w.write_batch(b).unwrap();
                        c[2] += 1;
                    }
                }
                drop(c);
                drop(m);
                if rnd() % 50 == 0 {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
            stop.store(true, Ordering::Release);
        });
        // Checkpoints.
        s.spawn(|| {
            while !stop.load(Ordering::Acquire) {
                std::thread::sleep(std::time::Duration::from_millis(150));
                w.checkpoint().unwrap();
                counts.lock().unwrap()[3] += 1;
            }
        });
        // In-process readers: everything a hit points at must be its own id's content.
        for t in 0..2u64 {
            let stop = &stop;
            let path = &path;
            s.spawn(move || {
                let r = Db::open_read_only(path, opts()).unwrap();
                let mut rounds = 0u64;
                let mut max_len = 0usize;
                while !stop.load(Ordering::Acquire) {
                    let q = vec_of(dim, (rounds * 31 + t) % 4000);
                    for h in r.search(&q, 5).unwrap() {
                        if let Some((v, m)) = r.get(h.id) {
                            assert_eq!(v, vec_of(dim, h.id), "reader {t}: vector of {}", h.id);
                            assert_eq!(
                                m["i"],
                                Value::Int(h.id as i64),
                                "reader {t}: meta of {}",
                                h.id
                            );
                        }
                    }
                    let n = r.len();
                    assert!(n + 2 >= max_len, "reader {t}: len {n} after {max_len}");
                    max_len = max_len.max(n);
                    let f = sanadb::Filter::ge("i", 500i64);
                    for h in r.search_where(&q, 3, &f).unwrap() {
                        assert!(h.id >= 500, "reader {t}: filter let {} through", h.id);
                    }
                    rounds += 1;
                }
                eprintln!("reader {t}: {rounds} rounds");
                assert!(rounds > 10);
            });
        }
    });
    std::fs::write(path.with_extension("stop"), b"").unwrap();
    let status = child.wait().unwrap();
    assert!(status.success(), "reader child failed: {status}");

    // Final agreement with the model, then after a reopen.
    let m = model.into_inner().unwrap();
    let c = counts.into_inner().unwrap();
    let st = w.stats();
    eprintln!(
        "soak {secs}s: {} inserts, {} deletes, {} batches, {} checkpoints ({} incremental, {} failed), {} records",
        c[0],
        c[1],
        c[2],
        c[3],
        st.incremental_checkpoints,
        st.failed_checkpoints,
        m.len()
    );
    assert_eq!(st.failed_checkpoints, 0);
    let agree = |db: &Db, who: &str| {
        assert_eq!(db.len(), m.len(), "{who}: len");
        for &id in m.iter().step_by(7) {
            let (v, meta) = db.get(id).unwrap_or_else(|| panic!("{who}: {id} missing"));
            assert_eq!(v, vec_of(dim, id), "{who}: vector of {id}");
            assert_eq!(meta["i"], Value::Int(id as i64), "{who}: meta of {id}");
        }
        for id in (0..5000u64).step_by(13) {
            assert_eq!(
                db.get(id).is_some(),
                m.contains(&id),
                "{who}: presence of {id}"
            );
        }
    };
    agree(&w, "writer");
    drop(w);
    let db = Db::open(&path, opts()).unwrap();
    agree(&db, "reopened");
    let idx: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_str().unwrap().to_string())
        .filter(|n| n.contains(".idx."))
        .collect();
    assert_eq!(idx.len(), 1, "one index file after reopen: {idx:?}");
}

#[test]
fn sparse_vectors_index_search_and_round_trip() {
    use sanadb::Filter;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sparse.sana");
    let dim = 4usize;
    let opts = || Options::new(dim as u32);
    let db = Db::open(&path, opts()).unwrap();
    let sp_of = |i: u64| -> Vec<(u32, f32)> {
        vec![
            (i as u32 % 13, 1.0 + (i % 5) as f32),
            (50 + i as u32 % 7, 0.5),
            (1000 + i as u32, 0.1),
        ]
    };
    let n = 400u64;
    for i in 0..n {
        db.insert(
            i,
            &vec_of(dim, i),
            HashMap::from([
                ("sp".to_string(), Value::Sparse(sp_of(i))),
                ("g".to_string(), Value::Int((i % 3) as i64)),
            ]),
        )
        .unwrap();
    }
    let q = [(3u32, 1.0f32), (52, 2.0)];
    let dot = |p: &[(u32, f32)]| -> f32 {
        q.iter()
            .map(|(qi, qw)| p.iter().find(|(i, _)| i == qi).map_or(0.0, |(_, w)| qw * w))
            .sum()
    };
    let brute = |upto: u64, pred: &dyn Fn(u64) -> bool| -> Vec<(u64, f32)> {
        let mut v: Vec<(u64, f32)> = (0..upto)
            .filter(|&i| pred(i))
            .map(|i| (i, dot(&sp_of(i))))
            .filter(|(_, s)| *s > 0.0)
            .collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        v.truncate(10);
        v
    };
    let got = |db: &Db, f: Option<&Filter>| -> Vec<(u64, f32)> {
        db.search_sparse(&q, 10, f)
            .unwrap()
            .into_iter()
            .map(|h| (h.id, -h.score))
            .collect()
    };
    let same = |a: &[(u64, f32)], b: &[(u64, f32)], stage: &str| {
        assert_eq!(a.len(), b.len(), "{stage}: {a:?} vs {b:?}");
        for (x, y) in a.iter().zip(b) {
            assert_eq!(x.0, y.0, "{stage}: {a:?} vs {b:?}");
            assert!((x.1 - y.1).abs() < 1e-5, "{stage}: {x:?} vs {y:?}");
        }
    };
    // Tail only (brute force), then indexed (inverted lists), then mixed.
    same(&got(&db, None), &brute(n, &|_| true), "tail");
    db.checkpoint().unwrap();
    same(&got(&db, None), &brute(n, &|_| true), "indexed");
    let f = Filter::eq("g", 1i64);
    same(
        &got(&db, Some(&f)),
        &brute(n, &|i| i % 3 == 1),
        "indexed, filtered",
    );
    for i in n..n + 50 {
        db.insert(
            i,
            &vec_of(dim, i),
            HashMap::from([("sp".to_string(), Value::Sparse(sp_of(i)))]),
        )
        .unwrap();
    }
    db.delete(3).unwrap();
    same(
        &got(&db, None),
        &brute(n + 50, &|i| i != 3),
        "index plus tail, one tombstone",
    );
    // Round trip from the index's forward section and from the tail.
    assert_eq!(db.get(7).unwrap().1["sp"], Value::Sparse(sp_of(7)));
    assert_eq!(db.get(n + 7).unwrap().1["sp"], Value::Sparse(sp_of(n + 7)));
    // Normalisation on insert: sorted, duplicates summed, zeros dropped.
    let messy = vec![(5, 1.0), (2, 1.0), (5, 2.0), (9, 0.0)];
    db.insert(
        9000,
        &vec_of(dim, 9000),
        HashMap::from([("sp".to_string(), Value::Sparse(messy))]),
    )
    .unwrap();
    assert_eq!(
        db.get(9000).unwrap().1["sp"],
        Value::Sparse(vec![(2, 1.0), (5, 3.0)])
    );
    let nan = HashMap::from([("sp".to_string(), Value::Sparse(vec![(1, f32::NAN)]))]);
    assert!(matches!(
        db.insert(9001, &vec_of(dim, 9001), nan),
        Err(sanadb::Error::InvalidVector)
    ));
    assert!(matches!(
        db.insert(9002, &vec_of(dim, 9002), meta(&[("sp", "text")])),
        Err(sanadb::Error::TypeMismatch { .. })
    ));
    let second = HashMap::from([("other".to_string(), Value::Sparse(vec![(1, 1.0)]))]);
    assert!(matches!(
        db.insert(9003, &vec_of(dim, 9003), second),
        Err(sanadb::Error::SecondSparseKey { .. })
    ));
    // Hybrid: the top sparse hit is in the fused list.
    let top = got(&db, None)[0].0;
    let hy = db
        .search_hybrid_sparse(&vec_of(dim, 1), &q, 10, None)
        .unwrap();
    assert_eq!(hy.len(), 10);
    assert!(hy.iter().any(|h| h.id == top));
    assert!(db.stats().sparse_searches >= 5);
    // Reopen and a read-only handle agree.
    let expect = got(&db, None);
    db.checkpoint().unwrap();
    same(&got(&db, None), &expect, "after second checkpoint");
    drop(db);
    let db = Db::open(&path, opts()).unwrap();
    same(&got(&db, None), &expect, "reopened");
    let r = Db::open_read_only(&path, opts()).unwrap();
    same(&got(&r, None), &expect, "reader");
    assert_eq!(
        r.get(9000).unwrap().1["sp"],
        Value::Sparse(vec![(2, 1.0), (5, 3.0)])
    );
}

#[test]
fn embedder_trait_adds_and_searches_by_text() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(
        dir.path().join("e.sana"),
        Options::new(16).text_field("body"),
    )
    .unwrap();
    // Any closure is an embedder: here a hashed bag of words.
    let model = |texts: &[&str]| -> sanadb::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0f32; 16];
                for w in t.split_whitespace() {
                    let h = w
                        .bytes()
                        .fold(7u64, |h, b| h.wrapping_mul(31).wrapping_add(b as u64));
                    v[(h % 16) as usize] += 1.0;
                }
                v
            })
            .collect())
    };
    db.add_text(
        1,
        "the borrow checker",
        meta(&[("lang", "rust")]),
        "body",
        &model,
    )
    .unwrap();
    db.add_text(
        2,
        "asyncio event loop",
        meta(&[("lang", "py")]),
        "body",
        &model,
    )
    .unwrap();
    db.add_text(
        3,
        "borrow checker rejects aliasing",
        HashMap::new(),
        "body",
        &model,
    )
    .unwrap();
    assert_eq!(
        db.get(3).unwrap().1["body"],
        "borrow checker rejects aliasing"
    );
    let hits = db
        .search_text_embedded("borrow checker", 2, None, &model)
        .unwrap();
    assert_eq!(
        hits.iter()
            .map(|h| h.id)
            .collect::<std::collections::HashSet<_>>(),
        [1, 3].into()
    );
    let hits = db
        .search_text_embedded(
            "event loop",
            1,
            Some(&sanadb::Filter::eq("lang", "py")),
            &model,
        )
        .unwrap();
    assert_eq!(hits[0].id, 2);
    assert_eq!(db.search_text("aliasing", 1, None).unwrap()[0].id, 3);
}

#[test]
fn store_properties_survive_checkpoints_reopen_and_reach_readers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("props.sana");
    let db = Db::open(&path, Options::new(4)).unwrap();
    assert!(db.properties().is_empty());
    db.set_property("embedding_model", "nomic-embed-text")
        .unwrap();
    db.set_property("owner", "rohith").unwrap();
    let reader = Db::open_read_only(&path, Options::new(4)).unwrap();
    assert_eq!(
        reader.property("embedding_model").as_deref(),
        Some("nomic-embed-text")
    );
    for i in 0..50u64 {
        db.insert(i, &vec_of(4, i), HashMap::new()).unwrap();
    }
    db.set_property("embedding_model", "bge-m3").unwrap(); // last write wins
    db.set_property("owner", "").unwrap(); // empty removes
    db.checkpoint().unwrap();
    assert_eq!(db.properties().len(), 1);
    assert_eq!(db.property("embedding_model").as_deref(), Some("bge-m3"));
    assert_eq!(
        reader.property("embedding_model").as_deref(),
        Some("bge-m3")
    );
    assert!(reader.property("owner").is_none());
    drop(db);
    let db = Db::open(&path, Options::new(4)).unwrap();
    assert_eq!(db.property("embedding_model").as_deref(), Some("bge-m3"));
    assert_eq!(db.len(), 50);
    // The compacted log carries the property after the snapshot frame.
    let log = std::fs::read(&path).unwrap();
    assert_eq!(log[16 + 8], 4, "first frame is the snapshot");
    let second = 16 + 8 + u32::from_le_bytes(log[16..20].try_into().unwrap()) as usize;
    assert_eq!(log[second + 8], 7, "then the property frame");
}
