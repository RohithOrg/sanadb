use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;

use sanadb::{Db, Metadata, Metric, Options, Value};

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
                let top = db.search(&v, 1).unwrap();
                assert_eq!(top[0].id, id);
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
                // Who holds it? Ask lsof before retrying, so a transient holder is named.
                let lock = format!("{}.lock", path.display());
                let who = std::process::Command::new("lsof")
                    .arg(&lock)
                    .output()
                    .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                    .unwrap_or_default();
                std::thread::sleep(std::time::Duration::from_millis(50));
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
    }
    eprintln!("reader child: {rounds} rounds, final len {max_len}");
    assert!(rounds > 10);
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
    while t.elapsed() < std::time::Duration::from_millis(1500) {
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
    assert!(rounds > 10);
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
/// and one overwrite before the checkpoint, ten inserts and a delete after).
/// Each must open, read back exactly, accept new writes, checkpoint into the
/// current layout and reopen.
#[test]
fn opens_stores_written_by_every_published_version() {
    let vec_of4 = |i: u64| -> Vec<f32> {
        (0..4u64)
            .map(|d| ((i * 7 + d * 13) % 101) as f32 / 101.0)
            .collect()
    };
    for version in ["0.1.1", "0.1.3", "0.1.4"] {
        let dir = tempfile::tempdir().unwrap();
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(format!("v{version}"));
        for name in ["fixture.sana", "fixture.sana.idx"] {
            std::fs::copy(src.join(name), dir.path().join(name)).unwrap();
        }
        let path = dir.path().join("fixture.sana");
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
            if version == "0.1.4" {
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
        check(&db, "opened", 68);
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
        check(&db, "after checkpoint and reopen", 69);
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
        if i % 5 == 0 {
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
