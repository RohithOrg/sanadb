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
    assert!(path.with_file_name("v.sana.idx").exists());

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
        let _ = std::fs::remove_file(dir.path().join("fuzz.sana.idx"));
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
    assert_eq!(bytes[13], 2, "header now at reader level 2");
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
