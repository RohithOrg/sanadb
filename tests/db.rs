use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;

use sanadb::{Db, Metadata, Metric, Options};

fn meta(pairs: &[(&str, &str)]) -> Metadata {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

fn vec_of(dim: usize, seed: u64) -> Vec<f32> {
    // Cheap deterministic pseudo-random vector.
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
    let mut db = Db::open(dir.path().join("v.sana"), Options::new(4)).unwrap();
    db.insert(1, &[0.0, 0.0, 0.0, 0.0], meta(&[("kind", "a")])).unwrap();
    db.insert(2, &[1.0, 0.0, 0.0, 0.0], meta(&[("kind", "b")])).unwrap();
    db.insert(3, &[5.0, 5.0, 5.0, 5.0], meta(&[("kind", "a")])).unwrap();

    let hits = db.search(&[0.9, 0.0, 0.0, 0.0], 2).unwrap();
    assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![2, 1]);

    let hits = db.search_filtered(&[0.9, 0.0, 0.0, 0.0], 2, |m| m["kind"] == "a").unwrap();
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
    let mut db = Db::open(dir.path().join("v.sana"), Options::new(2)).unwrap();
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
        let mut db = Db::open(&path, Options::new(3)).unwrap();
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
        let mut db = Db::open(&path, Options::new(2)).unwrap();
        db.insert(1, &[1.0, 1.0], HashMap::new()).unwrap();
        db.insert(2, &[2.0, 2.0], HashMap::new()).unwrap();
        db.sync().unwrap();
    }
    // Simulate a crash mid-write: append half a frame of junk.
    OpenOptions::new().append(true).open(&path).unwrap().write_all(&[0x20, 0, 0, 0, 1, 2, 3]).unwrap();
    let before = std::fs::metadata(&path).unwrap().len();
    let mut db = Db::open(&path, Options::new(2)).unwrap();
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
    let mut db = Db::open(&path, Options::new(8)).unwrap();
    for i in 0..200u64 {
        db.insert(i, &vec_of(8, i), meta(&[("i", &i.to_string())])).unwrap();
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
        .map(|(i, v)| (v.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum::<f32>(), i as u64))
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0));
    all[..k].iter().map(|e| e.1).collect()
}

#[test]
fn graph_recall_vs_brute_force() {
    let (dim, n, k) = (32usize, 5000u64, 10usize);
    let dir = tempfile::tempdir().unwrap();
    let mut db = Db::open(dir.path().join("v.sana"), Options::new(dim as u32)).unwrap();
    let vectors: Vec<Vec<f32>> = (0..n).map(|i| vec_of(dim, i)).collect();
    for (i, v) in vectors.iter().enumerate() {
        db.insert(i as u64, v, HashMap::new()).unwrap();
    }
    db.checkpoint().unwrap();
    assert_eq!(db.delta_len(), 0);
    assert_eq!(db.len(), n as usize);

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
    let mut db = Db::open(&path, Options::new(dim as u32)).unwrap();
    for i in 0..n {
        db.insert(i, &vec_of(dim, i), meta(&[("i", &i.to_string())])).unwrap();
    }
    db.checkpoint().unwrap();
    drop(db);

    // Log holds metadata only now: far smaller than the vectors it describes.
    let log_bytes = std::fs::metadata(&path).unwrap().len();
    assert!(log_bytes < n * dim as u64 * 4 / 2, "log is {log_bytes} bytes");
    assert!(path.with_file_name("v.sana.idx").exists());

    let mut db = Db::open(&path, Options::new(dim as u32)).unwrap();
    assert_eq!(db.len(), n as usize);
    assert_eq!(db.delta_len(), 0);
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
    assert!(db.search(&q7, 10).unwrap().iter().all(|h| h.id != 7), "old position of 7 still visible");
    let near_far = db.search(&far, 1).unwrap();
    assert_eq!(near_far[0].id, 7);
    assert_eq!(near_far[0].score, 0.0);
    drop(db);

    // Replay must re-derive the tombstones.
    let mut db = Db::open(&path, Options::new(dim as u32)).unwrap();
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
    let mut db = Db::open(dir.path().join("v.sana"), Options::new(dim as u32).search_beam(256)).unwrap();
    for i in 0..1000u64 {
        db.insert(i, &vec_of(dim, i), meta(&[("parity", if i % 2 == 0 { "even" } else { "odd" })])).unwrap();
    }
    db.checkpoint().unwrap();
    let q = vec_of(dim, 777_777);
    let hits = db.search_filtered(&q, 5, |m| m["parity"] == "odd").unwrap();
    assert_eq!(hits.len(), 5);
    assert!(hits.iter().all(|h| h.id % 2 == 1));
}

#[test]
fn selective_filter_fills_k_with_small_beam() {
    // 1% of records pass the filter; a beam of 16 visits far fewer than the
    // ~1000 nodes needed to see 10 of them. The adaptive retry must widen it.
    let (dim, n, k) = (16usize, 5000u64, 10usize);
    let dir = tempfile::tempdir().unwrap();
    let mut db = Db::open(dir.path().join("v.sana"), Options::new(dim as u32).search_beam(16)).unwrap();
    let vectors: Vec<Vec<f32>> = (0..n).map(|i| vec_of(dim, i)).collect();
    for (i, v) in vectors.iter().enumerate() {
        db.insert(i as u64, v, meta(&[("tag", if i % 100 == 0 { "rare" } else { "common" })])).unwrap();
    }
    db.checkpoint().unwrap();

    let mut hit = 0;
    let queries = 20;
    for qi in 0..queries {
        let q = vec_of(dim, 2_000_000 + qi);
        let rare: Vec<Vec<f32>> = vectors.iter().step_by(100).cloned().collect();
        let truth: Vec<u64> = brute_force_top(&rare, &q, k).iter().map(|i| i * 100).collect();
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
    let db = Db::open(dir.path().join("v.sana"), Options::new(dim as u32).search_beam(16).max_search_beam(16)).unwrap();
    let got = db.search_filtered(&vec_of(dim, 2_000_000), k, |m| m["tag"] == "rare").unwrap();
    assert!(got.len() < k);
}

#[test]
fn exact_search_matches_brute_force() {
    let dim = 32;
    let dir = tempfile::tempdir().unwrap();
    let mut db = Db::open(dir.path().join("v.sana"), Options::new(dim as u32).metric(Metric::Dot)).unwrap();
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
    assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), expect[..10].iter().map(|e| e.1).collect::<Vec<_>>());
}
