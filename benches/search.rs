use std::collections::HashMap;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::{Rng, SeedableRng, rngs::StdRng};
use sanadb::{Db, Options};

fn bench(c: &mut Criterion) {
    let dim = 128;
    let mut rng = StdRng::seed_from_u64(1);
    let dir = tempfile::tempdir().unwrap();

    let mut group = c.benchmark_group("insert");
    group.throughput(Throughput::Elements(1));
    group.bench_function("dim128_no_sync", |b| {
        let db = Db::open(dir.path().join("ins.sana"), Options::new(dim)).unwrap();
        let v: Vec<f32> = (0..dim).map(|_| rng.r#gen()).collect();
        let mut id = 0u64;
        b.iter(|| {
            db.insert(id, &v, HashMap::new()).unwrap();
            id += 1;
        });
    });
    group.finish();

    let mut group = c.benchmark_group("search_k10");
    for &n in &[10_000usize, 100_000] {
        let db = Db::open(dir.path().join(format!("s{n}.sana")), Options::new(dim)).unwrap();
        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.r#gen()).collect();
            let meta = HashMap::from([("bucket".to_string(), (i % 10).to_string())]);
            db.insert(i as u64, &v, meta).unwrap();
        }
        let q: Vec<f32> = (0..dim).map(|_| rng.r#gen()).collect();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("flat", n), &n, |b, _| b.iter(|| db.search(&q, 10).unwrap()));
        group.bench_with_input(BenchmarkId::new("flat_filtered_10pct", n), &n, |b, _| {
            b.iter(|| db.search_filtered(&q, 10, |m| m["bucket"] == "3").unwrap())
        });

        let t = std::time::Instant::now();
        db.checkpoint().unwrap();
        eprintln!("graph build n={n}: {:?}", t.elapsed());
        group.bench_with_input(BenchmarkId::new("graph", n), &n, |b, _| b.iter(|| db.search(&q, 10).unwrap()));
        group.bench_with_input(BenchmarkId::new("graph_filtered_10pct", n), &n, |b, _| {
            b.iter(|| db.search_filtered(&q, 10, |m| m["bucket"] == "3").unwrap())
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
