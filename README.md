# sanadb

A disk-first embedded vector + metadata store in Rust. Single file, crash-safe,
no server. Aimed at on-device and edge AI workloads that outgrow RAM.

[![CI](https://github.com/RohithOrg/sanadb/actions/workflows/ci.yml/badge.svg)](https://github.com/RohithOrg/sanadb/actions/workflows/ci.yml)

Status: pre-1.0. The on-disk formats may still change between versions, and
no release has been cut. Crash safety, concurrent access and real-data recall
are tested (see below); multi-process access, replication and encryption are
not provided.

## Usage

Rust:

```rust
use sanadb::{Db, Options, Metric};
use std::collections::HashMap;

let mut db = Db::open("vectors.sana", Options::new(768).metric(Metric::Dot))?;
db.insert(1, &embedding, HashMap::from([("doc".into(), "intro.md".into())]))?;
let hits = db.search_filtered(&query, 10, |m| m["doc"].ends_with(".md"))?;
db.sync()?;       // fsync: survive power loss
db.checkpoint()?; // build the graph index, compact the log
```

Python (`pip install maturin && cd python && maturin develop --release`):

```python
import numpy as np, sanadb

db = sanadb.Db("vectors.sana", dim=768, metric="dot")
db.insert(1, embedding, {"doc": "intro.md"})
db.insert_many(ids, np.asarray(vectors, dtype=np.float32), metas)
hits = db.search(query, k=10, filter={"doc": "intro.md"})   # [(id, score), ...]
hits = db.search(query, k=10, filter=lambda m: m["doc"].endswith(".md"))
db.checkpoint()
```

`Db` is thread-safe. Reads run concurrently; a checkpoint builds the new
index without blocking readers or writers, and the GIL is released during
inserts, searches with dict filters, and checkpoints.

## Design

Two files: `<path>` is the write-ahead log, `<path>.idx` is the graph index.

- **Log.** Header plus CRC-framed frames: insert (id, vector, metadata),
  delete (id), meta (id, metadata; the vector lives in the index). Open replays
  it; a torn tail is truncated. Writes hit the OS page cache immediately and
  survive a process crash; `sync()` or `Options::sync_every_write` fsync for
  power-loss safety.
- **Graph index.** A Vamana graph (the DiskANN layout): one fixed-size block
  per node holding the vector, neighbour count and up to R neighbour slots.
  The file is memory-mapped, so a query reads only the blocks it walks and the
  OS page cache holds the working set. The dataset does not have to fit in RAM;
  the id table (8 bytes per node) and metadata do.
- **Delta + tombstones.** Writes since the last checkpoint live in an exact
  in-RAM flat index. Graph records that were deleted or overwritten are hidden
  by a tombstone set rebuilt from the log on open. A search walks the graph,
  scans the delta, and merges.
- **Checkpoint.** Snapshots live vectors under a brief exclusive lock, builds
  the graph with no lock held, then swaps the index file in by rename, rewrites
  the log (metadata-only frames, plus full frames for records written during
  the build) and swaps that in. Index first, so the log never names a vector
  the index lacks. A crash between the two renames leaves a consistent,
  slightly redundant state. Writes that land during the build stay in the
  delta, identified by a per-record write sequence number.
- **Concurrency.** `Db` is `Send + Sync`: a `RwLock` guards the in-memory
  state, reads share it, writes take it exclusively, checkpoints serialise on
  a separate mutex. One process at a time; there is no file lock yet.
- **Flat index.** Contiguous `Vec<f32>`; distance loops use eight independent
  accumulators so the compiler emits packed FMA. Also the correctness baseline
  the graph is tested against (recall@10 ≥ 0.9 enforced, 0.996 measured).

- **Parallel build.** Node insertions run across all cores with a lock per
  adjacency list; no thread holds two locks. The graph is nondeterministic
  but recall is unchanged.
- **Adaptive filtered search.** A filter is applied to visited nodes. When
  fewer than `k` pass, the walk retries with the beam doubled up to
  `max_search_beam`, so a 1% filter still returns `k` results with the default
  beam. Cost grows with selectivity instead of silently returning short.

- **Product quantization.** Indexes with at least 1024 nodes also store a
  codebook (256 centroids per subspace, ~4 floats each) and one byte per
  subspace per node. The codes are loaded into RAM on open (32 bytes per node
  at dim 128, versus 512 for the vector). The walk scores neighbours from the
  codes, so expanding a node reads one block from disk, its adjacency, instead
  of R neighbour vectors. The final candidates are re-ranked with exact
  distances from the mapped file. `GraphParams::use_pq = false` turns it off.

Graph build still holds all vectors in RAM.

## Numbers

**Synthetic data.** Uniform random vectors from `rand`, the easiest case for
approximate search; recall on real embeddings will be lower. Apple Silicon
laptop, 10 cores, dim 128, k = 10, R = 32, build L = 100, search beam 64.
Everything in the OS page cache (hot).

| n | flat search | graph search | graph, 10% filter | graph build |
|---|---|---|---|---|
| 10,000 | 227 µs | 117 µs | 115 µs | 1.2 s |
| 100,000 | 3.8 ms | 140 µs | 146 µs | 14.4 s |

Graph search time is nearly flat in n. Hot-cache latency is the same with and
without PQ: the win PQ buys is fewer distinct disk blocks touched per query,
which only shows once the dataset outgrows the page cache. Not yet measured.

Recall@10 against brute force on the synthetic set, from the test suite:
0.98 (5K x 32, PQ), 0.99 exact walk vs 0.98 PQ walk (3K x 32), 0.93 with a
1% filter and a beam of 16.

**SIFT1M (public, real data).** 1,000,000 base vectors, dim 128, L2, the
10,000 published queries and published top-100 ground truth. Same laptop,
single query thread, hot page cache (the 744 MB index fits in RAM, so this
measures CPU cost, not disk). Run it yourself with
`cargo run --release --example sift1m -- data/sift`.

| | PQ on (32 B/node in RAM) | PQ off (walk reads full vectors) |
|---|---|---|
| build, 10 cores | 115 s | 101 s |
| index file | 744 MB | 712 MB |
| beam 32: recall@10, mean, p99 | 0.898, 365 µs, 2.9 ms | 0.925, 378 µs, 3.7 ms |
| beam 64 | 0.961, 217 µs, 310 µs | 0.969, 246 µs, 346 µs |
| beam 128 | 0.985, 394 µs, 634 µs | 0.988, 441 µs, 577 µs |
| beam 256 | 0.995, 694 µs, 910 µs | 0.995, 800 µs, 1.0 ms |

Beam 32 is the first pass after the index is opened, so its tail includes
first-touch page faults on the mapped file; later passes run warm. PQ costs
about 0.01 recall at a given beam and saves 10–15% CPU hot; its disk-read
saving is unmeasured until the index exceeds RAM.

## Roadmap

1. **Done:** WAL, checkpoint, flat index, metadata filter, tests, Criterion bench.
2. **Done:** Disk-resident Vamana graph index, memory-mapped, with delta +
   tombstones and atomic two-file checkpoint.
3. **Done:** Parallel build; adaptive beam for selective filters.
4. **Done:** Product-quantized codes in RAM for the walk, exact re-rank from disk.
5. **Done:** SIFT1M harness with published ground truth.
6. **Done:** Thread-safe `Db`, non-blocking checkpoint, Python bindings, CI.
7. Cold-cache measurement on a dataset larger than RAM (SIFT1B subset or
   Cohere 10M). This is the thesis of the project and is still unproven.
8. Sharded graph build so building, not just searching, scales past RAM.
9. File lock against a second process opening the same store.
10. ann-benchmarks and VectorDBBench adapters on top of the Python bindings.

## Development

```bash
cargo test --release
cargo clippy --all-targets -- -D warnings
cargo bench --bench search
cd python && maturin develop --release && pytest tests
```

See `RESEARCH.md` for the problem survey behind this direction.
