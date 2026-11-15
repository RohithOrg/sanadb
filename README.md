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
about 0.01 recall at a given beam and saves 10–15% CPU hot.

**Cold cache, SIFT1B prefix (public, real data).** First 5,000,000 vectors of
big-ann-benchmarks "bigann" (uint8, dim 128, L2), the 1000 first public
queries, ground truth by brute force. 16 GB laptop, NVMe, single query thread,
beam 64. The page cache was dropped with `sudo purge` before each cold pass;
the warm pass re-runs the same queries. Reproduce with `examples/bigann.rs`.

| | PQ on | PQ off |
|---|---|---|
| graph build, 10 cores | 731 s | ~720 s |
| index file | 3.55 GB | 3.40 GB |
| recall@10 | 0.927 | 0.937 |
| cold: mean / p50 / p99 | 8.6 ms / 5.6 ms / 39 ms | 21.9 ms / 4.5 ms / 187 ms |
| warm: mean / p50 / p99 | 270 µs / 264 µs / 372 µs | 289 µs / 283 µs / 404 µs |

Cold, every expanded node is a random 4 KB read. With PQ the walk reads only
the expanded node's block (its adjacency) and scores neighbours from codes in
RAM; without PQ it also reads each neighbour's block. That shows as 2.5x lower
mean and 4.8x lower p99 cold latency with PQ. The medians are close, so for
most queries the neighbour blocks were already pulled in by earlier hops;
the gain is in the tail. Those rows expand one node at a time (`io_width = 1`).

**Beam-width expansion.** `Options::io_width` expands several of the best
unexpanded candidates per step and issues `madvise(WILLNEED)` for all their
blocks first, so the kernel overlaps the reads. Same index, queries and
protocol as above, PQ on:

| io_width | cold mean / p50 / p99 | warm mean | recall@10 |
|---|---|---|---|
| 1 | 8.6 ms / 5.6 ms / 39 ms | 270 µs | 0.927 |
| 4 (default) | 2.75 ms / 2.7 ms / 4.2 ms | 323 µs | 0.932 |
| 8 | 2.26 ms / 2.2 ms / 3.4 ms | 366 µs | 0.937 |

Width 4 cuts cold mean 3.1x and p99 9x for a 20% warm-CPU cost; width 8 buys
a little more cold latency for another 13% warm. Cold is now about 8x warm,
down from 32x. The remaining gap is the ~2 ms of sequential dependent read
rounds a 64-beam walk needs; closing it means issuing reads for likely next
hops before they are chosen, or holding hot blocks in an explicit cache.

## Roadmap

1. **Done:** WAL, checkpoint, flat index, metadata filter, tests, Criterion bench.
2. **Done:** Disk-resident Vamana graph index, memory-mapped, with delta +
   tombstones and atomic two-file checkpoint.
3. **Done:** Parallel build; adaptive beam for selective filters.
4. **Done:** Product-quantized codes in RAM for the walk, exact re-rank from disk.
5. **Done:** SIFT1M harness with published ground truth.
6. **Done:** Thread-safe `Db`, non-blocking checkpoint, Python bindings, CI.
7. **Done:** Cold-cache measurement on a 5M SIFT1B prefix (3.5 GB index,
   page cache purged). PQ cuts cold mean latency 2.5x and p99 4.8x.
8. **Done:** Beam-width expansion with block prefetch. Cold mean 8.6 ms to
   2.75 ms, p99 39 ms to 4.2 ms at the default width.
9. Sharded graph build so building, not just searching, scales past RAM
   (5M took 731 s and ~7 GB; 10M would not fit a 16 GB machine).
10. File lock against a second process opening the same store.
11. ann-benchmarks and VectorDBBench adapters on top of the Python bindings.

## Development

```bash
cargo test --release
cargo clippy --all-targets -- -D warnings
cargo bench --bench search
cd python && maturin develop --release && pytest tests
```

See `RESEARCH.md` for the problem survey behind this direction.
