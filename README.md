# sanadb

A disk-first embedded vector + metadata store in Rust. Single file, crash-safe,
no server. Aimed at on-device and edge AI workloads that outgrow RAM.

Status: milestone 2 (disk-resident graph index). Not production-ready.

## Usage

```rust
use sanadb::{Db, Options, Metric};
use std::collections::HashMap;

let mut db = Db::open("vectors.sana", Options::new(768).metric(Metric::Dot))?;
db.insert(1, &embedding, HashMap::from([("doc".into(), "intro.md".into())]))?;
let hits = db.search_filtered(&query, 10, |m| m["doc"].ends_with(".md"))?;
db.sync()?;       // fsync: survive power loss
db.checkpoint()?; // compact the file
```

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
- **Checkpoint.** Rebuilds the graph over every live record, swaps the index
  file in by rename, then rewrites the log as metadata-only frames and swaps
  that in. Index first, so the log never names a vector the index lacks. A
  crash between the two renames leaves a consistent, slightly redundant state.
- **Flat index.** Contiguous `Vec<f32>`; distance loops use eight independent
  accumulators so the compiler emits packed FMA. Also the correctness baseline
  the graph is tested against (recall@10 ≥ 0.9 enforced, 0.996 measured).

Graph build currently holds all vectors in RAM and runs single-threaded. Search
is filter-after-walk: a very selective filter needs a larger `search_beam`.

## Numbers

Apple Silicon laptop, dim 128, k = 10, R = 32, build L = 100, search beam 64.

| n | flat search | graph search | graph build |
|---|---|---|---|
| 10,000 | 227 µs | 94 µs | 4.2 s |
| 100,000 | 3.66 ms | 130 µs | 54 s |

Graph search time is nearly flat in n; the build is the cost to attack next.

## Roadmap

1. **Done:** WAL, checkpoint, flat index, metadata filter, tests, Criterion bench.
2. **Done:** Disk-resident Vamana graph index, memory-mapped, with delta +
   tombstones and atomic two-file checkpoint.
3. Filter-aware graph traversal (keep walking until k accepted candidates).
4. Product-quantized vectors in RAM for neighbour distances, full vectors on
   disk only for re-ranking (the actual DiskANN trick). Cuts page faults per hop
   from R to 1.
5. Sharded, parallel graph build so building, not just searching, scales past RAM.
6. MVCC snapshots for concurrent readers alongside a single writer.
7. Benchmark against sqlite-vec, usearch and LanceDB on 10M x 768-dim vectors.

## Development

```bash
cargo test
cargo bench
```

See `RESEARCH.md` for the problem survey behind this direction.
