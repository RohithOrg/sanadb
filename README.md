# sanadb

An embedded vector store in Rust, with Python bindings. Crash-safe writes,
metadata filters, and a graph index that keeps working when the data is
larger than RAM. No server.

[![CI](https://github.com/RohithOrg/sanadb/actions/workflows/ci.yml/badge.svg)](https://github.com/RohithOrg/sanadb/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/sanadb.svg)](https://crates.io/crates/sanadb)
[![PyPI](https://img.shields.io/pypi/v/sanadb.svg)](https://pypi.org/project/sanadb/)

```bash
cargo add sanadb      # Rust
pip install sanadb    # Python, prebuilt wheels for Linux, macOS, Windows
```

## Should you use it

Probably not yet for anything that matters. This is a 0.x release: the file
formats may still change, and nobody has run it in production.

Where it sits among the alternatives:

- **faiss, hnswlib, usearch** are indexes, not stores. They live in RAM, have
  no log, no crash safety and no metadata; persistence and updates are your
  problem. If an index is all you need, use one of them.
- **Qdrant, Milvus, Weaviate** are servers. Different kind of thing.
- **LanceDB** is the closest comparison: embedded, disk-based, Rust with Python
  bindings, mature and funded. It uses IVF-PQ; sanadb uses a DiskANN-style
  graph. If you need something dependable this quarter, use LanceDB.

What sanadb is trying to be is the SQLite of vector search: one dependency, a
few files, a write-ahead log with honest fsync semantics, concurrent readers,
index rebuilds that do not block writers, metadata filters, and a disk-resident
graph index. Nothing small and embeddable does all of that today. The numbers
below are real and reproducible; the track record is not there yet.

## Guarantees

- A write that returned `Ok` survives a process crash. It survives power loss
  only after `sync()` or with `sync_every_write`.
- A checkpoint either completes or leaves the previous files untouched.
- Readers never see a partially written record or index.
- One open `Db` per store. A second open, from any process, fails with
  `Error::Locked`.
- Search over records written since the last checkpoint is exact. Search over
  the graph is approximate; recall depends on `search_beam` and is measured
  below on public datasets.

Not provided: replication, encryption, multi-process access, typed metadata,
range filters, format stability before 1.0.

## Usage

Rust:

```rust
use sanadb::{Db, Options, Metric};
use std::collections::HashMap;

// Metric::L2, Metric::Dot, or Metric::Cosine (normalises on insert and query).
let db = Db::open("vectors.sana", Options::new(768).metric(Metric::Cosine))?;
db.insert(1, &embedding, HashMap::from([("doc".into(), "intro.md".into())]))?;
let hits = db.search_filtered(&query, 10, |m| m["doc"].ends_with(".md"))?;
db.sync()?;       // fsync: survive power loss
db.checkpoint()?; // build the graph index, compact the log
```

Python:

```python
import numpy as np, sanadb

db = sanadb.Db("vectors.sana", dim=768, metric="cosine")   # or "l2", "dot"
db.insert(1, embedding, {"doc": "intro.md"})
db.insert_many(ids, np.asarray(vectors, dtype=np.float32), metas)
hits = db.search(query, k=10, filter={"doc": "intro.md"})   # [(id, score), ...]
hits = db.search(query, k=10, filter=lambda m: m["doc"].endswith(".md"))
db.checkpoint()
```

`Db` is thread-safe. Reads run concurrently, a checkpoint builds the new index
without blocking readers or writers, and the Python bindings release the GIL
during inserts, dict-filtered searches and checkpoints. Scores are distances:
lower is closer. For cosine they are the negative cosine similarity, in
`[-1, 1]`.

## How it works

A store is three files: `<path>` is the write-ahead log, `<path>.idx` the graph
index, `<path>.lock` an empty file that is `flock`ed while the store is open.

**Log.** CRC-framed records: insert (id, vector, metadata), delete (id), and
meta (id, metadata only; the vector lives in the index). Opening replays the
log and truncates a torn tail. Writes go to the OS page cache at once and
survive a process crash; `sync()` or `sync_every_write` fsyncs for power loss.

**Graph index.** A Vamana graph, the DiskANN layout: one fixed-size block per
node holding its vector, neighbour count and up to R neighbour slots. The file
is memory-mapped, so a query touches only the blocks it walks and the page
cache holds the working set. Per node, RAM holds the 8-byte id, the metadata,
and a 32-byte product-quantization code; the vectors and edges stay on disk.

**Product quantization.** Indexes of 1024 nodes or more carry a codebook (256
centroids per subspace of about 4 floats) and one byte per subspace per node.
The walk scores neighbours from these codes, so expanding a node costs one
block read, its adjacency, rather than R reads of neighbour vectors. The best
candidates are re-ranked with exact distances from the mapped file.

**Beam-width expansion.** The walk expands several candidates per step and
asks the kernel to prefetch all their blocks first (`madvise(WILLNEED)`), so a
cold query overlaps its disk reads instead of taking one page fault at a time.
`Options::io_width`, default 4.

**Delta and tombstones.** Writes since the last checkpoint live in an exact
in-RAM index. Graph records that were since deleted or overwritten are hidden
by a tombstone set rebuilt from the log on open. A search walks the graph,
scans the delta, merges.

**Checkpoint.** Writes the live vectors to a temp file under a brief exclusive
lock, memory-maps it, builds the new graph with no lock held, then swaps the
index in by rename, rewrites the log and swaps that in. Index first, so the
log never names a vector the index lacks. Writes that arrive during the build
stay in the delta, told apart by a per-record sequence number. Sets larger
than `shard_size` (default 1M) are split by k-means into overlapping shards,
each shard is built on its own and spilled to disk, and a final pass merges
and prunes each node's edges. Peak memory is one shard's adjacency, not the
whole graph's.

**Filters.** A filter is applied to visited nodes. If fewer than `k` pass, the
walk retries with the beam doubled, up to `max_search_beam`, so a 1% filter
still returns `k` results. The cost grows with selectivity instead of the
result silently coming back short.

**Concurrency.** A `RwLock` guards the in-memory state: reads share it, writes
take it, checkpoints serialise on a separate mutex and hold neither during
the build.

## Numbers

All on one laptop: Apple Silicon, 10 cores, 16 GB, NVMe. Single query thread.
Everything here is reproducible with the examples in this repo.

### SIFT1M

1,000,000 vectors, dim 128, L2. The 10,000 published queries and published
ground truth. Index fits in RAM, so this measures CPU, not disk.
`cargo run --release --example sift1m -- data/sift`

| | PQ on | PQ off |
|---|---|---|
| build | 115 s | 101 s |
| index file | 744 MB | 712 MB |
| beam 32: recall@10, mean, p99 | 0.898, 365 µs, 2.9 ms | 0.925, 378 µs, 3.7 ms |
| beam 64 | 0.961, 217 µs, 310 µs | 0.969, 246 µs, 346 µs |
| beam 128 | 0.985, 394 µs, 634 µs | 0.988, 441 µs, 577 µs |
| beam 256 | 0.995, 694 µs, 910 µs | 0.995, 800 µs, 1.0 ms |

Beam 32 was the first pass after opening the index, so its tail includes
first-touch page faults. PQ costs about 0.01 recall at a given beam and saves
10 to 15% CPU when the index is warm.

### Cold cache, 5M vectors from SIFT1B

First 5,000,000 vectors of big-ann-benchmarks "bigann" (uint8, dim 128, L2),
the first 1000 public queries, ground truth by brute force. The 3.5 GB index
was evicted with `sudo purge` before each cold pass; the warm pass repeats the
same queries. Beam 64. `examples/bigann.rs` has the exact steps.

| | PQ on, width 1 | PQ off, width 1 | PQ on, width 4 | PQ on, width 8 |
|---|---|---|---|---|
| recall@10 | 0.927 | 0.937 | 0.932 | 0.937 |
| cold mean | 8.6 ms | 21.9 ms | 2.75 ms | 2.26 ms |
| cold p50 | 5.6 ms | 4.5 ms | 2.7 ms | 2.2 ms |
| cold p99 | 39 ms | 187 ms | 4.2 ms | 3.4 ms |
| warm mean | 270 µs | 289 µs | 323 µs | 366 µs |

Reading this table: PQ matters in the tail, where the walk without it reads
every neighbour's block from disk. Beam width matters everywhere cold, because
it turns a chain of dependent page faults into batches; width 4 is the default
and costs 20% warm CPU. Cold is still about 8x warm. The rest is the few
rounds of dependent reads a 64-beam walk cannot avoid without speculative
prefetch or an explicit block cache.

### Build memory

Same 5M set. Peak resident memory of the whole process from `/usr/bin/time -l`,
which includes the 2.6 GB of bulk-loaded vectors the delta holds until the
first checkpoint.

| | single shard | 5 shards of 1M |
|---|---|---|
| build | 731 s | 1396 s |
| peak RSS | about 7 GB (estimated, not measured) | 4.47 GB |
| recall@10, beam 64 | 0.932 | 0.934 |

Sharding halves the build's memory at the cost of doubling its time, since
every vector is inserted into two shard graphs. Recall does not move. The
bulk-load delta is now the larger part of peak memory; fixing that is next.

### Synthetic

Uniform random vectors are the easiest case for approximate search and are
kept only as a regression check: `cargo bench --bench search`. At 100K x 128
the exact scan takes 3.8 ms and the graph 140 µs.

## Next

1. Periodic checkpoints during bulk load, so the delta stops being the memory ceiling.
2. ann-benchmarks and VectorDBBench adapters on the Python bindings, so the
   comparison with hnswlib, usearch, faiss and LanceDB is run by their harness,
   not ours.
3. e-means (Meta FAIR's mini-batch k-means) for PQ training and shard partitioning.
4. Format freeze and 1.0.
5. `sanadb studio`: a local web UI to open a store, map it in 2D, cluster,
   search, watch the beam walk, and edit metadata.

## Development

```bash
cargo test --release
cargo clippy --all-targets -- -D warnings
cargo bench --bench search
cd python && maturin develop --release && pytest tests
```

Releases are cut by tagging `vX.Y.Z`; the workflow publishes the crate and
the wheels. Licensed under MIT or Apache-2.0, at your option.
