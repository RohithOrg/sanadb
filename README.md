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
  `Error::Locked`. The lock is an advisory `flock` on `<path>.lock`: reliable
  on local filesystems, not on NFS, SMB or other network mounts, where two
  machines can both believe they hold it. Keep stores on local disks.
- Search over records written since the last checkpoint is exact. Search over
  the graph is approximate; recall depends on `search_beam` and is measured
  below on public datasets.

- A newer sanadb opens any store an older one wrote, converting it on first
  open if the layout changed. An older sanadb opens a newer store as long as
  the layout is one it knows, skipping record kinds it does not understand;
  if the layout is newer than it can read, it fails with
  `Error::NeedsNewerVersion` rather than misreading. `CHANGELOG.md` lists
  every format change and its upgrade notes.

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

**Delta and tombstones.** Writes since the last checkpoint are searched
exactly. The delta keeps only ids and byte offsets into the memory-mapped log,
16 bytes per record; the vectors are read from the log in place, through the
page cache. Graph records that were since deleted or overwritten are hidden by
a tombstone set rebuilt from the log on open. A search walks the graph, scans
the delta, merges.

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
| build | 77 s | 76 s |
| index file | 744 MB | 712 MB |
| beam 32: recall@10, mean, p99 | 0.914, 250 µs, 1.1 ms | |
| beam 64 | 0.966, 207 µs, 305 µs | 0.965, 311 µs |
| beam 128 | 0.987, 343 µs, 478 µs | 0.985, 506 µs |
| beam 256 | 0.995, 586 µs, 806 µs | 0.995, 840 µs |

Beam 32 was the first pass after opening the index, so its tail includes
first-touch page faults. PQ costs nothing in recall here and saves about a
third of the CPU when the index is warm. The PQ-off row was measured with
1,000 queries per beam rather than 10,000, so its cache was colder; treat it
as an upper bound.

### ann-benchmarks, against hnswlib, faiss HNSW and usearch

Run by the [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) harness
in local mode with the adapters in `benchmarks/ann-benchmarks/`, so the
queries, ground truth, timing and recall computation are theirs. The three
in-RAM HNSW libraries use the harness's own configs at M=16 and M=24,
efConstruction 500 (faiss trimmed to those two groups; usearch has no
upstream adapter, ours is in the same folder and forces f32 vectors so all
four hold the same data). Same laptop, single query thread, k=10, warm,
best of 2 runs. Queries per second at the nearest measured recall, higher is
better; a blank cell means that configuration has no point within 0.01 of
the row's recall. Raw CSVs: `benchmarks/ann-benchmarks/results/`.

#### sift-128-euclidean

1,000,000 SIFT vectors, dim 128, L2, their 10,000 queries.

| recall@10 | hnswlib M=16 | hnswlib M=24 | faiss HNSW M=16 | faiss HNSW M=24 | usearch M=16 | usearch M=24 | sanadb R=32 PQ | sanadb R=32 no PQ | sanadb R=48 PQ |
|---|---|---|---|---|---|---|---|---|---|
| ~0.93 | 5,938 (0.931) |  | 11,559 (0.939) |  | 5,119 (0.930) |  |  | 5,457 (0.934) | 4,579 (0.939) |
| ~0.96 |  | 4,797 (0.958) |  | 9,844 (0.958) |  | 3,991 (0.956) | 3,947 (0.964) | 4,229 (0.957) |  |
| ~0.98 | 3,452 (0.977) | 2,808 (0.989) | 6,498 (0.982) | 5,467 (0.990) | 2,936 (0.977) | 2,298 (0.988) | 2,800 (0.980) | 2,400 (0.983) | 3,003 (0.984) |
| ~0.99 | 2,520 (0.990) | 2,808 (0.989) | 4,332 (0.992) | 5,467 (0.990) | 2,102 (0.989) | 2,298 (0.988) | 1,696 (0.993) | 2,041 (0.989) | 2,229 (0.993) |
| ~0.996 | 1,639 (0.997) | 2,011 (0.996) | 2,796 (0.998) | 3,849 (0.996) | 1,357 (0.997) | 1,642 (0.995) | 1,348 (0.995) | 1,154 (0.996) | 1,774 (0.996) |
| ~0.999 | 922 (1.000) | 1,323 (0.999) | 1,425 (1.000) | 2,424 (0.999) | 749 (0.999) | 1,073 (0.999) | 743 (0.998) | 635 (0.998) | 1,029 (0.999) |
| build | 209 s | 266 s | 132 s | 764 s* | 252 s | 311 s | 119 s | 112 s | 222 s |

#### glove-100-angular

1,183,514 GloVe word vectors, dim 100, angular ground truth, their 10,000
queries. The sanadb adapter maps `angular` to `Metric::Cosine`. This dataset
is hard for every graph index; hnswlib reaches 0.98 only at ef = 800.

| recall@10 | hnswlib M=16 | hnswlib M=24 | faiss HNSW M=16 | faiss HNSW M=24 | usearch M=16 | usearch M=24 | sanadb R=32 PQ | sanadb R=32 no PQ | sanadb R=48 PQ |
|---|---|---|---|---|---|---|---|---|---|
| ~0.73 | 5,174 (0.728) |  | 9,619 (0.732) |  | 3,697 (0.729) |  |  | 3,843 (0.732) |  |
| ~0.81 | 2,979 (0.811) |  | 5,505 (0.816) |  | 2,237 (0.811) |  |  | 2,273 (0.810) |  |
| ~0.86 |  | 2,266 (0.853) | 3,873 (0.852) | 4,084 (0.859) |  | 1,667 (0.853) | 1,517 (0.858) | 1,314 (0.866) | 2,248 (0.860) |
| ~0.89 | 1,401 (0.889) | 1,610 (0.887) | 2,435 (0.889) | 2,854 (0.893) | 1,024 (0.887) | 1,187 (0.887) | 1,066 (0.885) | 927 (0.891) | 1,608 (0.890) |
| ~0.93 | 787 (0.933) | 1,047 (0.922) | 1,241 (0.930) | 1,807 (0.928) | 553 (0.932) | 764 (0.923) |  |  | 882 (0.929) |
| ~0.95 | 553 (0.952) | 577 (0.960) | 810 (0.948) |  | 380 (0.951) | 410 (0.959) |  |  | 697 (0.942) |
| build | 319 s | 434 s | 967 s* | 257 s | 388 s | 555 s | 378 s | 322 s | 658 s |

What the tables say:

- faiss HNSW is the fastest of the four here, about 2x hnswlib at equal
  recall on both datasets. The most likely reason is SIMD: faiss ships NEON
  kernels for Apple Silicon, while hnswlib's and sanadb's distance loops are
  SSE/AVX intrinsics with a scalar fallback (hnswlib) and plain Rust left to
  the autovectoriser (sanadb). On an x86 server the gap between faiss and
  hnswlib is small in ann-benchmarks' published plots; expect these ratios
  to shift there. sanadb has no hand-written SIMD yet; that is the next
  performance item.
- sanadb lands within 10 to 25% of hnswlib and usearch at equal recall on
  SIFT, and at R=48 matches hnswlib on GloVe up to recall 0.93. At R=32 it
  trails on GloVe by 20 to 30%, and PQ costs about 0.02 recall per beam on
  these 100-dim vectors (25 subspaces of 4). It does this while holding 32
  bytes of PQ codes per vector in RAM instead of the whole index; the other
  three keep every vector and every edge in memory.
- usearch was forced to f32 so that the four libraries hold the same data.
  Its default f16 and i8 modes are faster and are not measured here.
- `*` faiss build times are not comparable between its two groups: the
  upstream adapter calls `omp_set_num_threads(1)` after each build, and in
  the harness's local mode that setting persists into the next group's
  build, so the second group of each dataset built single-threaded (M=24 on
  SIFT, M=16 on GloVe). Builds are multi-threaded for every library
  otherwise; sanadb's are sharded above 1M vectors, which is why the R=48
  GloVe build is slow.
- The GloVe table exists because its first run found a real defect: before
  0.1.3 cosine recall at beam 64 was 0.24, since RobustPrune's alpha test
  misbehaves on negative distances and cosine used negative dot product
  internally. See CHANGELOG.
- The harness's "index size" column is omitted: it measures process memory
  growth during build, which is near zero for a memory-mapped index.

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

Same 5M set, `examples/bigann.rs build`, which streams the 5M inserts through
the normal write path with auto-checkpoint on. Peak resident memory of the
whole process from `/usr/bin/time -l`.

| | 0.1.1, single shard | 0.1.1, 5 shards of 1M | 0.1.3, 5 shards of 1M |
|---|---|---|---|
| insert 5M (with auto-checkpoints) | | | 1114 s |
| final checkpoint | 731 s | 1396 s | 856 s |
| max RSS | about 7 GB (estimated, not measured) | 4.47 GB | 5.13 GB |
| peak anonymous memory | | not recorded | 3.58 GB |
| recall@10, beam 64 | 0.932 | 0.934 | 0.936 |

Sharding halves the build's anonymous memory at the cost of inserting every
vector into two shard graphs. Recall does not move.

Reading the 0.1.3 column: the delta no longer holds vectors in RAM (it reads
them from the mapped log), so the 2.6 GB bulk-load share of the old peak is
gone and the anonymous footprint is 3.58 GB (macOS "peak memory footprint").
Max RSS went up, not down, because RSS also counts resident pages of
memory-mapped files, and this build maps the 2.4 GB vector snapshot and the
index it is writing; those pages are reclaimable and are not what limits the
machine. The final checkpoint is 39% faster than the 0.1.1 sharded build.
The insert row is the realistic streaming path: the geometric auto-checkpoint
rule built intermediate graphs at 250K, 500K, 1M, 2M and 4M records along the
way; a bulk load that sets `auto_checkpoint_delta(0)` skips those.

### Synthetic

Uniform random vectors are the easiest case for approximate search and are
kept only as a regression check: `cargo bench --bench search`. At 100K x 128
the exact scan takes 3.8 ms and the graph 140 µs.

## Next

1. Hand-written SIMD distance kernels (NEON and AVX2); the faiss gap above
   is mostly that. Close the R=32 gap on cosine data. A VectorDBBench client.
2. Typed metadata and filter-aware graph traversal.
3. Format freeze and 1.0.
4. `sanadb studio`: a local web UI to open a store, map it in 2D, cluster,
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
