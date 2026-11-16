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
use sanadb::{Db, Filter, Options, Metric};
use std::collections::HashMap;

// Metric::L2, Metric::Dot, or Metric::Cosine (normalises on insert and query).
let db = Db::open("vectors.sana", Options::new(768).metric(Metric::Cosine))?;
db.insert(1, &embedding, HashMap::from([("doc".into(), "intro.md".into()), ("year".into(), 2024i64.into())]))?;
let hits = db.search_where(&query, 10, &Filter::and([Filter::prefix("doc", "intro"), Filter::ge("year", 2020i64)]))?;
let hits = db.search_filtered(&query, 10, |m| m["doc"].as_str().is_some_and(|s| s.ends_with(".md")))?; // any closure
let batch = db.search_many(&queries, 10)?; // row-major `n * dim`, all cores
db.sync()?;       // fsync: survive power loss
db.checkpoint()?; // build the graph index, compact the log
```

Python:

```python
import numpy as np, sanadb

db = sanadb.Db("vectors.sana", dim=768, metric="cosine")   # or "l2", "dot"
db.insert(1, embedding, {"doc": "intro.md", "year": 2024, "public": True})
db.insert_many(ids, np.asarray(vectors, dtype=np.float32), metas)
hits = db.search(query, k=10, filter={"doc": "intro.md"})   # [(id, score), ...]
hits = db.search(query, k=10, filter={"year": {"$gte": 2020}, "lang": {"$in": ["en", "de"]}, "$not": {"public": False}})
hits = db.search(query, k=10, filter=lambda m: m["doc"].endswith(".md"))   # any callable, slower path
ids, scores = db.search_many(queries, k=10)   # (n, k) arrays, all cores
db.checkpoint()
```

`Db` is thread-safe. Reads run concurrently, a checkpoint builds the new index
without blocking readers or writers, and the Python bindings release the GIL
during inserts, dict-filtered searches, batch searches and checkpoints. Scores are distances:
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

**Metadata and filters.** Values are typed (`Str`, `Int`, `Float`, `Bool`);
the first value under a key fixes its type. At checkpoint each key becomes a
slot-ordered column in the index file (strings dictionary-coded, with a
presence bitmap and, up to 65,536 distinct values, posting lists). A
`Filter` expression runs on those columns: checking a visited node is an
array read. Per query one of two regimes is chosen from the posting lists.
If at most `exact_filter_scan` records (default 16,384) can match, they are
read and ranked exactly, so a rare filter gets recall 1.0 and never a short
answer. Otherwise the walk is steered by the filter: rejected nodes are
expanded for navigation but only passing nodes take result slots, so a 1%
filter costs a longer walk rather than repeated widened walks. Closure
filters cannot be inspected and keep the old path: applied to visited nodes,
beam doubled until `k` pass, up to `max_search_beam`.

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
| build | 75 s | 71 s |
| index file | 744 MB | 712 MB |
| beam 64: recall@10, mean, p99 | 0.966, 146 µs, 213 µs | 0.971, 225 µs, 319 µs |
| beam 128 | 0.987, 240 µs, 328 µs | 0.989, 371 µs, 490 µs |
| beam 256 | 0.995, 405 µs, 555 µs | 0.996, 623 µs, 832 µs |

10,000 queries per beam, index warm. PQ costs 0.001 to 0.005 recall here
and saves about 35% of the CPU: the walk reads 32 bytes of codes per
candidate instead of a 512-byte vector, and only the final `beam`
candidates are re-ranked exactly.

These are the numbers after profiling the walk (0.1.4), measured back to
back against the 0.1.3 binary on the same index under the same load: beam
64 went from 202 µs to 146 µs, beam 256 from 572 µs to 405 µs, recall
unchanged to four digits. The sampler had shown that sorted-list inserts,
hash-set visited checks and a metadata lookup per visited node were together
about 40% of a warm query, and none of it was distance arithmetic. The
`madvise` prefetch calls are another 15% on a warm index and stay, because
they are what makes a cold query fast; `io_width(1)` turns them off. Absolute numbers move 20% with what
else the machine is doing; the ratio holds. See CHANGELOG.

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
| ~0.93 | 5,938 (0.931) |  | 11,559 (0.939) |  | 5,119 (0.930) |  |  | 7,517 (0.935) |  |
| ~0.96 |  | 4,797 (0.958) |  | 9,844 (0.958) |  | 3,991 (0.956) | 8,209 (0.966) | 5,775 (0.961) |  |
| ~0.98 | 3,452 (0.977) | 2,808 (0.989) | 6,498 (0.982) | 5,467 (0.990) | 2,936 (0.977) | 2,298 (0.988) | 6,124 (0.981) | 3,514 (0.984) | 7,665 (0.975) |
| ~0.99 | 2,520 (0.990) | 2,808 (0.989) | 4,332 (0.992) | 5,467 (0.990) | 2,102 (0.989) | 2,298 (0.988) | 5,014 (0.988) | 2,883 (0.990) | 4,904 (0.994) |
| ~0.996 | 1,639 (0.997) | 2,011 (0.996) | 2,796 (0.998) | 3,849 (0.996) | 1,357 (0.997) | 1,642 (0.995) | 2,955 (0.996) | 1,716 (0.997) | 3,936 (0.997) |
| ~0.999 | 922 (1.000) | 1,323 (0.999) | 1,425 (1.000) | 2,424 (0.999) | 749 (0.999) | 1,073 (0.999) | 1,689 (0.999) | 1,073 (0.999) | 2,353 (0.999) |
| build | 209 s | 266 s | 132 s | 764 s* | 252 s | 311 s | 122 s | 118 s | 206 s |

#### glove-100-angular

1,183,514 GloVe word vectors, dim 100, angular ground truth, their 10,000
queries. The sanadb adapter maps `angular` to `Metric::Cosine`. This dataset
is hard for every graph index; hnswlib reaches 0.98 only at ef = 800.

| recall@10 | hnswlib M=16 | hnswlib M=24 | faiss HNSW M=16 | faiss HNSW M=24 | usearch M=16 | usearch M=24 | sanadb R=32 PQ | sanadb R=32 no PQ | sanadb R=48 PQ |
|---|---|---|---|---|---|---|---|---|---|
| ~0.73 | 5,174 (0.728) |  | 9,619 (0.732) |  | 3,697 (0.729) |  |  | 5,186 (0.731) |  |
| ~0.81 | 2,979 (0.811) |  | 5,505 (0.816) |  | 2,237 (0.811) |  |  | 2,986 (0.806) |  |
| ~0.86 |  | 2,266 (0.853) | 3,873 (0.852) | 4,084 (0.859) |  | 1,667 (0.853) | 2,957 (0.858) | 1,762 (0.866) | 4,202 (0.860) |
| ~0.89 | 1,401 (0.889) | 1,610 (0.887) | 2,435 (0.889) | 2,854 (0.893) | 1,024 (0.887) | 1,187 (0.887) | 2,113 (0.885) | 1,286 (0.891) | 3,110 (0.890) |
| ~0.93 | 787 (0.933) | 1,047 (0.922) | 1,241 (0.930) | 1,807 (0.928) | 553 (0.932) | 764 (0.923) |  |  | 1,717 (0.929) |
| ~0.95 | 553 (0.952) | 577 (0.960) | 810 (0.948) |  | 380 (0.951) | 410 (0.959) |  |  | 1,355 (0.941) |
| build | 319 s | 434 s | 967 s* | 257 s | 388 s | 555 s | 293 s | 298 s | 576 s |

What the tables say:

- faiss HNSW is the fastest of the four here, about 2x hnswlib at equal
  recall on both datasets. Part of that is SIMD: faiss ships NEON kernels
  for Apple Silicon, while hnswlib's distance code is SSE/AVX intrinsics
  with a scalar fallback. It is not the whole story: sanadb's plain-Rust
  distance loop already compiles to NEON here (18 ns per 128-dim distance,
  hand-written intrinsics were no faster), and its gap to faiss turned out to
  be walk bookkeeping, fixed in 0.1.4. On an x86 server the gap between faiss
  and hnswlib is small in ann-benchmarks' published plots; expect these
  ratios to shift there.
- sanadb is now ahead of hnswlib and usearch at every recall on both
  datasets, and trades places with faiss: on SIFT faiss leads below recall
  0.99 and sanadb R=48 leads above it; on GloVe sanadb R=48 leads from
  recall 0.86 up (at 0.93, 1,717 against faiss's 1,807 is the one anchor
  faiss keeps). It does this while holding 32 bytes of PQ codes per vector
  in RAM instead of the whole index; the other three keep every vector and
  every edge in memory. PQ costs about 0.005 recall per beam on these
  100-dim vectors (25 subspaces of 4) and is worth it for throughput. The
  sanadb columns were measured after the 0.1.4 walk rewrite; the other
  columns are from the same day on the same machine.
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

### Filtered search, SIFT1M with labels

`examples/filtered.rs`: every record gets four string labels with 10, 100,
1,000 and 10,000 distinct values, so an equality filter on each passes 10%,
1%, 0.1% and 0.01% of the set. 500 published queries, k=10, beam 64, warm;
ground truth is brute force over the records that pass. The column path is
`search_where` with a `Filter`; the closure path is `search_filtered`, which
cannot look inside the filter and widens the beam until `k` pass.

| selectivity | regime | recall@10 | mean | p99 | closure recall@10 | closure mean |
|---|---|---|---|---|---|---|
| 10% | filtered walk | 0.995 | 1.04 ms | 1.8 ms | 0.792 | 0.42 ms |
| 1% | exact scan, 10K matches | 1.000 | 1.34 ms | 1.6 ms | 0.900 | 3.96 ms |
| 0.1% | exact scan | 1.000 | 110 µs | 161 µs | 0.399 | 10.4 ms |
| 0.01% | exact scan | 1.000 | 9 µs | 14 µs | 0.044 | 10.5 ms |

The rare-filter rows are the point: the old path returned 4 of 10 right
answers at 0.1% and almost none at 0.01%, after 10 ms of widened walks; the
posting lists make those the cheapest queries in the table. The scan beats
the walk up to about 20,000 matches warm, which is where the default
`exact_filter_scan` sits; at 10% the scan would take 13 ms and the walk
wins. The 10% walk is 7x an unfiltered query; closing that is on the list.

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

Checked again for 0.1.4 with the walk rewrite, old and new binary each
after a `purge`, on a busier machine (Docker running): cold mean 3.23 ms
before and 3.20 ms after at width 4, warm 258 µs and 227 µs. The rewrite
does not touch the cold path; the gap to the 2.75 ms above is the machine.

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

1. Close the remaining gap to faiss HNSW: the heaps and the PQ table build
   are the next items in the profile. AVX2 kernels for x86, where the
   autovectoriser does less without `target-cpu`. Close the R=32 gap on
   cosine data. A VectorDBBench client.
2. Readers in other processes and metadata off the heap:
   [design note](docs/design/snapshot-and-tail.md). Shared lock for
   readers, exclusive for the one writer; a `Db` holds only the log tail in
   RAM.
3. Filtered search, next steps from the
   [design note](docs/design/typed-metadata-and-filtered-search.md): the
   metadata map still lives in RAM beside the columns; measure on the
   big-ann-benchmarks filter track (YFCC-10M); two-hop expansion through
   rejected nodes if the filtered walk's recall needs it.
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
