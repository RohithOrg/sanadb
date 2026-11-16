# Changelog

All notable changes to sanadb. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and versions follow [Semantic Versioning](https://semver.org/). Before 1.0,
minor versions may change on-disk formats; when they do, this file says so and
the new version migrates old stores on open.

## [Unreleased]

## [0.1.3] - 2026-11-16

### Added
- Incremental checkpoint. When a graph exists and the pending change (new
  writes plus deletes) is smaller than the live graph, and the result fits
  `shard_size`, the graph is updated in place: deleted nodes are consolidated
  out (neighbours inherit their edges and re-prune, as in FreshDiskANN) and
  new vectors are inserted. Cost follows the change, not the store. Larger
  changes still rebuild from scratch.
- `Options::auto_checkpoint_delta` (default 250,000): checkpoint when at
  least that many records sit in the delta and the delta is at least as large
  as the indexed graph, so a bulk load costs a geometric series of builds
  rather than one per N records.
- Crash-injection test (a child process is SIGKILLed mid-write and the store
  reopened) and a corruption fuzz test over the log decoder, both in the
  regular suite.
- Rust tests run on Windows in CI.
- ann-benchmarks comparison extended to faiss HNSW and usearch (README
  "ann-benchmarks"); the usearch adapter lives in `benchmarks/ann-benchmarks/`.

### Fixed
- Cosine recall. The cosine metric used negative dot product as its internal
  distance, and RobustPrune's alpha test (`alpha * d(best, s) > d(p, s)`)
  inverts on negative values, so the build pruned far too aggressively.
  Found by ann-benchmarks on glove-100-angular: recall@10 0.24 at beam 64
  against hnswlib's 0.81. Cosine now stores unit vectors and uses squared L2
  internally (same ranking), converting scores back to negative cosine at the
  API. All-zero vectors are rejected under cosine, since their cosine is
  undefined. Stores built with cosine before this fix should be re-checkpointed
  (`checkpoint()` after any write, or delete the `.idx` file and reopen) to
  rebuild the graph; their data is intact.

### Changed
- Build is about a third faster (SIFT1M: 115 s to 77 s on 10 cores) with
  slightly better recall per beam and 5 to 15% lower query latency:
  adjacency lists may grow to `GraphParams::prune_slack` (default 1.3) times
  R before being pruned, as DiskANN does, which on hub-heavy data such as word
  embeddings removed the dominant cost; the walk's visited set uses FxHash
  instead of SipHash; the walk reuses thread-local scratch buffers.
- PQ codebooks and shard partitions are trained with e-means (Meta FAIR's
  mini-batch k-means recipe, with its small-dimension batch law) seeded by
  k-means++, instead of 8 to 12 rounds of Lloyd. A test keeps its distortion
  within 5% of a 20-iteration Lloyd baseline on a PQ-sized problem.
- Re-measured the 5M SIFT1B-prefix sharded build (README "Build memory"):
  final checkpoint 1396 s to 856 s, peak anonymous memory 3.58 GB now that
  the delta reads vectors from the mapped log. Max RSS reads 5.13 GB because
  it counts resident pages of the mapped snapshot and index.
- Locks are `parking_lot`: a panic on one thread no longer poisons the store
  for the others. `Error::Poisoned` is gone.

## [0.1.2] - 2026-11-15

First version with Python wheels on PyPI: the 0.1.1 release published the
crate but its macOS Intel wheel job sat on a retired runner label and PyPI
never ran. Intel macOS wheels now build on `macos-15-intel`.

### Changed
- Log format is now `SANADB02`: frames are padded to 4 bytes so vectors sit
  at aligned offsets. The delta index reads vectors from the memory-mapped log
  instead of holding copies in RAM, so bulk-loading N vectors no longer needs
  N vectors of memory before the first checkpoint.

### Added
- Forward-compatibility rules, modelled on SQLite's read/write versions. Both
  file headers carry a reader level (currently 1). A build opens any file
  whose level it supports and skips record kinds it does not know, so a future
  version can add features without locking older readers out. A file whose
  level is higher than the build supports fails with
  `Error::NeedsNewerVersion` instead of being misread.

### Upgrading from 0.1.1
- Stores written by 0.1.1 (`SANADB01`) are converted automatically the first
  time they are opened. The conversion rewrites the log through a temp file
  and rename; a crash during it leaves the old file intact. The index sidecar
  is unchanged. There is no downgrade path: a converted store will not open
  with 0.1.1.

## [0.1.1] - 2026-11-15

First published release, on crates.io and PyPI. 0.1.0 was tagged but never
published.

### Added
- Write-ahead log with CRC-framed records, torn-tail truncation, `sync()` and
  `sync_every_write`.
- Memory-mapped Vamana (DiskANN-style) graph index built at checkpoint, with
  product-quantized codes in RAM for the walk and exact re-ranking from disk.
- Parallel graph build; sharded build above `shard_size` (default 1M) over a
  memory-mapped snapshot.
- Beam-width expansion with `madvise(WILLNEED)` prefetch on Unix
  (`Options::io_width`, default 4).
- Exact in-RAM delta for records written since the last checkpoint, with
  tombstones for deleted or overwritten graph records.
- Non-blocking checkpoint: readers and writers continue during the build.
- Metadata filters with adaptive beam widening for selective filters.
- Metrics: L2, dot product, cosine (normalises on insert and query).
- Rejection of NaN and infinite vector components.
- Thread-safe `Db` (`Send + Sync`), exclusive file lock per open store.
- Python bindings (`pip install sanadb`) with numpy input, dict or callable
  filters, and the GIL released during work.
- Examples reproducing the SIFT1M and SIFT1B-prefix measurements in the README.

### Known limits
- One process per store. No replication, encryption, typed metadata or range
  filters. Formats may change before 1.0 (with migration).

[Unreleased]: https://github.com/RohithOrg/sanadb/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/RohithOrg/sanadb/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/RohithOrg/sanadb/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/RohithOrg/sanadb/releases/tag/v0.1.1
