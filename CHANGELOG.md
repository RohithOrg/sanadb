# Changelog

All notable changes to sanadb. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and versions follow [Semantic Versioning](https://semver.org/). Before 1.0,
minor versions may change on-disk formats; when they do, this file says so and
the new version migrates old stores on open.

## [Unreleased]

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

[Unreleased]: https://github.com/RohithOrg/sanadb/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/RohithOrg/sanadb/releases/tag/v0.1.1
