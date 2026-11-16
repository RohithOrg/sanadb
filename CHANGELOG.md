# Changelog

All notable changes to sanadb. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and versions follow [Semantic Versioning](https://semver.org/). Before 1.0,
minor versions may change on-disk formats; when they do, this file says so and
the new version migrates old stores on open.

## [Unreleased]

Towards 0.2.0: the store holds nothing in RAM that scales with the index.
See `docs/design/snapshot-and-tail.md`.

### Changed
- Metadata of indexed records is read from the index's attribute columns;
  the in-RAM map holds only records written since the last checkpoint (and,
  for an index built before 0.1.4, everything until the next checkpoint).
  The compacted log no longer carries Meta frames, so after a checkpoint it
  is 16 bytes. Log reader level 3: 0.1.4 refuses such a log with
  `NeedsNewerVersion`; level-2 logs open unchanged, level 1 and SANADB01
  are migrated as before.
- The index file ends in a footer that names the attribute table and a new
  sorted id table; `slot(id)` is a binary search on the mapping and the
  per-record id vector and hash map are gone. 0.1.4 indexes (trailer, no id
  table) still open and build the map until the next checkpoint.
- Measured on the labelled SIFT1M store (1M records, four string attributes),
  `examples/opentime.rs`, after one checkpoint in the new layout:

  | | 0.1.4 | now |
  |---|---|---|
  | open | 949 ms | 4.8 ms |
  | resident memory after open | 864 MB | 63 MB |
  | log size after checkpoint | 66 MB | 16 B |
  | 1000 warm queries | 85 ms | 75 ms |

  The 63 MB is the PQ codes (32 MB) plus whatever pages the queries touch.

## [0.1.4] - 2026-11-16

This release changes the Rust `Metadata` type and the log format (reader
level 2); stores from 0.1.1 to 0.1.3 are converted on open, see Upgrading.

### Added
- Typed metadata. `Metadata` is `HashMap<String, Value>` with
  `Value::{Str, Int, Float, Bool}`. The first value written under a key fixes
  that key's type for the store; writing another type is
  `Error::TypeMismatch`. Python maps `str`, `int`, `float`, `bool` both ways
  (`bool` before `int`) and rejects other value types with `TypeError`.
- `Filter`, a filter expression (`Eq`, `Ne`, `In`, `Lt`, `Le`, `Gt`, `Ge`,
  `Prefix`, `Exists`, `And`, `Or`, `Not`), and `Db::search_where` /
  `search_many_where` that take one. In Python the `filter=` dict now
  accepts operators: `{"year": {"$gte": 2020}, "lang": {"$in": ["en",
  "de"]}, "$or": [...], "$not": {...}}`; a plain `{"k": v}` dict still means
  equality. Callables keep working on the old post-filter path.
- Attribute columns in the index file. At checkpoint every metadata key
  becomes a slot-ordered column (strings dictionary-coded) with a presence
  bitmap and, for keys with at most 65,536 distinct values, posting lists.
  `search_where` evaluates its filter on these columns with an array read per
  visited node, no hashing. The section is additive: 0.1.3 opens such an
  index and ignores it. An index without the section (built before this
  release) still works; `search_where` then evaluates filters from the
  metadata map and does not use the scan regime until the next checkpoint.
- Two filtered-search regimes chosen per query. When the posting lists put
  the match count at or below `Options::exact_filter_scan` (default 16,384),
  the matching records are read and ranked exactly: recall 1.0 and never a
  short result for a rare filter. Otherwise the graph is walked with the
  filter steering it: only passing nodes take result slots while rejected
  nodes are still expanded for navigation, so a 1% filter no longer needs
  the beam-doubling retries of the post-filter path. Measured on SIFT1M with
  synthetic labels (`examples/filtered.rs`, 500 queries, beam 64, warm),
  column path against the old post-filter closure path:

  | selectivity | regime | recall@10 | mean | old recall@10 | old mean |
  |---|---|---|---|---|---|
  | 10% | walk | 0.995 | 1.04 ms | 0.792 | 0.42 ms |
  | 1% | scan (10K matches) | 1.000 | 1.34 ms | 0.900 | 3.96 ms |
  | 0.1% | scan | 1.000 | 110 µs | 0.399 | 10.4 ms |
  | 0.01% | scan | 1.000 | 9 µs | 0.044 | 10.5 ms |

  The scan is faster than the filtered walk up to about 20,000 matches when
  the index is warm, which is where the default sits; at 10% (100K matches)
  the scan takes 13 ms and the walk wins. The filtered walk at 10% costs 7x
  an unfiltered query for 0.995 recall; its expansions are capped at 64 times
  the beam.
- `Db::search_many` and `search_many_filtered`: k-NN for a batch of queries
  (row-major slice) spread over all cores with rayon. In Python,
  `Db.search_many(queries, k, filter=None)` takes a 2-D float32 array and
  returns `(ids, scores)` as `(n, k)` arrays, padded with `-1` and `nan`,
  with the GIL released.

### Changed
- Warm queries are 27 to 30% faster on SIFT1M (beam 64: 202 µs to 146 µs,
  beam 256: 572 µs to 405 µs; 10,000 queries, old and new binary back to
  back on the same index, same recall), from profiling the walk rather than
  from SIMD: the distance loop already compiled to NEON. Three changes, each
  measured with the macOS sampler on a warm index:
  - The candidate list is two heaps (unexpanded candidates, and the `beam`
    best seen) instead of a sorted `Vec`; its shifting inserts were 18% of a
    query.
  - The visited set is a bitmap over slots with a touched-word list for
    clearing (n/8 bytes per thread) instead of a hash set; its inserts were
    14%.
  - Not changed, after measuring: the `madvise` prefetch calls are 15% of a
    warm query and a probe gate that skipped them when recent blocks read as
    resident was tried and dropped, since it cost 45% on a cold query in a
    purge-then-query A/B on the 5M index. `io_width(1)` remains the switch
    for stores known to be in RAM.
  - The PQ lookup sum uses four accumulators and the per-query table is
    built from a dimension-major copy of the codebook, so both vectorise.
- Unfiltered searches no longer look up each visited node's metadata (6%).
- `rustc-hash` is no longer a dependency.

### Upgrading from 0.1.1 to 0.1.3
- Rust: `Metadata` values are `Value`, not `String`. `"x".into()` still
  builds a value, `m["k"] == "x"` still compares; code that called string
  methods on a value uses `m["k"].as_str()`.
- Logs are converted the first time a store is opened (temp file and
  rename, as the 0.1.1 conversion was); every existing value becomes
  `Value::Str`, nothing is parsed from its spelling, so a key that held
  `"2024"` stays a string key. The index file is unchanged; the next
  checkpoint adds the attribute columns. No downgrade.
- Python: existing code is unchanged; dict filters with plain values behave
  as before, except that a value of the wrong type for a key matches nothing
  instead of comparing its string form.

### Known issues
- Under heavy CPU load, with the whole test suite running in parallel in one
  process, about one run in 25 saw `Error::Locked` when reopening a store
  that the same thread had just dropped, and once a record written before a
  checkpoint was missing after the following reopen. Neither reproduces with
  the test alone, with eight threads hammering `flock` (340,000 cycles, no
  false conflict), or with `lsof` run at the moment of failure (no holder).
  `Db::open` now retries a held lock for up to 200 ms before failing. The
  tests carry diagnostics that will name the holder and the lost frame if it
  happens again; if you see either outside the test suite, please open an
  issue with the output.

### Fixed
- A record deleted while a checkpoint was building now gets a Delete frame
  in the compacted log, so reopening the store replays its tombstone.
  Before, the stale graph node was hidden only by the per-node metadata
  lookup during search, and `index_stats().tombstones` under-counted after
  a reopen.

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

[Unreleased]: https://github.com/RohithOrg/sanadb/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/RohithOrg/sanadb/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/RohithOrg/sanadb/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/RohithOrg/sanadb/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/RohithOrg/sanadb/releases/tag/v0.1.1
