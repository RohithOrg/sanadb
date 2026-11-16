# Changelog

All notable changes to sanadb. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and versions follow [Semantic Versioning](https://semver.org/). Before 1.0,
minor versions may change on-disk formats; when they do, this file says so and
the new version migrates old stores on open.

## [Unreleased]

### Added
- `Db::neighbors(id)` (Python `Db.neighbors(id)`): the out-edges of a
  record's node in the graph index, for tooling that draws the graph.
- studio: click a point to expand its node: its graph edges are drawn
  to the records they reach, a popover shows the name, id and metadata
  with "Show its edges" and "Search from here", each neighbour expands
  in turn, a HUD counts nodes and edges with Clear. Works in 2D and 3D.
  `/api/neighbors?id=` returns the edges with projected positions.
- studio: a 3D map. The server computes a third principal component and
  the share of variance each axis carries; the page has a 2D / 3D switch,
  orbit by drag, zoom, shift-drag pan, a slow idle rotation that stops on
  touch and is off under reduced motion, and draws the walk, results,
  query, labels and hover in space. `/api/map` points gain a `z`,
  `/api/search` returns `query_xyz`, `/api/labels` a `z` per cluster.

## [0.2.4] - 2026-11-16

The release that reaches PyPI: 0.2.3 published to crates.io but its PyPI
upload failed (see Changed). Also the documentation split by language,
the docs refresh script, and the failure-modes page.

### Changed
- Release workflow: `sanadb` and `sanadb-studio` are uploaded to PyPI in
  two steps; one upload of both packages' files was refused in 0.2.3 (the
  trusted-publisher token is per project), so 0.2.3 never reached PyPI.
- Release workflow: the macOS arm64 wheel is cross-compiled on the Intel
  macOS runner; GitHub's arm64 macOS runners cancelled the job for lack
  of capacity twice during the 0.2.3 release.

### Added
- Python: `auto_checkpoint_delta=` and `auto_checkpoint_garbage=` on `Db`,
  the same policy options the Rust `Options` had.
- `examples/semantic_search.rs` (feature `embed-http`): semantic, filtered
  and hybrid search over text through an OpenAI-compatible embedding
  endpoint, with the same documents and output as the Python example.
- `docs/failure-modes.md` and `python/examples/failure_modes.py`: the lock
  held, a torn log, a file from a newer version, a damaged index and a
  wrong dimension, each produced on a real store with the library's output.
- `scripts/refresh-docs.py`: reruns every example program and splices its
  output into the pages that quote it, regenerates the Python API page,
  and retakes the studio screenshots with Playwright; `--check` fails if
  a page is out of date. The screenshots were retaken with it.
- Documentation split by language: `docs/python/` and `docs/rust/` each
  hold getting started, API, examples and recipes (RAG indexing,
  checkpoint policy, processes, backup, failure modes; moving from Chroma
  in Python). PyPI and crates.io now show language-specific READMEs
  (`python/README.md`, `docs/rust/README.md`); `scripts/pyi_to_md.py`
  generates the Python API page from the shipped stubs.
- `benchmarks/beir/scifact_dense.py`: end-to-end check with a real
  embedding model (all-MiniLM-L6-v2 on BEIR SciFact). Brute force over
  the model's vectors reproduces the published nDCG@10 of 0.645; sanadb's
  graph at the default beam returns 99.6% of brute force's top 10 and the
  same 0.645. Documented in `docs/python/examples.md` and `docs/benchmarks.md`.
- `python/examples/semantic_search.py` and `python/examples/embed_server.py`:
  a sentence-transformers walkthrough with its output, and the model as a
  local OpenAI-compatible endpoint for studio.

## [0.2.3] - 2026-11-16

### Added
- `pip install "sanadb[studio]"` installs the studio binary as a console
  script: a second PyPI package, `sanadb-studio`, holds only the
  executable (built by maturin from the binary crate, one wheel per
  platform), and the `studio` extra of `sanadb` pins it to the same
  version. `pip install sanadb` alone stays the library only.

## [0.2.2] - 2026-11-16

Store properties, and the studio polish that followed 0.2.1. No format
level change: the new frame kind is skipped by older readers.

### Added
- Store properties: `Db::set_property(key, value)`, `property(key)`,
  `properties()` (Python the same) keep strings with the file, written as
  log frame kind 7, which older readers skip; visible to readers at once,
  rewritten into the compacted log at every checkpoint. Studio records
  `embedding_model` and `embedding_dim` the first time an attached model
  writes a record, shows the recorded model when none is attached, and
  warns when the attached model differs from the recorded one.
- Studio: the walk's head travels along each segment at a pace
  proportional to the jump, with a continuous scrubber; an "Instant"
  speed draws the whole walk at once and is remembered; a record's vector
  can be replaced from the page, from new text (re-embedded) or pasted
  numbers, under the same id.

### Fixed
- Release workflow: the PyPI step took the studio binaries along with the
  wheels and failed; 0.2.1 therefore never reached PyPI. It downloads only
  wheels and the sdist now.

## [0.2.1] - 2026-11-16

sanadb studio, bring-your-own embedding model in Rust, Python and studio, and the small public API they needed. No format change. Published to crates.io; the PyPI upload failed (see 0.2.2).

### Added
- `sanadb-studio` (`cargo install sanadb-studio`, or a prebuilt binary
  from the GitHub release): a local web page for one store with a PCA map
  of a sample, every search mode, the graph walk drawn on the map, record
  inspection,
  metadata editing and deletion, sparse-vector queries, cluster labels
  from a metadata key, a files view, live counters, and light and dark
  chrome. See `docs/studio.md`.
- `Db::peek(path)` reads a store's dimension and metric from the file;
  `Db::ids()` lists live ids; `Db::keys()` lists metadata keys with their
  types; `Db::search_traced(query, k)` returns the hits and the nodes the
  graph walk expanded, in order. The index reader also accepts the
  32-byte footer that 0.2.0 development builds wrote, so stores built with
  them still open.
- Bring your own embedding model. Rust: the `Embedder` trait (implemented
  by any closure `Fn(&[&str]) -> Result<Vec<Vec<f32>>>`), `Db::add_text`
  and `Db::search_text_embedded`, and `OpenAiEmbedder` for any
  OpenAI-compatible `/v1/embeddings` endpoint behind the `embed-http`
  feature. Python: `Db(..., embedding_function=fn)` with `add(documents,
  ids, metas)` and `query(texts, k, filter)`. Studio: `--embed-url` with
  `--embed-model` (and `--embed-key-env`), or `--embed-cmd`; a "Text to
  embed" query box and an Add record form.
- Studio: a record's headline from a name key (`title`, `name`, `doc`,
  `path`, `file`, `url`, `label`, `text`, `body`) with the id labelled
  beneath; the score's meaning stated per metric; counts as 1M / 4K with
  exact values on hover; cluster labels drawn only where one value holds
  at least 20% of its cluster; monochrome instrument chrome with the
  accent reserved for data.

## [0.2.0] - 2026-11-16

The store holds nothing in RAM that scales with the index, readers can
open it from other processes, and writes can be batched atomically. See
`docs/design/snapshot-and-tail.md`. The log format moves to reader level 3,
so this is the first release 0.1.4 cannot read; see Upgrading below.

### Changed
- Metadata of indexed records is read from the index's attribute columns;
  the in-RAM map holds only records written since the last checkpoint (and,
  for an index built before 0.1.4, everything until the next checkpoint).
  The compacted log no longer carries Meta frames, so after a checkpoint it
  is 36 bytes: the header and the frame naming its index. Log reader level 3: 0.1.4 refuses such a log with
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
  | log size after checkpoint | 66 MB | 36 B |
  | 1000 warm queries | 85 ms | 75 ms |

  The 63 MB is the PQ codes (32 MB) plus whatever pages the queries touch.

### Upgrading from 0.1.x
- Open the store once with 0.2.0: a 0.1.1 or 0.1.2/0.1.3 log is rewritten
  as before, a 0.1.4 log is read as-is. The first checkpoint writes the
  index under its new name (`<path>.idx.<id>`), moves metadata into its
  columns and compacts the log to reader level 3. Until that checkpoint the
  store still opens with 0.1.4; after it, 0.1.4 refuses the log with
  `NeedsNewerVersion`. No downgrade.
- Rust API: `Metadata` and `Filter` are unchanged from 0.1.4. New:
  `Db::open_read_only`, `Db::refresh`, `Db::write_batch` with `Batch`,
  `Db::index_path`, `Options::storage` with `Storage`, `Error::ReadOnly`,
  `IndexStats::storage`. `GraphParams` gained a `storage` field, so code
  that builds it without `..Default::default()` adds one.
- Python: `Db(path, dim, read_only=True)`, `refresh()`, `delete_many`,
  `storage="int8"`; `insert_many` is now atomic.

### Added
- `Db::backup(dir)`: a consistent point-in-time copy of the store (index
  file, then the log's complete frames under the write lock) that `Db::open`
  or `open_read_only` accepts at `dir/<file name>`. Works from a reader too.
  Python: `backup(dir)`.
- `Db::open_read_only`: open a store beside its writer, in the same or
  another process, with no lock. Reads see a prefix of the writer's history
  and refresh before each call (`Db::refresh` forces it); writes return
  `Error::ReadOnly`. Python: `sanadb.Db(path, dim, read_only=True)`,
  `refresh()`, writes raise `PermissionError`. A test spawns a reader process
  that queries through 1,500 ms of inserts, deletes and checkpoints.
- Each checkpoint writes its index to a new file, `<path>.idx.<id>`, and the
  compacted log's first frame names it (frame kind 4). Nothing is renamed
  over an index any more, which is what made checkpoints fail on Windows
  while a file was mapped. The old index is deleted after the swap; if a
  reader still maps it (Windows refuses the delete) the next writer open
  collects it. `Db::index_path()` returns the current file. Stores whose
  index is still the unnumbered `<path>.idx` open as before.
- On x86-64, squared L2 and dot product use AVX2/FMA kernels when the CPU
  has them (detected once at runtime; the default x86-64 target is SSE2
  without FMA). A test checks them against the scalar reference on every
  length on the Linux CI runner. aarch64 keeps the autovectorised loop,
  which already compiles to NEON FMA. Query latency is memory-bound and is
  not expected to move; builds, exact scans and re-ranking are.
- Filtered walk at 10% selectivity measured across beams (README): beam 32
  gives recall 0.975 at 0.59 ms against 0.995 at 1.1 ms for beam 64.
- `Storage::I8` (`Options::storage`, Python `storage="int8"`): indexed vectors
  kept as 8-bit codes with a per-record min and scale, `dim + 8` bytes
  instead of `dim * 4`, so each hop of a cold query reads a quarter of the
  vector bytes. The walk is unchanged (PQ codes); exact re-ranking and
  `get` dequantise. The log stays `f32`, so recent writes are exact and the
  value stored is only rounded once (requantising a dequantised vector gives
  the same codes). Header byte 14 of the index names the storage; a 0.1.x
  reader rejects an int8 index as inconsistent geometry rather than
  misreading it. `IndexStats::storage` reports it. SIFT1M, PQ on, 10,000
  queries, same machine, f32 index against int8 index built from the same
  data:

  | | f32 | int8 |
  |---|---|---|
  | index file | 744 MB | 372 MB |
  | beam 64: recall@10, mean | 0.966, 161 µs | 0.960, 159 µs |
  | beam 256 | 0.995, 411 µs | 0.988, 460 µs |

  The vector is 128 of the 644 bytes an f32 block holds (the rest is the
  adjacency list), so the file halves rather than quarters; the recall cost
  is under one point at every beam.
- `Db::write_batch(Batch)`: inserts and deletes applied atomically. The log
  brackets them in `Begin`/`Commit` frames (kinds 5 and 6); replay applies a
  batch only on its `Commit`, a writer opening a log that ends mid-batch cuts
  the batch off, and a reader never sees part of one. Everything is validated
  before anything is written, including type consistency within the batch.
  Python `insert_many` is now one batch, and `delete_many(ids)` is new.
- Tests: the index file is now fuzzed like the log (400 random bit flips
  and truncations per run; every open is `Ok` or a corruption error, and an
  accepted index answers reads without panicking). Found and fixed: a
  neighbour slot past the end, an id-table slot past the end, and dictionary
  or posting offsets past the file all read out of bounds. Stores written by
  the published 0.1.1, 0.1.3 and 0.1.4 crates live in `tests/fixtures/`;
  each is opened, read back, written to, checkpointed into the current
  layout and reopened, by the writer and by a read-only `Db`.
- Keyword and hybrid search. `Options::text_field(key)` names a metadata
  key whose strings are indexed at checkpoint (lowercase alphanumeric
  tokens, BM25 with k1 = 1.2, b = 0.75). `Db::search_text(query, k, filter)`
  scores indexed records from the inverted index and records written since
  the last checkpoint from their metadata, with one set of corpus
  statistics; scores are negated BM25. `Db::search_hybrid(vector, text, k,
  filter)` fuses the dense and keyword lists by reciprocal rank (`1 / (60 +
  rank)`, each list fetched at `3k`). Python: `text_field=`, `search_text()`,
  `search(query, text=...)`. The text section lives in the index file after
  the id table, named by the footer's third slot; a 0.2.0 reader ignores it.
  On BEIR SciFact (5,183 documents, 300 test queries,
  `benchmarks/beir/scifact_bm25.py`), against the `rank_bm25` reference with
  the same tokens:

  | | nDCG@10 | Recall@100 | ms/query |
  |---|---|---|---|
  | sanadb `search_text` | 0.662 | 0.886 | 1.2 |
  | rank_bm25 (Okapi) | 0.654 | 0.876 | 30.6 |

  Published Anserini BM25 on SciFact is 0.665 nDCG@10; the small
  differences are the idf variant.
- `Filter::Contains(key, token)`: the key's string, tokenised like the text
  index, contains the token. On the text field it runs on the inverted index
  (membership by binary search over the term's postings, the postings as
  scan candidates); on another `Str` column the value is decoded and
  tokenised. Python: `{"tags": {"$contains": "t12"}}`. This is how tag sets
  are filtered: on the big-ann-benchmarks YFCC filter track at 1M records
  (`examples/yfcc.rs`) every selectivity bucket gets recall 1.0 (0.994 above
  10,000 matches) where post-filtering got 0.17 to 0.83; table in README.
- Learned sparse vectors. `Value::Sparse(Vec<(u32, f32)>)` is a metadata
  value (Python: an `(indices, weights)` pair of arrays); pairs are sorted,
  duplicates summed and zero weights dropped on insert, non-finite weights
  are `Error::InvalidVector`. One key per store may hold them
  (`Error::SecondSparseKey` otherwise); at checkpoint it is written to a
  sparse section of the index file (forward lists for `get`, inverted
  lists for search; log value kind 4, footer slot 4, see `docs/FORMAT.md`).
  `Db::search_sparse(query, k, filter)` scores by dot product over the
  inverted lists plus the tail; `Db::search_hybrid_sparse(vector, sparse,
  k, filter)` fuses it with the dense result by reciprocal rank. Python:
  `search_sparse()`, `search(query, sparse=...)`. No public-dataset quality
  number yet: that needs a SPLADE encoding of a BEIR set, which is on the
  list. MaxScore for long BM25 posting lists is not implemented; SciFact
  queries take 1.2 ms term-at-a-time and no measured workload needed it.
- `Db::stats()` (Python `Db.stats()`): operation counters since open:
  searches by regime (graph walk, filtered walk, exact scan, delta only),
  text searches, inserts, deletes, batches, checkpoints (total,
  incremental, failed, last duration) and read-only refreshes (appended,
  reloaded). Relaxed atomics; polling them costs nothing measurable.
- Cargo feature `tracing`: `tracing` events on target `sanadb` for
  checkpoints (incremental, duration, records), failed checkpoints, reader
  reloads and orphan index files removed at open. Off by default; the crate
  has no new mandatory dependency.
- Failure injection in the test suite (`sanadb::fault`, hidden from docs):
  a checkpoint whose index write fails part way, whose index fsync fails or
  whose log rename fails, and a failed log append, each leave the store
  serving everything acknowledged, with the old index current and no
  partial files; the next checkpoint succeeds and a reopen agrees.

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

[Unreleased]: https://github.com/RohithOrg/sanadb/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/RohithOrg/sanadb/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/RohithOrg/sanadb/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/RohithOrg/sanadb/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/RohithOrg/sanadb/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/RohithOrg/sanadb/releases/tag/v0.1.1
