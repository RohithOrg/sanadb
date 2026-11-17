# 104 questions, answered

A reviewer asked these questions of sanadb as a candidate for production
use. Each gets one of three answers. **Yes**: proven, with where the
proof lives. **Partly**: true in part, with what is missing. **No**: a
gap. The gaps feed the [roadmap](roadmap.md). Answers are as of 0.2.6,
17 November 2026; the test names are in `tests/db.rs`.

Tally: 37 yes, 50 partly, 17 no.

## 1. Purpose and positioning

1. **Who is the primary user.** Partly. The README says what it is and the goal ("the SQLite of vector search"); the audience, people embedding search in an application, an agent or a service without running a vector server, is implied, not stated. One sentence to add.
2. **What it solves better.** Yes. A single file, crash-safe by construction, with a disk-first graph index for sets larger than RAM, filters evaluated on index columns rather than after the fact, and keyword and sparse search in the same file. README, "What it is".
3. **Why over SQLite with sqlite-vec.** No, not written. The answer: sqlite-vec scans every vector per query, which is fine to about 100,000 rows and not beyond; sanadb's graph answers a million in 150 µs and five million from a cold disk in 3 ms, with metadata filters that stay exact. sqlite-vec wins when the data is small and already in SQLite. A comparison page is on the roadmap.
4. **Why over LanceDB.** Partly. The README names LanceDB as the closest embedded alternative and says it is the one to pick when dependability this quarter matters. No feature or number comparison.
5. **When not to use it.** Yes. README, "Should you use it" and "What sanadb does not do": no second writer, no replication, no encryption, no GPU, network filesystems, and anything that cannot afford a 0.x bug.
6. **What "SQLite-like" means.** Partly. Embedded (a library, one file per store), reliable by design (log, atomic checkpoint, fault injection), portable (little-endian format, opens on Linux, macOS, Windows) are each documented; installation is a `pip install` or `cargo add`. The four are not spelled out together.
7. **Understood in five minutes.** Partly. The README's quick start, feature table, numbers and guarantees are written for that; it has not been tested on someone who did not write it.

## 2. Data correctness

8. **Acknowledged write readable after a process crash.** Yes. `survives_being_killed_mid_write`, `crash_child_writes_until_killed`; [operations](operations.md), Durability.
9. **Insert, overwrite, delete identical before and after checkpoint and reopen.** Yes. `graph_survives_reopen_upsert_delete_and_recheckpoint`, `random_operations_agree_with_a_model`, `incremental_checkpoint_matches_full_rebuild`.
10. **Same id twice in one batch.** Partly. Frames replay in order inside the batch, so the last one wins, but there is no test or doc line saying so.
11. **Deleted or overwritten records reappearing after recovery.** Yes, they do not: `reopen_replays_log`, the crash tests, the model test.
12. **Vector and metadata updated atomically.** Yes. One log frame carries both; a reader sees the frame or not.
13. **Empty store, empty batch, missing id, k larger than the count.** Partly. Empty store: `checkpoint_of_empty_store_is_fine`, search returns nothing. Missing id: `get` returns `None`, `delete` returns false. k larger than the count: fewer results. An empty batch writes nothing. Only the first is a test; the rest are behaviour without a test or a doc line.
14. **NaN, infinity, zero-norm, extreme values.** Partly. NaN and infinity are refused (`InvalidVector`), a zero vector under cosine too; `cosine_normalises_and_rejects_non_finite`. Very large finite values are not tested.
15. **Quantized storage and `get()`.** Partly. `get` returns the dequantised vector under int8, stated in the Rust docs and the Python docstring; not in the operations page.

## 3. Durability and recovery

16. **What a successful write guarantees by default.** Yes. Survives a process crash (in the page cache), not power loss. [Operations](operations.md), Durability.
17. **What `sync()` adds.** Yes. fsync of the log: survives power loss from there; `sync_every_write` does it per write.
18. **A new store's directory entry.** Partly. The checkpoint's renames fsync the directory on Unix; the first creation of the log file does not fsync the directory, so a new, never-checkpointed store can vanish on power loss before its first `sync()`. Not documented.
19. **Crash at every checkpoint stage.** Partly. Fault injection covers index write (disk full mid-build), index fsync, log rename and log append (`injected_io_failures_leave_the_store_consistent`); a process kill is tested mid-write, not at each checkpoint stage.
20. **Partial write.** Yes. A torn tail is trimmed on open: `torn_tail_is_truncated_not_fatal`, shown on [failure modes](failure-modes.md).
21. **fsync failing after bytes reached disk.** Partly. The index fsync fault is injected and the store stays on the old index; a failing `sync()` on the log returns the error and the data stays in the page cache. Not written down.
22. **Same handle after an I/O error.** Yes. The fault tests keep using the handle after a failed checkpoint, append and rename; `failed_checkpoints` counts them. The failure-modes page should say "keep the handle".
23. **Torn final record versus corruption mid-log.** Partly. Both stop replay at the first frame that fails its CRC; a mid-log corruption truncates everything after it, which the store cannot tell from a torn tail. Documented on the failure-modes page as "trimmed there"; the loss should be reported, see 24.
24. **Recovery silently discarding acknowledged data.** Partly. A torn tail holds at most the write that had not returned, and a write that returned but was not synced can be lost to power loss, both documented. Mid-log corruption discards acknowledged frames after it without a message; a counter or a `tracing` event is missing.
25. **Recovery and migration safe to interrupt repeatedly.** Partly. Recovery is idempotent (it only truncates and re-reads). Migration from 0.1 layouts converts on open; stores from every version are fixtures, but interrupting a migration is not tested.
26. **Disk full during insert, checkpoint, migration, backup.** Partly. Checkpoint and insert are covered by fault injection; migration and backup are not.

## 4. Concurrency and process behaviour

27. **Readers seeing a partial batch.** Yes, they cannot: `Begin` and `Commit` frames, `write_batch_is_all_or_nothing`, the reader tests.
28. **Snapshot or per-operation consistency.** Yes. A reader sees a prefix of the writer's history, refreshed at the start of each read; there is no multi-operation snapshot. [Operations](operations.md), Concurrency.
29. **Checkpoint losing a concurrent write.** Yes, it does not: `writes_and_reads_during_checkpoint_are_kept`.
30. **How quickly readers see writes.** Yes. At their next read: the refresh check is per read, 244 µs per search on a reader in the throughput table.
31. **Slow reader keeping an older index.** Yes. On Unix the mapping outlives the unlink; on Windows the file stays until the reader releases it and the next writer open removes orphans (`orphan_index_files_are_collected_on_open`).
32. **Writer crash with readers open.** Partly. Readers keep serving the files they have; the next writer replays and collects. The reader-survives tests cover writes and checkpoints, not a writer kill with readers attached.
33. **Writer exclusion across operating systems.** Yes. `flock` on Unix, a file lock on Windows, `second_open_is_refused_until_first_is_dropped` runs on all three in CI. Network filesystems excluded, documented.
34. **The intermittent lock failure.** No. Seen about once in 25 loaded test-suite runs, covered by a 200 ms retry, not reproduced on demand. On the roadmap.
35. **Thread, process and filesystem safety distinguished.** Yes. [Operations](operations.md), Concurrency and processes.

## 5. Search quality

36. **Recall of the defaults on realistic embeddings.** Yes. all-MiniLM-L6-v2 on SciFact: 0.996 recall@10 against brute force at the default beam, same nDCG as brute force; GloVe in ann-benchmarks. [Benchmarks](benchmarks.md).
37. **Recall by dimension, size, metric, distribution.** Partly. 128-dim L2 (SIFT), 100-dim angular (GloVe), 384-dim cosine (SciFact), 192-dim uint8 (YFCC), 1M and 5M. Nothing at 768 or above.
38. **Recall after overwrites, deletes, incremental checkpoints.** Partly. `incremental_checkpoint_matches_full_rebuild` asserts the same results; no published number over a long churn.
39. **Evaluated against exact nearest neighbours.** Yes. Every table uses published ground truth or brute force.
40. **Duplicates, ties, zero results.** Partly. Ties break by id; duplicates are ordinary records; a query that passes nothing returns an empty list. Not documented.
41. **PQ hurting particular data.** Partly. Measured on SIFT (0.001 to 0.005 recall) and cold 5M; not on GloVe or text embeddings.
42. **int8 loss including the worst queries.** Partly. "Within a point on SIFT"; no per-query distribution.
43. **Approximate versus exact guarantees separated.** Yes. The tail and rare filters are exact; the walk is approximate with its recall measured; stated in getting started and operations.

## 6. Filtering and hybrid retrieval

44. **Filter correctness before and after checkpoint and reopen.** Yes. `filtered_graph_search_respects_filter`, `search_where_uses_columns_scan_and_filtered_walk`, `metadata_lives_in_columns_after_checkpoint`.
45. **Missing metadata, mixed types, compound expressions.** Yes. A missing key fails the comparison, a second type for a key is refused at write (`TypeMismatch`), `And`, `Or`, `Not` are tested.
46. **Fewer than k with enough matches.** Partly. The column path cannot (exact scan returns all; the filtered walk widens, `selective_filter_fills_k_with_small_beam`); the closure path can stop at `max_search_beam`, documented.
47. **Correlated filters versus random labels.** Yes. YFCC's real label sets next to SIFT's synthetic ones, both in [benchmarks](benchmarks.md).
48. **Scan-or-walk threshold accounting for dimension and cold storage.** No. Tuned on 128-dim SIFT warm, said so in the benchmarks. Roadmap.
49. **Keyword statistics after updates and deletions.** Partly. Rebuilt at every checkpoint; between checkpoints the tail's documents are scored with the index's statistics. Not documented.
50. **Sparse results against an independent dot product.** Partly. `sparse_vectors_index_search_and_round_trip` checks the round trip and ordering on small data; no brute-force comparison at scale.
51. **Hybrid actually improving relevance.** No. One worked example (the exact token `GOGC`), no BEIR number for hybrid. Roadmap.
52. **Hybrid candidate counts and fusion settings across workloads.** No. Reciprocal rank with fixed candidates; weighted fusion is on the roadmap.

## 7. Latency and throughput

53. **p50, p95, p99 and maximum during sustained writes.** No. p99 is reported warm without concurrent writes; the soak records none. Roadmap item 1.
54. **How long a checkpoint holds the exclusive lock.** Partly. The build holds neither lock; the swap at the end takes the write lock for the rename and remap. Not measured.
55. **How long an insert blocks on an automatic checkpoint.** Yes, and the answer is unfavourable: the checkpoint runs inside the insert that triggers it, so that insert takes as long as a full build (75 s at a million SIFT vectors). Now documented; moving it off the write path is on the roadmap.
56. **Query latency against the tail.** Partly. The tail is scanned exactly, cost linear in its size; the auto-checkpoint defaults keep it to a fraction of the index. No curve published.
57. **Ingestion outpacing checkpointing.** No. Not measured.
58. **Deletion-heavy traffic and traversal.** No. Tombstones are skipped in the walk; the cost is not measured.
59. **Throughput with durability on.** No. The throughput table is with fsync off, said so.
60. **Concurrent query scaling.** Partly. `search_many` on all cores: 25,800 queries per second; reads share a lock; latency under concurrent single searches is not tabled.
61. **First open, recovery, first query, steady state reported separately.** Partly. Cold versus warm per query, `examples/opentime.rs` for open time; recovery time is not reported.

## 8. Memory and disk

62. **Total memory, not just PQ codes.** Partly. Peak RSS of a 5M build is 5.13 GB (benchmarks, Build memory); query-time RSS is not tabled.
63. **Anonymous memory, resident mapped pages, metadata, reader tails.** Partly. Anonymous memory of the build is recorded; the rest is not.
64. **Memory at 384, 768, 1536, 3072 dimensions.** No.
65. **Index substantially larger than RAM.** No. The cold test evicts a 3.5 GB index from a 16 GB machine; a set that does not fit in RAM at all has not been run.
66. **Peak memory and temporary disk during checkpoint.** Partly. Temporary disk is one new index file, about `n × block` bytes, documented; peak memory only for the 5M build.
67. **Free space to rebuild.** Partly. Old index plus new index plus the log, so about twice the index; implied, not stated.
68. **Metadata, postings or sparse vectors dominating.** No. Not measured.
69. **Reader processes duplicating memory.** Partly. The index is shared through the mapping; each reader holds its own tail and metadata map. Not measured.
70. **Hard memory limit or OOM kill.** No. An OOM kill is a crash and is covered by the crash tests only in that sense.

## 9. Benchmark credibility

71. **Reproducing the numbers.** Yes, in principle: public datasets, the example programs and the ann-benchmarks adapter with their exact commands; `scripts/refresh-docs.py` reruns the example outputs. Not yet reproduced by someone else.
72. **Hardware, versions, settings, dataset checksums.** Partly. Hardware and the sanadb version per table, default release profile; dataset checksums are not recorded.
73. **Competitors at equivalent recall and durability.** Yes. ann-benchmarks compares at equal recall; the page says the others are in-RAM indexes without a log.
74. **Build, ingest, checkpoint, query measured separately.** Yes. Throughput table and SIFT1M table.
75. **What "cold" means.** Yes. Evicted before the pass, then the same queries warm; stated with the table.
76. **Repeated runs and variability.** Partly. Some re-runs are reported (0.1.4 against 0.1.3, GloVe re-measured in one session); no systematic variance.
77. **Realistic high-dimensional datasets.** Partly. SciFact at 384; nothing at 768 or 1536.
78. **Linux and x86 results.** No. Everything is Apple Silicon; x86 has kernels and tests but no table.

## 10. API and developer experience

79. **Install and run an example without troubleshooting.** Yes. Wheels for every platform, numpy the only dependency, the semantic search example with its output; verified in a fresh environment at each release.
80. **Safe defaults, docs matching code.** Partly. The outputs in the docs are regenerated by script and the Python API page is generated from the stubs; but this review found the Python auto-checkpoint defaults at 0 since 0.2.4 against Rust's 250,000 and 100,000, fixed on master and released next.
81. **Trade-offs visible in the API.** Yes. `sync()`, `checkpoint()`, `search_beam`, `storage`, `exact_filter_scan`.
82. **Actionable errors, retry versus reopen.** Partly. The failure-modes page has the table; "keep the handle after an I/O error, reopen only after `Corrupt`" is not stated in those words.
83. **Explicit resource cleanup in Python.** No. No `close()` or context manager; the lock is released when the object is collected. Roadmap.
84. **Large imports and queries bounded in memory.** Partly. `insert_many` materialises its arrays; `search_many` is bounded by `n × k`. Streaming import is not documented.
85. **Inspecting health without reading Rust.** Yes. `stats()`, `index_stats()`, `properties()`, studio, `tracing` events.
86. **Exporting data.** Partly. `ids()` and `get()` reach every record; no export command.

## 11. Compatibility and portability

87. **Every release opens previous stores.** Yes. `opens_stores_written_by_every_published_version` over `tests/fixtures/`.
88. **Older binary refusing newer formats without modifying.** Yes. `NeedsNewerVersion` is raised before anything is written; `unknown_frame_kinds_are_skipped_and_newer_levels_refused`.
89. **Migrations safe, documented, recoverable.** Partly. Documented per version in the changelog; a backup before upgrading is the recovery; interruption is untested (25).
90. **Moving stores between operating systems.** Partly. The format is fixed-endian with no paths inside, so it should; not tested as such.
91. **Endianness, alignment, integer widths, maximum sizes.** Yes. [FORMAT.md](FORMAT.md): little-endian throughout, u32 record count, 65,535-byte metadata limits.
92. **Wheels across platforms and Python versions.** Partly. abi3 wheels for Python 3.9 and later on seven targets; the test suite runs on 3.12 only.
93. **Optional features and bindings in CI.** Partly. Python runs in CI; the `embed-http` and `tracing` features are not built there.
94. **Fixtures covering every format.** Yes.

## 12. Testing and long-term trust

95. **Randomised tests against a reference model.** Yes. `random_operations_agree_with_a_model`.
96. **Failures at individual I/O boundaries, partial writes, post-commit.** Partly. Four injected points plus the torn tail; nothing after a `Commit` that then fails to fsync.
97. **Crash tests verifying the exact acknowledged set.** Partly. The child reports its progress and the parent checks every acknowledged id is present; it does not check that nothing unacknowledged appears.
98. **Fuzzing parsers for crashes, allocation, hangs.** Partly. 400 random bit-flips over an index and a log per run, asserting no panic; no coverage-guided fuzzer, no allocation or time bound.
99. **Long mixed workloads run routinely.** Partly. `scripts/soak.sh` exists and a short version runs in the suite; hours-long runs are manual.
100. **Backups during writes, restored independently.** Yes. `backup_is_a_consistent_point_in_time_copy` writes while backing up and opens the copy.
101. **External developers.** No. None yet.
102. **Policy for corruption reports and urgent fixes.** No. No support policy written.
103. **Another maintainer.** Partly. Architecture, design notes and the byte-level format are written for that; one person has written all of it.
104. **Evidence for 1.0.** Partly. The roadmap says a year of other people's workloads; concrete criteria (soak hours, external users, zero format changes over a period) are not listed.
