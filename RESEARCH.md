# sanadb — research notes (2026-11-15)

## Question
Is there room for a new, faster database written in Rust (in-memory or disk-based), and which
open problems are worth attacking as an open-source project?

## Open problems surfaced

1. **Object-storage-native engines.** Cloud databases can no longer rely on instance-local disks
   for WAL durability, so logs and SSTs move to S3-class storage. SlateDB (Rust) already does this,
   but it is single-writer and pays object-store latency on every durable write. Open: low-latency
   durable writes, multi-writer coordination, adaptive placement between local NVMe cache and
   object storage.
2. **LSM write amplification.** Still an active VLDB 2026 topic. Hard, research-grade, incremental.
3. **io_uring / thread-per-core in Rust.** Mixed evidence. Apache Iggy moved from Tokio to
   thread-per-core io_uring and later settled on a hybrid. An arXiv study (2512.04859) concludes
   io_uring is "not a panacea" for transactional engines on NVMe. Conviva's compio rewrite ran 60%
   slower than mmap. Rust runtime ecosystem is fragmented (Glommio unmaintained, monoio niche).
   Verdict: a legitimate engineering edge, but not a product by itself.
4. **Embedded vector search beyond RAM.** Most Rust embedded vector stores are memory-bound and
   degrade past roughly 1M vectors. Concurrency under load at the edge is unverified. Many projects
   are hobby-stage. Gap: a disk-resident (DiskANN-style) single-file engine with real durability,
   MVCC, and filtered search.
5. **Agent memory stores.** Crowded (Engram, MenteDB, CortexaDB, RetainDB, agentic-db, Mem0, Zep).
   Most are wrappers over Postgres/SQLite or early hobby code. Gap is maturity and engine-level
   design, not absence of projects. Consistency, provenance, access control, and context-aware
   data reduction remain open.
6. **Time-series storage cost.** Benchmarks ignore storage-level TCO (write amplification,
   compaction cost, space). No controlled cross-engine measurement existed before MSST 2026.

## Assessment

"Faster than everything" is not a defensible goal; RocksDB, LMDB, SQLite, and the newer Rust
engines (SurrealKV, Fjall, redb, SlateDB, Turso) are already tuned by full-time teams. A new
engine wins only by picking a workload where the incumbents are structurally wrong.

## Candidate directions (ranked)

### A. Disk-first embedded vector + metadata store (recommended)
Single file, scales past RAM via a DiskANN/Vamana graph laid out for NVMe, filtered ANN search,
MVCC snapshots, crash-safe WAL. Benchmark targets: LanceDB, Qdrant Edge, sqlite-vec, usearch.
Clear user: on-device and edge AI apps that outgrow RAM. Rust fits: no GC pauses, SIMD via
`std::arch`, zero-copy mmap or io_uring reads.

### B. Tiered object-storage KV with local NVMe write cache
SlateDB-shaped, but durable writes land on a local replicated NVMe log first and are shipped to
S3 asynchronously, giving sub-millisecond commits with bottomless storage. Harder (needs a
replication protocol), bigger payoff, longer runway.

### C. Agent memory engine at the storage layer
Vector + graph + temporal decay as native index types in one embedded engine, not a Postgres
wrapper. Market is noisy; differentiation would have to come from engine quality (direction A is
the foundation for this anyway).

## Suggested first milestone
Direction A, scoped to: single-file store, u32-keyed KV + one vector index, WAL + checkpoint,
k-NN query with simple metadata filter, Criterion benchmark against sqlite-vec and usearch on
10M 768-dim vectors on a laptop NVMe.

## Sources
- https://openproceedings.org/2026/conf/edbt/paper-T3.pdf
- https://vldb.org/2026/program.html
- https://www.msstconference.org/2026/papers/MSST26_paper_10.pdf
- https://grc.iit.edu/research/projects/storehub/report/
- https://surrealdb.com/static/whitepapers/surrealkv.pdf
- https://dbdb.io/browse?programming-language=rust
- https://arxiv.org/html/2512.04859v2
- https://dev.to/sospeter/how-apache-iggy-ditched-tokio-for-thread-per-core-iouring-and-what-it-cost-them-3m75
- https://tonbo.io/blog/exploring-better-async-rust-disk-io
- https://www.actian.com/blog/developer/comparing-embedded-vector-databases/
- https://thenewstack.io/slatedb-bottomless-databases-built-on-cloud-object-stores/
- https://docs.rs/crate/slatedb/0.15.0
- https://www.shaped.ai/blog/the-8-best-tools-for-ai-agent-memory-long-term-recall-2026-guide
