# Status, limits and roadmap

## Where it stands

0.2.0 is the first release with the on-disk format frozen, a byte-level
specification, readers in other processes, atomic batches, typed metadata
with filter-aware search, keyword and sparse search with hybrid fusion,
int8 storage, fault-injection and soak tests, and stores from every
published version opened by the test suite. The numbers in
[benchmarks.md](benchmarks.md) are measured, reproducible, and from one
laptop.

What it does not have is a track record. Nobody has run it in production;
the project is a few weeks old. Treat 0.2.0 as a serious beta: keep
backups (`Db::backup`), run the soak script against your own workload
shape, and read the limits below before depending on it.

## Limits as of 0.2.0

- One writer per store. Readers in any number of processes, but a second
  writer is refused. No streaming replication; replication is backup and
  file copy.
- No encryption inside the files; use the platform's.
- The writer lock is an advisory `flock`, which network filesystems do not
  honour reliably. Local disks only.
- A checkpoint rebuilds or updates the whole graph; there is no
  incremental index maintenance per write. The tail between checkpoints is
  searched exactly and should be kept to a fraction of the index.
- Filtered search at moderate selectivity (around 10% matching) costs
  several times an unfiltered query, because the walk needs `beam` passing
  nodes before it can stop. Rare filters are cheap (exact scan); common
  ones are cheap (plain walk).
- Keyword search is term-at-a-time BM25 with no MaxScore skipping; fine at
  thousands of documents per term, unmeasured at millions.
- Sparse vectors have no public-dataset quality number yet.
- Single query thread per `search`; `search_many` uses all cores across
  queries. No GPU; none planned, the design is memory-bound and disk-first.
- x86 has AVX2 kernels but no published numbers; everything measured so far
  is Apple Silicon.
- Rare: under a heavily loaded machine, `Db::open` has occasionally hit
  `Error::Locked` with no other writer alive (about once in 25 parallel
  test-suite runs). A retry window covers it; the cause is still being
  chased with timing diagnostics in the test suite.

## Roadmap

In rough order.

1. **0.2.0 release.** The checklist is done except a quiet re-measurement
   of the GloVe table and the cold int8 comparison at 5M.
2. **Hybrid search, the rest.** MaxScore for long posting lists, weighted
   fusion as an alternative to reciprocal rank, a BEIR number for sparse
   retrieval with a SPLADE encoding. Design note:
   [design/hybrid-search.md](design/hybrid-search.md).
3. **Filtered search at moderate selectivity.** A filter-aware edge budget
   at build (ACORN-style) or a smaller result heap for filtered walks;
   measure on the YFCC filter track at 10M.
4. **More comparisons.** The VectorDBBench client in
   `benchmarks/vectordbbench/` run against the harness; more
   ann-benchmarks datasets; a first x86 table.
5. **x86 kernels** for build, exact scan and re-ranking, and a fused int8
   kernel. Query latency is memory-bound and will not move; builds and
   scans will.
6. **1.0.** Same format, the promise extended to a major version; what
   remains is a year of other people's workloads.
7. **sanadb studio.** A local web UI to open a store, map it in 2D,
   cluster, search, watch the beam walk, and edit metadata.

## Not planned

- A server, a query language, multi-tenancy. Embed it; put your own
  service in front if you need one.
- A second writer or distributed consensus.
- Built-in encryption.
- GPU search.
