# Status, limits and roadmap

## Where it stands

0.2.x has the on-disk format frozen and specified byte for byte, readers
in other processes, atomic batches, typed metadata with filter-aware
search, keyword and sparse search with hybrid fusion, int8 storage,
store properties, an embedding hook in both languages, fault-injection
and soak tests, stores from every published version opened by the test
suite, and studio: a local page that maps a store in 2D or 3D, searches
it every way, draws the walk and the graph, and edits records. The
numbers in [benchmarks.md](benchmarks.md) are measured, reproducible,
and from one laptop; dense retrieval with a real model reproduces the
model's published quality.

What it does not have is a track record. Nobody has run it in production;
the project is a few weeks old. Treat 0.2.x as a serious beta: keep
backups (`Db::backup`), run the soak script against your own workload
shape, and read the limits below before depending on it. A reviewer's
104 questions, each answered with what is proven, what is partly there
and what is missing, are on [questions.md](questions.md).

## Limits as of 0.2.6

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
- Sparse vectors have no public-dataset quality number yet (dense
  retrieval with a real model has one: SciFact, `docs/python/examples.md`).
- Single query thread per `search`; `search_many` uses all cores across
  queries. No GPU; none planned, the design is memory-bound and disk-first.
- x86 has AVX2 kernels but no published numbers; everything measured so far
  is Apple Silicon.
- Rare: under a heavily loaded machine, `Db::open` has occasionally hit
  `Error::Locked` with no other writer alive (about once in 25 parallel
  test-suite runs). A retry window covers it; the cause is still being
  chased with timing diagnostics in the test suite.

## Roadmap

What stands between 0.2.6 and something you would run in production, in
the order it is being worked. The gaps in [questions.md](questions.md)
feed this list.

1. **Evidence under load.** A longer soak (hours, not minutes) with the
   writer, checkpoints and readers in separate processes on a store
   larger than RAM, with latency percentiles recorded during the writes;
   the rare `Locked`-with-no-writer case chased to its cause instead of a
   retry window; crash tests on Linux x86 as well as arm64. These
   produce the track record; nothing else does.
2. **Hybrid search, the rest.** MaxScore for long posting lists (keyword
   search at millions of documents is unmeasured), weighted fusion as an
   alternative to reciprocal rank, a BEIR number for hybrid and for
   sparse retrieval with a SPLADE encoding. Design note:
   [design/hybrid-search.md](design/hybrid-search.md).
3. **Filtered search at moderate selectivity.** A filter-aware edge
   budget at build (ACORN-style) or a smaller result heap for filtered
   walks; a scan-or-walk threshold that accounts for the vector width;
   measure on the YFCC filter track at 10M.
4. **Comparisons people ask for.** A page against SQLite with sqlite-vec
   and against LanceDB; the VectorDBBench client in
   `benchmarks/vectordbbench/` run against the harness; a 768- or
   1536-dimensional dataset; a first x86 table; the cold int8 comparison
   at 5M; memory at 384 to 3072 dimensions.
5. **Operations.** A `close()` and context manager in Python; an export
   command; a checkpoint that runs off the write path instead of inside
   the insert that triggers it; incremental index maintenance so a
   long-running store does not need a full rebuild per checkpoint; a
   documented backup and restore drill; a support policy for corruption
   reports and compatibility regressions.
6. **x86 kernels** for build, exact scan and re-ranking, and a fused int8
   kernel. Query latency is memory-bound and will not move; builds and
   scans will.
7. **Studio.** Import from JSONL and CSV; a diff of two stores; an
   embedding model chosen from the page rather than the command line.
8. **1.0.** Same format, the promise extended to a major version; what
   remains is a year of other people's workloads.

## Not planned

- A server, a query language, multi-tenancy. Embed it; put your own
  service in front if you need one.
- A second writer or distributed consensus.
- Built-in encryption.
- GPU search.
