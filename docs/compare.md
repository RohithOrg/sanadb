# Compared with the alternatives

Where sanadb sits next to the stores people ask about, written by the
sanadb side and saying so. The numbers quoted for sanadb are its own,
from [benchmarks](benchmarks.md); the statements about the others are
about their design, checked against their documentation on 17 November
2026, not measured here.

## SQLite with sqlite-vec

sqlite-vec is an extension that adds vector columns and a `MATCH`
operator to SQLite. It is the right choice when the vectors live beside
relational data that is already in SQLite, the set is small, and you
want one database file for everything.

- **Search.** sqlite-vec scans every vector for each query (a brute-force
  KNN over the virtual table). That is exact, and fast enough up to
  roughly 100,000 vectors of a few hundred dimensions; at a million it is
  hundreds of milliseconds per query and grows linearly. sanadb's graph
  answers a million in about 150 µs warm and five million from a cold
  disk in about 3 ms, with recall measured against brute force.
- **Filters.** In sqlite-vec a `WHERE` clause on other columns is
  applied by SQLite around the vector scan, so it is exact at any
  selectivity, at the scan's cost. sanadb evaluates filters on index
  columns: rare filters by exact scan of the matches, common ones by
  steering the walk; both measured.
- **Durability.** SQLite's, which is the gold standard; sanadb has its
  own log and atomic checkpoint, tested with fault injection but with
  nothing like SQLite's track record.
- **Keyword and sparse.** SQLite's FTS5 gives BM25 in the same file;
  sanadb has BM25 over one text field and learned sparse vectors, fused
  with the dense result in one call.
- **Use sqlite-vec when:** under 100K vectors, data already in SQLite,
  exact results wanted, SQL wanted. **Use sanadb when:** the set is
  hundreds of thousands to millions, queries need to stay in the
  microseconds, or hybrid search matters.

## LanceDB

LanceDB is an embedded, disk-based vector database over the Lance
columnar format, with Python, TypeScript and Rust bindings, an IVF-PQ
index, full-text search, and a cloud offering. It is the closest thing to
sanadb in shape and the one to pick today for anything that cannot afford
a 0.x bug.

- **Index.** LanceDB uses IVF with product quantization (and an HNSW
  option); sanadb uses a Vamana graph with PQ codes in RAM and the graph
  memory-mapped. Graphs give higher recall at equal latency on most
  datasets; IVF builds faster and scales out more simply.
- **Data model.** LanceDB is a columnar table store: any number of
  columns, Arrow types, versioned datasets, SQL-like filters, and a
  path to Lance's wider ecosystem. sanadb is one record type (id, vector,
  typed metadata) in one file, with a byte-level format specification.
- **Writes.** LanceDB appends new fragments and compacts; sanadb has a
  write-ahead log with atomic batches, a tail that is searched exactly
  until the next checkpoint, and readers in other processes that follow
  the writer live.
- **Maturity.** LanceDB has years of use, a company behind it and a
  cloud product. sanadb is weeks old with one author.
- **Use LanceDB when:** you need dependability now, many columns,
  Arrow interoperability, or a hosted option. **Use sanadb when:** a
  single file with a specified format, live cross-process readers, exact
  filters at low selectivity, or the smallest possible dependency
  footprint matters, and a 0.x risk is acceptable.

## Chroma

Chroma is the common first choice for RAG prototypes: an embedding
function on the collection, `add` and `query` in text, a local mode and
a server mode. sanadb's Python API deliberately matches that shape (see
[moving from Chroma](python/recipes.md#moving-from-chroma)). The
differences: sanadb is a single file with no server mode and no
multi-tenancy; its index is disk-first rather than in-memory HNSW, so it
keeps working past RAM; filters are evaluated on index columns; and it
ships a byte-level format and fault-injection tests. Chroma has a much
larger community and more integrations.

## faiss, hnswlib, usearch

These are indexes, not stores: in-RAM structures with save and load,
no log, no metadata, no crash safety, no readers in other processes.
Within that scope they are excellent and faster to build. sanadb is
compared against them at equal recall in ann-benchmarks' own harness
(ahead of hnswlib and usearch at every recall on SIFT, trading places
with faiss HNSW), while holding 32 bytes per vector in RAM instead of the
index. If you only need an index inside one process and can rebuild it
from source data, use one of these.

## Qdrant, Milvus, Weaviate, pgvector

Servers (pgvector is an extension to a server). They give you network
access, multi-tenancy, replication, authentication and horizontal scale,
none of which sanadb has or plans. If you need a shared service, run
one of these; put your own service in front of sanadb only when one
process owning one file is the shape you want.
