# sanadb

A disk-first embedded vector + metadata store in Rust. Single file, crash-safe,
no server. Aimed at on-device and edge AI workloads that outgrow RAM.

Status: milestone 1 (exact search, write-ahead log, checkpoint). Not production-ready.

## Usage

```rust
use sanadb::{Db, Options, Metric};
use std::collections::HashMap;

let mut db = Db::open("vectors.sana", Options::new(768).metric(Metric::Dot))?;
db.insert(1, &embedding, HashMap::from([("doc".into(), "intro.md".into())]))?;
let hits = db.search_filtered(&query, 10, |m| m["doc"].ends_with(".md"))?;
db.sync()?;       // fsync: survive power loss
db.checkpoint()?; // compact the file
```

## Design

- **One file, one log.** The file is a header plus CRC-framed insert/delete
  frames. Open replays it; a torn tail is truncated. `checkpoint` rewrites the
  file with only live records via temp-file + rename, so a crash mid-compaction
  keeps the old file intact.
- **Durability knobs.** Writes hit the OS page cache immediately (survive a
  process crash). `sync()` or `Options::sync_every_write` fsync for power-loss safety.
- **Exact flat index.** Vectors stored contiguously; distance loops use eight
  independent accumulators so the compiler emits packed FMA. This is the
  correctness baseline later indexes are measured against.

## Roadmap

1. **Done:** WAL, checkpoint, flat index, metadata filter, tests, Criterion bench.
2. Disk-resident graph index (Vamana/DiskANN style) laid out in page-sized
   blocks so the working set, not the dataset, must fit in RAM.
3. Filtered graph traversal (filter-aware search, not post-filtering).
4. MVCC snapshots for concurrent readers alongside a single writer.
5. Benchmark against sqlite-vec, usearch and LanceDB on 10M x 768-dim vectors.

## Development

```bash
cargo test
cargo bench
```

See `RESEARCH.md` for the problem survey behind this direction.
