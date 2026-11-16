# Rust recipes

Short, complete patterns for the situations that come up once the first
example works. Each one is the smallest version that is correct; the
reasons are in [operating a store](../operations.md).

## Implement `Embedder` over your own model

The trait has one method, and any closure of the right shape implements
it, so a model behind any Rust inference crate or HTTP API plugs in
without a wrapper type:

```rust
use sanadb::{Db, Embedder, Error, Result};

// A closure is an Embedder.
let model = |texts: &[&str]| -> Result<Vec<Vec<f32>>> {
    texts.iter().map(|t| my_model.encode(t).map_err(|e| Error::Io(std::io::Error::other(e)))).collect()
};
db.add_text(1, "Rust borrow checker explained", meta, "body", &model)?;

// Or a type, when it carries state such as a loaded model or a client.
struct Local(candle_model::Model);
impl Embedder for Local {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(self.0.encode_batch(texts))
    }
}
```

Record the model's name once with
`db.set_property("embedding_model", "...")`; studio and future readers
check it, and `add_text` keeps the text under the key you name so
`search_text` and `get` see it.

## Index a folder of documents for retrieval (RAG)

Chunk, embed in batches, write in atomic batches, checkpoint once at the
end. Ids are a hash of the file and chunk position so a re-run of the
same folder overwrites rather than duplicates; the chunk text and its
source stay with the vector so a hit can be shown and cited.

```rust
use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{Hash, Hasher};
use sanadb::{Batch, Db, Metric, Options, Value};

let db = Db::open("kb.sana", Options::new(384).metric(Metric::Cosine).text_field("text"))?;
db.set_property("embedding_model", "all-MiniLM-L6-v2")?;

let chunk_id = |path: &str, n: usize| -> u64 {
    let mut h = DefaultHasher::new();
    (path, n).hash(&mut h);
    h.finish() >> 1
};
for entry in walkdir::WalkDir::new("notes").into_iter().flatten().filter(|e| e.path().extension() == Some("md".as_ref())) {
    let path = entry.path().to_string_lossy().into_owned();
    let text = std::fs::read_to_string(entry.path())?;
    let pieces: Vec<&str> = text.as_bytes().chunks(800).map(|c| std::str::from_utf8(c).unwrap_or("")).collect();
    let vectors = model.embed(&pieces)?;                      // one request for the whole file
    let mut batch = Batch::new();
    for (n, (piece, v)) in pieces.iter().zip(&vectors).enumerate() {
        batch = batch.insert(chunk_id(&path, n), v, HashMap::from([
            ("text".to_string(), Value::from(*piece)),
            ("source".to_string(), Value::from(path.as_str())),
            ("chunk".to_string(), Value::from(n as i64)),
        ]));
    }
    db.write_batch(batch)?;                                   // the file lands whole or not at all
}
db.checkpoint()?;

for hit in db.search_text_embedded("how do I rotate the API key", 5, None, &model)? {
    let (_, meta) = db.get(hit.id).unwrap();
    println!("{:+.3} {}#{}", hit.score, meta["source"], meta["chunk"]);
}
```

(`chunks(800)` on bytes is a sketch; split on characters or sentences in
real code.) A changed file re-embeds its chunks under the same ids and
overwrites them; a removed file's chunks go with `delete`. Overwrites and
deletes accumulate in the log until the next checkpoint compacts them.

## When to checkpoint

The checkpoint is the one heavy operation: it builds the graph over all
records on every core and writes a new index file. Records written since
the last checkpoint are still searched, exactly, so correctness never
depends on it; latency does once the tail is large.

- **Bulk load:** insert everything, checkpoint once, then serve.
- **Steady trickle:** `Options::auto_checkpoint_delta(50_000)` (or
  whatever fraction of the set you accept being scanned) and the store
  decides; or call `checkpoint()` from a timer thread, it holds neither
  the read nor the write lock while it builds.
- **Many overwrites or deletes:** `Options::auto_checkpoint_garbage(100_000)`
  compacts once the log carries that many dead entries.
- **Read-only handles** never checkpoint; they follow the writer's.

`db.delta_len()` says how large the tail is; `db.stats().exact_scans`
rising fast is the sign that the tail is being scanned on every query.

## One writer, many readers

`Db` is `Send + Sync`; within a process, share one handle in an `Arc`
across threads (reads run in parallel, writes serialise). Across
processes, one writes and any number open the same path read-only and
follow it:

```rust
// the indexing process
let db = Db::open("kb.sana", opts())?;

// each serving process
let reader = Db::open_read_only("kb.sana", opts())?;
let hits = reader.search(&q, 10)?;
```

A second writer gets `Error::Locked` at open; a write on a read-only
handle returns `Error::ReadOnly`. The lock is an advisory `flock`, which
is reliable on local disks and not on NFS or SMB.

## Backup and restore

```rust
let dir = db.backup("/backups/kb-2026-11-16")?;      // consistent, while writes continue
```

The directory holds the log and the index file it names. Restore is
copying the directory back and opening it; the backup is itself a
complete store that opens read-only or as a new writer. Replication is
the same operation shipped over the network.

## Durability settings

A write that returned `Ok` survives a process crash. For power loss too,
either call `db.sync()` at the points that matter (after a batch, before
acknowledging to a client) or open with `Options::sync_every_write(true)`
and pay an fsync per write.

## Hybrid search for identifiers and names

Embeddings blur exact tokens: error codes, flags, product names. Pass the
literal tokens to `search_hybrid` and the dense result is fused with BM25
by reciprocal rank, so a record matching the exact token rises:

```rust
let q = model.embed(&["tuning the collector"])?.remove(0);
db.search_hybrid(&q, "GOGC", 5, None)?;
```

The store needs `text_field` set at open for this; `add_text` stores each
document's text there.

## Learned sparse vectors (SPLADE and friends)

One metadata key per store may hold `Value::Sparse(Vec<(u32, f32)>)`;
`search_sparse` runs a dot product over its inverted lists and
`search_hybrid_sparse` fuses it with the dense result:

```rust
db.insert(1, &dense, HashMap::from([("splade".to_string(), Value::Sparse(terms))]))?;
db.search_sparse(&q_terms, 10, None)?;
db.search_hybrid_sparse(&dense_q, &q_terms, 10, None)?;
```

## Observability

`db.stats()` returns counters since open; they are always kept. With the
`tracing` feature the crate emits events on target `sanadb` for opens,
checkpoints (with duration), failed checkpoints, reader reloads and
orphan index files removed at open, so a `tracing-subscriber` filter of
`sanadb=info` shows the store's life without any code in yours.

## When things go wrong

The full page, with each case produced on a real store, is
[failure modes](../failure-modes.md); this is the short table.

| what you see | what it means | what to do |
|---|---|---|
| `Error::Locked` | another writer has the path open | `open_read_only`, or close the other handle; a crashed process releases the lock when it dies |
| `Error::ReadOnly` | the handle is read-only | write from the writer process |
| `Error::DimMismatch` | a vector of the wrong length | the dimension is fixed at the first open; `Options::new` with another `dim` fails with `HeaderMismatch` too |
| `Error::TypeMismatch` | a metadata key used with a second type | keys are typed by their first value; use another key or convert |
| `Error::SecondSparseKey` | two keys hold sparse vectors | one sparse key per store |
| `Error::InvalidVector` | NaN or infinity, or all zeros under cosine | check the model's output |
| `Error::NeedsNewerVersion` | the file was written by a newer sanadb with a layout this one cannot read | upgrade; an older version never misreads |
| `Error::Corrupt(what)` | a CRC or geometry check failed on the index file | restore from `backup`; a torn log tail is truncated on open without an error |
| searches get slow after a long run without checkpoint | the tail is being scanned exactly on every query | `checkpoint()` or `auto_checkpoint_delta` |
| recall lower than expected | the beam is small for the data | raise `search_beam` (64 → 128) or `GraphParams::max_degree` (32 → 48) and checkpoint |
| results shorter than `k` before any error | fewer records pass the filter than asked for | that is the answer |
