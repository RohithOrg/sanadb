# Python recipes

Short, complete patterns for the situations that come up once the first
example works. Each one is the smallest version that is correct; the
reasons are in [operating a store](../operations.md).

## Index a folder of documents for retrieval (RAG)

Chunk, embed in batches, write in atomic batches, checkpoint once at the
end. Ids are the document's position so a re-run of the same folder
overwrites rather than duplicates; the chunk text and its source stay
with the vector so a hit can be shown and cited without a second store.

```python
import hashlib, pathlib
import sanadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
db = sanadb.Db("kb.sana", dim=384, metric="cosine", text_field="text",
               embedding_function=lambda t: model.encode(t, normalize_embeddings=True))
db.set_property("embedding_model", "all-MiniLM-L6-v2")

def chunks(text, size=800, overlap=100):
    for i in range(0, max(len(text) - overlap, 1), size - overlap):
        yield text[i:i + size]

def chunk_id(path, n):                       # stable across runs: same file + chunk → same id
    return int.from_bytes(hashlib.blake2b(f"{path}#{n}".encode(), digest_size=8).digest(), "big") >> 1

for path in pathlib.Path("notes").rglob("*.md"):
    pieces = list(chunks(path.read_text()))
    db.add(pieces,
           ids=[chunk_id(path, n) for n in range(len(pieces))],
           metas=[{"source": str(path), "chunk": n} for n in range(len(pieces))])
db.checkpoint()

for id_, score in db.query(["how do I rotate the API key"], k=5)[0]:
    _, meta = db.get(id_)
    print(f"{score:+.3f} {meta['source']}#{meta['chunk']}: {meta['text'][:80]}")
```

A changed file re-embeds its chunks under the same ids and overwrites
them; a deleted file's chunks are removed with
`db.delete_many([...])`. Overwrites and deletes accumulate in the log
until the next checkpoint compacts them.

## When to checkpoint

The checkpoint is the one heavy operation: it builds the graph over all
records on every core and writes a new index file. Records written since
the last checkpoint are still searched, exactly, so correctness never
depends on it; latency does once the tail is large.

- **Bulk load:** insert everything, checkpoint once, then serve.
- **Steady trickle:** open with `auto_checkpoint_delta=50_000` (or
  whatever fraction of the set you accept being scanned) and let the
  store decide; or call `checkpoint()` from a timer thread, it does not
  block readers or writers.
- **Many overwrites or deletes:** `auto_checkpoint_garbage=100_000`
  compacts once the log carries that many dead entries.
- **Read-only handles** never checkpoint; they follow the writer's.

```python
db = sanadb.Db("kb.sana", dim=384, metric="cosine", auto_checkpoint_delta=50_000)
```

`db.delta_len` says how large the tail is; `db.stats()["exact_scans"]`
rising fast is the sign that the tail is being scanned on every query.

## One writer, many readers

One process writes; any number of processes read, following the writer's
inserts and checkpoints without being told. A web app with worker
processes opens the store read-only in each worker:

```python
# indexer.py: the only process that writes
db = sanadb.Db("kb.sana", dim=384, metric="cosine", text_field="text")

# worker.py: as many as you like, any machine sharing the local disk is not supported
reader = sanadb.Db("kb.sana", dim=384, metric="cosine", text_field="text", read_only=True)
hits = reader.search(q, k=10)
```

A second writer on the same path fails at open with
`RuntimeError: store is open elsewhere (lock file held)`; a write on a
read-only handle raises `PermissionError`. The lock is an advisory
`flock`, which is reliable on local disks and not on NFS or SMB.

## Backup and restore

```python
copy = db.backup("/backups/kb-2026-11-16")      # consistent, while writes continue
```

The directory holds the log and the index file it names. Restore is
copying the directory back and opening it; the backup is itself a
complete store that opens read-only or as a new writer. Replication is
the same operation shipped over the network.

## Durability settings

A write that returned survives a process crash. For power loss too,
either call `db.sync()` at the points that matter (after a batch, before
acknowledging to a client) or open with `sync_every_write=True` and pay
an fsync per write (hundreds of writes a second instead of hundreds of
thousands).

## Hybrid search for identifiers and names

Embeddings blur exact tokens: error codes, flags, product names. Pass the
literal tokens as `text=` and the dense result is fused with BM25 by
reciprocal rank, so a record matching the exact token rises:

```python
q = model.encode(["tuning the collector"], normalize_embeddings=True)[0]
db.search(q, k=5, text="GOGC")
```

The store needs `text_field` set at open for this; `add` stores each
document's text there.

## Learned sparse vectors (SPLADE and friends)

One metadata key per store may hold a sparse vector, written as
`(indices, weights)`; `search_sparse` runs a dot product over its
inverted lists and `search(..., sparse=)` fuses it with the dense result:

```python
db.insert(1, dense, {"body": text, "splade": (indices, weights)})
db.search_sparse((q_idx, q_w), k=10)
db.search(dense_q, k=10, sparse=(q_idx, q_w))
```

## Moving from Chroma

The shapes line up: `embedding_function` on the constructor, `add` with
documents and metadata, `query` with texts and a `where`-style filter.
Differences that matter:

| Chroma | sanadb |
|---|---|
| `client.create_collection(name, embedding_function=f)` | `sanadb.Db(path, dim=..., metric="cosine", text_field="text", embedding_function=f)`; one file per collection |
| `collection.add(documents=, ids=, metadatas=)` | `db.add(documents, ids=, metas=)`; ids are integers, and are returned if you omit them |
| `collection.query(query_texts=, n_results=, where=)` | `db.query(texts, k=, filter=)`; same `$eq $ne $in $gt $gte $lt $lte $and $or` operators, plus `$prefix $contains $exists $not` |
| `collection.get(ids=)` | `db.get(id)` returns `(vector, metadata)`, the document text under the text field |
| `collection.update / upsert` | `insert` and `add` overwrite an existing id |
| distances (lower is closer) | scores: negative cosine similarity for `cosine`, squared L2 for `l2`, negative dot for `dot` |
| a server or a persistent client directory | a file; readers in other processes open it `read_only=True` |

Metadata values are `str`, `int`, `float`, `bool` in both; sanadb fixes
a key's type on first write and raises `TypeError` on a different type
later, where Chroma would store both.

## When things go wrong

The full page, with each case produced on a real store, is
[failure modes](../failure-modes.md); this is the short table.

| what you see | what it means | what to do |
|---|---|---|
| `RuntimeError: store is open elsewhere (lock file held)` | another writer has the path open | open `read_only=True`, or close the other handle; a crashed process releases the lock when it dies |
| `PermissionError` on insert | the handle is read-only | write from the writer process |
| `ValueError: dimension mismatch` | a vector of the wrong length | the dimension is fixed at the first open; `sanadb.Db` with another `dim` refuses too |
| `TypeError` on insert | a metadata key used with a second type | keys are typed by their first value; use another key or convert |
| `ValueError` naming a second sparse key | two keys hold sparse vectors | one sparse key per store |
| `RuntimeError: ... needs a newer version` | the file was written by a newer sanadb with a layout this one cannot read | upgrade; an older version never misreads |
| searches get slow after a long run without checkpoint | the tail is being scanned exactly on every query | `db.checkpoint()` or `auto_checkpoint_delta` |
| recall lower than expected | the beam is small for the data | raise `search_beam` (64 → 128) or `max_degree` (32 → 48) at open and checkpoint |
| results empty before any error | `k` larger than the records that pass the filter | that is the answer: fewer records pass than asked for |

Corruption is handled, not raised at random: a torn log tail is truncated
on open, a damaged index file fails to open with a `RuntimeError` naming
the check that failed, and a backup from before it opens unchanged.
