# sanadb studio

A local web page for one store: see what is in it, search it every way
the library can, watch the graph walk of a search on a map of the
vectors, and edit or delete records. One binary, no build step, nothing
leaves the machine.

![sanadb studio on a 1M-record store](img/studio.png)

## Install and open

```bash
pip install "sanadb[studio]"       # the binary as a console script; or cargo install sanadb-studio,
sanadb-studio data/vectors.sana    # or a prebuilt binary from the GitHub release
# sanadb studio 0.2.3: data/vectors.sana (1000000 records, dim 128, L2) at http://127.0.0.1:7070/
```

```
sanadb-studio <store.sana> [--port 7070] [--text-field KEY] [--read-only]
              [--embed-url URL --embed-model NAME [--embed-key-env VAR] | --embed-cmd CMD]
```

The dimension and metric come from the file. Pass `--text-field` with the
key the store was built with to enable keyword and hybrid search. Studio
opens the store as its writer; if another process holds it, studio falls
back to read-only and says so in the header. It binds to `127.0.0.1` and
has no authentication: one store, one machine, one person.

## 0. Attach an embedding model

Without a model the page can only search from a record that is already
in the store. With one, you type text. Two ways to attach it:

```bash
# any OpenAI-compatible /v1/embeddings endpoint: Ollama, LM Studio, vLLM,
# text-embeddings-inference, OpenAI itself (the key comes from an env var)
sanadb-studio docs.sana --embed-url http://localhost:11434/v1/embeddings --embed-model nomic-embed-text
sanadb-studio docs.sana --embed-url https://api.openai.com/v1/embeddings --embed-model text-embedding-3-small --embed-key-env OPENAI_API_KEY

# or any command: a JSON array of strings on stdin, a JSON array of float arrays on stdout
sanadb-studio docs.sana --embed-cmd "python embed.py"
```

The model's dimension has to match the store's; a mismatch is reported
on the first query. For a sentence-transformers model,
`python/examples/embed_server.py` in the repository serves it as such an
endpoint on localhost with one command. Studio writes the model's name into the store's
properties (`embedding_model`, `embedding_dim`) the first time one is
attached; later, the header shows which model the vectors came from even
with no model attached, and warns in red when a different one is. The header shows which model is attached, the query
panel gains a "Text to embed" box, and the **Add record** button in the
header takes text, optional metadata and an optional id, embeds the text,
keeps it under the text field (or `text`), and writes the record. Nothing
is sent anywhere except to the endpoint or command you named.

## 1. Read the header

![the header: file, records, dimension, metric, index state, tail, checkpoint](img/studio-1-open.png)

Left to right: the file, how many records are live, the dimension and
metric, the index (nodes, storage, PQ subspaces, records tombstoned since
the last checkpoint), how many records sit in the tail since the last
checkpoint, and the Checkpoint button. A store that has never
checkpointed shows "no index yet": everything still works, searched
exactly over the tail.

The map builds on open: a sample of up to 4,000 records projected to two
dimensions by PCA and coloured by k-means. It takes a few seconds on a
million records and is cached until the next checkpoint.

## 2. Search

![the query panel](img/studio-2-query.png)

- **Text to embed** (with a model attached) is the query most people
  mean: the text becomes a vector through the model and the search runs
  on that.
- **Record id** uses that record's vector as the query: the way to ask
  "what is near this one".
- **Keyword text** runs BM25 over the text field; with an id as well it
  becomes a hybrid search, fused by reciprocal rank.
- **Sparse vector** takes `index:weight` pairs (`3:1.0 52:2.0`), alone or
  fused with an id.
- **Filter** is JSON in the same grammar as the Python bindings:
  `{"lang": "rust"}`, `{"year": {"$gte": 2020}}`, `{"cat": {"$in":
  ["a", "b"]}}`, `{"$not": {...}}`, `{"$or": [...]}`. Operators: `$eq $ne
  $in $lt $le $gt $ge $prefix $contains $exists`.
- **k** is how many results; Enter in any field searches.
- **Draw the graph walk** asks for the walk trace (unfiltered searches
  from an id only).

The results header names the mode that ran (`nearest`, `filtered`,
`keyword`, `hybrid`, `sparse`, `walk`), the time the search took inside
the engine, and what the score column means for this store's metric.
Scores are the engine's own values, lower is closer:

| metric | score | identical vectors |
|---|---|---|
| `l2` | squared L2 distance | 0 |
| `cosine` | negative cosine similarity; vectors are normalised on insert and on query | −1 (1 is opposite) |
| `dot` | negative dot product of the vectors as stored | the most negative value |

Keyword results carry negated BM25, hybrid results the negated fused rank
score; both are ranks more than distances.

## 3. Watch the walk

![the map with the walk of one search](img/studio-3-walk.png)

The white ring is the query. The amber trail is every node the graph walk
expanded, in the order it expanded them: each segment carries an arrowhead
in the direction the walk went, the first twelve expansions are numbered,
the latest one is marked, and older segments fade. The transport bar
under the map title plays the walk at ¼× to 4×, pauses it, steps one
expansion at a time, replays from the start, and has a scrubber you can
drag; space, left and right arrow do the same from the keyboard. The
speed setting "Instant" draws the whole walk at once for people who do
not want the animation; the choice is remembered. The
blue-ringed points are the results. Scroll to zoom, drag to
pan, hover a point for its id, click a point to inspect it. Searches are
projected onto the same plane as the map, so a walk that heads across the
map before settling is a real property of the graph, not of the drawing.

## 4. Inspect and edit a record

![results table and the record panel](img/studio-4-results-record.png)

Click a result, a record in the list, or a point. The panel's headline is
the record's name when its metadata has one (the first of `title`,
`name`, `doc`, `path`, `file`, `url`, `label`, `text`, `body` holding a
string), otherwise "Record" and its id; the id is always shown beneath,
labelled. Below: the vector's dimensions, non-zero count and L2 norm, its
score against the current query, the map cluster it fell in, the full
values on demand, and the metadata as JSON you can edit. "Replace the vector" takes new text to re-embed (with
a model attached) or pasted numbers and rewrites the record under the
same id. **Save metadata** writes a new version of the record
to the log; it is visible to every reader at once and reaches the index
columns at the next checkpoint, like any write. **Delete** asks first.
**Search from this record** puts its id in the query and runs.

Types are fixed per key: the first value ever written under a key sets
it, and a save that changes a key's type is refused with the reason.

## 5. Filter

![a filtered search](img/studio-5-filtered.png)

With a filter the engine picks the regime per query: few matching
records are read exactly (recall 1.0, microseconds), many steer the
walk. The footer's `exact scans` and `filtered walks` counters tell you
which one ran.

## 6. Label the clusters

![cluster labels from a metadata key](img/studio-6-labels.png)

"Label clusters by" at the bottom right writes each cluster's most common
value of a key at its centroid, with the share of the cluster that holds
it. A cluster where one value reaches 80% means something; eight clusters
at 11 to 14% each, as on this synthetic set, mean the key is independent
of the vectors. That is the question this control exists to answer.

## 7. Files, keys, records

![the files tab](img/studio-7-files.png)

The left column's tabs: **Records** pages through every live id with its
metadata; **Keys** lists each metadata key with its type; **Files** shows
the log, the index and the lock with their sizes, hover a name for its
path, reader level and the index's snapshot id. The footer counts the
store's operations since studio opened it.

## Checkpoint

The Checkpoint button builds the graph index over every record and
compacts the log, on all cores, without blocking readers. The header's
tail count drops to zero, the index node count rises, and the map is
rebuilt. Run it after a bulk edit or import.

## Light and dark

The chrome follows the system setting; the toggle at the top right
overrides it and is remembered in the browser. The map viewport is dark
in both.

![dark chrome](img/studio-dark.png)

## What it is not

Not an admin console and not multi-store: one file, one process, no
users. Import from a file is on the roadmap.
