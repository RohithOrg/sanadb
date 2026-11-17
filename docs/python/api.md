# Python API

Generated from the package's type stubs (`python/sanadb.pyi`), which ship in the wheel, so an editor shows the same signatures and docstrings.

```python
import sanadb
```
Type stubs for the sanadb extension module.


## `Sparse`

```python
Sparse = tuple[npt.ArrayLike, npt.ArrayLike]
```
A sparse vector as `(indices, weights)`; returned as two numpy arrays.
One metadata key per store may hold them; `search_sparse` searches it.


## `Metadata`

```python
Metadata = dict[str, Union[str, int, float, bool, Sparse]]
```

## `Filter`

```python
Filter = Union[Mapping[str, Any], Callable[[Metadata], bool]]
```

## class `Db`

A disk-first vector + metadata store at `path` (a file; the index
lives beside it as `<path>.idx.<id>`).

### `__init__(path: str, dim: int, *, metric: str = 'l2', search_beam: int = 64, max_search_beam: int = 4096, io_width: int = 4, sync_every_write: bool = False, pq: bool = True, max_degree: int = 32, build_beam: int = 100, alpha: float = 1.2, read_only: bool = False, storage: str = 'f32', text_field: Optional[str] = None, embedding_function: Optional[Callable[[list[str]], npt.ArrayLike]] = None, auto_checkpoint_delta: int = 250000, auto_checkpoint_garbage: int = 100000) -> None`

`embedding_function` maps a list of texts to an (n, dim) array; it
enables `add()` and `query()`, like Chroma's embedding functions.
`auto_checkpoint_delta` checkpoints once that many records sit in the
tail (and the tail is at least as large as the index);
`auto_checkpoint_garbage` once the log holds that many more entries
than live records. 0 turns one off. The checkpoint runs inside the
insert that triggers it, so that one insert takes as long as a build.

### `dim` (property)

### `delta_len` (property)

Records written since the last checkpoint (searched exactly).

### `__len__() -> int`

### `neighbors(id: int) -> Optional[list[int]]`

The out-edges of `id`'s node in the graph index, as ids; None when
the record has no node (in the tail since the last checkpoint,
deleted, or absent). For tooling that draws the graph.

### `insert(id: int, vector: npt.ArrayLike, meta: Optional[Metadata] = None) -> None`

Insert or replace one record. A key's type is fixed by its first value.

### `insert_many(ids: Sequence[int], vectors: npt.ArrayLike, metas: Optional[Sequence[Optional[Metadata]]] = None) -> None`

Insert `(n, dim)` vectors atomically: all of them or none.

### `add(documents: Sequence[str], ids: Optional[Sequence[int]] = None, metas: Optional[Sequence[Optional[Metadata]]] = None) -> list[int]`

Embed the documents and insert them atomically; the text is kept
under the text field (or `text`). Returns the ids used.

### `query(texts: Sequence[str], k: int = 10, filter: Optional[Mapping[str, Any]] = None) -> list[list[tuple[int, float]]]`

Embed each text and search: one `[(id, score), ...]` per text.

### `set_property(key: str, value: str) -> None`

A string kept with the store, such as the embedding model's name;
visible to readers at once, kept across checkpoints. Empty removes it.

### `property(key: str) -> Optional[str]`

### `properties() -> dict[str, str]`

### `delete(id: int) -> bool`

### `delete_many(ids: Sequence[int]) -> None`

### `get(id: int) -> Optional[tuple[npt.NDArray[np.float32], Metadata]]`

### `search(query: npt.ArrayLike, k: int = 10, filter: Optional[Filter] = None, text: Optional[str] = None, sparse: Optional[Sparse] = None) -> list[tuple[int, float]]`

k nearest `(id, score)`, closest first; lower score is closer.

`filter` is a dict (`{"k": v}`, `{"k": {"$gte": 3}}`, `$eq $ne $in
$lt $le $gt $ge $prefix $contains $exists $and $or $not`) or a
callable on the metadata. With `text`, the result is the hybrid of
the vector search and a BM25 search on the store's text field,
fused by reciprocal rank; with `sparse`, of the vector search and a
dot-product search on the sparse field.

### `search_many(queries: npt.ArrayLike, k: int = 10, filter: Optional[Mapping[str, Any]] = None) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]`

`(n, dim)` queries on all cores: `(ids, scores)` of shape `(n, k)`,
`-1` ids where fewer than k matched.

### `search_text(text: str, k: int = 10, filter: Optional[Mapping[str, Any]] = None) -> list[tuple[int, float]]`

BM25 over the text field; scores are negated BM25.

### `search_sparse(query: Sparse, k: int = 10, filter: Optional[Mapping[str, Any]] = None) -> list[tuple[int, float]]`

Dot product over the sparse field; scores are negated dot products.

### `backup(dir: str) -> str`

Consistent point-in-time copy into `dir`; returns the copy's path.

### `refresh() -> None`

Read-only handles: pick up the writer's changes (reads do this themselves).

### `sync() -> None`

fsync the log.

### `checkpoint() -> None`

Build the graph index over every record and compact the log.

### `index_stats() -> Optional[dict[str, Any]]`

`nodes`, `pq_subspaces`, `tombstones`, `storage`; None before the first checkpoint.

### `stats() -> dict[str, int]`

Operation counters since open (searches by regime, writes,
checkpoints and their last duration, reader refreshes).

