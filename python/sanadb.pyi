"""Type stubs for the sanadb extension module."""
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

__version__: str

Sparse = tuple[npt.ArrayLike, npt.ArrayLike]
"""A sparse vector as `(indices, weights)`; returned as two numpy arrays.
One metadata key per store may hold them; `search_sparse` searches it."""
Metadata = dict[str, Union[str, int, float, bool, Sparse]]
Filter = Union[Mapping[str, Any], Callable[[Metadata], bool]]

class Db:
    """A disk-first vector + metadata store at `path` (a file; the index
    lives beside it as `<path>.idx.<id>`)."""

    def __init__(
        self,
        path: str,
        dim: int,
        *,
        metric: str = "l2",
        search_beam: int = 64,
        max_search_beam: int = 4096,
        io_width: int = 4,
        sync_every_write: bool = False,
        pq: bool = True,
        max_degree: int = 32,
        build_beam: int = 100,
        alpha: float = 1.2,
        read_only: bool = False,
        storage: str = "f32",
        text_field: Optional[str] = None,
        embedding_function: Optional[Callable[[list[str]], npt.ArrayLike]] = None,
        auto_checkpoint_delta: int = 0,
        auto_checkpoint_garbage: int = 0,
    ) -> None:
        """`embedding_function` maps a list of texts to an (n, dim) array; it
        enables `add()` and `query()`, like Chroma's embedding functions.
        `auto_checkpoint_delta` checkpoints once that many records sit in the
        tail; `auto_checkpoint_garbage` once the log holds that many more
        entries than live records. 0 is off."""
    @property
    def dim(self) -> int: ...
    @property
    def delta_len(self) -> int:
        """Records written since the last checkpoint (searched exactly)."""
    def __len__(self) -> int: ...
    def insert(self, id: int, vector: npt.ArrayLike, meta: Optional[Metadata] = None) -> None:
        """Insert or replace one record. A key's type is fixed by its first value."""
    def insert_many(
        self,
        ids: Sequence[int],
        vectors: npt.ArrayLike,
        metas: Optional[Sequence[Optional[Metadata]]] = None,
    ) -> None:
        """Insert `(n, dim)` vectors atomically: all of them or none."""
    def add(
        self,
        documents: Sequence[str],
        ids: Optional[Sequence[int]] = None,
        metas: Optional[Sequence[Optional[Metadata]]] = None,
    ) -> list[int]:
        """Embed the documents and insert them atomically; the text is kept
        under the text field (or `text`). Returns the ids used."""
    def query(
        self, texts: Sequence[str], k: int = 10, filter: Optional[Mapping[str, Any]] = None
    ) -> list[list[tuple[int, float]]]:
        """Embed each text and search: one `[(id, score), ...]` per text."""
    def set_property(self, key: str, value: str) -> None:
        """A string kept with the store, such as the embedding model's name;
        visible to readers at once, kept across checkpoints. Empty removes it."""
    def property(self, key: str) -> Optional[str]: ...
    def properties(self) -> dict[str, str]: ...
    def delete(self, id: int) -> bool: ...
    def delete_many(self, ids: Sequence[int]) -> None: ...
    def get(self, id: int) -> Optional[tuple[npt.NDArray[np.float32], Metadata]]: ...
    def search(
        self,
        query: npt.ArrayLike,
        k: int = 10,
        filter: Optional[Filter] = None,
        text: Optional[str] = None,
        sparse: Optional[Sparse] = None,
    ) -> list[tuple[int, float]]:
        """k nearest `(id, score)`, closest first; lower score is closer.

        `filter` is a dict (`{"k": v}`, `{"k": {"$gte": 3}}`, `$eq $ne $in
        $lt $le $gt $ge $prefix $contains $exists $and $or $not`) or a
        callable on the metadata. With `text`, the result is the hybrid of
        the vector search and a BM25 search on the store's text field,
        fused by reciprocal rank; with `sparse`, of the vector search and a
        dot-product search on the sparse field."""
    def search_many(
        self,
        queries: npt.ArrayLike,
        k: int = 10,
        filter: Optional[Mapping[str, Any]] = None,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        """`(n, dim)` queries on all cores: `(ids, scores)` of shape `(n, k)`,
        `-1` ids where fewer than k matched."""
    def search_text(
        self, text: str, k: int = 10, filter: Optional[Mapping[str, Any]] = None
    ) -> list[tuple[int, float]]:
        """BM25 over the text field; scores are negated BM25."""
    def search_sparse(
        self, query: Sparse, k: int = 10, filter: Optional[Mapping[str, Any]] = None
    ) -> list[tuple[int, float]]:
        """Dot product over the sparse field; scores are negated dot products."""
    def backup(self, dir: str) -> str:
        """Consistent point-in-time copy into `dir`; returns the copy's path."""
    def refresh(self) -> None:
        """Read-only handles: pick up the writer's changes (reads do this themselves)."""
    def sync(self) -> None:
        """fsync the log."""
    def checkpoint(self) -> None:
        """Build the graph index over every record and compact the log."""
    def index_stats(self) -> Optional[dict[str, Any]]:
        """`nodes`, `pq_subspaces`, `tombstones`, `storage`; None before the first checkpoint."""
    def stats(self) -> dict[str, int]:
        """Operation counters since open (searches by regime, writes,
        checkpoints and their last duration, reader refreshes)."""
