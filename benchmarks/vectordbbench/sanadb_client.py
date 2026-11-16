"""VectorDBBench client for sanadb (https://github.com/zilliztech/VectorDBBench).

VectorDBBench drives a database through a small abstract API: a config with
connection settings, a case config with index and search parameters, and a
client with `insert_embeddings`, `search_embedding`, `optimize` and a
`ready_to_load` check. This module implements that API over the sanadb
Python package. Register it by copying the file into
`vectordb_bench/backend/clients/sanadb/` and adding a `DB.Sanadb` entry in
`vectordb_bench/backend/clients/__init__.py` as for the other embedded
clients (see their `chroma` client for the shape); then

    vectordbbench sanadb --case-type Performance768D1M --path /tmp/sanadb-bench

The filter cases pass `filters={"id": N, "metadata": ">=N"}`, meaning
`id >= N`; that becomes a `{"id": {"$gte": N}}` sanadb filter on an `Int`
column, so the filtered cases run on the exact-scan or filtered-walk regime
rather than post-filtering.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterable

import numpy as np

try:  # the harness's base classes; absent when this file is imported standalone
    from vectordb_bench.backend.clients.api import DBCaseConfig, DBConfig, MetricType, VectorDB
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - lets the module be imported without the harness
    class VectorDB:  # type: ignore[no-redef]
        pass

    class DBConfig:  # type: ignore[no-redef]
        pass

    class DBCaseConfig:  # type: ignore[no-redef]
        pass

    class MetricType:  # type: ignore[no-redef]
        L2 = "L2"
        COSINE = "COSINE"
        IP = "IP"

    BaseModel = object  # type: ignore[assignment]

import sanadb


class SanadbConfig(DBConfig, BaseModel):  # type: ignore[misc]
    """Where the store lives. One file per collection under `path`."""

    db_label: str = ""
    path: str = "/tmp/sanadb-bench"

    def to_dict(self) -> dict:
        return {"path": self.path}


class SanadbIndexConfig(DBCaseConfig, BaseModel):  # type: ignore[misc]
    metric_type: MetricType | None = None
    max_degree: int = 32
    build_beam: int = 100
    search_beam: int = 64
    pq: bool = True
    storage: str = "f32"

    def index_param(self) -> dict:
        return {"max_degree": self.max_degree, "build_beam": self.build_beam, "pq": self.pq, "storage": self.storage}

    def search_param(self) -> dict:
        return {"search_beam": self.search_beam}


class Sanadb(VectorDB):  # type: ignore[misc]
    def __init__(self, dim: int, db_config: dict, db_case_config: SanadbIndexConfig, collection_name: str = "bench", drop_old: bool = False, **kwargs):
        self.dim = dim
        self.case = db_case_config
        os.makedirs(db_config["path"], exist_ok=True)
        self.path = os.path.join(db_config["path"], f"{collection_name}.sana")
        metric = {MetricType.L2: "l2", MetricType.COSINE: "cosine", MetricType.IP: "dot"}.get(getattr(db_case_config, "metric_type", None), "l2")
        self.metric = metric
        if drop_old:
            for name in os.listdir(db_config["path"]):
                if name.startswith(f"{collection_name}.sana"):
                    os.remove(os.path.join(db_config["path"], name))
        self.db: sanadb.Db | None = None

    def _open(self) -> sanadb.Db:
        p = self.case.index_param()
        return sanadb.Db(
            self.path,
            dim=self.dim,
            metric=self.metric,
            search_beam=self.case.search_beam,
            max_search_beam=max(self.case.search_beam, 4096),
            pq=p["pq"],
            max_degree=p["max_degree"],
            build_beam=p["build_beam"],
            storage=p["storage"],
        )

    @contextmanager
    def init(self):
        self.db = self._open()
        try:
            yield
        finally:
            self.db = None

    def ready_to_load(self) -> bool:
        return True

    def optimize(self, data_size: int | None = None) -> None:
        """Build the graph index: VectorDBBench calls this once after loading."""
        assert self.db is not None
        self.db.checkpoint()

    def insert_embeddings(self, embeddings: Iterable[list[float]], metadata: list[int], **kwargs) -> tuple[int, Exception | None]:
        assert self.db is not None
        try:
            vecs = np.asarray(list(embeddings), dtype=np.float32)
            ids = [int(i) for i in metadata]
            # VectorDBBench's filter cases select on the integer id, so the
            # id is stored as an Int attribute too.
            self.db.insert_many(ids, vecs, [{"id": i} for i in ids])
            return len(ids), None
        except Exception as e:  # the harness wants the error returned, not raised
            return 0, e

    def search_embedding(self, query: list[float], k: int = 100, filters: dict | None = None, timeout: int | None = None, **kwargs) -> list[int]:
        assert self.db is not None
        q = np.asarray(query, dtype=np.float32)
        flt = None
        if filters and "id" in filters:
            flt = {"id": {"$gte": int(filters["id"])}}
        return [i for i, _ in self.db.search(q, k=k, filter=flt)]
