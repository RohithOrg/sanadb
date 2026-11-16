"""ann-benchmarks adapter for sanadb.

Copy this directory to `ann_benchmarks/algorithms/sanadb/` in a checkout of
https://github.com/erikbern/ann-benchmarks and run, for example:

    python run.py --algorithm sanadb --dataset sift-128-euclidean --local

The store is built in a temp directory, checkpointed once (that is the index
build), and queried single-threaded like every other algorithm there.

Set `SANADB_ANNB_KEEP=<dir>` to build each configuration's store under that
directory instead and reuse it on later runs: a store whose index already
holds every vector skips the build, so a query-only rerun of a 1M set takes
seconds instead of a six-minute build. The build time reported for a reused
store is then not a build.
"""

import os
import shutil
import tempfile

import numpy as np
import sanadb

from ..base.module import BaseANN


class Sanadb(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self._param = method_param
        self._dir = None
        self._keep = False
        self._db = None
        self._beam = 64

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        self._dim = X.shape[1]
        keep = os.environ.get("SANADB_ANNB_KEEP")
        if keep:
            tag = "_".join(f"{k}{v}" for k, v in sorted(self._param.items()))
            self._dir = os.path.join(keep, f"{self._metric}_{len(X)}x{self._dim}_{tag}")
            os.makedirs(self._dir, exist_ok=True)
            self._keep = True
        else:
            self._dir = tempfile.mkdtemp(prefix="sanadb-annb-")
            self._keep = False
        self._path = os.path.join(self._dir, "index.sana")
        self._open(beam=self._beam)
        stats = self._db.index_stats()
        if stats and stats["nodes"] == len(X) and stats["tombstones"] == 0:
            print(f"sanadb: reusing {self._path}")
            return
        self._db.insert_many(list(range(len(X))), X)
        self._db.checkpoint()

    def _open(self, beam):
        self._db = None  # release the file lock before reopening
        self._db = sanadb.Db(
            self._path,
            dim=self._dim,
            metric=self._metric,
            search_beam=beam,
            max_search_beam=beam,
            io_width=self._param.get("io_width", 4),
            pq=self._param.get("pq", True),
            max_degree=self._param.get("R", 32),
            build_beam=self._param.get("L", 100),
            alpha=self._param.get("alpha", 1.2),
        )

    def set_query_arguments(self, beam):
        # Search parameters are fixed per Db, so reopen; the index is on disk
        # and the page cache keeps this cheap.
        self._beam = beam
        self._open(beam=beam)
        self.name = "sanadb(%s, beam=%d)" % (self._param, beam)

    def query(self, v, n):
        return np.array([h[0] for h in self._db.search(np.asarray(v, dtype=np.float32), k=n)], dtype=np.int64)

    def done(self):
        self._db = None
        if self._dir and not self._keep:
            shutil.rmtree(self._dir, ignore_errors=True)

    def __str__(self):
        return self.name
