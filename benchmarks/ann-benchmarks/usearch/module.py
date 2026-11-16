import numpy as np
from usearch.index import Index

from ..base.module import BaseANN


class USearch(BaseANN):
    """usearch HNSW, f32 vectors, single query thread (same footing as the hnswlib adapter)."""

    def __init__(self, metric, method_param):
        self.metric = {"angular": "cos", "euclidean": "l2sq"}[metric]
        self.method_param = method_param

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.index = Index(
            ndim=X.shape[1],
            metric=self.metric,
            dtype="f32",
            connectivity=self.method_param["M"],
            expansion_add=self.method_param["efConstruction"],
        )
        self.index.add(np.arange(len(X), dtype=np.uint64), X)

    def set_query_arguments(self, ef):
        self.index.expansion_search = ef
        self.name = "usearch (%s, 'efQuery': %s)" % (self.method_param, ef)

    def query(self, v, n):
        return self.index.search(np.asarray(v, dtype=np.float32), n, threads=1, exact=False).keys

    def freeIndex(self):
        del self.index
