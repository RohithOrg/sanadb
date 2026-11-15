import threading

import numpy as np
import pytest

import sanadb


@pytest.fixture
def db(tmp_path):
    return sanadb.Db(str(tmp_path / "v.sana"), dim=8)


def test_insert_search_get_delete(db):
    db.insert(1, np.zeros(8, dtype=np.float32), {"kind": "a"})
    db.insert(2, [1.0] + [0.0] * 7, {"kind": "b"})
    db.insert(3, np.full(8, 5.0), {"kind": "a"})  # float64 accepted
    assert len(db) == 3

    hits = db.search([0.9] + [0.0] * 7, k=2)
    assert [h[0] for h in hits] == [2, 1]
    assert hits[0][1] == pytest.approx(0.01, abs=1e-6)

    hits = db.search([0.9] + [0.0] * 7, k=2, filter={"kind": "a"})
    assert [h[0] for h in hits] == [1, 3]

    hits = db.search([0.9] + [0.0] * 7, k=5, filter=lambda m: m["kind"] == "b")
    assert [h[0] for h in hits] == [2]

    vec, meta = db.get(3)
    assert vec.dtype == np.float32 and vec.tolist() == [5.0] * 8
    assert meta == {"kind": "a"}

    assert db.delete(2) is True
    assert db.delete(2) is False
    assert db.get(2) is None
    assert len(db) == 2


def test_dim_mismatch_is_value_error(db):
    with pytest.raises(ValueError):
        db.insert(1, [0.0] * 7)
    with pytest.raises(ValueError):
        db.search([0.0] * 9)


def test_bad_metric(tmp_path):
    with pytest.raises(ValueError):
        sanadb.Db(str(tmp_path / "x.sana"), dim=4, metric="hamming")


def test_cosine_and_nan(tmp_path):
    db = sanadb.Db(str(tmp_path / "c.sana"), dim=3, metric="cosine")
    db.insert(1, [2.0, 0.0, 0.0])
    db.insert(2, [0.0, 1.0, 0.0])
    hits = db.search([9.0, 0.0, 0.0], k=2)
    assert [h[0] for h in hits] == [1, 2]
    assert hits[0][1] == pytest.approx(-1.0)
    assert db.get(1)[0].tolist() == [1.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        db.insert(3, [float("nan"), 0.0, 0.0])


def test_insert_many_checkpoint_reopen(tmp_path):
    path = str(tmp_path / "v.sana")
    rng = np.random.default_rng(0)
    n, dim = 3000, 16
    xs = rng.random((n, dim), dtype=np.float32)
    db = sanadb.Db(path, dim=dim)
    db.insert_many(list(range(n)), xs, [{"i": str(i)} for i in range(n)])
    assert len(db) == n
    assert db.index_stats() is None

    db.checkpoint()
    stats = db.index_stats()
    assert stats["nodes"] == n and stats["pq_subspaces"] == 4 and stats["tombstones"] == 0
    assert db.delta_len == 0

    # recall@10 against brute force on real random queries
    hit = 0
    for qi in range(50):
        q = rng.random(dim, dtype=np.float32)
        truth = np.argsort(((xs - q) ** 2).sum(axis=1))[:10]
        got = [h[0] for h in db.search(q, k=10)]
        hit += len(set(got) & set(truth.tolist()))
    assert hit / 500 >= 0.9

    del db
    db = sanadb.Db(path, dim=dim)
    assert len(db) == n
    vec, meta = db.get(42)
    assert np.allclose(vec, xs[42]) and meta == {"i": "42"}


def test_callable_filter_error_propagates(db):
    db.insert(1, [0.0] * 8, {"a": "1"})
    with pytest.raises(KeyError):
        db.search([0.0] * 8, k=1, filter=lambda m: m["missing"] == "x")


def test_threads_write_during_checkpoint(tmp_path):
    dim, n = 16, 20000
    db = sanadb.Db(str(tmp_path / "v.sana"), dim=dim)
    rng = np.random.default_rng(1)
    db.insert_many(list(range(n)), rng.random((n, dim), dtype=np.float32))
    db.checkpoint()

    t = threading.Thread(target=db.checkpoint)
    t.start()
    for i in range(n, n + 300):
        db.insert(i, rng.random(dim, dtype=np.float32), {"late": "1"})
    for i in range(0, 100):
        db.delete(i)
    t.join()

    assert len(db) == n + 300 - 100
    assert all(db.get(i) is None for i in range(100))
    assert all(db.get(i)[1] == {"late": "1"} for i in range(n, n + 300))
    db.checkpoint()
    assert db.delta_len == 0 and db.index_stats()["tombstones"] == 0
    assert len(db) == n + 300 - 100
