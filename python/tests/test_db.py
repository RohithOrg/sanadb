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


def test_search_many_matches_single(tmp_path):
    import numpy as np
    db = sanadb.Db(str(tmp_path / "many.sana"), dim=8)
    rng = np.random.default_rng(3)
    vecs = rng.random((2000, 8), dtype=np.float32)
    db.insert_many(list(range(2000)), vecs, [{"even": "yes"} if i % 2 == 0 else None for i in range(2000)])
    db.checkpoint()
    qs = rng.random((17, 8), dtype=np.float32)
    ids, scores = db.search_many(qs, k=5)
    assert ids.shape == (17, 5) and scores.shape == (17, 5)
    for row in range(17):
        single = db.search(qs[row], k=5)
        assert [i for i, _ in single] == ids[row].tolist()
        assert np.allclose([s for _, s in single], scores[row])
    ids, _ = db.search_many(qs, k=3, filter={"even": "yes"})
    assert (ids % 2 == 0).all()


def test_metadata_is_typed(db):
    db.insert(1, np.ones(8, dtype=np.float32), {"doc": "a.md", "year": 2024, "score": 0.5, "public": True})
    _, m = db.get(1)
    assert m == {"doc": "a.md", "year": 2024, "score": 0.5, "public": True}
    assert type(m["year"]) is int and type(m["public"]) is bool and type(m["score"]) is float
    with pytest.raises(TypeError):
        db.insert(2, np.ones(8, dtype=np.float32), {"year": "2024"})
    with pytest.raises(TypeError):
        db.insert(3, np.ones(8, dtype=np.float32), {"tags": ["a", "b"]})
    db.insert(4, np.zeros(8, dtype=np.float32), {"year": 1999})
    assert [i for i, _ in db.search(np.zeros(8, dtype=np.float32), k=1, filter={"year": 1999})] == [4]
    assert db.search(np.zeros(8, dtype=np.float32), k=1, filter={"year": "1999"}) == []


def test_filter_dict_operators(tmp_path):
    db = sanadb.Db(str(tmp_path / "f.sana"), dim=4)
    rng = np.random.default_rng(5)
    n = 3000
    vecs = rng.random((n, 4), dtype=np.float32)
    langs = ["en", "de", "fr"]
    metas = [{"lang": langs[i % 3], "year": 2000 + i % 20, "score": (i % 100) / 100, "rare": True} if i % 300 == 1 else {"lang": langs[i % 3], "year": 2000 + i % 20, "score": (i % 100) / 100} for i in range(n)]
    db.insert_many(list(range(n)), vecs, metas)
    db.checkpoint()
    q = vecs[17]

    def brute(pred, k=5):
        d = ((vecs - q) ** 2).sum(axis=1)
        ids = [i for i in np.argsort(d, kind="stable") if pred(metas[i])]
        return ids[:k]

    # Exact regime (10 matches through the posting list), equality and $exists.
    assert [i for i, _ in db.search(q, k=5, filter={"rare": True})] == brute(lambda m: m.get("rare") is True)
    assert [i for i, _ in db.search(q, k=5, filter={"rare": {"$exists": True}, "year": {"$gte": 2005, "$lt": 2015}})] == brute(
        lambda m: "rare" in m and 2005 <= m["year"] < 2015
    )
    assert [i for i, _ in db.search(q, k=5, filter={"$and": [{"rare": True}, {"lang": {"$in": ["de", "fr"]}}]})] == brute(
        lambda m: m.get("rare") is True and m["lang"] in ("de", "fr")
    )
    # Walk regime: every hit passes, k results come back.
    hits = db.search(q, k=5, filter={"lang": "de", "score": {"$lt": 0.5}})
    assert len(hits) == 5 and all(metas[i]["lang"] == "de" and metas[i]["score"] < 0.5 for i, _ in hits)
    hits = db.search(q, k=5, filter={"$or": [{"lang": "en"}, {"year": 2019}], "$not": {"lang": {"$prefix": "f"}}})
    assert len(hits) == 5 and all((metas[i]["lang"] == "en" or metas[i]["year"] == 2019) and not metas[i]["lang"].startswith("f") for i, _ in hits)
    assert db.search(q, k=5, filter={"lang": "zz"}) == []
    assert db.search(q, k=5, filter={"year": "2001"}) == [], "wrong type matches nothing"
    with pytest.raises(ValueError):
        db.search(q, k=5, filter={"year": {"$between": [1, 2]}})
    ids, _ = db.search_many(vecs[:3], k=5, filter={"rare": True})
    assert ids.shape == (3, 5) and all(metas[i].get("rare") is True for i in ids.ravel() if i >= 0)


def test_read_only_follows_writer(tmp_path):
    p = str(tmp_path / "ro.sana")
    w = sanadb.Db(p, dim=4)
    w.insert(1, np.ones(4, dtype=np.float32), {"k": "a"})
    r = sanadb.Db(p, dim=4, read_only=True)
    assert len(r) == 1 if hasattr(r, "__len__") else r.get(1) is not None
    w.insert(2, np.zeros(4, dtype=np.float32), {"k": "b"})
    r.refresh()
    assert r.get(2)[1] == {"k": "b"}
    w.checkpoint()
    assert [i for i, _ in r.search(np.zeros(4, dtype=np.float32), k=1)] == [2]
    with pytest.raises(PermissionError):
        r.insert(3, np.ones(4, dtype=np.float32))


def test_insert_many_is_atomic_and_delete_many(tmp_path):
    db = sanadb.Db(str(tmp_path / "b.sana"), dim=4)
    vecs = np.ones((3, 4), dtype=np.float32)
    with pytest.raises(TypeError):
        db.insert_many([1, 2, 3], vecs, [{"n": 1}, {"n": "x"}, None])
    assert db.get(1) is None
    db.insert_many([1, 2, 3], vecs, [{"n": 1}, {"n": 2}, None])
    assert len(db) == 3
    db.delete_many([1, 3, 99])
    assert len(db) == 1 and db.get(2) is not None
