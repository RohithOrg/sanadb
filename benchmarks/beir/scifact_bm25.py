"""BM25 on BEIR SciFact: sanadb's search_text against the rank_bm25 reference.

Both use sanadb's tokeniser (lowercase, split on non-alphanumerics) so the
comparison isolates the scoring and the index, not the text processing.
Reports nDCG@10 and Recall@100 over the test qrels, and the per-query time.

    python benchmarks/beir/scifact_bm25.py data/beir/scifact
"""
import json
import math
import os
import re
import sys
import tempfile
import time

import numpy as np
import sanadb
from rank_bm25 import BM25Okapi

TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(s: str) -> list[str]:
    # Mirrors sanadb::text::tokenize: alphanumeric runs, lowercased. The Rust
    # tokeniser splits on anything that is not char::is_alphanumeric, which
    # treats '_' as a separator; [^\W_]+ does the same.
    return [t.lower() for t in TOKEN.findall(s) if len(t) <= 64]


def ndcg_at_10(ranked: list[str], rel: dict[str, int]) -> float:
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def main(root: str) -> None:
    corpus = [json.loads(l) for l in open(f"{root}/corpus.jsonl")]
    queries = {q["_id"]: q["text"] for q in (json.loads(l) for l in open(f"{root}/queries.jsonl"))}
    qrels: dict[str, dict[str, int]] = {}
    with open(f"{root}/qrels/test.tsv") as f:
        next(f)
        for line in f:
            q, d, s = line.strip().split("\t")
            qrels.setdefault(q, {})[d] = int(s)
    docs = [(d["_id"], (d.get("title", "") + " " + d["text"]).strip()) for d in corpus]
    ids = [d for d, _ in docs]
    print(f"{len(docs)} documents, {len(qrels)} test queries")

    # Reference: rank_bm25 with the same tokens (k1=1.2, b=0.75 like sanadb).
    t = time.time()
    ref = BM25Okapi([tokenize(txt) for _, txt in docs], k1=1.2, b=0.75)
    print(f"rank_bm25 built in {time.time() - t:.1f}s")

    # sanadb: vectors are irrelevant here, use a 4-dim constant.
    tmp = tempfile.mkdtemp()
    db = sanadb.Db(os.path.join(tmp, "scifact.sana"), dim=4, text_field="body")
    t = time.time()
    db.insert_many(list(range(len(docs))), np.zeros((len(docs), 4), dtype=np.float32), [{"body": txt, "doc": d} for d, txt in docs])
    db.checkpoint()
    print(f"sanadb indexed in {time.time() - t:.1f}s")

    res = {"sanadb": [0.0, 0.0, 0.0], "rank_bm25": [0.0, 0.0, 0.0]}
    for qid, rel in qrels.items():
        qtext = queries[qid]
        t = time.time()
        hits = db.search_text(qtext, k=100)
        res["sanadb"][2] += time.time() - t
        ranked = [ids[i] for i, _ in hits]
        res["sanadb"][0] += ndcg_at_10(ranked, rel)
        res["sanadb"][1] += len(set(ranked[:100]) & set(rel)) / len(rel)
        t = time.time()
        scores = ref.get_scores(tokenize(qtext))
        top = np.argsort(-scores)[:100]
        res["rank_bm25"][2] += time.time() - t
        ranked = [ids[i] for i in top if scores[i] > 0]
        res["rank_bm25"][0] += ndcg_at_10(ranked, rel)
        res["rank_bm25"][1] += len(set(ranked[:100]) & set(rel)) / len(rel)
    n = len(qrels)
    print(f"{'system':<10} {'nDCG@10':>8} {'R@100':>7} {'ms/query':>9}")
    for name, (nd, r, secs) in res.items():
        print(f"{name:<10} {nd / n:8.4f} {r / n:7.4f} {1000 * secs / n:9.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/beir/scifact")
