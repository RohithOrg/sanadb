"""Dense retrieval on BEIR SciFact with a real embedding model, end to end.

    pip install sanadb sentence-transformers
    python benchmarks/beir/scifact_dense.py data/beir/scifact

Encodes the 5,183 documents and 300 test queries with all-MiniLM-L6-v2,
stores the document vectors in sanadb (cosine), checkpoints, and reports:

- recall@10 of the graph search against brute-force cosine over the same
  vectors (does the index lose anything the model found), per beam;
- nDCG@10 against the human relevance labels, for brute force and for
  sanadb (does the whole pipeline retrieve), next to the published number
  for this model on this dataset (0.645 in the sentence-transformers
  evaluations of all-MiniLM-L6-v2).
"""
import json
import math
import os
import sys
import tempfile
import time

import numpy as np
import sanadb
from sentence_transformers import SentenceTransformer

MODEL = "all-MiniLM-L6-v2"


def ndcg_at_10(ranked, rel):
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def main(root):
    corpus = [json.loads(l) for l in open(f"{root}/corpus.jsonl")]
    queries = {q["_id"]: q["text"] for q in (json.loads(l) for l in open(f"{root}/queries.jsonl"))}
    qrels = {}
    with open(f"{root}/qrels/test.tsv") as f:
        next(f)
        for line in f:
            q, d, s = line.strip().split("\t")
            qrels.setdefault(q, {})[d] = int(s)
    doc_ids = [d["_id"] for d in corpus]
    texts = [(d.get("title", "") + " " + d["text"]).strip() for d in corpus]
    qids = list(qrels)
    model = SentenceTransformer(MODEL)
    t = time.time()
    D = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    Q = model.encode([queries[q] for q in qids], normalize_embeddings=True, show_progress_bar=False)
    print(f"{len(texts)} documents and {len(qids)} queries encoded with {MODEL} ({D.shape[1]} dims) in {time.time() - t:.0f}s")

    # Brute-force truth over the same vectors: the best any index can do.
    truth = np.argsort(-(Q @ D.T), axis=1)[:, :10]

    tmp = tempfile.mkdtemp()
    db = sanadb.Db(os.path.join(tmp, "scifact.sana"), dim=D.shape[1], metric="cosine")
    db.set_property("embedding_model", MODEL)
    db.insert_many(list(range(len(texts))), D.astype(np.float32), [{"doc": d} for d in doc_ids])
    t = time.time()
    db.checkpoint()
    print(f"sanadb: inserted and indexed in {time.time() - t:.1f}s; {db.index_stats()}")

    nd_truth = sum(ndcg_at_10([doc_ids[i] for i in truth[k]], qrels[q]) for k, q in enumerate(qids)) / len(qids)
    print(f"\n{'beam':>5} {'recall@10 vs brute force':>26} {'nDCG@10':>8} {'µs/query':>9}")
    print(f"{'exact':>5} {'1.000':>26} {nd_truth:8.3f} {'':>9}   (brute force over the model's vectors)")
    for beam in (16, 32, 64, 128):
        db = sanadb.Db(os.path.join(tmp, "scifact.sana"), dim=D.shape[1], metric="cosine", search_beam=beam, max_search_beam=beam, read_only=True)
        hit, nd, t = 0, 0.0, time.time()
        for k, q in enumerate(qids):
            ids = [i for i, _ in db.search(Q[k].astype(np.float32), k=10)]
            hit += len(set(ids) & set(truth[k].tolist()))
            nd += ndcg_at_10([doc_ids[i] for i in ids], qrels[q])
        us = (time.time() - t) / len(qids) * 1e6
        print(f"{beam:>5} {hit / (10 * len(qids)):>26.3f} {nd / len(qids):8.3f} {us:9.0f}")
    print(f"\npublished nDCG@10 for {MODEL} on SciFact: 0.645 (sentence-transformers model card evaluations)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/beir/scifact")
