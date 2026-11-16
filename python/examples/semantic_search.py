"""Semantic search over your own text with a sentence-transformers model.

    pip install "sanadb[studio]" sentence-transformers
    python python/examples/semantic_search.py

The model turns text into 384-dimensional vectors; sanadb stores them with
the text and metadata, searches by meaning, filters on metadata, and
fuses the vector result with BM25 keyword matching. The model's name is
recorded in the store so a reader a year from now knows what produced the
vectors. Open the store afterwards in studio with the same model:

    python python/examples/embed_server.py &          # the model as a local endpoint
    sanadb-studio docs.sana --embed-url http://127.0.0.1:8089/v1/embeddings --embed-model all-MiniLM-L6-v2
"""
import os
import tempfile

import sanadb
from sentence_transformers import SentenceTransformer

MODEL = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL)

path = os.environ.get("SANADB_PATH", os.path.join(tempfile.mkdtemp(), "docs.sana"))
db = sanadb.Db(
    path,
    dim=model.get_sentence_embedding_dimension(),
    metric="cosine",
    text_field="body",
    embedding_function=lambda texts: model.encode(texts, normalize_embeddings=True),
)
db.set_property("embedding_model", MODEL)

docs = [
    ("The borrow checker rejects aliasing of mutable references", {"lang": "rust", "year": 2021}),
    ("Lifetimes tie a reference to the scope that owns the data", {"lang": "rust", "year": 2022}),
    ("asyncio runs coroutines on a single-threaded event loop", {"lang": "python", "year": 2020}),
    ("Tokio's select! cancels the branches that did not complete", {"lang": "rust", "year": 2023}),
    ("Goroutines are multiplexed onto OS threads by the Go scheduler", {"lang": "go", "year": 2019}),
    ("Garbage collection pauses are tuned with GOGC", {"lang": "go", "year": 2020}),
    ("Pattern matching on tagged unions catches missing cases at compile time", {"lang": "rust", "year": 2022}),
    ("The GIL serialises bytecode execution across threads", {"lang": "python", "year": 2018}),
]
ids = db.add([t for t, _ in docs], metas=[m for _, m in docs])
print(f"{len(ids)} documents embedded with {MODEL} into {path}")

def show(title, hits):
    print(f"\n{title}")
    for id_, score in hits:
        _, meta = db.get(id_)
        print(f"  {score:+.3f}  [{meta['lang']}]  {meta['body']}")

# Search by meaning: the query shares no words with the sentence it finds.
show("why can't I have two mutable references", db.query(["why can't I have two mutable references"], k=3)[0])
show("exhaustive match on an enum", db.query(["exhaustive match on an enum"], k=3)[0])

# Meaning plus a metadata filter, evaluated on the index columns.
show("running many tasks on one thread, Python only", db.query(["running many tasks on one thread"], k=3, filter={"lang": "python"})[0])

# Hybrid: the vector result fused with BM25 on the text, for exact terms
# that embed poorly (identifiers, flags, names).
q = model.encode(["tuning the collector"], normalize_embeddings=True)[0]
show("hybrid: 'GOGC' as a keyword with a semantic query", db.search(q, k=3, text="GOGC"))

db.checkpoint()
print(f"\nindex built: {db.index_stats()}; properties: {db.properties()}")
print("scores are negative cosine similarity: -1 identical, lower is closer")
