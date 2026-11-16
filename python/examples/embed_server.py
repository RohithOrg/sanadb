"""A sentence-transformers model as a local OpenAI-compatible embeddings
endpoint, for `sanadb-studio --embed-url` or the Rust `OpenAiEmbedder`.

    pip install sentence-transformers
    python python/examples/embed_server.py            # http://127.0.0.1:8089/v1/embeddings
    sanadb-studio docs.sana --embed-url http://127.0.0.1:8089/v1/embeddings --embed-model all-MiniLM-L6-v2

Loads the model once; each request embeds the `input` strings and answers
in the shape OpenAI, Ollama and the others use: {"data": [{"index": i,
"embedding": [...]}, ...]}. Binds to localhost only.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from sentence_transformers import SentenceTransformer

MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
PORT = int(os.environ.get("EMBED_PORT", "8089"))
model = SentenceTransformer(MODEL)
print(f"{MODEL}: {model.get_sentence_embedding_dimension()} dimensions at http://127.0.0.1:{PORT}/v1/embeddings", file=sys.stderr)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/embeddings":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
        vecs = model.encode(texts, normalize_embeddings=True)
        out = {"model": MODEL, "data": [{"index": i, "embedding": v.tolist()} for i, v in enumerate(vecs)]}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass


HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
