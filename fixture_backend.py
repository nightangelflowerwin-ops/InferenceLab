"""Explicitly fake chat/embedding backend for local proxy demonstrations.

It echoes input and uses hashed character bigrams. It is not an LLM or a semantic
embedding model and cannot demonstrate model quality or inference performance.
"""
import argparse
from collections import Counter
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import time


def fixture_embedding(text, dimensions=128):
    values = [0] * dimensions
    normalized = " ".join(text.lower().split())
    for offset in range(max(1, len(normalized) - 1)):
        gram = normalized[offset:offset + 2]
        bucket = int.from_bytes(hashlib.sha256(gram.encode()).digest()[:4], "little") % dimensions
        values[bucket] += 1
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


def handler(name):
    class Fixture(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, result):
            data = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                return self.respond(200, {"fixture": True, "name": name})
            if self.path == "/v1/models":
                return self.respond(200, {"data": [{"id": name, "object": "model"}], "fixture": True})
            self.respond(404, {"error": "Unknown path"})

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1024 * 1024:
                    raise ValueError("Bounded body required")
                self.connection.settimeout(10)
                body = json.loads(self.rfile.read(size))
                if self.path == "/v1/embeddings":
                    text = body["input"]
                    if not isinstance(text, str):
                        raise ValueError("String input required")
                    return self.respond(200, {"data": [{"embedding": fixture_embedding(text), "index": 0}],
                        "model": body.get("model", name), "fixture": True, "embedding_kind": "hashed_character_bigrams"})
                if self.path != "/v1/chat/completions":
                    return self.respond(404, {"error": "Unknown path"})
                query = body["messages"][-1]["content"]
                if not isinstance(query, str):
                    raise ValueError("Fixture only accepts plain text")
                content = f"[{name} fixture, no model inference] " + query
            except (ValueError, KeyError, IndexError, TypeError):
                return self.respond(400, {"error": "Invalid fixture request"})
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for word in content.split():
                    self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {"content": word + " "}}]}) + "\n\n").encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                return
            self.respond(200, {"id": "fixture-" + name, "object": "chat.completion", "created": int(time.time()),
                "model": body.get("model", name), "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"}], "fixture": True})
    return Fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--name", default="local-fixture")
    args = parser.parse_args()
    print(f"FAKE backend: http://127.0.0.1:{args.port}; no model loaded", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), handler(args.name)).serve_forever()


if __name__ == "__main__":
    main()
