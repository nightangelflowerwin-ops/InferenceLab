import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from complexity_gateway import ComplexityPolicy, evaluate, gateway_handler
from semantic_cache import SemanticCache, SQLiteVectors, SQLiteResponses, HTTPEmbedder


class GatewayTests(unittest.TestCase):
    def test_policy_length_entropy_tools_and_reasoning(self):
        policy = ComplexityPolicy()
        def body(text):
            return {"messages": [{"role": "user", "content": text}]}
        self.assertEqual(policy.classify(body("Hello"))["route"], "local")
        self.assertEqual(policy.classify(body("Prove this theorem"))["route"], "complex")
        self.assertIn("length", policy.classify(body("x" * 3000))["reasons"])
        self.assertIn("byte_entropy", policy.classify(body("".join(chr(n) for n in range(32, 127)) * 4))["reasons"])
        self.assertIn("tool_request", policy.classify({**body("hi"), "tools": ["fixture"]})["reasons"])
        result = evaluate(policy, [{"id": "hello", "expected": "local", "request": body("Hello")},
                                   {"id": "proof", "expected": "complex", "request": body("Prove a theorem")}])
        self.assertEqual(result["accuracy"], 1)

    def test_http_forward_stream_cache_isolation_and_failure(self):
        counts = {"local": 0, "complex": 0, "embedding": 0}
        def fixture_handler(name):
            class Fixture(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass
                def do_POST(self):
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    if self.path == "/v1/embeddings":
                        counts["embedding"] += 1
                        result = {"data": [{"embedding": [1, 0]}]}
                    else:
                        counts[name] += 1
                        if body.get("stream"):
                            self.send_response(200)
                            self.send_header("Content-Type", "text/event-stream")
                            self.end_headers()
                            self.wfile.write(b'data: {"choices":[{"delta":{"content":"fixture"}}]}\n\ndata: [DONE]\n\n')
                            return
                        result = {"id": "fixture", "model": body["model"],
                                  "choices": [{"message": {"content": name + " fixture"}}],
                                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
                    encoded = json.dumps(result).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)
            return Fixture
        servers, threads = [], []
        with tempfile.TemporaryDirectory() as folder:
            try:
                backends = {}
                for name in ("local", "complex"):
                    server = ThreadingHTTPServer(("127.0.0.1", 0), fixture_handler(name))
                    servers.append(server)
                    thread = threading.Thread(target=server.serve_forever)
                    threads.append(thread)
                    thread.start()
                    backends[name] = dict(url=f"http://127.0.0.1:{server.server_port}", model=name + "-model", revision="fixture-v1")
                path = Path(folder) / "cache.db"
                cache = SemanticCache(SQLiteVectors(path), SQLiteResponses(path),
                    HTTPEmbedder(backends["local"]["url"], "fixture-embedding", "fixture-v1"))
                gateway = ThreadingHTTPServer(("127.0.0.1", 0), gateway_handler(ComplexityPolicy(), backends, cache, "fixture-v1"))
                servers.append(gateway)
                thread = threading.Thread(target=gateway.serve_forever)
                threads.append(thread)
                thread.start()
                base = f"http://127.0.0.1:{gateway.server_port}"
                def post(text, tenant="a", extra=None, optin=True):
                    body = {"model": "alias", "temperature": 0, "messages": [{"role": "user", "content": text}], **(extra or {})}
                    return urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions",
                        data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "X-Tenant-ID": tenant,
                        "X-Allow-Semantic-Cache": str(optin).lower()}), timeout=5)
                with post("Define a cat") as response:
                    self.assertEqual(response.headers["X-Cache"], "miss")
                    self.assertEqual(json.load(response)["model"], "local-model")
                with post("What is a cat") as response:
                    self.assertEqual(response.headers["X-Cache"], "hit")
                    result = json.load(response)
                    self.assertNotIn("usage", result)
                    self.assertIn("original_usage", result["cache"])
                self.assertEqual(counts["local"], 1)
                with post("What is a cat", tenant="b") as response:
                    self.assertEqual(response.headers["X-Cache"], "miss")
                    response.read()
                with post("What is a cat", extra={"stream": True}) as response:
                    self.assertEqual(response.headers["X-Cache"], "bypass")
                    self.assertIn(b"[DONE]", response.read())
                with post("Prove a theorem", optin=False) as response:
                    self.assertEqual(response.headers["X-Inference-Route"], "complex")
                    response.read()
                self.assertEqual(counts["complex"], 1)
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    post("hello", tenant="")
                self.assertEqual(caught.exception.code, 400)
                caught.exception.close()
                with post("Define 2 cats") as response:
                    self.assertEqual(response.headers["X-Cache"], "miss")
                    response.read()
            finally:
                for server in reversed(servers):
                    server.shutdown()
                    server.server_close()
                for thread in threads:
                    thread.join()


if __name__ == "__main__":
    unittest.main()
