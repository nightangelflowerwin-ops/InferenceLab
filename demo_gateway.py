"""Run a temporary end-to-end fixture gateway demo without model/API access."""
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import urllib.request

from complexity_gateway import ComplexityPolicy, gateway_handler
from fixture_backend import handler
from semantic_cache import SQLiteVectors, SQLiteResponses, HTTPEmbedder, SemanticCache


def run_demo():
    servers, threads = [], []
    with tempfile.TemporaryDirectory() as directory:
        try:
            backends = {}
            for name in ("local", "complex"):
                server = ThreadingHTTPServer(("127.0.0.1", 0), handler(name + "-fixture"))
                servers.append(server)
                thread = threading.Thread(target=server.serve_forever)
                threads.append(thread)
                thread.start()
                backends[name] = {"url": f"http://127.0.0.1:{server.server_port}", "model": name + "-fixture", "revision": "fixture-v1"}
            db = Path(directory) / "cache.db"
            cache = SemanticCache(SQLiteVectors(db), SQLiteResponses(db),
                HTTPEmbedder(backends["local"]["url"], "hashed-bigram-fixture", "fixture-v1"))
            gateway = ThreadingHTTPServer(("127.0.0.1", 0), gateway_handler(ComplexityPolicy(), backends, cache, "fixture-v1"))
            servers.append(gateway)
            thread = threading.Thread(target=gateway.serve_forever)
            threads.append(thread)
            thread.start()
            observations = []
            for query in ("What is a cat?", "What is a cat?", "Prove a theorem"):
                body = {"model": "route-alias", "temperature": 0, "messages": [{"role": "user", "content": query}]}
                request = urllib.request.Request(f"http://127.0.0.1:{gateway.server_port}/v1/chat/completions",
                    data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "X-Tenant-ID": "demo",
                    "X-Allow-Semantic-Cache": "true"})
                with urllib.request.urlopen(request, timeout=5) as response:
                    observations.append({"query": query, "route": response.headers["X-Inference-Route"],
                        "cache": response.headers["X-Cache"], "response": json.load(response)})
            assert [o["cache"] for o in observations[:2]] == ["miss", "hit"]
            assert observations[-1]["route"] == "complex"
            return {"evidence": "Actual loopback proxy/cache operations with echo backends and lexical fixture embeddings; no LLM, semantic embedding model or inference speedup measured", "observations": observations}
        finally:
            for server in reversed(servers):
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_demo()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Verified local route, complex route, cache miss and cache hit using fixtures")
