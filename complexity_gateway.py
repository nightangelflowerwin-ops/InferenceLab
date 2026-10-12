"""Loopback complexity-routing gateway with opt-in semantic response caching.

No external providers, API keys, cloud purchases or model downloads are used.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import uuid

from semantic_cache import (SemanticCache, SQLiteVectors, SQLiteResponses, RedisResponses,
                            QdrantVectors, HTTPEmbedder, identity, local_url)


@dataclass
class ComplexityPolicy:
    token_limit: int = 512
    entropy_threshold: float = 5.5

    def classify(self, body):
        if self.token_limit < 1 or not 0 <= self.entropy_threshold <= 8:
            raise ValueError("Invalid routing policy")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Nonempty messages list required")
        text_parts = []
        multimodal = False
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in ("system", "developer", "user", "assistant", "tool"):
                raise ValueError("Invalid message")
            content = message.get("content")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                multimodal = True
            elif content is not None:
                raise ValueError("Invalid message content")
        text = "\n".join(text_parts)
        encoded = text.encode("utf-8")
        counts = Counter(encoded)
        entropy = -sum(n / len(encoded) * math.log2(n / len(encoded)) for n in counts.values()) if encoded else 0
        estimate = math.ceil(len(encoded) / 4)
        cues = re.findall(r"\b(?:prove|derive|debug|optimize|tradeoffs|trade-offs|compare|reasoning|algorithm|architecture)\b", text.lower())
        reasons = []
        if estimate > self.token_limit:
            reasons.append("length")
        if entropy >= self.entropy_threshold and estimate >= 64:
            reasons.append("byte_entropy")
        if cues:
            reasons.append("reasoning_cue")
        if body.get("tools") or body.get("functions"):
            reasons.append("tool_request")
        if multimodal:
            reasons.append("multimodal")
        return {"route": "complex" if reasons else "local", "reasons": reasons or ["small_text"],
                "estimated_input_tokens": estimate, "byte_entropy_bits": entropy,
                "caveat": "Byte entropy and UTF-8 length are heuristics, not tokenizer counts or reasoning difficulty."}


def evaluate(policy, cases):
    confusion = {"local": {"local": 0, "complex": 0}, "complex": {"local": 0, "complex": 0}}
    details = []
    for case in cases:
        expected = case["expected"]
        if expected not in confusion:
            raise ValueError("Unknown reference route")
        result = policy.classify(case["request"])
        confusion[expected][result["route"]] += 1
        details.append({"id": case["id"], "expected": expected, **result})
    correct = sum(confusion[k][k] for k in confusion)
    return dict(cases=len(cases), accuracy=correct / len(cases) if cases else None,
                confusion=confusion, details=details,
                evidence="Agreement with supplied routing labels; does not measure model answer quality or API cost savings.")


def gateway_handler(policy, backends, cache=None, embedding_revision=None, max_inflight=16):
    for backend in backends.values():
        local_url(backend["url"])
        if not backend["model"] or not backend["revision"]:
            raise ValueError("Every route needs model and revision")
    capacity = threading.BoundedSemaphore(max_inflight)
    counters = Counter()
    lock = threading.Lock()

    class Gateway(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def respond(self, status, payload, extra=None):
            data = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for key, value in (extra or {}).items():
                self.send_header(key, str(value))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/health":
                return self.respond(404, {"error": "Unknown path"})
            with lock:
                snapshot = dict(counters)
            self.respond(200, {"configured_routes": sorted(backends), "counters": snapshot,
                               "cache_enabled": cache is not None})

        def do_POST(self):
            self.close_connection = True
            if self.path != "/v1/chat/completions":
                return self.respond(404, {"error": "Unknown path"})
            if not capacity.acquire(blocking=False):
                return self.respond(429, {"error": "Gateway capacity reached"})
            try:
                self.forward()
            finally:
                capacity.release()

        def forward(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1024 * 1024:
                    return self.respond(413, {"error": "Bounded request body required"})
                self.connection.settimeout(30)
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("JSON object required")
                json.dumps(body, allow_nan=False)
                decision = policy.classify(body)
                tenant = self.headers.get("X-Tenant-ID")
                if not tenant:
                    raise ValueError("Trusted X-Tenant-ID required")
            except (ValueError, TypeError):
                return self.respond(400, {"error": "Invalid request or missing tenant"})
            route = decision["route"]
            if route not in backends:
                return self.respond(503, {"error": "Selected route has no configured local backend", "routing": decision})
            backend = backends[route]
            routed = {**body, "model": backend["model"]}
            headers = {"X-Inference-Route": route, "X-Estimated-Input-Tokens": decision["estimated_input_tokens"],
                       "X-Cache": "bypass"}
            with lock:
                counters[route + "_requests"] += 1
            embedding, cache_id = None, None
            if cache and self.headers.get("X-Allow-Semantic-Cache", "").lower() == "true":
                try:
                    cache_id = identity(tenant, backend["revision"], embedding_revision, routed)
                    if cache_id:
                        response, embedding, score = cache.lookup(*cache_id)
                        if response is not None:
                            # Cached usage describes the original inference request,
                            # not prompt tokens or GPU work for this cache lookup.
                            response = json.loads(json.dumps(response))
                            original_usage = response.pop("usage", None)
                            response.update(id="chatcmpl-cache-" + uuid.uuid4().hex, created=int(time.time()),
                                cache={"hit": True, "similarity": score, "original_usage": original_usage})
                            with lock:
                                counters["cache_hits"] += 1
                            headers["X-Cache"] = "hit"
                            return self.respond(200, response, headers)
                        headers["X-Cache"] = "miss"
                except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
                    cache_id, embedding = None, None
                    headers["X-Cache"] = "unavailable"
                    with lock:
                        counters["cache_errors"] += 1
            request = urllib.request.Request(backend["url"].rstrip("/") + "/v1/chat/completions",
                data=json.dumps(routed, allow_nan=False).encode(),
                headers={"Content-Type": "application/json", "X-Tenant-ID": tenant})
            try:
                upstream = urllib.request.urlopen(request, timeout=60)
            except urllib.error.HTTPError as error:
                status = error.code
                error.close()
                return self.respond(status, {"error": "Selected upstream rejected request"}, headers)
            except OSError:
                return self.respond(502, {"error": "Selected upstream unavailable; request not replayed"}, headers)
            with upstream:
                if routed.get("stream"):
                    self.send_response(upstream.status)
                    self.send_header("Content-Type", upstream.headers.get("Content-Type", "text/event-stream"))
                    self.send_header("Connection", "close")
                    for key, value in headers.items():
                        self.send_header(key, str(value))
                    self.end_headers()
                    try:
                        while chunk := upstream.read1(8192):
                            self.wfile.write(chunk)
                            self.wfile.flush()
                    except OSError:
                        pass  # Partial streams are never replayed or cached.
                    return
                try:
                    raw = upstream.read(2 * 1024 * 1024 + 1)
                    if len(raw) > 2 * 1024 * 1024:
                        raise ValueError("Upstream response exceeds bound")
                    response = json.loads(raw)
                    if not isinstance(response, dict) or "error" in response or not isinstance(response.get("choices"), list):
                        raise ValueError("Invalid completion response")
                except (OSError, ValueError):
                    return self.respond(502, {"error": "Invalid upstream response"}, headers)
            if cache_id and embedding is not None:
                try:
                    cache.put(cache_id[0], embedding, response)
                except (OSError, ValueError, TypeError, sqlite3.Error):
                    with lock:
                        counters["cache_errors"] += 1
            return self.respond(200, response, headers)

    return Gateway


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["serve", "evaluate"])
    parser.add_argument("--local-url")
    parser.add_argument("--local-model", default="local-8b")
    parser.add_argument("--local-revision")
    parser.add_argument("--complex-url", help="Optional loopback complex-model backend; external APIs are disabled")
    parser.add_argument("--complex-model", default="complex-fixture")
    parser.add_argument("--complex-revision")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--token-limit", type=int, default=512)
    parser.add_argument("--cache-db", type=Path)
    parser.add_argument("--embedding-url")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-revision")
    parser.add_argument("--redis-port", type=int)
    parser.add_argument("--qdrant-url")
    args = parser.parse_args()
    policy = ComplexityPolicy(token_limit=args.token_limit)
    if args.action == "evaluate":
        if not args.cases or not args.out:
            parser.error("evaluate requires --cases and --out")
        result = evaluate(policy, json.loads(args.cases.read_text(encoding="utf-8")))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "details"}, indent=2))
        return
    if not args.local_url or not args.local_revision:
        parser.error("serve requires --local-url and --local-revision")
    backends = {"local": {"url": args.local_url, "model": args.local_model, "revision": args.local_revision}}
    if args.complex_url:
        if not args.complex_revision:
            parser.error("--complex-revision required")
        backends["complex"] = {"url": args.complex_url, "model": args.complex_model, "revision": args.complex_revision}
    cache = None
    if args.cache_db:
        args.cache_db.parent.mkdir(parents=True, exist_ok=True)
        if not all((args.embedding_url, args.embedding_model, args.embedding_revision)):
            parser.error("Cache requires local embedding URL, model and revision")
        vectors = QdrantVectors(args.qdrant_url) if args.qdrant_url else SQLiteVectors(args.cache_db)
        responses = RedisResponses(port=args.redis_port) if args.redis_port else SQLiteResponses(args.cache_db)
        cache = SemanticCache(vectors, responses, HTTPEmbedder(args.embedding_url, args.embedding_model, args.embedding_revision))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), gateway_handler(policy, backends, cache, args.embedding_revision))
    print(f"Loopback gateway at http://127.0.0.1:{args.port}; configured routes: {list(backends)}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
