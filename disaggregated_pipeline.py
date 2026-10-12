"""Separate loopback prefill/decode pools with signed CPU fixture-state handoff.

The recurrence is a tiny deterministic fixture, not a transformer or GPU KV cache.
"""
import argparse
from contextlib import contextmanager
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

from semantic_cache import local_url


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def validated_job(body):
    if not isinstance(body, dict):
        raise ValueError("Job object required")
    result = {key: body[key] for key in ("request_id", "tenant", "model_revision", "tokens", "max_new_tokens")}
    for key in ("request_id", "tenant", "model_revision"):
        if not isinstance(result[key], str) or not 1 <= len(result[key]) <= 256:
            raise ValueError("Bounded nonempty identity fields required")
    tokens = result["tokens"]
    if not isinstance(tokens, list) or len(tokens) > 4096 or any(
            isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < 256 for token in tokens):
        raise ValueError("Fixture token IDs must be 0..255; prompt limit 4096")
    count = result["max_new_tokens"]
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 256:
        raise ValueError("Generation limit must be 1..256")
    return result


def advance(state, token):
    return (state * 1664525 + token + 1013904223) & 0xffffffff


def prefill(tokens):
    state = 0
    for token in tokens:
        state = advance(state, token)
    return state


def decode(state, count):
    tokens = []
    for _ in range(count):
        token = (state ^ (state >> 16)) % 256
        tokens.append(token)
        state = advance(state, token)
    return tokens


def make_handoff(job, secret, clock=time.time, ttl=30):
    job = validated_job(job)
    envelope = {"request_digest": digest(job), "tenant": job["tenant"],
                "model_revision": job["model_revision"], "token_count": len(job["tokens"]),
                "rolling_state": prefill(job["tokens"]), "expires": clock() + ttl,
                "format": "cpu-fixture-state-v1"}
    return {"envelope": envelope, "signature": hmac.new(secret, canonical(envelope), hashlib.sha256).hexdigest()}


def verify_handoff(job, handoff, secret, clock=time.time):
    job = validated_job(job)
    envelope = handoff["envelope"]
    expected = hmac.new(secret, canonical(envelope), hashlib.sha256).hexdigest()
    if not isinstance(handoff.get("signature"), str) or not hmac.compare_digest(expected, handoff["signature"]):
        raise ValueError("Handoff integrity failed")
    if envelope.get("format") != "cpu-fixture-state-v1" or envelope.get("request_digest") != digest(job):
        raise ValueError("Handoff belongs to another request or format")
    if envelope.get("tenant") != job["tenant"] or envelope.get("model_revision") != job["model_revision"]:
        raise ValueError("Handoff tenant/revision differs")
    expiry = envelope.get("expires")
    if not isinstance(expiry, (int, float)) or not clock() < expiry < float("inf"):
        raise ValueError("Handoff expired or invalid")
    state = envelope.get("rolling_state")
    if isinstance(state, bool) or not isinstance(state, int) or not 0 <= state <= 0xffffffff or envelope.get("token_count") != len(job["tokens"]):
        raise ValueError("Invalid fixture state")
    return state


def call(url, path, body, timeout=10):
    request = urllib.request.Request(local_url(url) + path, data=canonical(body),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read(2 * 1024 * 1024 + 1)
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("Worker response exceeds bound")
    return json.loads(data)


class WorkerPool:
    def __init__(self, workers, per_worker_capacity=4):
        if not workers or per_worker_capacity < 1:
            raise ValueError("Nonempty bounded worker pool required")
        self.workers = {name: local_url(url) for name, url in workers.items()}
        self.capacity = per_worker_capacity
        self.active = {name: 0 for name in workers}
        self.healthy, self.lock = set(workers), threading.Lock()

    @contextmanager
    def reserve(self):
        with self.lock:
            available = [name for name in self.healthy if self.active[name] < self.capacity]
            if not available:
                raise RuntimeError("No available worker in pool")
            name = min(available, key=lambda candidate: (self.active[candidate], candidate))
            self.active[name] += 1
        try:
            yield name, self.workers[name]
        finally:
            with self.lock:
                self.active[name] -= 1

    def health(self, name, healthy):
        with self.lock:
            if name not in self.workers:
                raise ValueError("Unknown worker")
            if healthy:
                self.healthy.add(name)
            else:
                self.healthy.discard(name)


class Coordinator:
    def __init__(self, producers, consumers):
        self.producers, self.consumers = producers, consumers
        self.requests, self.lock = {}, threading.Lock()

    def execute(self, body):
        job = validated_job(body)
        key, fingerprint = (job["tenant"], job["request_id"]), digest(job)
        with self.lock:
            old = self.requests.get(key)
            if old:
                if old["digest"] != fingerprint:
                    raise ValueError("Request ID reused with different work")
                if old["status"] == "done":
                    return old["response"]
                raise RuntimeError("Request is active or has uncertain completion; no automatic replay")
            if len(self.requests) >= 10000:
                raise RuntimeError("Coordinator request ledger capacity reached")
            self.requests[key] = {"digest": fingerprint, "status": "active"}
        start, entered_decode = time.perf_counter(), False
        try:
            transfer = None
            for _ in range(len(self.producers.workers)):
                with self.producers.reserve() as (producer, url):
                    try:
                        transfer = call(url, "/prefill", job)
                        break
                    except (OSError, ValueError):
                        self.producers.health(producer, False)
            if transfer is None:
                raise RuntimeError("Prefill pool failed")
            prefill_end = time.perf_counter()
            with self.consumers.reserve() as (consumer, url):
                entered_decode = True
                # Once decode may have begun, timeout/failure is never replayed.
                try:
                    result = call(url, "/decode", {"job": job, "handoff": transfer})
                except (OSError, ValueError):
                    self.consumers.health(consumer, False)
                    raise
            output_tokens = result.get("tokens") if isinstance(result, dict) else None
            if not isinstance(output_tokens, list) or len(output_tokens) != job["max_new_tokens"] or any(
                    isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < 256 for token in output_tokens):
                raise ValueError("Decode response differs from token contract")
            response = {"request_id": job["request_id"], "tokens": result["tokens"],
                        "producer": producer, "consumer": consumer,
                        "handoff_bytes": len(canonical(transfer)), "prefill_and_fetch_ms": (prefill_end - start) * 1000,
                        "decode_and_transfer_ms": (time.perf_counter() - prefill_end) * 1000,
                        "fixture": True, "evidence": "Actual loopback phase separation with signed CPU fixture state; no GPU KV transfer or inference"}
            with self.lock:
                self.requests[key] = {"digest": fingerprint, "status": "done", "response": response}
            return response
        except (OSError, ValueError, KeyError, RuntimeError):
            with self.lock:
                if entered_decode:
                    self.requests[key]["status"] = "uncertain"
                else:
                    # A failed prefill has not produced user-visible generation.
                    self.requests.pop(key, None)
            raise


def json_handler(operation):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            self.respond(200 if self.path == "/health" else 404, {"fixture": True})
        def respond(self, status, result):
            data = canonical(result)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1024 * 1024:
                    raise ValueError("Bounded JSON body required")
                self.connection.settimeout(10)
                body = json.loads(self.rfile.read(size))
                self.respond(200, operation(self.path, body))
            except (ValueError, KeyError, TypeError):
                self.respond(400, {"error": "Invalid job or handoff"})
            except (RuntimeError, OSError):
                self.respond(503, {"error": "Pool unavailable or request completion uncertain"})
    return Handler


def worker_handler(phase, secret, revision):
    def operation(path, body):
        if phase == "prefill" and path == "/prefill":
            job = validated_job(body)
            if job["model_revision"] != revision:
                raise ValueError("Worker revision differs")
            return make_handoff(job, secret)
        if phase == "decode" and path == "/decode":
            job = validated_job(body["job"])
            if job["model_revision"] != revision:
                raise ValueError("Worker revision differs")
            state = verify_handoff(job, body["handoff"], secret)
            return {"tokens": decode(state, job["max_new_tokens"]), "fixture": True}
        raise ValueError("Wrong phase endpoint")
    return json_handler(operation)


def demo():
    servers, threads, secret = [], [], os.urandom(32)
    try:
        pools = []
        for phase in ("prefill", "decode"):
            workers = {}
            for index in range(2):
                server = ThreadingHTTPServer(("127.0.0.1", 0), worker_handler(phase, secret, "fixture-v1"))
                servers.append(server)
                thread = threading.Thread(target=server.serve_forever)
                threads.append(thread)
                thread.start()
                workers[f"{phase}-{index}"] = f"http://127.0.0.1:{server.server_port}"
            pools.append(WorkerPool(workers))
        coordinator = Coordinator(*pools)
        result = coordinator.execute({"request_id": "demo", "tenant": "demo", "model_revision": "fixture-v1",
                                      "tokens": [1, 2, 3, 4], "max_new_tokens": 8})
        assert result["tokens"] == decode(prefill([1, 2, 3, 4]), 8)
        return result
    finally:
        for server in reversed(servers):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["demo", "prefill", "decode", "proxy"])
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--revision", default="fixture-v1")
    parser.add_argument("--producer", action="append", help="ID=http://127.0.0.1:PORT")
    parser.add_argument("--consumer", action="append", help="ID=http://127.0.0.1:PORT")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.action == "demo":
        result = demo()
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return
    if args.action == "proxy":
        if not args.producer or not args.consumer:
            parser.error("Separate producer and consumer pools required")
        coordinator = Coordinator(WorkerPool(dict(value.split("=", 1) for value in args.producer)),
                                  WorkerPool(dict(value.split("=", 1) for value in args.consumer)))
        def operation(path, body):
            if path != "/generate":
                raise ValueError("Unknown endpoint")
            return coordinator.execute(body)
        handler = json_handler(operation)
    else:
        secret = os.environ.get("INFERENCELAB_TRANSFER_SECRET")
        if not secret or len(secret) < 16:
            parser.error("Set a shared local INFERENCELAB_TRANSFER_SECRET of at least 16 characters")
        handler = worker_handler(args.action, secret.encode(), args.revision)
    print(f"CPU fixture {args.action} at http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
