"""Local OpenAI-compatible prefix routing proxy with expiring residency reports.

Reported cache residency is an integration hint, not proof that a vLLM block exists.
"""
import argparse
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import time
import urllib.error
import urllib.request


def prefix_key(tenant, model_revision, body):
    prefix = []
    for message in body.get("messages", []):
        if message.get("role") not in ("system", "developer"):
            break
        prefix.append(message)
    value = [tenant, model_revision, body.get("model"), prefix]
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


class Router:
    def __init__(self, replicas, ttl=30, limit=10000, clock=time.monotonic):
        self.replicas = replicas
        self.ttl, self.limit, self.clock = ttl, limit, clock
        self.healthy = set(replicas)
        self.resident = {}
        self.lock = threading.Lock()
        self.stats = dict(reported_hit=0, affinity=0, failed=0)

    def report(self, replica, key, present=True):
        with self.lock:
            if replica not in self.replicas:
                raise ValueError("Unknown replica")
            if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
                raise ValueError("Invalid prefix key")
            slot = (replica, key)
            if present:
                self.resident[slot] = self.clock() + self.ttl
                while len(self.resident) > self.limit:
                    del self.resident[min(self.resident, key=self.resident.get)]
            else:
                self.resident.pop(slot, None)

    def health(self, replica, healthy):
        with self.lock:
            if replica not in self.replicas:
                raise ValueError("Unknown replica")
            if healthy:
                self.healthy.add(replica)
            else:
                self.healthy.discard(replica)
                self.resident = {k: v for k, v in self.resident.items() if k[0] != replica}

    def route(self, key):
        with self.lock:
            self.resident = {k: expiry for k, expiry in self.resident.items() if expiry > self.clock()}
            if not self.healthy:
                raise RuntimeError("No healthy replicas")
            holders = {replica for replica, prefix in self.resident if prefix == key and replica in self.healthy}
            reason = "reported_hit" if holders else "affinity"
            pool = holders or self.healthy
            selected = max(pool, key=lambda replica: hashlib.sha256(f"{key}:{replica}".encode()).digest())
            self.stats[reason] += 1
            return selected, reason


def handler(router, revision, admin_token):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass  # Do not log prompts or caller identifiers.

        def respond(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/health":
                return self.respond(404, {"error": "Unknown path"})
            with router.lock:
                snapshot = dict(healthy_replicas=len(router.healthy), counters=dict(router.stats))
            self.respond(200 if snapshot["healthy_replicas"] else 503, snapshot)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1048576:
                    self.close_connection = True
                    return self.respond(413, {"error": "Request must be between 1 byte and 1 MiB"})
                self.connection.settimeout(30)
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("JSON object required")
                if self.path == "/admin/events":
                    supplied = self.headers.get("Authorization", "")
                    if not hmac.compare_digest(supplied, f"Bearer {admin_token}"):
                        return self.respond(401, {"error": "Unauthorized"})
                    if body["event"] == "health":
                        if not isinstance(body["healthy"], bool):
                            raise ValueError("healthy must be boolean")
                        router.health(body["replica"], body["healthy"])
                    elif body["event"] in ("resident", "evicted"):
                        router.report(body["replica"], body["key"], body["event"] == "resident")
                    else:
                        raise ValueError("Unknown event")
                    return self.respond(200, {"accepted": True})
                if self.path != "/v1/chat/completions":
                    return self.respond(404, {"error": "Unknown path"})
                tenant = self.headers.get("X-Tenant-ID")
                if not tenant:
                    return self.respond(400, {"error": "X-Tenant-ID required from a trusted local caller"})
                if not isinstance(body.get("messages"), list) or not body.get("model"):
                    raise ValueError("messages list and model required")
                key = prefix_key(tenant, revision, body)
                replica, reason = router.route(key)
            except (ValueError, KeyError, TypeError, AttributeError):
                return self.respond(400, {"error": "Invalid request or event"})
            except RuntimeError:
                return self.respond(503, {"error": "No healthy replicas"})
            request = urllib.request.Request(router.replicas[replica] + self.path,
                data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
                "X-Prefix-Key": key, "X-Tenant-ID": tenant})
            try:
                upstream = urllib.request.urlopen(request, timeout=60)
            except urllib.error.HTTPError as error:
                error.close()
                return self.respond(error.code, {"error": "Upstream rejected request"})
            except OSError:
                router.health(replica, False)
                with router.lock:
                    router.stats["failed"] += 1
                return self.respond(502, {"error": "Upstream unavailable; no automatic POST replay"})
            with upstream:
                self.send_response(upstream.status)
                self.send_header("Content-Type", upstream.headers.get("Content-Type", "application/json"))
                self.send_header("X-Routing-Reason", reason)
                self.send_header("X-Replica", replica)
                self.send_header("X-Prefix-Key", key)
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                try:
                    while chunk := upstream.read1(8192):
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except OSError:
                    with router.lock:
                        router.stats["failed"] += 1
                    # Headers/output may already be sent. Close rather than replay.

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replica", action="append", required=True, help="ID=http://127.0.0.1:PORT")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    replicas = dict(value.split("=", 1) for value in args.replica)
    from urllib.parse import urlsplit
    for url in replicas.values():
        parsed = urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost") or parsed.path not in ("", "/"):
            parser.error("This local prototype accepts loopback HTTP replicas only")
    token = os.environ.get("PREFIX_ADMIN_TOKEN")
    if not token:
        parser.error("Set PREFIX_ADMIN_TOKEN for residency and health reports")
    router = Router({name: url.rstrip("/") for name, url in replicas.items()})
    print(f"Local proxy: http://127.0.0.1:{args.port}; cache residency requires explicit events", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), handler(router, args.model_revision, token)).serve_forever()


if __name__ == "__main__":
    main()
