import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prefix_router import Router, handler, prefix_key


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.router = Router({"a": "http://127.0.0.1:1", "b": "http://127.0.0.1:2"},
                             ttl=10, clock=lambda: self.now)
        self.body = {"model": "fixture", "messages": [
            {"role": "system", "content": "Shared instructions"},
            {"role": "user", "content": "First question"}]}
        self.key = prefix_key("tenant-a", "revision-1", self.body)

    def test_prefix_identity_and_isolation(self):
        different = json.loads(json.dumps(self.body))
        different["messages"][1]["content"] = "Another question"
        self.assertEqual(self.key, prefix_key("tenant-a", "revision-1", different))
        self.assertNotEqual(self.key, prefix_key("tenant-b", "revision-1", self.body))
        self.assertNotEqual(self.key, prefix_key("tenant-a", "revision-2", self.body))
        different["messages"][0]["content"] += " "
        self.assertNotEqual(self.key, prefix_key("tenant-a", "revision-1", different))

    def test_residency_eviction_expiry_health(self):
        initial, reason = self.router.route(self.key)
        self.assertEqual(reason, "affinity")
        other = "b" if initial == "a" else "a"
        self.router.report(other, self.key)
        self.assertEqual(self.router.route(self.key), (other, "reported_hit"))
        self.router.report(other, self.key, False)
        self.assertEqual(self.router.route(self.key), (initial, "affinity"))
        self.router.report(other, self.key)
        self.now = 11
        self.assertEqual(self.router.route(self.key), (initial, "affinity"))
        self.router.health(initial, False)
        self.assertEqual(self.router.route(self.key)[0], other)
        self.router.health(other, False)
        with self.assertRaises(RuntimeError):
            self.router.route(self.key)

    def test_bounded_events(self):
        self.router.limit = 2
        for index in range(10):
            self.router.report("a", f"{index:064x}")
        self.assertEqual(len(self.router.resident), 2)

    def test_http_stream_admin_and_rerouting(self):
        class Backend(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"fixture"}}]}\n\n')
                self.wfile.flush()
                self.wfile.write(b'data: [DONE]\n\n')

        servers = []
        try:
            for replica in ("a", "b"):
                backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
                servers.append(backend)
                threading.Thread(target=backend.serve_forever, daemon=True).start()
                self.router.replicas[replica] = f"http://127.0.0.1:{backend.server_port}"
            proxy = ThreadingHTTPServer(("127.0.0.1", 0), handler(self.router, "revision-1", "test-secret"))
            servers.append(proxy)
            threading.Thread(target=proxy.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{proxy.server_port}"
            def post(path, payload, headers=None):
                request = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json", **(headers or {})})
                return urllib.request.urlopen(request, timeout=5)
            with post("/v1/chat/completions", self.body, {"X-Tenant-ID": "tenant-a"}) as response:
                first = response.headers["X-Replica"]
                self.assertEqual(response.headers["X-Routing-Reason"], "affinity")
                self.assertIn(b"data: [DONE]", response.read())
            target = "b" if first == "a" else "a"
            event = dict(event="resident", replica=target, key=self.key)
            with self.assertRaises(urllib.error.HTTPError) as caught:
                post("/admin/events", event)
            self.assertEqual(caught.exception.code, 401)
            caught.exception.close()
            with post("/admin/events", event, {"Authorization": "Bearer test-secret"}) as response:
                self.assertEqual(response.status, 200)
            with post("/v1/chat/completions", self.body, {"X-Tenant-ID": "tenant-a"}) as response:
                self.assertEqual(response.headers["X-Replica"], target)
                self.assertEqual(response.headers["X-Routing-Reason"], "reported_hit")
                self.assertIn(b"[DONE]", response.read())
            with self.assertRaises(urllib.error.HTTPError) as caught:
                post("/v1/chat/completions", self.body)
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()
        finally:
            for server in reversed(servers):
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
