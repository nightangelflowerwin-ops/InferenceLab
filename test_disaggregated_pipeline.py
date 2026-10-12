import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest

from disaggregated_pipeline import (Coordinator, WorkerPool, demo, decode, prefill,
                                    make_handoff, verify_handoff, worker_handler)


class DisaggregationTests(unittest.TestCase):
    def setUp(self):
        self.job = {"request_id": "r", "tenant": "tenant", "model_revision": "fixture-v1",
                    "tokens": [1, 2, 3], "max_new_tokens": 4}
        self.secret = b"local-test-handoff-secret"

    def test_signed_handoff_integrity_request_revision_and_expiry(self):
        handoff = make_handoff(self.job, self.secret, clock=lambda: 100)
        self.assertEqual(verify_handoff(self.job, handoff, self.secret, clock=lambda: 101), prefill([1, 2, 3]))
        corrupt = copy.deepcopy(handoff)
        corrupt["envelope"]["rolling_state"] += 1
        with self.assertRaises(ValueError):
            verify_handoff(self.job, corrupt, self.secret, clock=lambda: 101)
        with self.assertRaises(ValueError):
            verify_handoff(self.job, handoff, self.secret, clock=lambda: 130)
        for changed in ({"tenant": "other"}, {"model_revision": "other"}, {"tokens": [9]}):
            with self.assertRaises(ValueError):
                verify_handoff({**self.job, **changed}, handoff, self.secret, clock=lambda: 101)

    def test_pool_capacity_health_and_least_active_routing(self):
        pool = WorkerPool({"a": "http://127.0.0.1:1", "b": "http://127.0.0.1:2"}, per_worker_capacity=1)
        with pool.reserve() as first:
            with pool.reserve() as second:
                self.assertNotEqual(first[0], second[0])
                with self.assertRaises(RuntimeError):
                    with pool.reserve():
                        pass
        self.assertEqual(pool.active, {"a": 0, "b": 0})
        pool.health("a", False)
        with pool.reserve() as selected:
            self.assertEqual(selected[0], "b")

    def test_real_http_separate_pool_demo_matches_monolithic_fixture(self):
        result = demo()
        self.assertTrue(result["producer"].startswith("prefill-"))
        self.assertTrue(result["consumer"].startswith("decode-"))
        self.assertEqual(result["tokens"], decode(prefill([1, 2, 3, 4]), 8))
        self.assertGreater(result["handoff_bytes"], 0)

    def test_prefill_network_failure_failover_and_no_decode_replay(self):
        servers, threads = [], []
        try:
            urls = {}
            for phase in ("prefill", "decode"):
                server = ThreadingHTTPServer(("127.0.0.1", 0), worker_handler(phase, self.secret, "fixture-v1"))
                servers.append(server)
                thread = threading.Thread(target=server.serve_forever)
                threads.append(thread)
                thread.start()
                urls[phase] = f"http://127.0.0.1:{server.server_port}"
            # Closed ephemeral port represents an unreachable local producer.
            import socket
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                dead = f"http://127.0.0.1:{sock.getsockname()[1]}"
            producers = WorkerPool({"a-dead": dead, "b-live": urls["prefill"]})
            coordinator = Coordinator(producers, WorkerPool({"decode": urls["decode"]}))
            response = coordinator.execute(self.job)
            self.assertEqual(response["producer"], "b-live")
            self.assertNotIn("a-dead", producers.healthy)
            self.assertEqual(coordinator.execute(self.job), response)
            with self.assertRaises(ValueError):
                coordinator.execute({**self.job, "tokens": [9]})
            failing = Coordinator(WorkerPool({"prefill": urls["prefill"]}), WorkerPool({"decode": dead}))
            with self.assertRaises(OSError):
                failing.execute(self.job)
            self.assertEqual(failing.requests[("tenant", "r")]["status"], "uncertain")
            with self.assertRaises(RuntimeError):
                failing.execute(self.job)
        finally:
            for server in reversed(servers):
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join()


if __name__ == "__main__":
    unittest.main()
