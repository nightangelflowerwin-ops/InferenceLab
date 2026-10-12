from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from queue_control import TokenQueue, ScalePolicy, LocalAutoscaler, demo, metrics_handler, worker


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.now = 0
        self.path = Path(self.temp.name) / "queue.db"
        self.queue = TokenQueue(self.path, clock=lambda: self.now)

    def test_idempotency_restart_and_metrics(self):
        self.assertTrue(self.queue.enqueue("j", 100, {"seconds": 0}))
        self.assertFalse(self.queue.enqueue("j", 100, {"seconds": 0}))
        with self.assertRaises(ValueError):
            self.queue.enqueue("j", 101, {})
        self.assertEqual(TokenQueue(self.path, clock=lambda: self.now).snapshot()["pending_tokens"], 100)
        server = ThreadingHTTPServer(("127.0.0.1", 0), metrics_handler(self.queue))
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/metrics") as response:
                self.assertIn(b"inference_queue_pending_tokens 100", response.read())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_atomic_claims_across_connections(self):
        for n in range(20):
            self.queue.enqueue(str(n), 10, {})
            self.queue.register("w" + str(n))
        def claim(n):
            queue = TokenQueue(self.path, clock=lambda: 0)
            return queue.claim("w" + str(n))
        with ThreadPoolExecutor(max_workers=8) as pool:
            leases = list(pool.map(claim, range(20)))
        self.assertEqual(len({lease["id"] for lease in leases}), 20)
        self.assertEqual(self.queue.snapshot()["leased_tokens"], 200)

    def test_fencing_after_expiry_and_dead_letter(self):
        self.queue.enqueue("j", 100, {})
        self.queue.register("a")
        self.queue.register("b")
        old = self.queue.claim("a", 1)
        self.now = 2
        new = self.queue.claim("b", 1)
        self.assertGreater(new["epoch"], old["epoch"])
        self.assertFalse(self.queue.complete(old, "stale"))
        self.assertFalse(self.queue.renew(old))
        self.now = 4
        third = self.queue.claim("a", 1)
        self.now = 6
        self.assertIsNone(self.queue.claim("b"))
        self.assertEqual(self.queue.snapshot()["dead_jobs"], 1)
        self.assertFalse(self.queue.complete(third, "late"))

    def test_drain_finishes_current_job_and_prevents_new_claim(self):
        self.queue.enqueue("j1", 100, {})
        self.queue.enqueue("j2", 100, {})
        self.queue.register("a")
        lease = self.queue.claim("a")
        self.queue.drain("a")
        self.assertIsNone(self.queue.claim("a"))
        self.assertEqual(self.queue.snapshot()["draining_workers"], 1)
        self.assertTrue(self.queue.complete(lease, "finished"))
        self.assertEqual(self.queue.snapshot()["draining_workers"], 0)
        self.assertEqual(self.queue.snapshot()["pending_jobs"], 1)

    def test_scaling_cooldown_and_busy_floor(self):
        policy = ScalePolicy(tokens_per_worker=100, maximum=4, cooldown=10)
        self.assertEqual(policy.desired(dict(pending_tokens=201, busy_workers=1), 1, 0), 4)
        self.assertEqual(policy.desired(dict(pending_tokens=0, busy_workers=0), 4, 5), 4)
        self.assertEqual(policy.desired(dict(pending_tokens=0, busy_workers=0), 4, 11), 0)
        self.assertEqual(policy.desired(dict(pending_tokens=0, busy_workers=2), 2, 20), 2)

    def test_actual_fixture_worker_shutdown_drain(self):
        queue = TokenQueue(self.path)
        queue.enqueue("long", 100, {"seconds": 0.15})
        queue.enqueue("next", 100, {"seconds": 0})
        event = threading.Event()
        thread = threading.Thread(target=worker, args=(queue, "live", event))
        thread.start()
        deadline = time.monotonic() + 2
        while not queue.snapshot()["leased_jobs"] and time.monotonic() < deadline:
            time.sleep(.005)
        event.set()
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(queue.snapshot()["done_jobs"], 1)
        self.assertEqual(queue.snapshot()["pending_jobs"], 1)

    def test_demo_completes_and_drains(self):
        result = demo(self.path)
        self.assertEqual(result["desired_workers"], 3)
        self.assertEqual(result["final"]["done_jobs"], 3)
        self.assertEqual(result["final"]["ready_workers"], 0)

    def test_autoscaler_starts_workers_executes_and_scales_to_zero(self):
        queue = TokenQueue(self.path)
        queue.enqueue("work", 2048, {"seconds": .02})
        scaler = LocalAutoscaler(queue, ScalePolicy(cooldown=0))
        try:
            self.assertEqual(scaler.tick()["desired"], 1)
            deadline = time.monotonic() + 3
            while queue.snapshot()["done_jobs"] != 1 and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(queue.snapshot()["done_jobs"], 1)
            self.assertEqual(scaler.tick()["desired"], 0)
            deadline = time.monotonic() + 3
            while scaler.workers and time.monotonic() < deadline:
                scaler.tick()
                time.sleep(.01)
            self.assertEqual(scaler.workers, {})
        finally:
            scaler.close()


if __name__ == "__main__":
    unittest.main()
