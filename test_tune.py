"""Offline orchestration tests. These do not benchmark GPU inference."""
import copy
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

import tune


class TuningTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path(__file__).with_name("config.json").read_text())
        self.config["revision"] = "a" * 40

    def test_grid_and_baseline(self):
        jobs = tune.make_plan(self.config, Path("results"))
        self.assertEqual(len(jobs), 54)
        self.assertEqual(jobs, tune.make_plan(self.config, Path("results")))
        self.assertTrue(any(not j["candidate"]["chunked"] for j in jobs))
        self.assertTrue(all(j["candidate"]["chunked"] or j["candidate"]["tokens"] >= 4096 for j in jobs))

    def test_revision_and_context_validation(self):
        self.config["revision"] = "main"
        with self.assertRaises(ValueError):
            tune.make_plan(self.config, Path("results"))
        self.config["revision"] = "a" * 40
        self.config["workloads"][0]["input"] = 4096
        with self.assertRaises(ValueError):
            tune.make_plan(self.config, Path("results"))

    def test_real_readiness_and_process_cleanup(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        # A small HTTP fixture supplies readiness only; no fake model output.
        code = ("from http.server import HTTPServer,BaseHTTPRequestHandler\n"
                "class H(BaseHTTPRequestHandler):\n"
                " def do_GET(self):\n"
                "  self.send_response(200 if self.path=='/health' else 404); self.end_headers()\n"
                f"HTTPServer(('127.0.0.1',{port}),H).serve_forever()")
        process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            tune.ready(process, port, 10)
        finally:
            tune.stop(process)
        self.assertIsNotNone(process.poll())

    def test_dead_server_is_failure(self):
        process = subprocess.Popen([sys.executable, "-c", "pass"])
        process.wait()
        with self.assertRaises(RuntimeError):
            tune.ready(process, 1, 2)

    def test_reports_reject_failed_incomplete_and_slo_violations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = tune.make_plan(self.config, root)[:4]
            # Give fixtures one comparable workload for selection.
            for job in jobs:
                job["workload"]["name"] = "fixture"
            tune.dump(root / "plan.json", jobs)
            for index, job in enumerate(jobs):
                out = root / job["name"]
                out.mkdir()
                tune.dump(out / "status.json", {"status": "failed" if index == 1 else "passed"})
                for repeat in range(2 if index == 2 else 3):
                    tune.dump(out / f"metrics-{repeat}.json", dict(completed=128,
                        output_throughput=100 + 1000 * index,
                        p95_ttft_ms=1500 if index == 3 else 50, p95_itl_ms=10))
            rows = tune.summarize(root, 1000)
            self.assertEqual(len(rows), 8)
            recommendation = json.loads((root / "recommendations.json").read_text())
            self.assertEqual(recommendation["best_per_workload"]["fixture"]["candidate"], jobs[0]["name"])


if __name__ == "__main__":
    unittest.main()
