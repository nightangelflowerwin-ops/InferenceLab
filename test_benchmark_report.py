import copy
import unittest

from benchmark_report import validate_rows, summarize, nondominated, markdown


def fixture_rows():
    rows = []
    for candidate, duration, latency, price in (("a", 10, 10, 1), ("b", 12, 20, 2), ("c", 8, 15, 2)):
        for repeat in range(3):
            rows.append(dict(run_id=f"{candidate}-{repeat}", engine="fixture", candidate=candidate,
                engine_version="fixture-v1", hardware="fixture-hardware", model_revision="fixture-model",
                tokenizer_revision="fixture-tokenizer", workload="fixture-workload", source_kind="fixture",
                cost_source="invented price", input_tokens=32, output_tokens=128, request_count=8, completed=8,
                concurrency=8, warmup_requests=8, seed=42, request_rate="inf", arrival_process="closed-loop", sampling_policy="greedy-fixture",
                total_output_tokens=1024, duration_s=duration, billed_seconds=duration,
                p95_ttft_ms=latency, p95_itl_ms=2, hourly_cost_usd=price))
    return rows


class BenchmarkTests(unittest.TestCase):
    def test_cost_arithmetic_frontier_and_repeated_run_intervals(self):
        rows = fixture_rows()
        parsed = validate_rows(rows)
        self.assertAlmostEqual(parsed[0]["window_cost_usd"], 10 / 3600)
        self.assertAlmostEqual(parsed[0]["output_tokens_s"], 102.4)
        report = summarize(rows)
        points = report["contexts"][0]["candidates"]
        self.assertEqual({p["candidate"] for p in points if p["three_objective_pareto"]}, {"a", "c"})
        self.assertEqual(points[0]["p95_ttft_ms_bootstrap_interval"], [10, 10])
        self.assertIn("SYNTHETIC FIXTURE DATA", markdown(report))

    def test_failed_or_under_repeated_candidate_excluded(self):
        rows = fixture_rows()
        rows[0]["completed"] = 7
        rows[0]["total_output_tokens"] = 896
        report = summarize(rows)
        point = next(p for p in report["contexts"][0]["candidates"] if p["candidate"] == "a")
        self.assertFalse(point["eligible"])
        self.assertFalse(point["three_objective_pareto"])
        report = summarize(fixture_rows()[:2])
        self.assertFalse(report["contexts"][0]["candidates"][0]["eligible"])

    def test_hardware_workload_source_and_revision_are_not_pooled(self):
        rows = fixture_rows()
        for field, value in (("hardware", "other"), ("model_revision", "other"),
                             ("workload", "other"), ("source_kind", "caller-supplied-hardware"),
                             ("concurrency", 4), ("request_rate", "10"), ("seed", 7), ("sampling_policy", "other")):
            changed = copy.deepcopy(rows)
            changed[0][field] = value
            if field == "source_kind":
                changed[0].update(model_revision="a" * 40, tokenizer_revision="b" * 40)
            self.assertEqual(len(summarize(changed)["contexts"]), 2)

    def test_invalid_units_counts_duplicates_and_nonfinite_values(self):
        rows = fixture_rows()
        with self.assertRaises(ValueError):
            validate_rows(rows + [rows[0]])
        for field, value in (("duration_s", 0), ("hourly_cost_usd", float("nan")),
                             ("billed_seconds", 1), ("completed", 9), ("total_output_tokens", 2000),
                             ("p95_ttft_ms", 1000000),
                             ("request_count", True), ("source_kind", "verified-real-gpu")):
            changed = copy.deepcopy(rows)
            changed[0][field] = value
            with self.assertRaises(ValueError):
                validate_rows(changed)


if __name__ == "__main__":
    unittest.main()
