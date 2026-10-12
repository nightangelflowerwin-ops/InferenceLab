import copy
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import json
import unittest

from fp8_harness import (calibration_report, corpus_manifest, compare_scores,
                         decode_e4m3, encode_e4m3)


class FP8Tests(unittest.TestCase):
    def test_preparation_plan_without_gpu_dependencies(self):
        root = Path(__file__).parent
        script = root / "projects/06-fp8-calibration/modelopt_prepare.py"
        compile(script.read_text(encoding="utf-8"), str(script), "exec")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plan.json"
            subprocess.run([sys.executable, str(script), "--corpus", str(root / "projects/06-fp8-calibration/corpus-fixture.json"),
                "--model-snapshot", "/unavailable/model", "--out", str(output), "--plan-only"],
                check=True, capture_output=True, timeout=10)
            plan = json.loads(output.read_text())
            self.assertEqual(plan["export"], "modelopt.torch.export.export_hf_checkpoint")
            self.assertEqual(len(plan["corpus_manifest"]["documents"]), 4)
    def traces(self):
        documents = [{"id": "cal", "split": "calibration", "text": "calibration text"}] + [
            {"id": str(n), "split": "evaluation", "text": "evaluation text " + str(n)} for n in range(3)]
        manifest = corpus_manifest(documents, "fixture-tokenizer", "fixture-tokenizer-v1", "fixture-model-v1")
        baseline = {"corpus_manifest": manifest, "corpus_sha256": manifest["sha256"],
                    "tokenizer_revision": manifest["tokenizer_revision"], "model_revision": manifest["model_revision"],
                    "evaluation_policy": "fixture next-token scoring, no truncation",
                    "scores": [{"document_id": str(n), "token_index": 0, "token_id": n, "logprob": -math.log(2)} for n in range(3)]}
        quantized = copy.deepcopy(baseline)
        for record in quantized["scores"]:
            record["logprob"] = -math.log(4)
        return baseline, quantized

    def test_all_finite_codes_roundtrip_signed_zero_and_ties(self):
        for code in range(256):
            value = decode_e4m3(code)
            if math.isfinite(value):
                self.assertEqual(encode_e4m3(value), code)
            else:
                self.assertIn(code, (127, 255))
        self.assertEqual(decode_e4m3(1), 2 ** -9)
        self.assertEqual(decode_e4m3(126), 448)
        self.assertEqual(encode_e4m3(1.0625), 56)
        self.assertEqual(encode_e4m3(2 ** -10), 0)
        self.assertEqual(encode_e4m3(-0.), 128)
        self.assertEqual(decode_e4m3(encode_e4m3(1000)), 448)

    def test_heldout_clipping_and_zero_calibration(self):
        report = calibration_report([0, 1, -1], [0, 1, -1, 2])
        self.assertEqual(report["heldout_clipped_fraction"], .25)
        self.assertEqual(report["heldout_max_absolute_error"], 1)
        self.assertAlmostEqual(report["heldout_rmse"], .5)
        report = calibration_report([0, 0], [0, 1])
        self.assertEqual(report["scale"], 1)
        self.assertEqual(report["heldout_clipped_fraction"], 0)

    def test_token_weighted_perplexity_and_paired_bootstrap(self):
        baseline, quantized = self.traces()
        result = compare_scores(baseline, quantized)
        self.assertAlmostEqual(result["baseline_perplexity"], 2)
        self.assertAlmostEqual(result["quantized_perplexity"], 4)
        self.assertAlmostEqual(result["perplexity_delta_quantized_minus_baseline"], 2)
        self.assertEqual(result["evaluated_tokens"], 3)
        for delta in result["paired_document_bootstrap_delta_95_percent_interval"]:
            self.assertAlmostEqual(delta, 2)

    def test_alignment_provenance_masks_and_split_leakage_rejected(self):
        baseline, quantized = self.traces()
        for field, value in (("tokenizer_revision", "other"), ("corpus_sha256", "wrong")):
            changed = copy.deepcopy(quantized)
            changed[field] = value
            with self.assertRaises(ValueError):
                compare_scores(baseline, changed)
        for field, value in (("token_id", 100), ("evaluate", False), ("logprob", float("nan"))):
            changed = copy.deepcopy(quantized)
            changed["scores"][0][field] = value
            with self.assertRaises(ValueError):
                compare_scores(baseline, changed)
        with self.assertRaises(ValueError):
            corpus_manifest([{"id": "a", "split": "calibration", "text": "same"},
                             {"id": "b", "split": "evaluation", "text": "same"}], "t", "tr", "mr")
        changed = copy.deepcopy(baseline)
        changed["scores"][0]["document_id"] = "cal"
        with self.assertRaises(ValueError):
            compare_scores(changed, copy.deepcopy(changed))


if __name__ == "__main__":
    unittest.main()
