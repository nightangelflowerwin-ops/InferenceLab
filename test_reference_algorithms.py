import ast
from collections import Counter
import json
import math
from pathlib import Path
import random
import unittest

from rmsnorm_reference import rmsnorm
from speculative_reference import step, generate, fixture_evaluation, serving_plan


class RMSNormTests(unittest.TestCase):
    def test_known_values_weights_and_zero(self):
        result = rmsnorm([[3, 4], [0, 0]], [1, 2], epsilon=.5)
        self.assertAlmostEqual(result[0][0], 3 / math.sqrt(13))
        self.assertAlmostEqual(result[0][1], 8 / math.sqrt(13))
        self.assertEqual(result[1], [0, 0])

    def test_odd_width_scale_symmetry_and_invalid_contract(self):
        x = [[1, -2, 3]]
        positive = rmsnorm(x, [1, 1, 1])
        negative = rmsnorm([[-1, 2, -3]], [1, 1, 1])
        self.assertEqual(positive, [[-v for v in negative[0]]])
        self.assertAlmostEqual(sum(v * v for v in positive[0]) / 3, 1, places=5)
        for rows, weights, epsilon in (([[1, 2]], [1], 1e-6), ([[float("nan")]], [1], 1e-6), ([[1]], [1], 0)):
            with self.assertRaises(ValueError):
                rmsnorm(rows, weights, epsilon)

    def test_gpu_source_syntax_only(self):
        path = Path(__file__).parent / "projects/03-triton-kernel/rmsnorm_triton.py"
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        ast.parse(source)
        # This checks Python source parsing, not Triton compilation or GPU behavior.

    def test_cpu_oracle_avoids_square_overflow(self):
        result = rmsnorm([[1e200, 1e200]], [1, 1])
        self.assertEqual(result, [[1, 1]])


class SpeculativeTests(unittest.TestCase):
    def test_identical_models_accept_all_and_one_batch_callback(self):
        calls = []
        def target(contexts):
            calls.append(contexts)
            return [[1, 0] for _ in contexts]
        result = step([9], lambda _: [1, 0], target, 4, random.Random(42))
        self.assertEqual(result["tokens"], [0] * 5)
        self.assertEqual(result["accepted"], 4)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], [[9], [9, 0], [9, 0, 0], [9, 0, 0, 0], [9, 0, 0, 0, 0]])

    def test_disjoint_support_rejection_uses_target_residual(self):
        result = step([], lambda _: [0, 1], lambda contexts: [[1, 0] for _ in contexts], 3, random.Random(1))
        self.assertTrue(result["rejected"])
        self.assertEqual(result["tokens"], [0])
        self.assertEqual(result["accepted"], 0)

    def test_seeded_distribution_matches_target_statistically(self):
        result = fixture_evaluation(20000)
        self.assertLess(abs(result["observed_first_token_distribution"][0] - .8), .015)

    def test_bad_distributions_and_vocabulary_rejected(self):
        for draft in ([0, 0], [-1, 2], [float("nan"), 1]):
            with self.assertRaises(ValueError):
                step([], lambda _: draft, lambda contexts: [[1, 0] for _ in contexts], 2, random.Random(1))
        with self.assertRaises(ValueError):
            step([], lambda _: [1, 0], lambda contexts: [[1, 0, 0] for _ in contexts], 2, random.Random(1))
        plan = serving_plan("/models/target-immutable", "/models/draft-immutable")
        self.assertEqual(json.loads(plan[-1])["method"], "draft_model")
        self.assertIn("127.0.0.1", plan)

    def test_token_budget_and_eos_boundaries(self):
        draft = lambda _: [1, 0]
        target = lambda contexts: [[1, 0] for _ in contexts]
        for count in (0, 1, 2, 5, 9):
            result = generate([], draft, target, count, lookahead=4)
            self.assertEqual(len(result["tokens"]), count)
        result = generate([], draft, target, 10, lookahead=4, eos=0)
        self.assertEqual(result["tokens"], [0])
        self.assertTrue(result["stopped_on_eos"])

    def test_context_dependent_joint_distribution(self):
        def target(contexts):
            return [[.7, .3] if not context else [.2, .8] if context[0] == 0 else [.6, .4]
                    for context in contexts]
        observed = Counter(tuple(generate([], lambda _: [.4, .6], target, 2, seed=seed)["tokens"])
                           for seed in range(10000))
        expected = {(0, 0): .14, (0, 1): .56, (1, 0): .18, (1, 1): .12}
        for sequence, probability in expected.items():
            self.assertLess(abs(observed[sequence] / 10000 - probability), .02)


if __name__ == "__main__":
    unittest.main()
