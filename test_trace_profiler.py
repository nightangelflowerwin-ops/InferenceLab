import copy
import unittest

from trace_profiler import moe_profile, collective_profile, placement_plan


class TraceTests(unittest.TestCase):
    def events(self, choices):
        return [dict(request_id="r", layer=0, step=0, token_id=n, experts=experts)
                for n, experts in enumerate(choices)]

    def test_balanced_and_imbalanced_experts(self):
        balanced = moe_profile(self.events([[0], [1], [2], [3]]), 4)["layers"]["0"]
        self.assertEqual(balanced["coefficient_of_variation"], 0)
        self.assertEqual(balanced["gini"], 0)
        self.assertAlmostEqual(balanced["effective_experts"], 4)
        imbalanced = moe_profile(self.events([[0]] * 4), 4)["layers"]["0"]
        self.assertEqual(imbalanced["max_to_mean"], 4)
        self.assertEqual(imbalanced["idle_experts"], 3)
        self.assertEqual(imbalanced["gini"], .75)

    def test_topk_counts_and_rank_mapping(self):
        events = self.events([[0, 1], [1, 2]])
        for event in events:
            event.update(expert_ranks=[0, 1], communication_bytes=100, communication_ms=2)
        layer = moe_profile(events, 4)["layers"]["0"]
        self.assertEqual(layer["assignments"], 4)
        self.assertEqual(layer["load"], [1, 2, 1, 0])
        self.assertEqual(layer["communication_bytes"], 200)
        self.assertEqual(layer["rank_assignments"], {0: 2, 1: 2})

    def test_duplicate_and_invalid_trace_rejected(self):
        events = self.events([[0]])
        with self.assertRaises(ValueError):
            moe_profile(events + events, 4)
        with self.assertRaises(ValueError):
            moe_profile(self.events([[4]]), 4)
        with self.assertRaises(ValueError):
            moe_profile(self.events([[0, 0]]), 4)

    def test_placement_preserves_experts_and_assignment_count(self):
        plan = placement_plan([100, 100, 10, 10], 2)
        self.assertEqual(plan["projected_rank_assignments"], [110, 110])
        self.assertEqual(set(plan["expert_to_rank"]), {"0", "1", "2", "3"})
        self.assertEqual(sum(plan["projected_rank_assignments"]), 220)
        self.assertEqual(plan["projected_max_to_mean"], 1)

    def test_collective_slowest_rank_and_bandwidth(self):
        events = [dict(run_id="r", sample_id=sample, rank=rank, world_size=4,
                       bytes=1000000, duration_ms=rank + 1, operation="all_reduce")
                  for sample in range(2) for rank in range(4)]
        group = collective_profile(events)["groups"][0]
        self.assertEqual(group["p50_slowest_rank_ms"], 4)
        self.assertEqual(group["slowest_rank"], "3")
        self.assertEqual(group["median_algorithm_GB_s"], .25)
        self.assertEqual(group["estimated_collective_bus_GB_s"], .375)
        with self.assertRaises(ValueError):
            collective_profile(events[:-1])
        with self.assertRaises(ValueError):
            collective_profile(events + [events[0]])


if __name__ == "__main__":
    unittest.main()
