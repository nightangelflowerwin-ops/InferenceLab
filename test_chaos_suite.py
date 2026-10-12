import unittest

from chaos_suite import RecoveringCluster, FixtureFault, run_suite, checksum


class ChaosTests(unittest.TestCase):
    def test_all_isolated_faults_recover(self):
        result = run_suite()
        self.assertEqual(len(result["cases"]), 3)
        self.assertTrue(all(case["recovered"] for case in result["cases"]))

    def test_corrupt_checkpoint_recomputed(self):
        cluster = RecoveringCluster()
        cluster.execute("warm", "context")
        cluster.checkpoints[checksum("context")]["data"] = '{}'
        cluster.inject("a", "simulated_ecc")
        answer = cluster.execute("retry", "context")
        self.assertFalse(answer["fixture_cache_hit"])
        self.assertEqual(answer["answer_checksum"], checksum("context"))
        self.assertTrue(any(event["event"] == "checkpoint_rejected" for event in cluster.events))

    def test_all_replicas_fail_and_request_fencing(self):
        cluster = RecoveringCluster()
        cluster.execute("j", "one")
        with self.assertRaises(ValueError):
            cluster.execute("j", "two")
        cluster.inject("a", "partition")
        cluster.inject("b", "simulated_oom")
        with self.assertRaises(FixtureFault):
            cluster.execute("another", "three")


if __name__ == "__main__":
    unittest.main()
