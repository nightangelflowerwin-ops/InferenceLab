import json
from pathlib import Path
import unittest


class ManifestTests(unittest.TestCase):
    def test_keda_metric_contract(self):
        file = Path(__file__).parent / "projects/07-gpu-autoscaler/keda-scaledobject.json"
        resource = json.loads(file.read_text())
        self.assertEqual(resource["kind"], "ScaledObject")
        spec = resource["spec"]
        self.assertLessEqual(spec["minReplicaCount"], spec["maxReplicaCount"])
        self.assertGreater(spec["cooldownPeriod"], 0)
        trigger = spec["triggers"][0]
        self.assertEqual(trigger["type"], "prometheus")
        self.assertIn("inference_queue_leased_tokens", trigger["metadata"]["query"])
        self.assertEqual(trigger["metadata"]["ignoreNullValues"], "false")
        self.assertGreater(float(trigger["metadata"]["threshold"]), 0)


if __name__ == "__main__":
    unittest.main()
