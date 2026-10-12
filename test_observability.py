import base64
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import unittest
import urllib.request

from observability import parse_samples, assess, fixture_handler


class ObservabilityTests(unittest.TestCase):
    def test_fixture_alerts_and_distinct_activity_occupancy(self):
        file = Path(__file__).parent / 'projects/11-dcgm-dashboard/fixture.prom'
        result = assess(parse_samples(file.read_text()))['devices'][0]
        self.assertTrue(result['fixture'])
        self.assertIn('temperature_above_configured_threshold', result['alerts'])
        self.assertIn('software_oom_counter_nonzero', result['alerts'])
        self.assertIn('framebuffer_utilization_above_95_percent', result['alerts'])
        self.assertEqual(result['values']['DCGM_FI_PROF_SM_OCCUPANCY'], .15)
        self.assertEqual(result['values']['DCGM_FI_PROF_SM_ACTIVE'], .8)

    def test_missing_nan_sentinel_staleness_and_label_parsing(self):
        samples = parse_samples('DCGM_FI_DEV_GPU_TEMP{gpu="a",name="quoted\\\" label"} 85 1000\n'
                                'DCGM_FI_DEV_FB_USED{gpu="a"} NaN\n'
                                'DCGM_FI_DEV_FB_FREE{gpu="a"} 9223372036854775794\n')
        self.assertEqual(samples[0]['labels']['name'], 'quoted" label')
        result = assess(samples, now=100)['devices'][0]
        self.assertIn('timestamped_samples_stale', result['alerts'])
        self.assertIsNone(result['framebuffer_used_fraction'])
        self.assertIn('telemetry_incomplete', result['alerts'])
        self.assertNotIn('DCGM_FI_DEV_FB_FREE', result['values'])
        with self.assertRaises(ValueError):
            parse_samples('x{gpu="a",gpu="b"} 1')
        with self.assertRaises(ValueError):
            parse_samples('x 1\nx 2')

    def test_real_http_fixture_exporter(self):
        file = Path(__file__).parent / 'projects/11-dcgm-dashboard/fixture.prom'
        server = ThreadingHTTPServer(('127.0.0.1', 0), fixture_handler(file))
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}/metrics') as response:
                result = assess(parse_samples(response.read().decode()))
            self.assertTrue(result['devices'][0]['fixture'])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_mig_identity_and_ambiguous_series_do_not_become_zero(self):
        text = ('DCGM_FI_DEV_GPU_TEMP{UUID="GPU-a",GPU_I_ID="1"} 80\n'
                'DCGM_FI_DEV_GPU_TEMP{UUID="GPU-a",GPU_I_ID="2"} 90\n')
        self.assertEqual(len(assess(parse_samples(text))['devices']), 2)
        text = ('DCGM_FI_DEV_GPU_TEMP{gpu="a",job="one"} 80\n'
                'DCGM_FI_DEV_GPU_TEMP{gpu="a",job="two"} 90\n')
        result = assess(parse_samples(text))['devices'][0]
        self.assertNotIn('DCGM_FI_DEV_GPU_TEMP', result['values'])
        self.assertIn('ambiguous_series:DCGM_FI_DEV_GPU_TEMP', result['alerts'])

    def test_dashboard_query_contract_and_no_zero_fallback(self):
        root = Path(__file__).parent / 'projects/11-dcgm-dashboard'
        dashboard = json.loads((root/'dashboard.json').read_text())
        expressions = [target['expr'] for panel in dashboard['panels'] for target in panel['targets']]
        self.assertEqual(len(dashboard['panels']), 9)
        self.assertIn('DCGM_FI_PROF_SM_OCCUPANCY', expressions)
        self.assertIn('inference_gpu_oom_total', expressions)
        self.assertIn('inference_lab_fixture', expressions)
        self.assertTrue(all('or vector(0)' not in expression for expression in expressions))


@unittest.skipUnless(os.environ.get('INFERENCELAB_GRAFANA_URL'), 'Real Grafana service is not configured')
class GrafanaServiceTests(unittest.TestCase):
    def test_real_dashboard_import_and_readback(self):
        root = Path(__file__).parent / 'projects/11-dcgm-dashboard'
        dashboard = json.loads((root/'dashboard.json').read_text())
        base = os.environ['INFERENCELAB_GRAFANA_URL']
        authorization = base64.b64encode(('admin:' + os.environ['INFERENCELAB_GRAFANA_PASSWORD']).encode()).decode()
        request = urllib.request.Request(base + '/api/dashboards/db',
            data=json.dumps({'dashboard':dashboard,'overwrite':True}).encode(),
            headers={'Content-Type':'application/json','Authorization':'Basic ' + authorization})
        with urllib.request.urlopen(request, timeout=10) as response:
            self.assertEqual(json.load(response)['status'], 'success')
        request = urllib.request.Request(base + '/api/dashboards/uid/inferencelab-gpu', headers={'Authorization':'Basic ' + authorization})
        with urllib.request.urlopen(request, timeout=10) as response:
            loaded = json.load(response)['dashboard']
        self.assertEqual(len(loaded['panels']), 9)
        self.assertEqual(loaded['panels'][2]['targets'][0]['expr'], 'DCGM_FI_PROF_SM_OCCUPANCY')


if __name__ == '__main__':
    unittest.main()
