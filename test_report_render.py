import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('INFERENCELAB_RENDER_TEST'), 'Optional Matplotlib rendering environment is not configured')
class ReportRenderTests(unittest.TestCase):
    def test_real_report_and_three_plots(self):
        root=Path(__file__).parent
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable,str(root/'benchmark_report.py'),
                str(root/'projects/15-public-benchmarks/benchmark-fixture.csv'),'--out',directory,'--plot'],
                check=True,capture_output=True,timeout=60)
            destination=Path(directory)
            report=json.loads((destination/'report.json').read_text())
            self.assertEqual(len(report['contexts']),3)
            from PIL import Image
            for image in destination.glob('*.png'):
                with Image.open(image) as rendered:
                    rendered.verify()
                self.assertGreater(image.stat().st_size,10000)
            self.assertEqual(len(list(destination.glob('*.png'))),3)
            text=(destination/'report.md').read_text()
            self.assertIn('SYNTHETIC FIXTURE DATA',text)
            self.assertIn('context-1-pareto.png',text)


if __name__=='__main__':
    unittest.main()
