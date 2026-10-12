import tempfile
from pathlib import Path
import unittest

from job_runner import submit, text_report, recovery_demo, jobs
from queue_control import TokenQueue


class JobRunnerTests(unittest.TestCase):
    def test_real_counts_submission_identity_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'jobs.db'
            queue = TokenQueue(path)
            self.assertTrue(submit(queue, 'a.txt', "Hello hello\nworld", 'same')['inserted'])
            self.assertFalse(submit(TokenQueue(path), 'a.txt', "Hello hello\nworld", 'same')['inserted'])
            with self.assertRaises(ValueError):
                submit(queue, 'a.txt', 'changed', 'same')
            result = text_report({'name': 'a.txt', 'text': "Hello hello\nworld"})
            self.assertEqual(result['words'], 3)
            self.assertEqual(result['lines'], 2)
            self.assertEqual(result['top_words'][0], ['hello', 2])
            self.assertEqual(len(jobs(queue)), 1)

    def test_actual_process_kill_and_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            result = recovery_demo(Path(directory) / 'jobs.db')
            self.assertTrue(result['recovered'])
            self.assertEqual(result['final']['attempts'], 2)
            self.assertEqual(result['final']['result']['words'], 7)


if __name__ == '__main__':
    unittest.main()
