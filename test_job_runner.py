import tempfile
from pathlib import Path
import unittest

from job_runner import submit, text_report, recovery_demo, jobs, ProcessPool
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

    def test_managed_workers_exit_when_supervisor_session_expires(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TokenQueue(Path(directory)/'jobs.db')
            pool = ProcessPool(queue,1)
            try:
                pool.tick()
                with queue.transaction() as db:
                    db.execute('UPDATE runner_sessions SET expires=0 WHERE id=?',(pool.session,))
                process = next(iter(pool.processes.values()))
                self.assertEqual(process.wait(timeout=10),0)
            finally:
                pool.close()


if __name__ == '__main__':
    unittest.main()
