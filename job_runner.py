"""Reliable single-host text jobs using durable fenced leases and worker processes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import uuid

from queue_control import TokenQueue

MAX_TEXT_BYTES = 1024 * 1024


def submit(queue, name, text, job_id=None, delay=0):
    if not isinstance(name, str) or not 1 <= len(name) <= 200:
        raise ValueError('A filename between 1 and 200 characters is required')
    if not isinstance(text, str) or len(text.encode('utf-8')) > MAX_TEXT_BYTES:
        raise ValueError('Each UTF-8 text file must be at most 1 MiB')
    if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not 0 <= delay <= 10:
        raise ValueError('Demonstration delay must be between 0 and 10 seconds')
    job_id = job_id or uuid.uuid4().hex
    if not isinstance(job_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', job_id):
        raise ValueError('Job ID must contain 1 to 80 letters, numbers, underscores or hyphens')
    payload = {'name': name, 'text': text, 'delay': delay}
    inserted = queue.enqueue(job_id, max(1, len(text)), payload)
    return {'id': job_id, 'inserted': inserted}


def text_report(payload):
    text = payload['text']
    words = re.findall(r"\w+(?:['’]\w+)*", text, flags=re.UNICODE)
    counts = Counter(word.casefold() for word in words)
    return {'name': payload['name'], 'words': len(words), 'characters': len(text),
            'lines': len(text.splitlines()), 'utf8_bytes': len(text.encode('utf-8')),
            'sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
            'top_words': [[word, count] for word, count in counts.most_common(10)]}


def jobs(queue, limit=100):
    with queue.transaction() as db:
        rows = db.execute('SELECT * FROM jobs ORDER BY created DESC,id LIMIT ?', (limit,)).fetchall()
        return [{'id': row['id'], 'name': json.loads(row['payload'])['name'], 'state': row['state'],
                 'worker': row['owner'], 'attempts': row['epoch'], 'created': row['created'],
                 'result': json.loads(row['result']) if row['result'] else None} for row in rows]


def run_worker(db_path, worker_id, lease_seconds=2):
    queue = TokenQueue(db_path)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    queue.register(worker_id)
    try:
        while not stop.is_set():
            lease = queue.claim(worker_id, lease_seconds)
            if lease is None:
                with queue.transaction() as db:
                    state = db.execute('SELECT state FROM workers WHERE id=?', (worker_id,)).fetchone()[0]
                if state != 'ready':
                    break
                stop.wait(.05)
                continue
            finished, lost = threading.Event(), threading.Event()
            def heartbeat():
                while not finished.wait(lease_seconds / 3):
                    if not queue.renew(lease, lease_seconds):
                        lost.set()
                        break
            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()
            try:
                # A bounded delay makes crash recovery visible; counts use actual submitted text.
                time.sleep(lease['payload'].get('delay', 0))
                report = text_report(lease['payload'])
                if not lost.is_set():
                    queue.complete(lease, report)
            except (KeyError, TypeError, ValueError) as error:
                queue.complete(lease, {'error': type(error).__name__ + ': invalid text job'})
            finally:
                finished.set()
                thread.join(timeout=lease_seconds)
    finally:
        queue.drain(worker_id)


class ProcessPool:
    def __init__(self, queue, count=3):
        if not 1 <= count <= 8:
            raise ValueError('Choose 1 to 8 worker processes')
        self.queue, self.count = queue, count
        self.processes = {}
        self.lock = threading.RLock()
        self.closing = False

    def tick(self):
        with self.lock:
            for worker_id, process in list(self.processes.items()):
                if process.poll() is not None:
                    self.queue.drain(worker_id)
                    del self.processes[worker_id]
            while not self.closing and len(self.processes) < self.count:
                worker_id = 'process-' + uuid.uuid4().hex[:12]
                flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker',
                    '--db', str(Path(self.queue.path).resolve()), '--id', worker_id],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=flags)
                self.processes[worker_id] = process

    def crash_busy(self):
        with self.lock:
            for job in jobs(self.queue):
                process = self.processes.get(job['worker'])
                if job['state'] == 'leased' and process and process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                    return {'worker': job['worker'], 'job': job['id'], 'pid': process.pid}
            raise ValueError('No busy managed worker. Add a recovery demo job first.')

    def status(self):
        with self.lock:
            return [{'id': key, 'pid': proc.pid, 'alive': proc.poll() is None}
                    for key, proc in self.processes.items()]

    def close(self):
        with self.lock:
            self.closing = True
            processes = list(self.processes.items())
            for worker_id, _ in processes:
                self.queue.drain(worker_id)
        deadline = time.monotonic() + 15
        for _, process in processes:
            try:
                process.wait(timeout=max(.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def wait_job(queue, job_id, state, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = next((job for job in jobs(queue) if job['id'] == job_id), None)
        if row and row['state'] == state:
            return row
        time.sleep(.05)
    raise TimeoutError(f'Job did not reach {state}')


def recovery_demo(path):
    queue = TokenQueue(path)
    pool = ProcessPool(queue, 1)
    job_id = 'recovery-' + uuid.uuid4().hex[:12]
    text = 'Recovery works. Recovery keeps the correct report.'
    submit(queue, 'recovery-demo.txt', text, job_id, delay=3)
    try:
        pool.tick()
        first = wait_job(queue, job_id, 'leased')
        killed = pool.crash_busy()
        pool.tick()
        final = wait_job(queue, job_id, 'done')
        assert final['attempts'] == 2 and final['result'] == text_report({'name': 'recovery-demo.txt', 'text': text})
        return {'job': job_id, 'first_attempt': first['attempts'], 'killed_process': killed,
                'final': final, 'recovered': True,
                'scope': 'Actual killed worker process, expired lease and recomputed real text report on one host. SQLite coordinator remains available; no multi-machine availability claim.'}
    finally:
        pool.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['worker', 'submit', 'status', 'demo'])
    parser.add_argument('--db', type=Path, default=Path('results/job-runner/jobs.db'))
    parser.add_argument('--id', default='process-' + uuid.uuid4().hex[:12])
    parser.add_argument('--folder', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    if args.action == 'worker':
        run_worker(args.db, args.id)
        return
    queue = TokenQueue(args.db)
    if args.action == 'submit':
        if not args.folder or not args.folder.is_dir():
            parser.error('--folder must name a directory containing UTF-8 .txt files')
        files = sorted(args.folder.glob('*.txt'))
        if not 1 <= len(files) <= 100:
            parser.error('Choose a folder containing 1 to 100 .txt files')
        # Validate the whole batch before creating any jobs.
        batch = []
        for file in files:
            with file.open('rb') as stream:
                content = stream.read(MAX_TEXT_BYTES + 1)
            if len(content) > MAX_TEXT_BYTES:
                parser.error('Each file must be at most 1 MiB')
            batch.append((file.name, content.decode('utf-8')))
        if sum(len(text.encode('utf-8')) for _, text in batch) > 10 * MAX_TEXT_BYTES:
            parser.error('Folder batch must be at most 10 MiB')
        result = [submit(queue, name, text) for name, text in batch]
    elif args.action == 'demo':
        result = recovery_demo(args.db)
    else:
        result = {'summary': queue.snapshot(), 'jobs': jobs(queue)}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
