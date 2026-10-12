"""Durable local token queue, fenced leases, graceful drain and scaling policy.

No GPU, Kubernetes API, cloud nodes or billing actions are performed.
"""
import argparse
from contextlib import closing, contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import signal
import sqlite3
import threading
import time
import uuid


class TokenQueue:
    def __init__(self, path, clock=time.time, max_attempts=3):
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.path, self.clock, self.max_attempts = str(path), clock, max_attempts
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS workers(id TEXT PRIMARY KEY, state TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs(
                    id TEXT PRIMARY KEY, tokens INTEGER NOT NULL, payload TEXT NOT NULL,
                    state TEXT NOT NULL, owner TEXT, expires REAL, epoch INTEGER NOT NULL DEFAULT 0,
                    result TEXT, created REAL NOT NULL);
            """)

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN IMMEDIATE")
            self._expire(db)
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _expire(self, db):
        db.execute("""UPDATE jobs SET state=CASE WHEN epoch>=? THEN 'dead' ELSE 'pending' END,
                   owner=NULL,expires=NULL WHERE state='leased' AND expires<=?""",
                   (self.max_attempts, self.clock()))
        db.execute("""UPDATE workers SET state='stopped' WHERE state='draining' AND NOT EXISTS
                   (SELECT 1 FROM jobs WHERE owner=workers.id AND state='leased')""")

    def enqueue(self, job_id, tokens, payload):
        if not job_id or isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise ValueError("Nonempty job ID and positive integer token estimate required")
        encoded = json.dumps(payload, sort_keys=True)
        with self.transaction() as db:
            old = db.execute("SELECT tokens,payload FROM jobs WHERE id=?", (job_id,)).fetchone()
            if old:
                if old["tokens"] != tokens or old["payload"] != encoded:
                    raise ValueError("Idempotency key reused with different work")
                return False
            db.execute("INSERT INTO jobs(id,tokens,payload,state,created) VALUES(?,?,?,'pending',?)",
                       (job_id, tokens, encoded, self.clock()))
            return True

    def register(self, worker):
        if not worker:
            raise ValueError("Worker ID required")
        with self.transaction() as db:
            old = db.execute("SELECT state FROM workers WHERE id=?", (worker,)).fetchone()
            if old:
                return old["state"]
            db.execute("INSERT INTO workers VALUES(?,'ready')", (worker,))
            return "ready"

    def claim(self, worker, lease_seconds=30):
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("Positive finite lease duration required")
        with self.transaction() as db:
            row = db.execute("SELECT state FROM workers WHERE id=?", (worker,)).fetchone()
            if not row or row["state"] != "ready":
                return None
            if db.execute("SELECT 1 FROM jobs WHERE owner=? AND state='leased'", (worker,)).fetchone():
                return None
            job = db.execute("SELECT * FROM jobs WHERE state='pending' ORDER BY created,id LIMIT 1").fetchone()
            if job is None:
                return None
            epoch = job["epoch"] + 1
            db.execute("UPDATE jobs SET state='leased',owner=?,expires=?,epoch=? WHERE id=?",
                       (worker, self.clock() + lease_seconds, epoch, job["id"]))
            return dict(id=job["id"], tokens=job["tokens"], payload=json.loads(job["payload"]),
                        epoch=epoch, owner=worker)

    def renew(self, lease, seconds=30):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Positive finite lease duration required")
        with self.transaction() as db:
            cursor = db.execute("""UPDATE jobs SET expires=? WHERE id=? AND owner=? AND epoch=?
                AND state='leased'""", (self.clock() + seconds, lease["id"], lease["owner"], lease["epoch"]))
            return cursor.rowcount == 1

    def complete(self, lease, result):
        with self.transaction() as db:
            cursor = db.execute("""UPDATE jobs SET state='done',result=?,owner=NULL,expires=NULL
                WHERE id=? AND owner=? AND epoch=? AND state='leased'""",
                (json.dumps(result), lease["id"], lease["owner"], lease["epoch"]))
            self._expire(db)
            return cursor.rowcount == 1

    def drain(self, worker):
        with self.transaction() as db:
            db.execute("UPDATE workers SET state='draining' WHERE id=? AND state='ready'", (worker,))
            self._expire(db)

    def snapshot(self):
        with self.transaction() as db:
            result = dict(pending_tokens=0, leased_tokens=0, pending_jobs=0, leased_jobs=0, done_jobs=0,
                          dead_jobs=0, busy_workers=0, ready_workers=0, draining_workers=0)
            for row in db.execute("SELECT state,count(*) AS n,sum(tokens) AS tokens FROM jobs GROUP BY state"):
                result[{"pending": "pending_jobs", "leased": "leased_jobs", "done": "done_jobs", "dead": "dead_jobs"}[row["state"]]] = row["n"]
                if row["state"] in ("pending", "leased"):
                    result[row["state"] + "_tokens"] = row["tokens"]
            result["busy_workers"] = db.execute("SELECT count(DISTINCT owner) FROM jobs WHERE state='leased'").fetchone()[0]
            for row in db.execute("SELECT state,count(*) AS n FROM workers GROUP BY state"):
                if row["state"] in ("ready", "draining"):
                    result[row["state"] + "_workers"] = row["n"]
            return result


@dataclass
class ScalePolicy:
    tokens_per_worker: int = 2048
    minimum: int = 0
    maximum: int = 8
    cooldown: float = 30
    last_nonempty: float | None = None

    def desired(self, snapshot, current, now):
        if self.tokens_per_worker <= 0 or not 0 <= self.minimum <= self.maximum or self.cooldown < 0:
            raise ValueError("Invalid scaling policy")
        pending, busy = snapshot["pending_tokens"], snapshot["busy_workers"]
        demand = math.ceil(pending / self.tokens_per_worker) + busy
        target = max(self.minimum, min(self.maximum, demand), busy)
        if pending or busy:
            self.last_nonempty = now
        if target < current and self.last_nonempty is not None and now - self.last_nonempty < self.cooldown:
            return current
        return target


def metrics_handler(queue):
    class Metrics(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path not in ("/metrics", "/health"):
                self.send_error(404)
                return
            snapshot = queue.snapshot()
            if self.path == "/metrics":
                lines = []
                for key, value in snapshot.items():
                    name = "inference_queue_" + key
                    lines.extend([f"# TYPE {name} gauge", f"{name} {value}"])
                data = ("\n".join(lines) + "\n").encode()
                mime = "text/plain; version=0.0.4"
            else:
                data, mime = json.dumps(snapshot).encode(), "application/json"
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    return Metrics


def worker(queue, worker_id, stop_event, idle_timeout=None):
    """Fixture worker: completes jobs after payload.seconds. It does not infer text."""
    queue.register(worker_id)
    idle_since = time.monotonic()
    while not stop_event.is_set():
        lease = queue.claim(worker_id, 5)
        if lease is None:
            if idle_timeout is not None and time.monotonic() - idle_since > idle_timeout:
                break
            stop_event.wait(0.05)
            continue
        idle_since = time.monotonic()
        try:
            seconds = float(lease["payload"].get("seconds", 0))
        except (ValueError, TypeError, AttributeError):
            queue.complete(lease, {"error": "Invalid fixture payload"})
            continue
        if not math.isfinite(seconds) or not 0 <= seconds <= 120:
            queue.complete(lease, {"error": "Invalid fixture duration"})
            continue
        deadline = time.monotonic() + seconds
        valid = True
        while time.monotonic() < deadline:
            if stop_event.is_set():
                queue.drain(worker_id)
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
            if not queue.renew(lease, 5):
                valid = False
                break
        if valid:
            queue.complete(lease, {"fixture": True, "worker": worker_id})
    queue.drain(worker_id)


def demo(path):
    queue = TokenQueue(path)
    for index, tokens in enumerate((1024, 3072, 512)):
        queue.enqueue(f"fixture-{index}", tokens, {"seconds": 0.02})
    before = queue.snapshot()
    policy = ScalePolicy()
    target = policy.desired(before, 0, 0)
    threads = [threading.Thread(target=worker, args=(queue, f"fixture-worker-{n}", threading.Event(), 0.1))
               for n in range(target)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return dict(evidence="local fixture execution, not GPU inference or cluster autoscaling",
                initial=before, desired_workers=target, final=queue.snapshot())


class LocalAutoscaler:
    """Scale actual CPU fixture worker threads; no external node provisioning."""
    def __init__(self, queue, policy=None, clock=time.monotonic):
        self.queue, self.policy, self.clock = queue, policy or ScalePolicy(), clock
        self.workers = {}

    def tick(self):
        self.workers = {key: value for key, value in self.workers.items() if value[0].is_alive()}
        snapshot = self.queue.snapshot()
        desired = self.policy.desired(snapshot, len(self.workers), self.clock())
        for _ in range(max(0, desired - len(self.workers))):
            worker_id = "local-" + uuid.uuid4().hex
            event = threading.Event()
            thread = threading.Thread(target=worker, args=(self.queue, worker_id, event), daemon=True)
            self.workers[worker_id] = (thread, event)
            thread.start()
        candidates = [key for key, (_, event) in self.workers.items() if not event.is_set()]
        for key in sorted(candidates)[:max(0, len(candidates) - desired)]:
            self.queue.drain(key)
            self.workers[key][1].set()
        return dict(desired=desired, live=len(self.workers), queue=snapshot)

    def close(self):
        for key, (_, event) in self.workers.items():
            self.queue.drain(key)
            event.set()
        for thread, _ in self.workers.values():
            thread.join(timeout=125)
            if thread.is_alive():
                raise RuntimeError("Fixture worker failed to drain within its maximum duration")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["enqueue", "snapshot", "metrics", "worker", "autoscale", "demo"])
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--id", default="local-worker")
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--seconds", type=float, default=1)
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--tokens-per-worker", type=int, default=2048)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--cooldown", type=float, default=30)
    args = parser.parse_args()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    queue = TokenQueue(args.db)
    if args.action == "enqueue":
        print(json.dumps({"inserted": queue.enqueue(args.id, args.tokens, {"seconds": args.seconds})}))
    elif args.action == "snapshot":
        print(json.dumps(queue.snapshot(), indent=2))
    elif args.action == "demo":
        print(json.dumps(demo(args.db), indent=2))
    elif args.action == "metrics":
        ThreadingHTTPServer(("127.0.0.1", args.port), metrics_handler(queue)).serve_forever()
    else:
        event = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: event.set())
        signal.signal(signal.SIGTERM, lambda *_: event.set())
        if args.action == "autoscale":
            scaler = LocalAutoscaler(queue, ScalePolicy(tokens_per_worker=args.tokens_per_worker,
                maximum=args.max_workers, cooldown=args.cooldown))
            try:
                while not event.is_set():
                    print(json.dumps(scaler.tick()), flush=True)
                    event.wait(.5)
            finally:
                scaler.close()
        else:
            worker(queue, args.id, event)


if __name__ == "__main__":
    main()
