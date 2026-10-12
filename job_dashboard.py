"""Loopback dashboard and supervised worker processes. Python standard library only."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import signal
import threading
import time
import webbrowser

from job_runner import ProcessPool, jobs, submit
from queue_control import TokenQueue


def handler(queue, pool, token, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def trusted(self):
            return self.headers.get('Host') in (f'127.0.0.1:{port}', f'localhost:{port}')

        def send(self, value, status=200, mime='application/json'):
            data = value if isinstance(value, bytes) else json.dumps(value).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if not self.trusted():
                self.send({'error': 'Unrecognized local host'}, 403)
                return
            if self.path == '/':
                self.send((Path(__file__).parent / 'job_dashboard.html').read_bytes(), mime='text/html; charset=utf-8')
            elif self.path == '/app.js':
                self.send((Path(__file__).parent / 'job_dashboard.js').read_bytes(), mime='text/javascript; charset=utf-8')
            elif self.path == '/style.css':
                self.send((Path(__file__).parent / 'job_dashboard.css').read_bytes(), mime='text/css; charset=utf-8')
            elif self.path == '/api/state':
                self.send({'summary': queue.snapshot(), 'jobs': jobs(queue), 'workers': pool.status()})
            elif self.path == '/api/reports':
                self.send([job for job in jobs(queue, 5000) if job['result']])
            elif self.path == '/api/session':
                self.send({'token': token})
            else:
                self.send({'error': 'Not found'}, 404)

        def do_POST(self):
            self.connection.settimeout(5)
            origins = (None, f'http://127.0.0.1:{port}', f'http://localhost:{port}')
            if not self.trusted() or self.headers.get('Origin') not in origins or not secrets.compare_digest(self.headers.get('X-Job-Token', ''), token):
                self.send({'error': 'Open the local dashboard to authorize this request'}, 403)
                return
            try:
                count = int(self.headers.get('Content-Length', '0'))
                if not 0 < count <= 7 * 1024 * 1024 or self.headers.get('Transfer-Encoding'):
                    raise ValueError('Request size is outside the allowed limit')
                data = json.loads(self.rfile.read(count))
                if not isinstance(data, dict):
                    raise ValueError('A JSON object is required')
                if self.path == '/api/jobs':
                    # Dashboard is deliberately for bounded local sessions, not unbounded ingestion.
                    with queue.transaction() as db:
                        if db.execute('SELECT count(*) FROM jobs').fetchone()[0] >= 5000:
                            raise ValueError('This session has 5000 jobs; use a fresh database')
                    result = submit(queue, data.get('name'), data.get('text'), data.get('id'))
                elif self.path == '/api/demo':
                    result = submit(queue, 'recovery-demo.txt', 'A worker can crash. Another worker recovers this real text report.', delay=6)
                elif self.path == '/api/crash':
                    result = pool.crash_busy()
                else:
                    self.send({'error': 'Not found'}, 404)
                    return
                self.send(result, 201)
            except (ValueError, TypeError, UnicodeError, KeyError) as error:
                self.send({'error': str(error)}, 400)
    return Handler


def serve(db, port=8765, workers=3, open_browser=True):
    db.parent.mkdir(parents=True, exist_ok=True)
    queue = TokenQueue(db)
    pool = ProcessPool(queue, workers)
    server = ThreadingHTTPServer(('127.0.0.1', port), BaseHTTPRequestHandler)
    server.daemon_threads = True
    server.RequestHandlerClass = handler(queue, pool, secrets.token_hex(32), server.server_port)
    stop = threading.Event()
    def request_stop(*_):
        stop.set()
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    def supervise():
        while not stop.wait(.2):
            pool.tick()
        server.shutdown()
    thread = threading.Thread(target=supervise)
    pool.tick()
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    print('Job runner dashboard: ' + url, flush=True)
    if open_browser:
        webbrowser.open(url, new=1)
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        stop.set()
        thread.join(timeout=5)
        server.server_close()
        pool.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=Path('results/job-runner/jobs.db'))
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    serve(args.db, args.port, args.workers, not args.no_browser)
