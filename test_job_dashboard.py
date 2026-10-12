from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
from pathlib import Path
import tempfile
import threading
import urllib.error
import urllib.request
import unittest

from job_dashboard import handler
from job_runner import ProcessPool, wait_job
from queue_control import TokenQueue


class DashboardTests(unittest.TestCase):
    def test_http_submission_reports_identity_and_local_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TokenQueue(Path(directory)/'jobs.db')
            pool = ProcessPool(queue, 1)
            server = ThreadingHTTPServer(('127.0.0.1',0), BaseHTTPRequestHandler)
            server.RequestHandlerClass = handler(queue,pool,'test-token',server.server_port)
            thread = threading.Thread(target=server.serve_forever)
            pool.tick()
            thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            def get(path):
                with urllib.request.urlopen(base+path,timeout=5) as response:
                    return response.read()
            def post(body,token='test-token',origin=None):
                headers={'Content-Type':'application/json','X-Job-Token':token}
                if origin: headers['Origin']=origin
                request=urllib.request.Request(base+'/api/jobs',data=json.dumps(body).encode(),headers=headers)
                with urllib.request.urlopen(request,timeout=5) as response:
                    return json.load(response)
            try:
                self.assertIn(b'Keep the work moving',get('/'))
                self.assertIn(b'textContent',get('/app.js'))
                for token,origin in [('wrong',None),('test-token','https://example.com')]:
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        post({'name':'a.txt','text':'hello'},token,origin)
                    self.assertEqual(error.exception.code,403)
                    error.exception.close()
                body={'id':'http-job','name':'<script>.txt','text':'Hello hello world'}
                self.assertTrue(post(body)['inserted'])
                self.assertFalse(post(body)['inserted'])
                completed=wait_job(queue,'http-job','done')
                self.assertEqual(completed['result']['words'],3)
                reports=json.loads(get('/api/reports'))
                self.assertEqual(reports[0]['name'],'<script>.txt')
                self.assertEqual(json.loads(get('/api/state'))['summary']['done_jobs'],1)
                with self.assertRaises(urllib.error.HTTPError) as error:
                    post({'id':'bad','name':'x','text':'a'*(1024*1024+1)})
                self.assertEqual(error.exception.code,400)
                error.exception.close()
            finally:
                server.shutdown();server.server_close();thread.join()
                pool.close()


if __name__=='__main__':
    unittest.main()
