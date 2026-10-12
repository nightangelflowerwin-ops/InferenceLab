import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import io
import json
import socket
import struct
import threading
import unittest

from range_stream import RangeReader,shards
from weight_stream import fixture,load_stream


class RangeTests(unittest.TestCase):
    def server(self,mode):
        blob=fixture()
        requests=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):
                pass
            def do_GET(self):
                requested=self.headers.get('Range')
                requests.append((requested,self.headers.get('If-Match')))
                start=int(requested.split('=')[1].split('-')[0]) if requested else 0
                status=206 if requested and mode!='ignore' else 200
                self.send_response(status)
                self.send_header('ETag','"changed"' if requested and mode=='changed' else '"fixture-v1"')
                data=blob[start:] if status==206 else blob
                self.send_header('Content-Length',str(len(data)))
                if status==206:
                    self.send_header('Content-Range',f'bytes {start}-{len(blob)-1}/{len(blob)}')
                self.end_headers()
                if len(requests)==1:
                    self.wfile.write(data[:len(data)//2])
                    self.wfile.flush()
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.close_connection=True
                else:
                    self.wfile.write(data)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f'http://127.0.0.1:{server.server_port}/object',requests,blob

    def test_interrupted_header_resumes_without_disk_or_duplicate_bytes(self):
        url,requests,blob=self.server('resume')
        with RangeReader(url) as reader:
            buffer=load_stream(reader,expected_sha256=hashlib.sha256(blob).hexdigest())
            self.assertEqual(reader.connections,2)
        self.assertEqual(struct.unpack('<4f',buffer.tensor('weight')),(1,2,3,4))
        self.assertEqual(requests[1],(f'bytes={len(blob)//2}-','"fixture-v1"'))

    def test_changed_object_or_ignored_range_fails_closed(self):
        for mode in ('changed','ignore'):
            url,_,_=self.server(mode)
            with RangeReader(url) as reader:
                with self.assertRaises(ValueError):
                    load_stream(reader)

    def test_shard_index_names_bytes_checksums_and_bounds(self):
        first=fixture()
        header=json.dumps({'other':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
        second=struct.pack('<Q',len(header))+header+struct.pack('<f',9)
        blobs={'a.safetensors':first,'b.safetensors':second}
        index={'metadata':{'total_size':20},'weight_map':{'weight':'a.safetensors','other':'b.safetensors'},
               'shard_sha256':{name:hashlib.sha256(blob).hexdigest() for name,blob in blobs.items()}}
        loaded=list(shards(index,lambda name:io.BytesIO(blobs[name]),max_payload_bytes=16,max_total_bytes=20))
        self.assertEqual([name for name,_ in loaded],['a.safetensors','b.safetensors'])
        self.assertEqual(struct.unpack('<f',loaded[1][1].tensor('other')),(9,))
        broken=json.loads(json.dumps(index))
        broken['weight_map']['wrong']='b.safetensors'
        with self.assertRaises(ValueError):
            list(shards(broken,lambda name:io.BytesIO(blobs[name])))
        broken=json.loads(json.dumps(index))
        broken['metadata']['total_size']=19
        with self.assertRaises(ValueError):
            list(shards(broken,lambda name:io.BytesIO(blobs[name])))
        with self.assertRaises(ValueError):
            list(shards({'metadata':{'total_size':0},'weight_map':{'w':'../escape.safetensors'}},lambda _:None))


if __name__=='__main__':
    unittest.main()
