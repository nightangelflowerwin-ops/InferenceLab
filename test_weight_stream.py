import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import struct
import threading
import unittest
import urllib.request

from weight_stream import fixture, load_stream


class WeightStreamTests(unittest.TestCase):
    def test_chunked_load_views_and_checksum(self):
        blob = fixture()
        weight = load_stream(io.BytesIO(blob), 16, hashlib.sha256(blob).hexdigest(), chunk_size=3)
        view = weight.tensor("weight")
        self.assertTrue(view.readonly)
        self.assertIs(view.obj, weight.payload)
        self.assertEqual(struct.unpack("<4f", view), (1, 2, 3, 4))
        with self.assertRaises(ValueError):
            load_stream(io.BytesIO(blob), expected_sha256="0" * 64)

    def test_truncation_memory_bound_and_trailing_data(self):
        blob = fixture()
        for bad in (blob[:7], blob[:-1], blob + b"extra"):
            with self.assertRaises(ValueError):
                load_stream(io.BytesIO(bad))
        with self.assertRaises(ValueError):
            load_stream(io.BytesIO(blob), max_payload_bytes=15)

    def test_invalid_offset_and_shape_before_allocation(self):
        for info in (dict(dtype="F32", shape=[4], data_offsets=[4, 20]),
                     dict(dtype="F32", shape=[5], data_offsets=[0, 16]),
                     dict(dtype="F32", shape=[-1], data_offsets=[0, 16])):
            header = json.dumps({"w": info}).encode()
            with self.assertRaises(ValueError):
                load_stream(io.BytesIO(struct.pack("<Q", len(header)) + header + bytes(16)))

    def test_real_http_response_stream_without_disk(self):
        blob = fixture()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(blob)))
                self.end_headers()
                for offset in range(0, len(blob), 5):
                    self.wfile.write(blob[offset:offset + 5])
                    self.wfile.flush()
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/weights", timeout=5) as response:
                buffer = load_stream(response)
            self.assertEqual(struct.unpack("<4f", buffer.tensor("weight")), (1, 2, 3, 4))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
