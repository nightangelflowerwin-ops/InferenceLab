"""Stream safetensors from a file or HTTP response into bounded CPU memory.

No model code executes. GPU upload and GPU-direct I/O are not implemented.
"""
import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import time
from urllib.parse import urlsplit
import urllib.request


DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
               "I16": 2, "U16": 2, "F16": 2, "BF16": 2, "I32": 4, "U32": 4,
               "F32": 4, "I64": 8, "U64": 8, "F64": 8}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON header key")
        result[key] = value
    return result


def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        block = stream.read(size - len(data))
        if not block:
            raise ValueError("Truncated safetensors stream")
        data.extend(block)
    return bytes(data)


@dataclass
class WeightBuffer:
    header: dict
    payload: bytearray
    sha256: str

    def tensor(self, name):
        start, end = self.header[name]["data_offsets"]
        return memoryview(self.payload)[start:end].toreadonly()


def load_stream(stream, max_payload_bytes=256 * 1024 * 1024, expected_sha256=None,
                chunk_size=1024 * 1024, max_header_bytes=1024 * 1024):
    if max_payload_bytes < 0 or chunk_size <= 0 or max_header_bytes <= 0:
        raise ValueError("Invalid memory or chunk bound")
    prefix = read_exact(stream, 8)
    header_size = struct.unpack("<Q", prefix)[0]
    if not 2 <= header_size <= max_header_bytes:
        raise ValueError("Header size exceeds bound or is invalid")
    raw_header = read_exact(stream, header_size)
    if not raw_header.startswith(b"{"):
        raise ValueError("Safetensors header must begin with an object")
    header = json.loads(raw_header, object_pairs_hook=unique_object)
    if not isinstance(header, dict):
        raise ValueError("Header object required")
    intervals = []
    for name, info in header.items():
        if name == "__metadata__":
            if not isinstance(info, dict) or any(not isinstance(v, str) for v in info.values()):
                raise ValueError("Metadata must map strings to strings")
            continue
        if not isinstance(info, dict) or set(info) != {"dtype", "shape", "data_offsets"}:
            raise ValueError("Malformed tensor metadata")
        dtype = info["dtype"]
        shape, offsets = info["shape"], info["data_offsets"]
        if dtype not in DTYPE_BYTES or not isinstance(shape, list) or any(
                isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in shape):
            raise ValueError("Unsupported dtype or invalid dimensions")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(
                isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in offsets):
            raise ValueError("Invalid tensor offsets")
        start, end = offsets
        if end < start or end - start != math.prod(shape) * DTYPE_BYTES[dtype]:
            raise ValueError("Tensor byte count differs from shape/dtype")
        if end > max_payload_bytes:
            raise ValueError("Payload exceeds memory bound")
        intervals.append((start, end))
    # The safe format must cover the entire payload without overlap or holes.
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError("Payload contains gaps or overlapping tensors")
        cursor = end
    payload = bytearray(cursor)
    digest = hashlib.sha256(prefix + raw_header)
    view = memoryview(payload)
    offset = 0
    while offset < cursor:
        count = min(chunk_size, cursor - offset)
        block = stream.read(count)
        if not block:
            raise ValueError("Truncated tensor payload")
        view[offset:offset + len(block)] = block
        digest.update(block)
        offset += len(block)
    if stream.read(1):
        raise ValueError("Unexpected trailing bytes")
    sha = digest.hexdigest()
    if expected_sha256 is not None and sha != expected_sha256.lower():
        raise ValueError("Weight checksum mismatch")
    return WeightBuffer(header, payload, sha)


def fixture():
    tensor = struct.pack("<4f", 1, 2, 3, 4)
    header = json.dumps({"weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}},
                        separators=(",", ":")).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    return struct.pack("<Q", len(header)) + header + tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path)
    source.add_argument("--url", help="HTTPS safetensors object URL; HTTP only on loopback")
    source.add_argument("--fixture", action="store_true")
    parser.add_argument("--max-mib", type=int, default=256)
    parser.add_argument("--sha256")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    if args.file:
        stream, kind = args.file.open("rb"), "local_file"
    elif args.fixture:
        stream, kind = io.BytesIO(fixture()), "synthetic_fixture"
    else:
        parsed = urlsplit(args.url)
        if parsed.username or parsed.password or parsed.fragment or not (
            parsed.scheme == "https" or parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost")):
            parser.error("External objects require HTTPS; credentials in URL userinfo are unsupported")
        stream, kind = urllib.request.urlopen(args.url, timeout=30), "http_object"
    with stream:
        buffer = load_stream(stream, args.max_mib * 1024 * 1024, args.sha256)
    report = dict(source_kind=kind, tensors=len([n for n in buffer.header if n != "__metadata__"]),
        payload_bytes=len(buffer.payload), sha256=buffer.sha256,
        cpu_load_ms=(time.perf_counter() - start) * 1000,
        evidence="Bounded CPU-memory loader. No temporary disk copy, GPU transfer, GPU-direct I/O or inference measured.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
