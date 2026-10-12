# Checkpointing and Weight Streaming Engine

Status: bounded safetensors CPU-memory streaming loader implemented and tested with local files, in-memory fixtures and a real localhost HTTP response. GPU upload, actual S3 cold-start comparisons and GPU-direct I/O remain pending.

```powershell
python weight_stream.py --fixture --max-mib 1 --out results/weight-fixture.json
python weight_stream.py --file YOUR_WEIGHTS.safetensors --max-mib 256 --sha256 EXPECTED_FILE_SHA256
```

For an object you control, `--url` accepts an HTTPS object URL, including a presigned S3 URL. No credentials are created or stored, and reports omit the URL. HTTP is allowed only for loopback fixtures. This is a standard HTTP read, not an authenticated S3 SDK implementation. Do not commit presigned URLs or account credentials.

The loader validates the header length, unique tensor names, supported dtype, shape/byte agreement and contiguous nonoverlapping offsets before allocating the bounded payload. It reads chunks into one CPU bytearray and exposes read-only tensor memoryviews without another tensor copy. A SHA-256 check can reject changed files. Truncation and trailing bytes are rejected. It never executes model code or creates a temporary weight file.

Limits: the whole supported payload must fit the chosen CPU-memory cap. Peak memory also includes a bounded header and one chunk; it is not a guarantee about the process's total RSS. The supported dtype subset is explicitly listed in source; sub-byte formats are not supported. Checksums are integrity checks, not source authentication. No resume/range fetching, sharded model index, GPU upload, inference warmup or GPU-ready cold-start timing is implemented yet.

The reported CPU load duration includes source opening and reading. For meaningful cold-start comparisons, use the same file and hash, repeat with controlled OS caching, and separately measure network download, parsing, CPU copies, GPU transfer and inference readiness. The tiny fixture is a correctness exercise, not a throughput benchmark.

Tests cover split reads, zero-copy views, checksum mismatch, bad offsets/shapes, memory limits, truncated/trailing data and chunked localhost HTTP loading. [Format reference](https://github.com/safetensors/safetensors#format).
