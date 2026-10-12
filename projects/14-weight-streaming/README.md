# Checkpointing and Weight Streaming Engine

Status: bounded safetensors CPU-memory streaming loader implemented and tested with local files, in-memory fixtures and a real localhost HTTP response. GPU upload, actual S3 cold-start comparisons and GPU-direct I/O remain pending.

```powershell
python weight_stream.py --fixture --max-mib 1 --out results/weight-fixture.json
python weight_stream.py --file YOUR_WEIGHTS.safetensors --max-mib 256 --sha256 EXPECTED_FILE_SHA256
```

For an object you control, `--url` accepts an HTTPS object URL, including a presigned S3 URL. No credentials are created or stored, and reports omit the URL. HTTP is allowed only for loopback fixtures. This is a standard HTTP read, not an authenticated S3 SDK implementation. Do not commit presigned URLs or account credentials.

The loader validates the header length, unique tensor names, supported dtype, shape/byte agreement and contiguous nonoverlapping offsets before allocating the bounded payload. It reads chunks into one CPU bytearray and exposes read-only tensor memoryviews without another tensor copy. A SHA-256 check can reject changed files. Truncation and trailing bytes are rejected. It never executes model code or creates a temporary weight file.

`range_stream.py` adds resumable HTTP reads pinned to a strong opaque ETag. A dropped connection reconnects with `Range` and `If-Match`; changed objects, ignored ranges, inconsistent sizes and exhausted retry budgets fail closed. The server must provide Content-Length, a strong ETag and byte-range support. Tests interrupt an actual localhost response during its header bytes and verify the reconstructed file hash and exact resume offset.

```powershell
python range_stream.py --url YOUR_OBJECT_URL --sha256 EXPECTED_SHA256 --max-mib 256 --out results/resumed-object.json
python range_stream.py --index model.safetensors.index.json --directory YOUR_SHARD_DIRECTORY --max-mib 256 --max-total-mib 1024 --out results/shards.json
```

Sharded indexes use `weight_map` and bounded `metadata.total_size`. Safe flat filenames, exact tensor membership, aggregate bytes and optional `shard_sha256` hashes are checked. `--object-url-map` replaces `--directory` with a local JSON map of shard names to individual signed object URLs. Reports omit URLs. No temporary weight file is created for HTTP loading.

Limits: each supported shard payload must fit the chosen CPU-memory cap. Peak memory also includes a bounded header and one chunk; it is not a guarantee about total process RSS. Consumers retaining yielded buffers can accumulate memory. The supported dtype subset is explicitly listed in source; sub-byte formats are not supported. Checksums and ETags provide integrity/object identity, not source authentication. GPU upload, inference warmup and GPU-ready cold-start timing remain unimplemented.

The reported CPU load duration includes source opening and reading. For meaningful cold-start comparisons, use the same file and hash, repeat with controlled OS caching, and separately measure network download, parsing, CPU copies, GPU transfer and inference readiness. The tiny fixture is a correctness exercise, not a throughput benchmark.

Tests cover split reads, zero-copy views, checksum mismatch, bad offsets/shapes, memory limits, truncated/trailing data and chunked localhost HTTP loading. [Format reference](https://github.com/safetensors/safetensors#format).
