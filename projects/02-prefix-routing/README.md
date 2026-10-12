# KV Cache Prefix Routing Proxy

Status: implemented local proxy with offline and HTTP integration tests. Real vLLM residency integration and GPU TTFT measurements are pending.

The proxy hashes the initial contiguous system/developer messages together with tenant, model and supplied model revision. Rendezvous hashing supplies stable placement when there are no valid cache reports. Explicit residency reports route requests to a reported holder; eviction, TTL expiry and health changes invalidate reports. An inference success does not create a residency report.

```powershell
$env:PREFIX_ADMIN_TOKEN = [guid]::NewGuid().ToString()
python prefix_router.py --replica a=http://127.0.0.1:8001 --replica b=http://127.0.0.1:8002 --model-revision YOUR_MODEL_COMMIT --port 8080
```

Send OpenAI chat-completion requests to `http://127.0.0.1:8080/v1/chat/completions` with `X-Tenant-ID` supplied by a trusted local caller. Streaming responses are forwarded incrementally. Response headers expose `X-Replica`, `X-Prefix-Key`, and `X-Routing-Reason` for debugging. `/health` exposes aggregate routing counters without prompt contents.

An integration can POST events to `/admin/events` with `Authorization: Bearer <PREFIX_ADMIN_TOKEN>`:

```json
{"event":"resident","replica":"a","key":"64_character_lowercase_prefix_hash"}
```

Events: `resident`, `evicted`, or `health` (with `replica` and a boolean `healthy`). Reports expire after 30 seconds and the table is bounded to 10,000 entries. Refresh reports only from actual engine evidence. Failed connections mark replicas unhealthy. Explicit health reports restore them; automatic POST replay is avoided because a timeout can occur after generation starts.

## Verification

Run `python -m unittest discover -s . -p "test_*.py" -v` from the repository root. Tests cover stable affinity, different-tenant and revision isolation, exact whitespace, expiry, eviction, bounded residency, unhealthy replicas, authenticated event ingestion, and actual HTTP/SSE forwarding between two fixture backends. Fixture output is not model inference.

## Boundaries

This is a loopback development proxy, not a production internet gateway. The tenant header is trusted input, not authentication. The prefix hash is a routing identity, not the engine's token/block hash. Production integration needs tokenizer/chat-template awareness, real engine events, tenant authentication, bounded active connections, background health checks and load-aware spillover. Backends must enforce their own tenant-safe cache isolation. The proxy does not implement cache migration or claim that residency hints guarantee hits. No TTFT reduction has been measured.

Reference: https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/
