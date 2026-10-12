# Complexity Routing AI Gateway

Status: loopback complexity-routing HTTP/SSE gateway and routing-label evaluator implemented and tested. Running actual local 8B inference, frontier API access and model-quality/cost validation remain pending.

```powershell
python demo_gateway.py --out results/gateway-demo.json
python complexity_gateway.py evaluate --cases projects/08-complexity-gateway/routing-fixtures.json --out results/routing-evaluation.json
```

The demo starts temporary local echo backends, a lexical fixture embedding endpoint and the gateway; it verifies local/complex routes and a real cache miss/hit. Fixtures load no model and make no API calls. The routing dataset includes counterexamples, and the default heuristic agrees with only 6/8 supplied labels. That is a software/policy demonstration, not a model-quality benchmark.

The policy combines UTF-8 byte-length token estimates, byte entropy, reasoning cue words and tool/multimodal flags. These features are cheap, but do not reliably measure reasoning difficulty. A short hard query can be misrouted, and a simple comparison can trigger the complex route. Production use needs a held-out labeled corpus and actual answer-quality evaluation. No fixed speed or cost saving is claimed.

For persistent local demonstrations run three terminals:

```powershell
python fixture_backend.py --port 8001 --name local-fixture
python fixture_backend.py --port 8002 --name complex-fixture
python complexity_gateway.py serve --local-url http://127.0.0.1:8001 --local-model local-fixture --local-revision fixture-v1 --complex-url http://127.0.0.1:8002 --complex-model complex-fixture --complex-revision fixture-v1 --cache-db results/gateway.db --embedding-url http://127.0.0.1:8001 --embedding-model hashed-bigram-fixture --embedding-revision fixture-v1
```

Send OpenAI-compatible chat requests to `http://127.0.0.1:8082/v1/chat/completions` with `X-Tenant-ID`. Response headers expose the route, estimated input tokens and cache state. Streaming is passed through and never cached or replayed. JSON requests/responses and active forwarding are bounded. A missing complex backend returns 503 instead of silently changing the selected route. A backend error is returned without retrying a potentially started generation.

All service URLs are loopback-only. External/frontier API execution is deliberately outside the current free local goal. Replace fixture URLs with owned local model endpoints to test CPU inference later. Tenant headers are trusted local input, not internet authentication. The embedding revision must identify the actual loaded model; the gateway cannot attest its deployment. [Cache policy and service adapters](../10-semantic-cache/README.md).
