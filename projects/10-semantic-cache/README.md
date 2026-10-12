# Semantic Cache Layer

Status: opt-in response-cache proxy integration, SQLite cosine index/TTL response store, real Redis RESP and Qdrant HTTP adapters, and labeled-reuse evaluator implemented. Local HTTP/SQLite tests pass. Dedicated CI verifies real Redis/Qdrant services. Real semantic embeddings and model-answer-quality evaluation remain pending.

```powershell
python demo_gateway.py --out results/gateway-demo.json
python cache_evaluation.py projects/10-semantic-cache/reuse-fixture.json --out results/reuse-evaluation.json
```

Enable cache use per request with `X-Allow-Semantic-Cache: true`. Only non-streaming single-output plain-text requests with explicit `temperature: 0` and no tools/functions are eligible. Similarity must be strictly greater than 0.98. Namespace identity includes tenant, routed model/revision, embedding revision, all generation settings, all earlier turns and number/negation guards for the final query. Cache lookup and response TTL are checked independently; an orphan vector is a miss. Cached responses omit current-request usage and expose original inference usage separately.

SQLite adapters provide a bounded local exact cosine index and response store, capped at 10,000 entries each. TTL cleanup occurs during writes/searches. Redis uses binary-safe RESP GET/SET with EX expiry and an internal expiry envelope. Qdrant searches with namespace and expiry filters and cosine scoring; expired points are deleted during writes. It does not have the SQLite adapter's total-entry cap; configure service storage limits and retention for sustained deployment.

Use the gateway's `--redis-port 6379 --qdrant-url http://127.0.0.1:6333` to select real services. Supply `--cache-db` plus local embedding URL/model/revision to enable the cache; the database holds the fallback adapters when service flags are absent. The Python adapters need no third-party package. Redis 7.4.2 and Qdrant 1.13.6 are pinned in the dedicated Linux CI service job. The laptop has neither service running; local tests exercise SQLite, wire parsing and actual HTTP proxy/embedding fixtures, while CI exercises real service storage and filtered search.

The fixture embedder is hashed character bigrams, not a semantic model. For real caching supply a local semantic embedding service and evaluate held-out queries with acceptable answer IDs. The public synthetic reuse fixture intentionally includes a high-similarity wrong answer: a 98% threshold alone allows incorrect reuse. Numbers/negation guards reduce some mistakes but cannot guarantee equivalence. Do not treat similarity as proof of correctness or cache arbitrary requests by default.

Cache-service/embedding failures fall through to the selected completion backend. Partial streams are not cached. Endpoints and Redis are loopback-only; this prototype trusts tenant headers and does not offer production authentication, cache-stampede coalescing or distributed rate limits. No GPU/cache latency reduction has been measured.

References: [Redis RESP](https://redis.io/docs/latest/develop/reference/protocol-spec/), [Redis SET](https://redis.io/docs/latest/commands/set/), [Qdrant filtered search](https://qdrant.tech/documentation/search-patterns/vector-search-filtering/).
