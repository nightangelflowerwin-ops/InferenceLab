# Disaggregated Prefill Decode Cluster

Status: separate CPU fixture worker pools, checked state handoff and loopback coordinator implemented and tested. Real vLLM prefill/decode, GPU KV transfer and hardware optimization remain pending.

```powershell
python disaggregated_pipeline.py demo --out results/disaggregated-fixture.json
```

The demo starts two prefill and two decode HTTP workers. Prefill computes a deterministic rolling state from fixture token IDs; decode consumes that state and generates fixture token IDs. A monolithic calculation verifies the same output. The handoff is HMAC-signed, request/tenant/revision-bound and expiring. Pool admission is capacity-bounded and chooses the least active healthy worker.

Persistent fixture services can be run in separate terminals with the same local `INFERENCELAB_TRANSFER_SECRET` (at least 16 characters):

```powershell
python disaggregated_pipeline.py prefill --port 8101 --revision fixture-v1
python disaggregated_pipeline.py decode --port 8102 --revision fixture-v1
python disaggregated_pipeline.py proxy --port 8090 --producer p=http://127.0.0.1:8101 --consumer d=http://127.0.0.1:8102
```

POST `/generate` on port 8090 with `request_id`, `tenant`, `model_revision`, integer `tokens` in 0..255 and `max_new_tokens` in 1..256. This is a custom fixture API, not an OpenAI model endpoint. The ledger binds each tenant/request ID to identical work and returns completed results on repeat. Failed prefills can fail over; decode failures mark completion uncertain and are not automatically replayed. A manual health report is available through the pool class; no automatic health reconciliation is claimed.

The coordinator ledger is in memory, capped at 10,000 requests, and does not survive process loss. Handoff byte counts and phase durations measure CPU HTTP/control paths, not GPU tensors, network fabric bandwidth or model latency. Production needs durable request state, engine-specific tensor layouts, compatible model/tokenizer versions, transfer transport and admission/deadline control. Actual hardware-phase optimization is excluded from the laptop evidence.

Reference: [vLLM disaggregated prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/). Integrating its KV connector requires a GPU-side adapter; this CPU fixture is independent of that engine protocol.
