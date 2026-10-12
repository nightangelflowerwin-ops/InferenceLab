# Inference Chaos and Fault Tolerance Suite

Status: isolated CPU fixture fault/recovery harness implemented and tested. Real network isolation, model-worker OOM recovery and GPU KV migration remain pending.

```powershell
python chaos_suite.py --out results/chaos-fixture.json
```

Three scenarios inject fixture admission failures representing an unreachable replica, an ECC signal and an OOM signal. They never disconnect real networks, allocate excessive memory or induce hardware errors. The controller detects failure, removes the replica, recovers checksummed CPU fixture state onto another replica and preserves the expected response checksum. Repeated request IDs return the completed fixture response; conflicting context is rejected.

The cache is serialized context and a counter, not real KV tensors. Checkpoints live in controller memory and survive replica failure, not controller-process loss. Corrupt checkpoints are rejected and recomputed. ECC-tainted replica state is cleared. Restored replicas start empty. All-replica failure returns an explicit error.

Recorded timings are CPU control-path measurements only. This sequential harness injects before output and does not prove partially streamed recovery. Additional non-GPU work remains: actual localhost network-fault injection, durable checkpoints and partial-stream/idempotency tests. GPU fault signals and GPU tensor transfer require hardware validation.
