# Inference Chaos and Fault Tolerance Suite

Status: CPU fixture faults, durable restart recovery and real localhost socket interruption/resume implemented and tested. Model-worker OOM recovery and GPU KV migration remain pending.

```powershell
python chaos_suite.py --out results/chaos-fixture.json
python recovery_store.py --db results/fresh-recovery.db --out results/network-recovery.json
```

Three scenarios inject fixture admission failures representing an unreachable replica, an ECC signal and an OOM signal. They never disconnect real networks, allocate excessive memory or induce hardware errors. The controller detects failure, removes the replica, recovers checksummed CPU fixture state onto another replica and preserves the expected response checksum. Repeated request IDs return the completed fixture response; conflicting context is rejected.

The cache is serialized context and a counter, not real KV tensors. The default controller uses memory; supplying `RecoveringCluster(store=RecoveryStore(path))` adds a SQLite WAL checkpoint/completion ledger that survives controller restart. Completion and checkpoint commit together. Corrupt checkpoints are rejected and recomputed; corrupt completed responses fail closed. ECC-tainted replica state is cleared. Restored replicas start empty. All-replica failure returns an explicit error.

`recovery_store.py` starts two actual loopback HTTP servers, cuts the first socket after two SSE events, and resumes through the second using an explicit event ID. Use a fresh database for each demo. Tests verify request identity, restart reuse, corruption handling and all five resumed event IDs. This exercises a shared local durable fixture-output ledger: the toy output is prepared before streaming. Client event-ID deduplication is still required; this is not exactly-once delivery or recovery of a transformer midway through generation.

Recorded timings are CPU control-path measurements only. SQLite is single-host storage with bounded ledger rows, and checksums provide integrity rather than authentication. No firewall/network partition, real ECC error, large allocation, GPU fault signal or GPU tensor transfer is induced. These require hardware validation.
