# Multi Node Tensor Parallelism

Status: collective trace validation and bottleneck analysis implemented and tested. Multi-node setup, NCCL collection and RDMA profiling remain pending.

```powershell
python trace_profiler.py collective projects/09-tensor-parallelism/fixture-collectives.jsonl --source-kind fixture --out results/collective-fixture.json
```

The four-rank fixture is synthetic. Supply JSONL events with `run_id`, `sample_id`, `rank`, `world_size`, `operation`, `bytes` and `duration_ms`. Every round must include every rank. Duplicate rank events, nonfinite durations and missing ranks are rejected. Each `(run_id,sample_id,rank)` identifies one unique operation sample.

Operations: `all_reduce`, `all_gather`, `reduce_scatter`, `send_recv`. Bytes represent the logical message size: all-reduce payload, total gathered output for all-gather, total input for reduce-scatter, transmitted bytes for send/receive. Algorithm GB/s uses decimal gigabytes and the slowest rank; estimated bus GB/s uses collective normalization, not NIC counters.

Reports group operation, size and world size, then show median/p95 slowest-rank durations, per-rank medians, slowest rank and derived bandwidth. Real studies must capture topology, engine/model revisions, NCCL transport, RDMA link counters and rank-to-GPU mappings. A slow rank alone does not distinguish GPU, PCIe, RDMA or scheduling causes.

Remaining non-GPU preparation: multi-node launch configuration and trace collector adapter. No NCCL or RDMA deployment is claimed.
