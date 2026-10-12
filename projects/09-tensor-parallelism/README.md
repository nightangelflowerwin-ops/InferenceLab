# Multi Node Tensor Parallelism

Status: collective analyzer, four-node launch-plan generator, per-rank trace combiner and prepared CUDA/NCCL collector implemented. Local planning/combination/help/syntax tests pass. Actual multi-node model serving, collector GPU execution and RDMA measurements remain pending.

```powershell
python trace_profiler.py collective projects/09-tensor-parallelism/fixture-collectives.jsonl --source-kind fixture --out results/collective-fixture.json
```

The four-rank fixture is synthetic. Supply JSONL events with `run_id`, `sample_id`, `rank`, `world_size`, `operation`, `bytes` and `duration_ms`. Every round must include every rank. Duplicate rank events, nonfinite durations and missing ranks are rejected. Each `(run_id,sample_id,rank)` identifies one unique operation sample.

Operations: `all_reduce`, `all_gather`, `reduce_scatter`, `send_recv`. Bytes represent the logical message size: all-reduce payload, total gathered output for all-gather, total input for reduce-scatter, transmitted bytes for send/receive. Algorithm GB/s uses decimal gigabytes and the slowest rank; estimated bus GB/s uses collective normalization, not NIC counters.

Reports group operation, size and world size, then show median/p95 slowest-rank durations, per-rank medians, slowest rank and derived bandwidth. Real studies must capture topology, engine/model revisions, NCCL transport, RDMA link counters and rank-to-GPU mappings. A slow rank alone does not distinguish GPU, PCIe, RDMA or scheduling causes.

## Four-node preparation

```powershell
python cluster_plan.py plan projects/09-tensor-parallelism/plan-fixture.json --out results/four-node-plan.json
```

The fixture addresses, model path and network names are illustrative. The generator requires four unique private IPv4 addresses and checks that the stated attention-head count divides across the tensor-parallel world size. This is a necessary shape check, not proof that the model/runtime supports the configuration or fits memory.

The plan emits per-node Ray and torchrun argument arrays plus environment settings, and a vLLM head-node serving command. It never runs SSH, starts Ray, downloads models or changes network settings. On actual hardware verify pinned software, identical immutable snapshots, node/HCA/interface names, GPU mapping and model-specific parallelism support first. `NCCL_IB_DISABLE=0` enables consideration of InfiniBand/RoCE transport; it does not prove that NCCL used RDMA. Retain NCCL debug logs and NIC counters.

[collect_nccl.py](collect_nccl.py) is prepared source for per-rank float32 all-reduce timing. It warms up, places barriers outside CUDA-event timing, checks numerical results and writes one exclusive JSONL file per rank. Source parsing and `--help` run without PyTorch installed; the actual collector requires CUDA/NCCL and has not executed on the laptop.

```powershell
python cluster_plan.py combine results/nccl-traces/rank-0.jsonl results/nccl-traces/rank-1.jsonl results/nccl-traces/rank-2.jsonl results/nccl-traces/rank-3.jsonl --out results/combined.jsonl
python trace_profiler.py collective results/combined.jsonl --source-kind hardware --out results/collective-report.json
```

The combiner checks complete rank coverage, unique samples and valid size/time units before writing. Use all rank files when `gpus_per_node` exceeds one. A hardware source label is caller provenance, not independent attestation. No 70B deployment, throughput or network bottleneck was measured.

References: [vLLM multi-node serving](https://docs.vllm.ai/en/latest/examples/ray_serving/multi-node-serving/), [PyTorch distributed launch](https://docs.pytorch.org/docs/stable/elastic/run.html).
