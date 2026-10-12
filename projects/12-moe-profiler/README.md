# MoE Load Profiler

Status: trace validation, expert/rank load statistics and offline placement proposals implemented and tested. Real expert-parallel deployment and hardware communication improvements remain pending.

```powershell
python trace_profiler.py moe projects/12-moe-profiler/fixture-routing.jsonl --experts 4 --ranks 2 --source-kind fixture --out results/moe-fixture.json
```

The trace is synthetic. Real JSONL events require `request_id`, `step`, `layer`, `token_id` and `experts` (selected IDs). Optional `expert_ranks` must align with the expert list. Optional `communication_bytes` and `communication_ms` must appear together. Durations are summed event observations; overlapping events are not wall time. Supply total expert count so unused experts remain visible. Duplicate token events and invalid IDs are rejected.

Reports include top-k assignment counts, per-expert and observed-rank loads, max/mean imbalance, coefficient of variation, Gini, effective expert count, idle experts and hottest experts. Placement proposals greedily move whole experts between ranks without changing token expert selection. These assignment-count estimates do not model memory, topology, weight movement or communication. A single hot expert cannot be split without a separate replication policy.

For actual studies collect traces after warmup across workloads and repetitions, with engine/model revisions, total experts, top-k, rank-to-node mapping and per-rank timing. Compare projected balance with measured before/after latency. This tool does not hook into a GPU engine or claim an improvement from fixture values.
