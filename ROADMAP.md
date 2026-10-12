# Project Status

All 15 projects are indexed here for public visibility. Planned entries are requirements, not shipped projects. GPU-dependent work remains unmeasured.

| Project | Status |
| --- | --- |
| [High Throughput Serving Engine Tuning](projects/01-serving-tuning/README.md) | Local harness tested; GPU benchmark pending |
| [KV Cache Prefix Routing Proxy](projects/02-prefix-routing/README.md) | Local proxy and HTTP integration tests implemented; GPU cache evidence pending |
| [Custom Fused Triton Kernel](projects/03-triton-kernel/README.md) | CPU oracle tested; Triton source and benchmark runner prepared; GPU compilation/correctness pending |
| [Speculative Decoding Pipeline](projects/04-speculative-decoding/README.md) | CPU rejection-sampling and generation reference tested; serving argv prepared; model execution pending |
| [Disaggregated Prefill Decode Cluster](projects/05-prefill-decode/README.md) | Separate loopback pools and signed CPU-state handoff tested; GPU connector/transfer pending |
| [FP8 Quantization and Calibration Harness](projects/06-fp8-calibration/README.md) | CPU FP8/corpus/perplexity tools tested; ModelOpt export prepared; GPU calibration/serving pending |
| [Queue Based GPU Autoscaler](projects/07-gpu-autoscaler/README.md) | Local durable queue, fixture autoscaler, drain workers and metrics tested; KEDA integration template supplied |
| [Complexity Routing AI Gateway](projects/08-complexity-gateway/README.md) | Loopback routing/SSE gateway and policy evaluator implemented; real model quality pending |
| [Multi Node Tensor Parallelism](projects/09-tensor-parallelism/README.md) | Analyzer, four-node planner and trace combiner tested; NCCL collector prepared; GPU/RDMA execution pending |
| [Semantic Cache Layer](projects/10-semantic-cache/README.md) | SQLite cache, Redis/Qdrant adapters and labeled reuse evaluator implemented; real service CI and semantic model evidence separated |
| [DCGM Observability and Throttling Dashboard](projects/11-dcgm-dashboard/README.md) | Snapshot checks, fixture exporter, Grafana/Prometheus stack and CI validation implemented; DCGM hardware pending |
| [MoE Load Profiler](projects/12-moe-profiler/README.md) | Trace analyzer and offline placement proposals tested; GPU collection pending |
| [Inference Chaos and Fault Tolerance Suite](projects/13-inference-chaos/README.md) | CPU fixture faults, SQLite restart recovery and real localhost interrupted-stream resume tested; GPU recovery pending |
| [Checkpointing and Weight Streaming Engine](projects/14-weight-streaming/README.md) | Bounded CPU-memory loader, ETag-pinned HTTP resume and sharded indexes tested; GPU upload pending |
| [Public Inference Benchmark Teardown](projects/15-public-benchmarks/README.md) | Repeated-run/Pareto reporter and three methodology articles implemented; charts are synthetic; real GPU comparisons pending |
