# Project Status

All 15 projects are indexed here for public visibility. Planned entries are requirements, not shipped projects. GPU-dependent work remains unmeasured.

| Project | Status |
| --- | --- |
| [High Throughput Serving Engine Tuning](projects/01-serving-tuning/README.md) | Local harness tested; GPU benchmark pending |
| [KV Cache Prefix Routing Proxy](projects/02-prefix-routing/README.md) | Local proxy and HTTP integration tests implemented; GPU cache evidence pending |
| [Custom Fused Triton Kernel](projects/03-triton-kernel/README.md) | Planned |
| [Speculative Decoding Pipeline](projects/04-speculative-decoding/README.md) | Planned |
| [Disaggregated Prefill Decode Cluster](projects/05-prefill-decode/README.md) | Planned |
| [FP8 Quantization and Calibration Harness](projects/06-fp8-calibration/README.md) | Planned |
| [Queue Based GPU Autoscaler](projects/07-gpu-autoscaler/README.md) | Local durable queue, fixture autoscaler, drain workers and metrics tested; KEDA integration template supplied |
| [Complexity Routing AI Gateway](projects/08-complexity-gateway/README.md) | Planned |
| [Multi Node Tensor Parallelism](projects/09-tensor-parallelism/README.md) | Collective trace analyzer tested; launch/collector preparation remains |
| [Semantic Cache Layer](projects/10-semantic-cache/README.md) | Planned |
| [DCGM Observability and Throttling Dashboard](projects/11-dcgm-dashboard/README.md) | Planned |
| [MoE Load Profiler](projects/12-moe-profiler/README.md) | Trace analyzer and offline placement proposals tested; GPU collection pending |
| [Inference Chaos and Fault Tolerance Suite](projects/13-inference-chaos/README.md) | CPU fixture recovery suite tested; network and durable recovery work remains |
| [Checkpointing and Weight Streaming Engine](projects/14-weight-streaming/README.md) | Bounded CPU-memory safetensors streamer tested; resume/sharding and GPU upload remain |
| [Public Inference Benchmark Teardown](projects/15-public-benchmarks/README.md) | Planned |
