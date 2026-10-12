# InferenceLab

[![Local correctness checks](https://github.com/nightangelflowerwin-ops/InferenceLab/actions/workflows/tests.yml/badge.svg)](https://github.com/nightangelflowerwin-ops/InferenceLab/actions/workflows/tests.yml)

An AI inference infrastructure portfolio: serving experiments, cache routing, GPU optimization and reproducible performance evidence.

**Implemented:** vLLM experiment runner, local prefix-routing proxy, durable token queue with fixture autoscaling, MoE/collective trace analyzers, isolated CPU fault/recovery suite and bounded CPU-memory weight streamer. **Measured GPU results:** none yet.

[Explore all 15 projects and their status](ROADMAP.md) | [Prefix routing implementation](projects/02-prefix-routing/README.md)

Started 2026-10-12. Project 1: high-throughput vLLM serving-engine tuning.

## Current evidence

Local machine: Windows, Intel UHD Graphics 620, approximately 31.8 GiB system RAM, Python 3.11.9. No NVIDIA GPU or nvidia-smi detected. CUDA deployment and inference performance measurements are pending. Offline tests verify planning, HTTP readiness, process cleanup, and report filtering. Test fixture numbers are never published as inference measurements.

## Run locally now

```powershell
python -m unittest discover -s . -p "test_*.py" -v
```

The shipped config deliberately requires an immutable model revision. Set `revision` to the model repository commit you intend to measure. `RTX 4090` is an illustrative GPU target, not this laptop's hardware or a recommendation to buy hardware. Change it to the actual GPU before running.

## Run when GPU access is available

Use a Linux CUDA machine with a compatible, explicitly pinned vLLM installation. Record its package version and CUDA/PyTorch environment (`pip freeze > requirements-measured.txt`). Consult the vLLM GPU installation matrix for that machine rather than installing a guessed wheel here. Use the installed version's `vllm serve --help` and `vllm bench serve --help` to verify the generated flags before spending time on the full sweep. The runner records the vLLM version, driver, GPU, configuration, commands and UTC timestamp.

```bash
# From this project directory, after editing config.json:
python tune.py plan --out plans/first
# Review plans/first/plan.json, then use a fresh result directory:
python tune.py run --out results/first --ttft-slo-ms 1000
python tune.py report --out results/first --ttft-slo-ms 1000
```

Each candidate starts a fresh server on loopback. Prefix caching is disabled to isolate scheduler effects. vLLM's continuous batching is exercised through concurrent requests; it is not a separate on/off switch. Sweep chunked prefill, sequence limits, token batch budgets and GPU memory utilization. The unchunked candidates with 4096-token budgets provide explicit comparison baselines. These are controlled baselines, not a claim to reproduce every engine default.

Three seeded workloads separate short prompts, long prefills and heavy decode. Each has 128 requests, 8 warmups and 3 measured repeats. The 54 candidate/workload pairs yield 162 measured runs. This is a closed-loop saturation study using a concurrency cap and infinite arrival rate. Add finite-rate workloads to study production queueing and SLOs. Random tokens are synthetic workload shapes, not quality evaluations.

Artifacts include server logs, benchmark logs, raw per-request JSON, sampled GPU memory/utilization/temperature/power, summary CSV and workload-specific recommendations. Startup failures, timeouts and incomplete request runs are excluded. Recommendations require all repeats to pass the TTFT constraint and rank median output throughput separately for each workload. Inspect ITL and per-request errors before choosing a production configuration. A TTFT-only constraint does not guarantee all service objectives.

The model server is revision-pinned. For exact tokenizer reproducibility, predownload that same snapshot and set `model` to its absolute snapshot directory; the benchmark CLI otherwise resolves the tokenizer repository independently. Downloads, model loading and GPU allocation occur only in `run` mode on the GPU host. No paid services or API keys are used.

Limitations: no GPU deployment has been validated on this laptop; CLI compatibility must be checked against the selected vLLM release. GPU telemetry covers all visible devices and is not per-process memory attribution. Readiness is HTTP health, and a competing process could still race to claim the port after the initial check. Linux cleanup signals the server and benchmark process groups; independently detached engine workers may still require inspection. Run on an otherwise idle dedicated GPU for reliable comparisons. Keep raw failures and document any retried runs.

## Evidence standard

Do not claim a fixed VRAM waste percentage, TTFT reduction or speculative speedup without a measured comparison. Reports need hardware, immutable model/tokenizer versions, engine versions, workload, warmup, repeats, failure rate, TTFT/ITL distributions and output tokens/sec. Cost curves require documented actual pricing; local experiments can report power and elapsed time without inventing cloud cost.

Official references checked while building:

- https://docs.vllm.ai/en/latest/cli/serve/
- https://docs.vllm.ai/en/latest/cli/bench/serve/
- https://docs.vllm.ai/en/latest/configuration/optimization/

## Portfolio sequence

| Project | Implemented or planned laptop work | Hardware evidence still needed |
| --- | --- | --- |
| 1. Serving tuning | Harness implemented, offline validation | Deploy and measure on a specific CUDA GPU |
| 2. Prefix routing | Local HTTP/SSE proxy and routing tests implemented | Prove cache hits and TTFT with real replicas |
| 3. Fused RMSNorm | Planned: Kernel source and numerical test design | Triton correctness and GPU timing |
| 4. Speculative decode | Planned: Acceptance/speed methodology | Compatible draft/target deployment and quality checks |
| 5. Prefill/decode pools | Planned: Routing and failure contract | Real KV transfer and separate GPU pools |
| 6. FP8 calibration | Planned: Corpus and quality evaluation harness | Supported GPU and TensorRT-LLM measurements |
| 7. Autoscaling | Durable queue, local autoscaler, drain tests and metrics implemented | GPU Kubernetes scale-out and interruption tests |
| 8. Complexity gateway | Planned: Routing policy and offline evaluation | Local model quality; frontier calls require separate authorization |
| 9. Tensor parallelism | Collective analyzer implemented; launch/collector preparation pending | Four GPU nodes with RDMA hardware |
| 10. Semantic caching | Planned: Local cache, namespace and stale-answer tests | Quality evaluation beyond a similarity threshold |
| 11. GPU observability | Planned: Dashboard definitions | NVIDIA DCGM metrics and thermal tests |
| 12. MoE profiling | Trace analysis and placement proposal tools implemented | Expert-parallel deployment |
| 13. Chaos suite | Isolated fixture recovery implemented; real network tests pending | Recovery tests on an owned GPU cluster |
| 14. Weight streaming | CPU safetensors stream loader implemented; resume/sharding pending | GPU transfer and cold-start measurement |
| 15. Public teardown | Planned: Reproducible report templates | Three measured comparisons before publication |

Supporting implementations now cover projects 1, 2, 7, 9, 12, 13 and 14. Several still have additional non-GPU work; none claims real GPU performance. See ROADMAP.md and each project page for exact remaining work. [Local fixture evidence](evidence/README.md) is available for inspection.
