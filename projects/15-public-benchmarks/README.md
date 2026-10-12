# Public Inference Benchmark Teardown

Status: repeated-run validation, cost arithmetic, bootstrap summaries, Pareto analysis, Markdown reporting, plotting and three public methodology articles implemented. Shipped charts are synthetic. Real GPU comparisons remain pending.

```powershell
python benchmark_report.py projects/15-public-benchmarks/benchmark-fixture.csv --out results/comparison
```

Optional plotting uses Matplotlib pinned in `requirements-report.txt`. Install it in an isolated Python environment and add `--plot`. Reports preserve source labels, input hashes, comparability metadata, failures and repeat requirements. Only fully completed candidates with enough repeats are eligible for frontier selection.

CSV columns are defined by `FIELDS` in [benchmark_report.py](../../benchmark_report.py). They cover engine/version/candidate, hardware, model/tokenizer identifiers, workload shape, concurrency, arrival rate/process, sampling policy, warmup, seed, request counts, observed output totals, benchmark/billed seconds, per-run p95 TTFT/ITL and supplied footprint price/source. Source labels are `fixture` or `caller-supplied-hardware`; the latter requires immutable commit/content identifiers, but the tool does not attest hardware provenance.

Outputs include cost per million observed output tokens, median throughput, median per-run p95 values, run-bootstrap intervals, a three-objective Pareto set and optional pairwise projection curves. Hardware, revision, workload and source mismatches are not pooled. Three repeats offer weak statistical evidence; publish raw traces before making production claims.

Methodology articles:

1. [Serving-engine comparison controls](01-serving-methodology.md)
2. [Cost and Pareto interpretation](02-cost-and-pareto.md)
3. [Reproducibility and quality](03-reproducibility-and-quality.md)

Inspect the [synthetic generated report and charts](reports/report.md). These articles are methodology/software evidence, not the three measured hardware comparisons originally proposed. GPU runs remain necessary for that claim.
