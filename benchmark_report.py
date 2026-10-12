"""Comparable repeated-run summaries, Pareto analysis and transparent reports.

The shipped CSV contains invented fixture measurements, not engine benchmarks.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics


TEXT_FIELDS = ("run_id", "engine", "candidate", "engine_version", "hardware", "model_revision",
               "tokenizer_revision", "workload", "source_kind", "cost_source", "request_rate", "arrival_process", "sampling_policy")
INTEGER_FIELDS = ("input_tokens", "output_tokens", "request_count", "completed", "total_output_tokens", "concurrency", "warmup_requests", "seed")
FLOAT_FIELDS = ("duration_s", "billed_seconds", "p95_ttft_ms", "p95_itl_ms", "hourly_cost_usd")
FIELDS = TEXT_FIELDS + INTEGER_FIELDS + FLOAT_FIELDS
CONTEXT_FIELDS = ("hardware", "model_revision", "tokenizer_revision", "workload", "input_tokens",
                  "output_tokens", "request_count", "source_kind", "concurrency", "request_rate", "arrival_process", "sampling_policy", "warmup_requests", "seed")


def validate_rows(rows):
    result, seen = [], set()
    for row in rows:
        parsed = {}
        for field in TEXT_FIELDS:
            value = row.get(field)
            if not isinstance(value, str) or not value or len(value) > 1024 or "\n" in value or "\r" in value:
                raise ValueError(f"Bounded nonempty text required: {field}")
            parsed[field] = value
        if parsed["source_kind"] not in ("fixture", "caller-supplied-hardware"):
            raise ValueError("Explicit fixture/caller-supplied-hardware source kind required")
        if parsed["source_kind"] == "caller-supplied-hardware":
            for field in ("model_revision", "tokenizer_revision"):
                if not re.fullmatch(r"(?:[0-9a-fA-F]{40}|sha256:[0-9a-fA-F]{64})", parsed[field]):
                    raise ValueError("Hardware comparisons require immutable commit/content identifiers")
        if parsed["run_id"] in seen:
            raise ValueError("Duplicate run ID")
        seen.add(parsed["run_id"])
        for field in INTEGER_FIELDS:
            value = row.get(field)
            if isinstance(value, bool):
                raise ValueError("Boolean is not a token/request count")
            integer = int(value)
            if isinstance(value, float) and value != integer or isinstance(value, str) and str(integer) != value:
                raise ValueError(f"Integer count required: {field}")
            if integer < 0:
                raise ValueError("Counts must be nonnegative")
            parsed[field] = integer
        for field in FLOAT_FIELDS:
            value = row.get(field)
            if isinstance(value, bool):
                raise ValueError("Boolean is not a measurement")
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Finite nonnegative measurement required: {field}")
            parsed[field] = value
        if parsed["request_count"] < 1 or parsed["output_tokens"] < 1 or parsed["duration_s"] <= 0:
            raise ValueError("Positive request/output/window counts required")
        if parsed["concurrency"] < 1 or parsed["arrival_process"] not in ("closed-loop", "poisson", "trace-replay"):
            raise ValueError("Positive concurrency and explicit arrival process required")
        if parsed["request_rate"] != "inf":
            rate = float(parsed["request_rate"])
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError("Positive finite request rate or inf required")
        if parsed["billed_seconds"] < parsed["duration_s"] or parsed["completed"] > parsed["request_count"]:
            raise ValueError("Invalid billed window or completed count")
        if parsed["total_output_tokens"] > parsed["completed"] * parsed["output_tokens"]:
            raise ValueError("Observed output exceeds declared per-request maximum")
        parsed["output_tokens_s"] = parsed["total_output_tokens"] / parsed["duration_s"]
        parsed["window_cost_usd"] = parsed["hourly_cost_usd"] * parsed["billed_seconds"] / 3600
        parsed["usd_per_million_output_tokens"] = parsed["window_cost_usd"] * 1e6 / parsed["total_output_tokens"] if parsed["total_output_tokens"] else None
        parsed["complete"] = parsed["completed"] == parsed["request_count"] and parsed["total_output_tokens"] > 0
        if parsed["complete"] and max(parsed["p95_ttft_ms"], parsed["p95_itl_ms"]) > parsed["duration_s"] * 1000:
            raise ValueError("Latency exceeds complete-run window; check time units and measurement scope")
        result.append(parsed)
    if not result:
        raise ValueError("Benchmark records required")
    return result


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def bootstrap_median(values, seed=42, repeats=500):
    rng = random.Random(seed)
    estimates = [statistics.median(rng.choices(values, k=len(values))) for _ in range(repeats)]
    return [percentile(estimates, .025), percentile(estimates, .975)]


def nondominated(points, objectives):
    def coordinates(point):
        return tuple(point[field] if minimize else -point[field] for field, minimize in objectives)
    return [point for point in points if not any(
        all(a <= b for a, b in zip(coordinates(other), coordinates(point))) and
        any(a < b for a, b in zip(coordinates(other), coordinates(point))) for other in points if other is not point)]


def summarize(rows, minimum_repeats=3):
    if minimum_repeats < 1:
        raise ValueError("Positive repeat requirement")
    contexts = defaultdict(lambda: defaultdict(list))
    for row in validate_rows(rows):
        context = tuple(row[field] for field in CONTEXT_FIELDS)
        contexts[context][(row["engine"], row["candidate"], row["engine_version"])].append(row)
    reports = []
    for context, candidates in sorted(contexts.items()):
        summary = []
        for (engine, candidate, version), runs in sorted(candidates.items()):
            successful = [run for run in runs if run["complete"]]
            entry = {"engine": engine, "candidate": candidate, "engine_version": version,
                     "runs": len(runs), "complete_runs": len(successful),
                     "eligible": len(runs) >= minimum_repeats and len(successful) == len(runs),
                     "failure_rate": 1 - sum(run["completed"] for run in runs) / sum(run["request_count"] for run in runs),
                     "run_ids": [run["run_id"] for run in runs], "cost_sources": sorted({run["cost_source"] for run in runs})}
            if successful:
                for field in ("output_tokens_s", "p95_ttft_ms", "p95_itl_ms", "usd_per_million_output_tokens"):
                    values = [run[field] for run in successful]
                    entry[field] = statistics.median(values)
                    entry[field + "_bootstrap_interval"] = bootstrap_median(values)
            summary.append(entry)
        eligible = [entry for entry in summary if entry["eligible"]]
        frontier = nondominated(eligible, [("usd_per_million_output_tokens", True), ("p95_ttft_ms", True), ("output_tokens_s", False)])
        frontier_ids = {(point["engine"], point["candidate"], point["engine_version"]) for point in frontier}
        for entry in summary:
            entry["three_objective_pareto"] = (entry["engine"], entry["candidate"], entry["engine_version"]) in frontier_ids
        reports.append({"context": dict(zip(CONTEXT_FIELDS, context)), "candidates": summary})
    return {"contexts": reports, "minimum_repeats": minimum_repeats, "bootstrap_seed": 42,
            "evidence": "Derived analysis of supplied rows. A source label is caller provenance, not independent hardware attestation. Fixture records must never be cited as engine measurements."}


def markdown(report):
    lines = ["# Inference comparison report", "", "This is derived analysis of supplied records. Prices and hardware provenance are not independently verified.", ""]
    for index, context in enumerate(report["contexts"], 1):
        metadata = context["context"]
        fixture = metadata["source_kind"] == "fixture"
        lines.extend([f"## Context {index}: {metadata['workload']}", "",
            "**SYNTHETIC FIXTURE DATA: no engine measurements or real prices.**" if fixture else "**Caller-supplied hardware records: verify raw traces before publication claims.**",
            "", "```json", json.dumps(metadata, indent=2), "```", "",
            "| Candidate | Complete runs | Output tokens/s | Median run p95 TTFT (ms) | USD / million output tokens | Eligible | 3-objective Pareto |",
            "| --- | --- | --- | --- | --- | --- | --- |"])
        for point in context["candidates"]:
            label = (point["engine"] + ": " + point["candidate"]).replace("|", "\\|")
            def value(field):
                return f"{point[field]:.3f}" if field in point else "unavailable"
            lines.append(f"| {label} | {point['complete_runs']}/{point['runs']} | {value('output_tokens_s')} | {value('p95_ttft_ms')} | {value('usd_per_million_output_tokens')} | {point['eligible']} | {point['three_objective_pareto']} |")
        lines.extend(["", "Eligible candidates require every recorded run to complete and at least the requested repeat count. Failed/incomplete candidates remain visible but are excluded from Pareto selection.", ""])
    lines.extend(["## Interpretation limits", "",
        "Each latency summary is a median of per-run p95 values, not a pooled-request p95. Bootstrap intervals resample whole runs; three repeats offer weak uncertainty evidence. Raw per-request traces are needed to inspect tails and failures.", "",
        "Cost uses the supplied total-footprint hourly price and billed seconds, divided by successful observed output tokens. It excludes costs not included in those inputs. Billed time must cover the benchmark window; idle/warmup/startup must be included if billed. Costs are not current price quotations.", "",
        "Pareto comparisons stay within matching hardware, model/tokenizer revisions, workload shape and source kind. A frontier says that no eligible candidate dominates all three measured axes; it does not establish answer quality, causal improvement, universal superiority or a production recommendation.", ""])
    return "\n".join(lines)


def plot(report, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    axes = [("usd_per_million_output_tokens", True, "p95_ttft_ms", True, "USD / million output tokens", "Median of run p95 TTFT (ms)"),
            ("output_tokens_s", False, "p95_ttft_ms", True, "Output tokens/s", "Median of run p95 TTFT (ms)"),
            ("usd_per_million_output_tokens", True, "output_tokens_s", False, "USD / million output tokens", "Output tokens/s")]
    for index, context in enumerate(report["contexts"], 1):
        points = [point for point in context["candidates"] if point["eligible"]]
        if not points:
            continue
        figure, panels = plt.subplots(1, 3, figsize=(15, 5))
        colors = plt.get_cmap("tab10")
        for panel, (x, min_x, y, min_y, xlabel, ylabel) in zip(panels, axes):
            for n, point in enumerate(points):
                xci, yci = point[x + "_bootstrap_interval"], point[y + "_bootstrap_interval"]
                panel.errorbar(point[x], point[y], xerr=[[max(0, point[x] - xci[0])], [max(0, xci[1] - point[x])]],
                    yerr=[[max(0, point[y] - yci[0])], [max(0, yci[1] - point[y])]], marker="o", capsize=3,
                    color=colors(n % 10), label=point["engine"] + ": " + point["candidate"])
            frontier = sorted(nondominated(points, [(x, min_x), (y, min_y)]), key=lambda point: point[x])
            panel.plot([point[x] for point in frontier], [point[y] for point in frontier], "--", color="black", alpha=.55)
            panel.set_xlabel(xlabel)
            panel.set_ylabel(ylabel)
            panel.grid(alpha=.2)
        handles, labels = panels[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)), fontsize=8)
        fixture = context["context"]["source_kind"] == "fixture"
        figure.suptitle(context["context"]["workload"] + (" | SYNTHETIC DATA, not engine measurements" if fixture else " | Caller-supplied records, provenance unverified"), fontsize=12, fontweight="bold")
        figure.text(.5, .12, "Dashed curves: two-objective Pareto projections. Error bars: run-resampled median intervals.", ha="center", fontsize=8)
        figure.tight_layout(rect=(0, .18, 1, .92))
        path = destination / f"context-{index}-pareto.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--minimum-repeats", type=int, default=3)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8", newline="") as file:
        report = summarize(list(csv.DictReader(file)), args.minimum_repeats)
    report["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (args.out / "report.md").write_text(markdown(report), encoding="utf-8")
    if args.plot:
        paths = plot(report, args.out)
        with (args.out / "report.md").open("a", encoding="utf-8") as file:
            file.write("\n## Pareto charts\n\n")
            for path in paths:
                file.write(f"![Explicitly labeled comparison projection]({path.name})\n\n")
    print(f"Generated {len(report['contexts'])} comparable context reports; source provenance remains explicit")


if __name__ == "__main__":
    main()
