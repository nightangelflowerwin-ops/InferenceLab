"""Validate and analyze supplied MoE routing and collective communication traces."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def placement_plan(loads, ranks):
    """Greedy placement of fixed expert workloads; does not alter token routing."""
    if ranks < 1 or not loads or any(number(load, "expert load") != int(load) for load in loads):
        raise ValueError("Positive rank count and integer expert loads required")
    totals, assignment = [0] * ranks, {}
    for expert in sorted(range(len(loads)), key=lambda e: (-loads[e], e)):
        rank = min(range(ranks), key=lambda r: (totals[r], r))
        assignment[str(expert)] = rank
        totals[rank] += loads[expert]
    mean = sum(totals) / ranks
    return dict(expert_to_rank=assignment, projected_rank_assignments=totals,
                projected_max_to_mean=max(totals) / mean if mean else 0,
                caveat="Offline assignment-count heuristic only. Weight memory, topology, communication and expert replication are not modeled; a single hot expert cannot be split by placement.")


def moe_profile(events, experts):
    if experts < 1:
        raise ValueError("experts must be positive")
    layers, seen = {}, set()
    for event in events:
        key = (event["request_id"], integer(event["step"], "step"),
               integer(event["layer"], "layer"), integer(event["token_id"], "token_id"))
        if key in seen:
            raise ValueError("Duplicate token routing event")
        seen.add(key)
        assigned = event["experts"]
        if not isinstance(assigned, list) or not assigned or len(set(assigned)) != len(assigned):
            raise ValueError("Nonempty unique expert list required")
        for expert in assigned:
            integer(expert, "expert")
            if expert >= experts:
                raise ValueError("Expert outside configured range")
        layer = layers.setdefault(event["layer"], dict(load=[0] * experts, tokens=0,
            rank_assignments=Counter(), communication_bytes=0, communication_ms=0,
            communication_events=0))
        layer["tokens"] += 1
        for expert in assigned:
            layer["load"][expert] += 1
        ranks = event.get("expert_ranks")
        if ranks is not None:
            if not isinstance(ranks, list) or len(ranks) != len(assigned):
                raise ValueError("expert_ranks must align with experts")
            for rank in ranks:
                layer["rank_assignments"][integer(rank, "rank")] += 1
        if "communication_bytes" in event or "communication_ms" in event:
            layer["communication_bytes"] += integer(event["communication_bytes"], "communication_bytes")
            layer["communication_ms"] += number(event["communication_ms"], "communication_ms")
            layer["communication_events"] += 1
    output = {}
    for layer_id, layer in sorted(layers.items()):
        loads = layer["load"]
        total = sum(loads)
        mean = total / experts
        entropy = -sum((n / total) * math.log(n / total) for n in loads if n)
        # Sorted O(E log E) Gini, including experts that received no tokens.
        gini = sum((2 * i - experts - 1) * n for i, n in enumerate(sorted(loads), 1)) / (experts * total)
        output[str(layer_id)] = {**layer, "rank_assignments": dict(layer["rank_assignments"]),
            "assignments": total, "max_to_mean": max(loads) / mean,
            "coefficient_of_variation": statistics.pstdev(loads) / mean,
            "gini": gini, "effective_experts": math.exp(entropy),
            "idle_experts": sum(n == 0 for n in loads),
            "hot_expert_ids": [i for i, n in enumerate(loads) if n == max(loads)]}
    return dict(event_count=len(seen), experts=experts, layers=output,
        caveat="Communication durations are summed event observations, not wall time or end-to-end bandwidth. No unobserved experts/ranks are inferred beyond configured expert count.")


def collective_profile(events):
    groups, identities = defaultdict(list), set()
    for event in events:
        rank = integer(event["rank"], "rank")
        size = integer(event["bytes"], "bytes")
        world = integer(event["world_size"], "world_size")
        duration = number(event["duration_ms"], "duration_ms")
        if world < 2 or rank >= world or duration == 0 or not size:
            raise ValueError("Positive size/time and valid multi-rank collective required")
        operation = event["operation"]
        if operation not in ("all_reduce", "all_gather", "reduce_scatter", "send_recv"):
            raise ValueError("Unknown collective")
        identity = (event["run_id"], event["sample_id"], rank)
        if identity in identities:
            raise ValueError("Duplicate collective rank sample")
        identities.add(identity)
        groups[(operation, size, world)].append(event)
    results = []
    for (operation, size, world), samples in sorted(groups.items()):
        rounds = defaultdict(dict)
        for sample in samples:
            rounds[(sample["run_id"], sample["sample_id"])][sample["rank"]] = sample["duration_ms"]
        incomplete = sum(set(values) != set(range(world)) for values in rounds.values())
        if incomplete:
            raise ValueError("Incomplete collective: every rank is required for bottleneck analysis")
        slowest = [max(values.values()) for values in rounds.values()]
        bandwidth = [size / (ms / 1000) / 1e9 for ms in slowest]
        factor = 2 * (world - 1) / world if operation == "all_reduce" else (
            (world - 1) / world if operation in ("all_gather", "reduce_scatter") else 1)
        by_rank = {str(rank): statistics.median(values[rank] for values in rounds.values())
                   for rank in range(world)}
        results.append(dict(operation=operation, bytes=size, world_size=world, rounds=len(rounds),
            p50_slowest_rank_ms=statistics.median(slowest), p95_slowest_rank_ms=percentile(slowest, .95),
            median_algorithm_GB_s=statistics.median(bandwidth),
            estimated_collective_bus_GB_s=statistics.median(bandwidth) * factor,
            median_rank_ms=by_rank, slowest_rank=max(by_rank, key=by_rank.get)))
    return dict(groups=results, caveat="Bandwidth is derived from supplied size/time. The bus estimate uses collective normalization, not measured NIC throughput; it does not diagnose RDMA or PCIe by itself.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["moe", "collective"])
    parser.add_argument("input", type=Path)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--ranks", type=int, help="Optional offline expert placement proposal")
    parser.add_argument("--source-kind", choices=["fixture", "unverified", "hardware"], default="unverified")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as file:
        events = [json.loads(line) for line in file if line.strip()]
    result = moe_profile(events, args.experts) if args.kind == "moe" else collective_profile(events)
    if args.kind == "moe" and args.ranks:
        for layer in result["layers"].values():
            layer["placement_proposal"] = placement_plan(layer["load"], args.ranks)
    result["source_file"] = args.input.name
    result["evidence"] = "Analysis of caller-supplied traces; hardware provenance must accompany the input"
    result["source_kind"] = args.source_kind
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
