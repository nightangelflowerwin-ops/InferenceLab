"""CPU reference for exact categorical speculative rejection sampling.

Callbacks represent distributions, not deployed draft/target language models.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random


def distribution(values):
    if not isinstance(values, (list, tuple)) or not values or any(
            isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or n < 0 for n in values):
        raise ValueError("Finite nonnegative distribution required")
    total = math.fsum(values)
    if total <= 0 or not math.isfinite(total):
        raise ValueError("Positive finite probability mass required")
    return [n / total for n in values]


def sample(probabilities, rng):
    value, cumulative = rng.random(), 0
    positive = [index for index, probability in enumerate(probabilities) if probability > 0]
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if value < cumulative:
            return index
    return positive[-1]


def step(prefix, draft, target_batch, lookahead, rng):
    if isinstance(lookahead, bool) or not isinstance(lookahead, int) or not 1 <= lookahead <= 64:
        raise ValueError("lookahead must be 1..64")
    proposed, qs, contexts = [], [], []
    for _ in range(lookahead):
        context = list(prefix) + proposed
        contexts.append(context)
        q = distribution(draft(context))
        if qs and len(q) != len(qs[0]):
            raise ValueError("Draft vocabulary changed")
        qs.append(q)
        proposed.append(sample(q, rng))
    contexts.append(list(prefix) + proposed)
    raw = target_batch(contexts)
    if not isinstance(raw, list) or len(raw) != lookahead + 1:
        raise ValueError("Target batch must score every proposal prefix and bonus position")
    ps = [distribution(p) for p in raw]
    if any(len(p) != len(qs[0]) for p in ps):
        raise ValueError("Shared draft/target vocabulary required")
    emitted = []
    for token, q, p in zip(proposed, qs, ps):
        if rng.random() < min(1, p[token] / q[token]):
            emitted.append(token)
        else:
            residual = [max(pn - qn, 0) for pn, qn in zip(p, q)]
            emitted.append(sample(distribution(residual), rng))
            return dict(tokens=emitted, accepted=len(emitted) - 1, proposed=lookahead,
                        rejected=True, target_batch_calls=1)
    emitted.append(sample(ps[-1], rng))
    return dict(tokens=emitted, accepted=lookahead, proposed=lookahead,
                rejected=False, target_batch_calls=1)


def generate(prefix, draft, target_batch, maximum_tokens, lookahead=4, eos=None, seed=42):
    if isinstance(maximum_tokens, bool) or not isinstance(maximum_tokens, int) or maximum_tokens < 0:
        raise ValueError("Nonnegative token limit required")
    if not 1 <= lookahead <= 64:
        raise ValueError("lookahead must be 1..64")
    rng, tokens, batches, accepted, proposed = random.Random(seed), [], 0, 0, 0
    while len(tokens) < maximum_tokens:
        context = list(prefix) + tokens
        remaining = maximum_tokens - len(tokens)
        if remaining == 1:
            probabilities = target_batch([context])
            if len(probabilities) != 1:
                raise ValueError("Target must return one final-position distribution")
            emitted = [sample(distribution(probabilities[0]), rng)]
        else:
            result = step(context, draft, target_batch, min(lookahead, remaining - 1), rng)
            emitted = result["tokens"]
            accepted += result["accepted"]
            proposed += result["proposed"]
        batches += 1
        for token in emitted:
            tokens.append(token)
            if token == eos:
                return dict(tokens=tokens, target_batch_calls=batches, accepted=accepted, proposed=proposed, stopped_on_eos=True)
    return dict(tokens=tokens, target_batch_calls=batches, accepted=accepted, proposed=proposed, stopped_on_eos=False)


def fixture_evaluation(trials=20000, seed=42):
    rng = random.Random(seed)
    observed, accepted, proposed = Counter(), 0, 0
    p, q = [.8, .2], [.2, .8]
    for _ in range(trials):
        result = step([], lambda _: q, lambda contexts: [p for _ in contexts], 4, rng)
        observed[result["tokens"][0]] += 1
        accepted += result["accepted"]
        proposed += result["proposed"]
    return dict(trials=trials, seed=seed, target_distribution=p, draft_distribution=q,
        observed_first_token_distribution=[observed[index] / trials for index in range(2)],
        accepted_proposal_fraction=accepted / proposed,
        evidence="CPU categorical distribution fixture. Does not establish real LLM quality, batched GPU execution or decode speedup.")


def serving_plan(target_snapshot, draft_snapshot, lookahead=4):
    if not target_snapshot or not draft_snapshot or not 1 <= lookahead <= 64:
        raise ValueError("Immutable local snapshots and bounded lookahead required")
    return ["vllm", "serve", target_snapshot, "--host", "127.0.0.1", "--port", "8000",
            "--speculative-config", json.dumps({"model": draft_snapshot, "method": "draft_model",
                                               "num_speculative_tokens": lookahead})]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--target-snapshot")
    parser.add_argument("--draft-snapshot")
    args = parser.parse_args()
    if not 1 <= args.trials <= 1000000:
        parser.error("--trials must be 1..1000000")
    result = fixture_evaluation(args.trials)
    if args.target_snapshot or args.draft_snapshot:
        result["unexecuted_serving_argv"] = serving_plan(args.target_snapshot, args.draft_snapshot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
