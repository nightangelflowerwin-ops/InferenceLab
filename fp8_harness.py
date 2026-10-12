"""CPU E4M3FN reference calibration and token-aligned perplexity comparison.

Reference scales are not exported TensorRT/ModelOpt quantization artifacts.
"""
import argparse
from bisect import bisect_left
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Finite numeric values required")
    return value


def decode_e4m3(code):
    if isinstance(code, bool) or not isinstance(code, int) or not 0 <= code <= 255:
        raise ValueError("One byte required")
    sign, exponent, fraction = (-1 if code & 128 else 1), (code >> 3) & 15, code & 7
    if exponent == 15 and fraction == 7:
        return float("nan")
    value = fraction / 8 * 2 ** -6 if exponent == 0 else (1 + fraction / 8) * 2 ** (exponent - 7)
    return math.copysign(value, sign)


POSITIVE_VALUES = [decode_e4m3(code) for code in range(127)]


def encode_e4m3(value):
    finite(value)
    sign = 128 if math.copysign(1, value) < 0 else 0
    absolute = min(abs(value), 448)
    upper = bisect_left(POSITIVE_VALUES, absolute)
    if upper == 0:
        chosen = 0
    elif upper == len(POSITIVE_VALUES):
        chosen = 126
    else:
        lower = upper - 1
        distances = (absolute - POSITIVE_VALUES[lower], POSITIVE_VALUES[upper] - absolute)
        chosen = lower if distances[0] < distances[1] else upper if distances[1] < distances[0] else (
            lower if lower % 2 == 0 else upper)
    return sign | chosen


def percentile(values, fraction):
    if not values or not 0 <= fraction <= 1:
        raise ValueError("Nonempty values and bounded percentile required")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def calibration_report(training, heldout, clipping_percentile=100):
    if not training or not heldout or not 0 < clipping_percentile <= 100:
        raise ValueError("Separate nonempty calibration and held-out values required")
    for value in training + heldout:
        finite(value)
    maximum = percentile([abs(value) for value in training], clipping_percentile / 100)
    scale = maximum / 448 if maximum else 1
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("Scale cannot be represented")
    reconstructed = []
    inference_bound = maximum if maximum else 448
    for value in heldout:
        # Saturate before division to avoid overflow for extreme held-out outliers.
        clipped = min(max(value, -inference_bound), inference_bound)
        reconstructed.append(decode_e4m3(encode_e4m3(clipped / scale)) * scale)
    errors = [estimate - actual for estimate, actual in zip(reconstructed, heldout)]
    bound = max(abs(error) for error in errors)
    rmse = bound * math.sqrt(math.fsum((error / bound) ** 2 for error in errors) / len(errors)) if bound else 0
    return {"dtype": "E4M3FN CPU reference", "scale": scale, "calibration_absolute_bound": maximum,
            "inference_absolute_clip_bound": inference_bound,
            "clipping_percentile": clipping_percentile, "calibration_count": len(training),
            "heldout_count": len(heldout), "heldout_clipped_fraction": sum(abs(value) > inference_bound for value in heldout) / len(heldout),
            "heldout_rmse": rmse, "heldout_max_absolute_error": bound,
            "evidence": "CPU scalar rounding/clipping study, not model calibration, perplexity or GPU throughput"}


def corpus_manifest(documents, tokenizer, tokenizer_revision, model_revision):
    if not all(isinstance(value, str) and value for value in (tokenizer, tokenizer_revision, model_revision)):
        raise ValueError("Tokenizer/model provenance required")
    ids, texts, entries = set(), {}, []
    for document in documents:
        key, text, split = document["id"], document["text"], document["split"]
        if not isinstance(key, str) or not key or key in ids or not isinstance(text, str) or not text:
            raise ValueError("Unique IDs and exact nonempty text required")
        if split not in ("calibration", "evaluation"):
            raise ValueError("Explicit calibration/evaluation split required")
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        if text_hash in texts and texts[text_hash] != split:
            raise ValueError("Calibration/evaluation text leakage")
        ids.add(key)
        texts[text_hash] = split
        entries.append({"id": key, "split": split, "text_sha256": text_hash, "utf8_bytes": len(text.encode())})
    if {entry["split"] for entry in entries} != {"calibration", "evaluation"}:
        raise ValueError("Both corpus splits required")
    manifest = {"tokenizer": tokenizer, "tokenizer_revision": tokenizer_revision,
                "model_revision": model_revision, "documents": sorted(entries, key=lambda entry: entry["id"])}
    manifest["sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return manifest


def validated_scores(records):
    result = {}
    for record in records:
        key = record["document_id"], record["token_index"]
        if not isinstance(key[0], str) or not key[0] or isinstance(key[1], bool) or not isinstance(key[1], int) or key[1] < 0 or key in result:
            raise ValueError("Unique document/token-index scores required")
        token = record["token_id"]
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise ValueError("Nonnegative token IDs required")
        logprob = finite(record["logprob"])
        if logprob > 0 or not isinstance(record.get("evaluate", True), bool):
            raise ValueError("Nonpositive log probability and boolean mask required")
        result[key] = (token, logprob, record.get("evaluate", True))
    if not result or not any(record[2] for record in result.values()):
        raise ValueError("At least one evaluated token required")
    return result


def perplexity(values):
    nll = -math.fsum(values) / len(values)
    if nll > 700:
        raise ValueError("Perplexity overflows supported report range")
    return math.exp(nll)


def compare_scores(baseline, quantized, bootstrap_repeats=500, seed=42):
    for field in ("corpus_sha256", "tokenizer_revision", "model_revision", "evaluation_policy"):
        if not baseline.get(field) or baseline.get(field) != quantized.get(field):
            raise ValueError(f"Paired comparison provenance differs: {field}")
    manifest = baseline.get("corpus_manifest")
    if not isinstance(manifest, dict) or manifest != quantized.get("corpus_manifest"):
        raise ValueError("The same complete corpus manifest must accompany both traces")
    unsigned_manifest = {key: value for key, value in manifest.items() if key != "sha256"}
    expected_hash = hashlib.sha256(json.dumps(unsigned_manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if expected_hash != manifest.get("sha256") or expected_hash != baseline["corpus_sha256"]:
        raise ValueError("Corpus manifest digest differs")
    if manifest.get("model_revision") != baseline["model_revision"] or manifest.get("tokenizer_revision") != baseline["tokenizer_revision"]:
        raise ValueError("Score/model tokenizer provenance differs from corpus manifest")
    first, second = validated_scores(baseline["scores"]), validated_scores(quantized["scores"])
    if set(first) != set(second):
        raise ValueError("Scored token positions differ")
    evaluation_ids = {entry["id"] for entry in manifest["documents"] if entry["split"] == "evaluation"}
    if any(key[0] not in evaluation_ids for key in first):
        raise ValueError("Scores must refer to evaluation documents only")
    documents = defaultdict(lambda: [[], []])
    for key in sorted(first):
        token, logprob, mask = first[key]
        qtoken, qlogprob, qmask = second[key]
        if token != qtoken or mask != qmask:
            raise ValueError("Token IDs or evaluation masks differ")
        if mask:
            documents[key[0]][0].append(logprob)
            documents[key[0]][1].append(qlogprob)
    base_values = [value for pair in documents.values() for value in pair[0]]
    quant_values = [value for pair in documents.values() for value in pair[1]]
    base_ppl, quant_ppl = perplexity(base_values), perplexity(quant_values)
    interval = None
    if len(documents) >= 3 and bootstrap_repeats > 0:
        if not 1 <= bootstrap_repeats <= 10000:
            raise ValueError("Bootstrap count must be 1..10000")
        rng, pairs, deltas = random.Random(seed), list(documents.values()), []
        for _ in range(bootstrap_repeats):
            selected = [rng.choice(pairs) for _ in pairs]
            deltas.append(perplexity([v for pair in selected for v in pair[1]]) -
                          perplexity([v for pair in selected for v in pair[0]]))
        interval = [percentile(deltas, .025), percentile(deltas, .975)]
    return {"evaluated_tokens": len(base_values), "evaluated_documents": len(documents),
            "baseline_perplexity": base_ppl, "quantized_perplexity": quant_ppl,
            "perplexity_delta_quantized_minus_baseline": quant_ppl - base_ppl,
            "relative_perplexity_change": quant_ppl / base_ppl - 1,
            "paired_document_bootstrap_delta_95_percent_interval": interval,
            "seed": seed, "provenance": {field: baseline[field] for field in ("corpus_sha256", "tokenizer_revision", "model_revision", "evaluation_policy")},
            "evidence": "Arithmetic on supplied aligned next-token log probabilities; model execution and trace provenance require separate evidence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["calibrate", "compare", "manifest"])
    parser.add_argument("input", type=Path)
    parser.add_argument("--quantized", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    if args.action == "calibrate":
        result = calibration_report(source["calibration_values"], source["heldout_values"], source.get("clipping_percentile", 100))
    elif args.action == "manifest":
        result = corpus_manifest(source["documents"], source["tokenizer"], source["tokenizer_revision"], source["model_revision"])
    else:
        if not args.quantized:
            parser.error("--quantized score file required")
        result = compare_scores(source, json.loads(args.quantized.read_text(encoding="utf-8")))
    result["source_kind"] = source.get("source_kind", "unverified")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
