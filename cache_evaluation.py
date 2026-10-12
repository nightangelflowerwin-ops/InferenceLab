"""Evaluate cache hit and incorrect-reuse rates on supplied labeled vectors."""
import argparse
import json
from pathlib import Path
import tempfile

from semantic_cache import SQLiteVectors, SQLiteResponses, SemanticCache


def evaluate(corpus, queries, thresholds):
    if not queries:
        raise ValueError("Held-out queries required")
    results = []
    for threshold in thresholds:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.db"
            index, responses = SQLiteVectors(path), SQLiteResponses(path)
            cache = SemanticCache(index, responses, lambda _: [1, 0], threshold=threshold, clock=lambda: 0)
            for entry in corpus:
                cache.put(entry["namespace"], entry["embedding"], {"answer_id": entry["answer_id"]})
            details, hits, incorrect = [], 0, 0
            for query in queries:
                cache.embedder = lambda _, embedding=query["embedding"]: embedding
                response, _, score = cache.lookup(query["namespace"], query["id"])
                hit = response is not None
                wrong = hit and response["answer_id"] not in query["acceptable_answer_ids"]
                hits += hit
                incorrect += wrong
                details.append(dict(query_id=query["id"], hit=hit, incorrect_reuse=bool(wrong), similarity=score))
            results.append(dict(threshold=threshold, queries=len(queries), hits=hits, incorrect_reuses=incorrect,
                hit_rate=hits / len(queries), incorrect_reuse_rate=incorrect / len(queries),
                incorrect_fraction_of_hits=incorrect / hits if hits else None, details=details))
    return {"results": results, "evidence": "Labeled vector replay only. Similarity does not establish answer equivalence; input provenance determines whether this is fixture or real embedding evidence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    result = evaluate(source["corpus"], source["queries"], source["thresholds"])
    result["source_kind"] = source.get("source_kind", "unverified")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"results": [{k: v for k, v in row.items() if k != "details"} for row in result["results"]]}, indent=2))


if __name__ == "__main__":
    main()
