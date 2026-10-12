"""Prepared ModelOpt FP8 calibration/export script; CUDA execution unverified.

Plan mode validates the exact local corpus without loading or downloading models.
Current TensorRT-LLM uses unified HF exports with its PyTorch backend.
"""
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fp8_harness import corpus_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_sequence_length < 1:
        parser.error("Positive batch and sequence limits required")
    source = json.loads(args.corpus.read_text(encoding="utf-8"))
    manifest = corpus_manifest(source["documents"], source["tokenizer"], source["tokenizer_revision"], source["model_revision"])
    plan = {"corpus_manifest": manifest, "model_snapshot": str(args.model_snapshot),
            "batch_size": args.batch_size, "max_sequence_length": args.max_sequence_length,
            "seed": 42, "export": "modelopt.torch.export.export_hf_checkpoint",
            "serving_backend": "TensorRT-LLM PyTorch",
            "evidence": "Prepared configuration only; no CUDA calibration or serving run"}
    if args.plan_only:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print("Wrote corpus-validated preparation plan; no model loaded")
        return
    if not args.model_snapshot.is_dir():
        parser.error("Existing immutable local model snapshot required")
    if source.get("source_kind", "").startswith("synthetic"):
        parser.error("Replace synthetic corpus fixtures before calibrating a real model")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint
    if not torch.cuda.is_available():
        parser.error("CUDA required for this prepared calibration path")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model_snapshot, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        parser.error("Tokenizer must define a pad or EOS token")
    model = AutoModelForCausalLM.from_pretrained(args.model_snapshot, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.bfloat16, device_map="auto").eval()
    texts = [document["text"] for document in source["documents"] if document["split"] == "calibration"]
    input_device = model.get_input_embeddings().weight.device
    def forward_loop(calibrating_model):
        with torch.inference_mode():
            for offset in range(0, len(texts), args.batch_size):
                batch = tokenizer(texts[offset:offset + args.batch_size], padding=True, truncation=True,
                                  max_length=args.max_sequence_length, return_tensors="pt")
                batch = {key: value.to(input_device) for key, value in batch.items()}
                calibrating_model(**batch)
    quantized = mtq.quantize(model, copy.deepcopy(mtq.FP8_DEFAULT_CFG), forward_loop)
    with torch.inference_mode():
        export_hf_checkpoint(quantized, export_dir=str(args.out))
    tokenizer.save_pretrained(args.out)
    plan["evidence"] = "ModelOpt calibration/export completed; TensorRT-LLM serving and quality are separate checks"
    plan["software"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "nvidia-modelopt")}
    (args.out / "inference-lab-calibration.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print("Exported quantized HF checkpoint; run TensorRT-LLM serving and paired quality checks separately")


if __name__ == "__main__":
    main()
