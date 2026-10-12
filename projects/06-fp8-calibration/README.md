# FP8 Quantization and Calibration Harness

Status: CPU E4M3FN reference, scale/clipping evaluation, exact-text corpus manifests, paired perplexity comparison and ModelOpt export preparation implemented and tested. Real model FP8 calibration, TensorRT-LLM serving and GPU throughput remain pending.

```powershell
python fp8_harness.py calibrate projects/06-fp8-calibration/activation-fixture.json --out results/fp8-scalar.json
python fp8_harness.py manifest projects/06-fp8-calibration/corpus-fixture.json --out results/corpus-manifest.json
python projects/06-fp8-calibration/modelopt_prepare.py --corpus projects/06-fp8-calibration/corpus-fixture.json --model-snapshot /models/immutable-snapshot --plan-only --out results/fp8-plan.json
```

The CPU reference implements E4M3FN's finite range, subnormals, signed zero, saturation and round-to-nearest-even. It fits a per-tensor scale from calibration scalars and reports held-out clipping/error. Tests round-trip every finite byte code and check boundary ties. These scalar statistics are not ModelOpt calibration tensors or a quantized model.

Corpus manifests preserve exact text hashes, IDs, split assignments, tokenizer/model revisions and a complete manifest digest. Calibration/evaluation duplicate text is rejected. Perplexity comparison requires the same manifest, token IDs, token positions, evaluation masks and policy. It evaluates held-out documents only, uses a token-weighted mean log probability, and reports quantized-minus-baseline delta and a paired document-bootstrap interval when at least three documents are supplied.

```powershell
python fp8_harness.py compare baseline-scores.json --quantized fp8-scores.json --out results/perplexity-comparison.json
```

Each score file must carry `corpus_manifest`, `corpus_sha256`, `model_revision`, `tokenizer_revision`, `evaluation_policy` and `scores`. Records contain `document_id`, `token_index`, `token_id`, finite nonpositive `logprob` and optional boolean `evaluate`. The scorer validates alignment but cannot attest that a caller's log probabilities came from a real model. A synthetic comparison demonstrates arithmetic only. Larger perplexity usually means worse prediction; the reported delta is signed, not described as a guaranteed drop.

`modelopt_prepare.py --plan-only` runs on the laptop without GPU imports. Its optional CUDA path loads an existing local snapshot with remote code/downloads disabled, calibrates only the calibration split, exports a unified Hugging Face checkpoint and records software versions/corpus provenance. That path is prepared source, not GPU-tested. Replace fixtures, pin the runtime/snapshot and independently verify tokenizer provenance before executing it. The script refuses the shipped synthetic corpus for actual model calibration.

Current ModelOpt guidance uses `export_hf_checkpoint` for TensorRT-LLM's PyTorch backend; the legacy TensorRT-engine checkpoint path is deprecated. The prepared export is therefore not a `trtllm-build` engine artifact. Serving, actual-model perplexity and throughput are separate hardware checks. [Current deployment guidance](https://nvidia.github.io/Model-Optimizer/deployment/1_tensorrt_llm.html), [unified export](https://nvidia.github.io/Model-Optimizer/deployment/3_unified_hf.html).
