# Speculative Decoding Pipeline

Status: CPU categorical speculative-sampling reference, batched verification callback contract, generation/EOS boundaries and vLLM serving-argument preparation implemented and tested. Actual draft/target model deployment and decode measurements remain pending.

```powershell
python speculative_reference.py --out results/speculative-fixture.json
```

The draft callback proposes autoregressive tokens and their categorical distributions. One target callback scores every proposal prefix and the bonus position. Acceptance uses the target/draft probability ratio. Rejection samples from the positive residual distribution; complete acceptance adds a target bonus token. Generation respects token limits and terminates at EOS. All callbacks must share the same token-ID vocabulary and apply compatible sampling transforms.

A seeded 20,000-trial fixture with different draft/target probabilities checks the first emitted token distribution against the target. Tests also cover identical distributions, disjoint support, residual rejection, invalid distributions, vocabulary mismatch, one callback per verification round, token limits and EOS. This statistical check supports the reference algorithm; it does not establish full LLM output equivalence or quality.

For future serving-argument preparation:

```powershell
python speculative_reference.py --target-snapshot /models/immutable-target-snapshot --draft-snapshot /models/immutable-draft-snapshot --out results/speculative-plan.json
```

The output includes an unexecuted `vllm serve` argv using `--speculative-config`. Snapshot paths must resolve to immutable, compatible models on the GPU machine. The script does not download models, inspect their parameter counts or launch serving. A 2B draft/70B target setup remains unbuilt and unmeasured. Draft generation is autoregressive; the target callback is a batch contract, not proof of one real GPU forward pass.

No 2x speedup or zero quality loss is claimed for a deployed model. Real validation requires draft acceptance, target-only comparison under the same sampling distribution, tokenizer/vocabulary checks, TTFT/ITL, memory and end-to-end decode throughput. [vLLM draft-model configuration](https://docs.vllm.ai/en/latest/features/speculative_decoding/draft_model/).
