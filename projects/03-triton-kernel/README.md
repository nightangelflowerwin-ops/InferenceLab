# Custom Fused Triton Kernel

Status: CPU numerical oracle, prepared fused Triton RMSNorm source and GPU benchmark/correctness runner implemented. CPU oracle tests and Python source parsing pass. Triton compilation, GPU correctness and timings remain unverified until CUDA execution.

The prepared kernel reduces each contiguous row in float32, applies the RMS scale and weights, and stores the result. It covers widths 1..8192, odd-width masking, float16/bfloat16/float32, explicit epsilon and empty row batches. It accepts matching contiguous CUDA input/weight tensors.

Run on a compatible CUDA machine with pinned PyTorch and Triton:

```bash
python projects/03-triton-kernel/rmsnorm_triton.py --out results/rmsnorm-gpu.json
```

The runner compares against a float32 formula and native PyTorch RMSNorm before timing either implementation with Triton's benchmark helper. It records hardware/software versions, shapes, dtype, tolerance, maximum absolute error and timings. Allocation is included for both implementations; model-level gains need a separate integration study. Unsupported bfloat16 hardware is skipped explicitly.

The laptop uses the dependency-free [CPU oracle](../../rmsnorm_reference.py) for known-value, symmetry, zero, width and overflow checks. Parsing the Triton file proves Python syntax only. It does not validate Triton code generation, numerical tolerances on GPU or performance. No speedup is claimed.

Reference: [Triton normalization/reduction tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html). The kernel here implements RMSNorm without mean subtraction; it is an original project implementation.
