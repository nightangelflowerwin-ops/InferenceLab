"""Prepared CUDA RMSNorm implementation and native PyTorch benchmark.

GPU correctness and timings are unverified on the CPU-only development laptop.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.testing


@triton.jit
def _rmsnorm(X, W, Y, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    values = tl.load(X + row * N + columns, columns < N, other=0).to(tl.float32)
    weights = tl.load(W + columns, columns < N, other=0).to(tl.float32)
    mean_square = tl.sum(values * values, axis=0) / N
    normalized = values * tl.rsqrt(mean_square + EPS) * weights
    tl.store(Y + row * N + columns, normalized, columns < N)


def fused_rmsnorm(x, weight, epsilon=1e-6):
    if x.ndim != 2 or weight.ndim != 1 or x.shape[1] != weight.numel():
        raise ValueError("Expected contiguous [rows,width] input and [width] weights")
    if not x.is_cuda or not weight.is_cuda or x.device != weight.device:
        raise ValueError("Both tensors must share a CUDA device")
    if not x.is_contiguous() or not weight.is_contiguous() or x.dtype != weight.dtype:
        raise ValueError("Matching dtype and contiguous tensors required")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Unsupported dtype")
    if not 0 < x.shape[1] <= 8192 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Width must be 1..8192 and epsilon positive/finite")
    output = torch.empty_like(x)
    if x.shape[0]:
        block = triton.next_power_of_2(x.shape[1])
        _rmsnorm[(x.shape[0],)](x, weight, output, x.shape[1], epsilon, block,
                              num_warps=4 if block <= 4096 else 8)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required; use the CPU oracle for local mathematical checks")
    torch.manual_seed(42)
    records = []
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            continue
        for rows, width in ((1, 127), (32, 768), (128, 4096), (32, 8192)):
            x = torch.randn(rows, width, device="cuda", dtype=dtype)
            weight = torch.randn(width, device="cuda", dtype=dtype)
            epsilon = 1e-6
            reference = (x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + epsilon) * weight.float()).to(dtype)
            actual = fused_rmsnorm(x, weight, epsilon)
            tolerance = .02 if dtype == torch.bfloat16 else .003 if dtype == torch.float16 else 1e-5
            torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)
            native = lambda: torch.nn.functional.rms_norm(x, (width,), weight, epsilon)
            torch.testing.assert_close(native(), reference, rtol=tolerance, atol=tolerance)
            fused_ms = triton.testing.do_bench(lambda: fused_rmsnorm(x, weight, epsilon), warmup=100, rep=500)
            native_ms = triton.testing.do_bench(native, warmup=100, rep=500)
            records.append(dict(rows=rows, width=width, dtype=str(dtype), fused_ms=fused_ms,
                native_pytorch_ms=native_ms, speedup=native_ms / fused_ms,
                max_absolute_error=float((actual.float() - reference.float()).abs().max()),
                correctness_tolerance=tolerance))
    result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
                  seed=42, epsilon=1e-6, records=records,
                  evidence="GPU execution results produced only when this script runs successfully on CUDA; allocation is included in each implementation's timing")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
