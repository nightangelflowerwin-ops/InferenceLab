"""Dependency-free CPU mathematical RMSNorm oracle; not a speed benchmark."""
import math


def rmsnorm(rows, weight, epsilon=1e-6):
    if not weight or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Positive epsilon and nonempty weights required")
    if any(not isinstance(n, (int, float)) or isinstance(n, bool) or not math.isfinite(n) for n in weight):
        raise ValueError("Finite numeric weights required")
    result = []
    for row in rows:
        if len(row) != len(weight) or any(not isinstance(n, (int, float)) or isinstance(n, bool) or not math.isfinite(n) for n in row):
            raise ValueError("Row width or values differ from contract")
        maximum = max(abs(value) for value in row)
        rms = maximum * math.sqrt(math.fsum((value / maximum) ** 2 for value in row) / len(weight)) if maximum else 0
        denominator = math.hypot(rms, math.sqrt(epsilon))
        result.append([value / denominator * scale for value, scale in zip(row, weight)])
    return result
