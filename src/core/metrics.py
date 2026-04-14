"""Hardware constants and metric calculations (MBU, TPS, TPOT, ITL)."""

from __future__ import annotations

import statistics
from typing import Dict, List

# Peak memory bandwidth (GB/s) per NVIDIA GPU
GPU_BANDWIDTH_GBPS: Dict[str, float] = {
    "rtx-5090":   1792.0,
    "rtx-5080":    960.0,
    "rtx-5070ti":  896.0,
    "rtx-5070":    672.0,
    "rtx-4090":   1008.0,
    "rtx-4080":    716.8,
    "rtx-4070ti":  504.0,
    "rtx-4070":    504.0,
    "rtx-3090":    936.2,
    "rtx-3080":    760.3,
    "a100-40":    1555.0,
    "a100-80":    2039.0,
    "h100":       3350.0,
    "l40s":        864.0,
}

# Peak memory bandwidth (GB/s) per Apple Silicon chip
CHIP_BANDWIDTH_GBPS: Dict[str, float] = {
    "m1":         68.0,
    "m1-pro":    200.0,
    "m1-max":    400.0,
    "m1-ultra":  800.0,
    "m2":        100.0,
    "m2-pro":    200.0,
    "m2-max":    400.0,
    "m2-ultra":  800.0,
    "m3":        100.0,
    "m3-pro":    150.0,
    "m3-max":    300.0,
    "m4":        120.0,
    "m4-pro":    273.0,
    "m4-max":    546.0,
}

# FP16 model size estimates (GB) for common models
MODEL_SIZE_ESTIMATES_GB: Dict[str, float] = {
    "Qwen/Qwen2.5-3B-Instruct":              6.0,
    "Qwen/Qwen2.5-7B-Instruct":             14.0,
    "Qwen/Qwen2.5-14B-Instruct":            28.0,
    "Qwen/Qwen2.5-72B-Instruct":           144.0,
    "meta-llama/Llama-3.1-8B-Instruct":     16.0,
    "meta-llama/Llama-3.1-70B-Instruct":   140.0,
    "mistralai/Mistral-7B-Instruct-v0.3":   14.0,
}


def get_model_size_gb(model: str) -> float:
    """Return FP16 size estimate for *model*; fall back to param-count heuristic."""
    if model in MODEL_SIZE_ESTIMATES_GB:
        return MODEL_SIZE_ESTIMATES_GB[model]
    for token in model.lower().replace("-", " ").replace("_", " ").split():
        if token.endswith("b"):
            try:
                params = float(token[:-1])
                return round(params * 2.0, 1)  # FP16 ≈ 2 bytes/param
            except ValueError:
                continue
    return 0.0


def compute_mbu(model_size_gb: float, tps: float, peak_bandwidth_gbps: float) -> float:
    """
    Model Bandwidth Utilisation (%).

    Each autoregressive decode step reads all model weights from VRAM:
      bandwidth_needed = model_size_gb × tps
      MBU% = bandwidth_needed / peak_bandwidth_gbps × 100
    """
    if peak_bandwidth_gbps <= 0 or tps <= 0 or model_size_gb <= 0:
        return 0.0
    return min((model_size_gb * tps) / peak_bandwidth_gbps * 100, 100.0)


def percentile(data: List[float], p: float) -> float:
    """Return the *p*-th percentile of *data* (0–100)."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Return mean/median/p95/min/max/stdev for *values*."""
    if not values:
        return {"mean": 0, "median": 0, "p95": 0, "min": 0, "max": 0, "stdev": 0}
    return {
        "mean":   round(statistics.mean(values),   2),
        "median": round(statistics.median(values),  2),
        "p95":    round(percentile(values, 95),     2),
        "min":    round(min(values),                2),
        "max":    round(max(values),                2),
        "stdev":  round(statistics.stdev(values) if len(values) > 1 else 0.0, 2),
    }
