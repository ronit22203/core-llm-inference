"""GPU stats collection via nvidia-smi."""

from __future__ import annotations

import subprocess
from typing import Dict


def get_gpu_stats() -> Dict[str, object]:
    """Read first GPU's stats via nvidia-smi; return zeroed dict on failure."""
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.free,memory.total,"
                "temperature.gpu,power.draw,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        parts = [p.strip() for p in raw.splitlines()[0].split(",")]
        used_mb  = float(parts[1])
        free_mb  = float(parts[2])
        total_mb = float(parts[3])
        return {
            "gpu_name":        parts[0],
            "vram_used_gb":    round(used_mb  / 1024, 2),
            "vram_free_gb":    round(free_mb  / 1024, 2),
            "vram_total_gb":   round(total_mb / 1024, 2),
            "gpu_temp_c":      int(float(parts[4])),
            "gpu_power_w":     round(float(parts[5]), 1),
            "gpu_utilization": int(float(parts[6])),
        }
    except Exception as exc:
        return {
            "gpu_name": "unknown",
            "vram_used_gb": 0.0,
            "vram_free_gb": 0.0,
            "vram_total_gb": 0.0,
            "gpu_temp_c": 0,
            "gpu_power_w": 0.0,
            "gpu_utilization": 0,
            "error": str(exc),
        }
