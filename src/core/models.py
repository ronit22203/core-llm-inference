"""Pydantic data models for benchmark requests and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BenchResult:
    """Single-query benchmark result."""

    query_id: int
    category: str
    query: str
    model_response: str
    query_snippet: str
    model: str
    # Core inference metrics
    ttft_ms: float          # Time to First Token (prefill latency)
    tps: float              # Output tokens per second
    tpot_ms: float          # Time Per Output Token
    itl_ms: float           # Inter-Token Latency (≈ TPOT for single stream)
    # Token counts
    prompt_tokens: int
    output_tokens: int
    # Timing breakdown (ms)
    total_duration_ms: float
    generation_ms: float
    # GPU VRAM (nvidia-smi)
    vram_used_gb: float
    vram_free_gb: float
    vram_total_gb: float
    # Efficiency
    mbu_pct: float
    model_size_gb: float
    # GPU metadata
    gpu_name: str = ""
    gpu_temp_c: int = 0
    gpu_power_w: float = 0.0
    gpu_utilization: int = 0
    error: Optional[str] = None
