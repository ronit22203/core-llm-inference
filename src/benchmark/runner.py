"""Benchmark runner: orchestrates single and sequential query execution."""

from __future__ import annotations

import statistics
from typing import List, Optional

from src.core.client import stream_chat
from src.core.metrics import compute_mbu, get_model_size_gb
from src.core.models import BenchResult
from src.monitoring.gpu_collector import get_gpu_stats


def benchmark_query(
    query: dict,
    model: str,
    base_url: str,
    model_size_gb: float,
    bandwidth_gbps: float,
) -> BenchResult:
    """Run a single query against the server and return a populated BenchResult."""

    def _error(exc: Exception) -> BenchResult:
        return BenchResult(
            query_id=query["id"], category=query["category"],
            query=query["query"], model_response="",
            query_snippet=query["query"][:70], model=model,
            ttft_ms=0, tps=0, tpot_ms=0, itl_ms=0,
            prompt_tokens=0, output_tokens=0,
            total_duration_ms=0, generation_ms=0,
            vram_used_gb=0, vram_free_gb=0, vram_total_gb=0,
            mbu_pct=0, model_size_gb=model_size_gb, error=str(exc),
        )

    try:
        (response_text, token_times, ttft_ms,
         prompt_tokens, output_tokens, total_time_s) = stream_chat(
            base_url, model, query["query"]
        )
    except Exception as exc:
        return _error(exc)

    if output_tokens == 0 and not response_text:
        return _error(Exception("Empty response — no tokens generated"))

    gpu = get_gpu_stats()
    total_ms      = total_time_s * 1000
    generation_ms = total_ms - ttft_ms if ttft_ms > 0 else total_ms
    tps           = output_tokens / (generation_ms / 1000) if generation_ms > 0 else 0.0
    tpot_ms       = generation_ms / output_tokens if output_tokens > 0 else 0.0
    itl_ms        = statistics.mean(token_times) if token_times else tpot_ms
    mbu_pct       = compute_mbu(model_size_gb, tps, bandwidth_gbps)

    snippet = query["query"][:70] + ("…" if len(query["query"]) > 70 else "")

    return BenchResult(
        query_id=query["id"],
        category=query["category"],
        query=query["query"],
        model_response=response_text,
        query_snippet=snippet,
        model=model,
        ttft_ms=round(ttft_ms,       2),
        tps=round(tps,               2),
        tpot_ms=round(tpot_ms,       2),
        itl_ms=round(itl_ms,         2),
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_duration_ms=round(total_ms,      2),
        generation_ms=round(generation_ms,     2),
        vram_used_gb=gpu["vram_used_gb"],
        vram_free_gb=gpu["vram_free_gb"],
        vram_total_gb=gpu["vram_total_gb"],
        mbu_pct=round(mbu_pct,       2),
        model_size_gb=round(model_size_gb, 3),
        gpu_name=gpu["gpu_name"],
        gpu_temp_c=gpu["gpu_temp_c"],
        gpu_power_w=gpu["gpu_power_w"],
        gpu_utilization=gpu["gpu_utilization"],
    )


def run_benchmark(
    queries: List[dict],
    model: str,
    base_url: str,
    model_size_gb: Optional[float] = None,
    bandwidth_gbps: float = 0.0,
    on_result=None,
) -> List[BenchResult]:
    """
    Run *queries* sequentially and return results.

    Parameters
    ----------
    on_result:
        Optional callback(index, total, result) called after each query.
    """
    if model_size_gb is None:
        model_size_gb = get_model_size_gb(model)

    results: List[BenchResult] = []
    total = len(queries)
    for i, query in enumerate(queries, 1):
        result = benchmark_query(query, model, base_url, model_size_gb, bandwidth_gbps)
        results.append(result)
        if on_result:
            on_result(i, total, result)

    return results
