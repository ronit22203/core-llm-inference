"""Push benchmark metrics to a Prometheus Pushgateway."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from src.core.models import BenchResult


def push_metrics(
    results: List["BenchResult"],
    pushgateway_url: str,
    job: str = "llm_benchmark",
) -> None:
    """Push per-run summary gauges to *pushgateway_url*."""
    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
    except ImportError:
        raise ImportError(
            "prometheus_client is required for Prometheus push. "
            "Install with: pip install prometheus-client"
        )

    good = [r for r in results if not r.error]
    if not good:
        return

    import statistics

    registry = CollectorRegistry()

    def gauge(name: str, desc: str, value: float) -> None:
        g = Gauge(name, desc, registry=registry)
        g.set(value)

    gauge("llm_ttft_mean_ms",       "Mean TTFT (ms)",             statistics.mean(r.ttft_ms  for r in good))
    gauge("llm_tps_mean",           "Mean tokens per second",     statistics.mean(r.tps      for r in good))
    gauge("llm_tpot_mean_ms",       "Mean TPOT (ms)",             statistics.mean(r.tpot_ms  for r in good))
    gauge("llm_mbu_mean_pct",       "Mean MBU (%)",               statistics.mean(r.mbu_pct  for r in good))
    gauge("llm_output_tokens_mean", "Mean output tokens",         statistics.mean(r.output_tokens for r in good))
    gauge("llm_gpu_power_mean_w",   "Mean GPU power draw (W)",    statistics.mean(r.gpu_power_w   for r in good))
    gauge("llm_gpu_temp_mean_c",    "Mean GPU temperature (°C)",  statistics.mean(float(r.gpu_temp_c) for r in good))
    gauge("llm_queries_total",      "Total queries run",          len(results))
    gauge("llm_queries_failed",     "Failed queries",             sum(1 for r in results if r.error))

    push_to_gateway(pushgateway_url, job=job, registry=registry)
