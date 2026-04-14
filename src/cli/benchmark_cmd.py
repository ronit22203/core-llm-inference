"""benchmark sub-command: run inference benchmark against a live server."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import typer

from src.benchmark.exporters import export_jsonl
from src.benchmark.queries import filter_by_category, load_queries, sample_queries
from src.benchmark.reporters import print_progress, print_summary
from src.benchmark.runner import run_benchmark
from src.config import load_config
from src.core.client import check_server, list_models, stream_chat
from src.core.metrics import GPU_BANDWIDTH_GBPS, get_model_size_gb
from src.monitoring.gpu_collector import get_gpu_stats

app = typer.Typer()


@app.command()
def benchmark(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file"),
    model: Optional[str]  = typer.Option(None, "--model",  "-m", help="Model name"),
    url: Optional[str]    = typer.Option(None, "--url",         help="Server base URL"),
    num_queries: Optional[int]  = typer.Option(None, "--queries", "-n"),
    all_queries: bool     = typer.Option(False, "--all-queries",  help="Run all queries"),
    gpu: Optional[str]    = typer.Option(None, "--gpu",          help="GPU type key for MBU calc"),
    model_size_gb: Optional[float] = typer.Option(None, "--model-size-gb"),
    categories: Optional[List[str]] = typer.Option(None, "--category", "-C", help="Filter categories"),
    warmup: Optional[bool] = typer.Option(None, "--warmup/--no-warmup"),
    seed: Optional[int]   = typer.Option(None, "--seed"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
    no_jsonl: bool        = typer.Option(False, "--no-jsonl",    help="Disable JSONL export"),
    prometheus: bool      = typer.Option(False, "--prometheus",  help="Push metrics to Pushgateway"),
    pushgateway: Optional[str] = typer.Option(None, "--pushgateway"),
) -> None:
    """Run an LLM inference benchmark against a live SGLang/vLLM server."""
    cfg = load_config(config_file)

    # CLI flags override config
    _model       = model       or cfg.get("model", "Qwen/Qwen2.5-7B-Instruct")
    _url         = url         or f"http://{cfg['server']['host']}:{cfg['server']['port']}"
    _url         = _url.replace("0.0.0.0", "localhost")
    _gpu         = gpu         or cfg["gpu"]["type"]
    _n           = num_queries or cfg["benchmark"]["num_queries"]
    _seed        = seed        or cfg["benchmark"]["seed"]
    _warmup      = warmup if warmup is not None else cfg["benchmark"]["warmup"]
    _cats        = list(categories) if categories else cfg["benchmark"].get("categories", [])
    _out_dir     = output_dir  or Path(cfg["benchmark"]["output_dir"])
    _model_size  = model_size_gb or cfg.get("model_size_gb") or get_model_size_gb(_model)
    _bw          = cfg["gpu"].get("peak_bandwidth_gbps") or GPU_BANDWIDTH_GBPS.get(_gpu, 0.0)
    _pushgw      = pushgateway or cfg["monitoring"]["prometheus_pushgateway"]

    # Preflight
    if not check_server(_url):
        typer.echo(f"❌  Server not reachable at {_url}", err=True)
        raise typer.Exit(1)

    # Load queries
    queries = load_queries()
    queries = filter_by_category(queries, _cats)
    if not queries:
        typer.echo(f"❌  No queries found for categories: {_cats}", err=True)
        raise typer.Exit(1)
    queries = sample_queries(queries, _n, seed=_seed, all_queries=all_queries)

    # Print header
    gpu0 = get_gpu_stats()
    typer.echo()
    typer.echo("  ╔════════════════════════════════════════════════════════════╗")
    typer.echo(f"  ║  SGLang Benchmark  │  {_model:<36}  ║")
    typer.echo("  ╚════════════════════════════════════════════════════════════╝")
    typer.echo(f"  GPU:    {gpu0['gpu_name']}  →  {_bw} GB/s peak bandwidth")
    typer.echo(f"  Model:  {_model}  (~{_model_size:.1f} GB)")
    typer.echo(f"  Server: {_url}")
    typer.echo(f"  Queries: {len(queries)}")
    typer.echo()

    # Warmup
    if _warmup:
        typer.echo("  ⏳  Warming up …")
        try:
            stream_chat(_url, _model, "Hello", timeout=60)
            typer.echo("  ✓   Warmup complete\n")
        except Exception as exc:
            typer.echo(f"  ⚠   Warmup failed: {exc}\n")

    # Run
    typer.echo("─" * 110)
    typer.echo(
        f"  {'#':>6}  S  {'QID':>4}  {'Category':<22}"
        f"  {'TTFT':>8}  {'TPS':>7}  {'TPOT':>8}  {'MBU':>6}  {'VRAM':>6}  {'Temp':>5}"
    )
    typer.echo("─" * 110)

    results = run_benchmark(
        queries, _model, _url, _model_size, _bw,
        on_result=lambda i, total, r: print_progress(i, total, r),
    )

    typer.echo("─" * 110)
    typer.echo()
    print_summary(results, _bw)

    # JSONL export
    if not no_jsonl and cfg["monitoring"].get("log_jsonl", True):
        meta = {
            "engine":         cfg.get("engine", "sglang"),
            "model":          _model,
            "gpu":            _gpu,
            "gpu_name":       gpu0["gpu_name"],
            "bandwidth_gbps": _bw,
            "model_size_gb":  _model_size,
            "server_url":     _url,
            "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        path = export_jsonl(results, _out_dir, meta=meta)
        typer.echo(f"  💾  Results saved → {path.resolve()}")
        typer.echo()

    # Prometheus push
    if prometheus:
        from src.monitoring.prometheus import push_metrics
        try:
            push_metrics(results, _pushgw)
            typer.echo(f"  📊  Metrics pushed → {_pushgw}")
        except Exception as exc:
            typer.echo(f"  ⚠   Prometheus push failed: {exc}", err=True)
