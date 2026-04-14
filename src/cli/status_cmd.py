"""status sub-command: server health, VRAM, and cache stats."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from src.config import load_config
from src.core.client import check_server, list_models
from src.monitoring.gpu_collector import get_gpu_stats
from src.server.health import get_cache_stats

app = typer.Typer()


@app.command()
def status(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c"),
    url: Optional[str]   = typer.Option(None, "--url"),
    cache: bool          = typer.Option(False, "--cache", help="Show KV-cache stats"),
    gpu: bool            = typer.Option(True,  "--gpu/--no-gpu"),
) -> None:
    """Show server health, GPU stats, and optional cache info."""
    cfg = load_config(config_file)
    _url = url or f"http://localhost:{cfg['server']['port']}"

    # Server health
    alive = check_server(_url)
    state = "✓  online" if alive else "✗  unreachable"
    typer.echo(f"\n  Server  {_url}  [{state}]")

    if alive:
        try:
            models = list_models(_url)
            for m in models:
                typer.echo(f"    model: {m.get('id', 'unknown')}")
        except Exception:
            pass

        if cache:
            stats = get_cache_stats(_url)
            if stats:
                typer.echo(f"\n  Cache stats:")
                for k, v in stats.items():
                    typer.echo(f"    {k}: {v}")
            else:
                typer.echo("  Cache stats: unavailable")

    # GPU
    if gpu:
        g = get_gpu_stats()
        typer.echo(f"\n  GPU  {g['gpu_name']}")
        typer.echo(f"    VRAM:  {g['vram_used_gb']:.1f} / {g['vram_total_gb']:.0f} GB  "
                   f"({g['vram_free_gb']:.1f} GB free)")
        typer.echo(f"    Temp:  {g['gpu_temp_c']}°C   Power: {g['gpu_power_w']}W   "
                   f"Util: {g['gpu_utilization']}%")

    typer.echo()
