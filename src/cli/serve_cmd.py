"""serve sub-command: launch an SGLang inference server."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from src.config import load_config
from src.server.launcher import SGLangServer

app = typer.Typer()


@app.command()
def serve(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c"),
    model: Optional[str]  = typer.Option(None, "--model",  "-m"),
    host: Optional[str]   = typer.Option(None, "--host"),
    port: Optional[int]   = typer.Option(None, "--port"),
    detach: bool          = typer.Option(False, "--detach", help="Start server and exit"),
) -> None:
    """Launch an SGLang inference server."""
    cfg = load_config(config_file)

    _model = model or cfg["model"]
    _host  = host  or cfg["server"]["host"]
    _port  = port  or cfg["server"]["port"]

    typer.echo(f"  🚀  Starting SGLang server: {_model} on {_host}:{_port}")

    server = SGLangServer(_model, host=_host, port=_port)

    if detach:
        import subprocess, sys
        subprocess.Popen(
            [sys.executable, "-m", "sglang.launch_server",
             "--model-path", _model, "--host", _host, "--port", str(_port)],
        )
        typer.echo(f"  ✓   Server started (detached) — http://{_host}:{_port}")
        return

    try:
        with server:
            typer.echo(f"  ✓   Server ready at http://{_host}:{_port}")
            typer.echo("  Press Ctrl+C to stop.")
            import time
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        typer.echo("\n  🛑  Server stopped.")
