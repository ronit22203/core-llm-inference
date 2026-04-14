"""CLI entry point — core-llm-inference."""

from __future__ import annotations

import typer

from src.cli.benchmark_cmd import benchmark
from src.cli.serve_cmd import serve
from src.cli.status_cmd import status

app = typer.Typer(
    name="core-llm-inference",
    help="LLM inference benchmarking and server management for SGLang/vLLM.",
    add_completion=False,
)

app.command("benchmark")(benchmark)
app.command("serve")(serve)
app.command("status")(status)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
