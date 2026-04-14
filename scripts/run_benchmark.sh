#!/usr/bin/env bash
# Run the SGLang benchmark using the new CLI entry point.
# Usage: ./scripts/run_benchmark.sh [--all-queries] [--queries N] [-- extra args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

core-llm-inference benchmark "$@"
