"""JSONL export for benchmark results."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from src.core.models import BenchResult


def export_jsonl(
    results: List["BenchResult"],
    output_dir: str | Path,
    meta: Optional[Dict] = None,
) -> Path:
    """Write *results* to a timestamped JSONL file in *output_dir*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if results:
        model      = results[0].model
        engine     = (meta or {}).get("engine", "unknown")
        safe_model = model.replace("/", "_").replace(":", "_")
        timestamp  = time.strftime("%Y%m%d_%H%M%S")
        filename   = f"benchmark_results_{engine}_{safe_model}_{timestamp}.jsonl"
    else:
        filename = f"benchmark_results_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

    path = output_dir / filename
    with open(path, "w") as f:
        for r in results:
            record = asdict(r)
            if meta:
                record["_meta"] = meta
            f.write(json.dumps(record) + "\n")

    return path
