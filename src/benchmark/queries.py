"""Load and filter benchmark queries from query.json."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional

# Default path — same directory as this package's parent utils/ folder
_DEFAULT_QUERY_FILE = Path(__file__).parents[3] / "utils" / "query.json"


def load_queries(query_file: Optional[Path] = None) -> List[dict]:
    """Return all medical queries from *query_file*."""
    path = Path(query_file) if query_file else _DEFAULT_QUERY_FILE
    with open(path) as f:
        return json.load(f)["medical_queries"]


def filter_by_category(queries: List[dict], categories: List[str]) -> List[dict]:
    """Return only queries whose category is in *categories*."""
    if not categories:
        return queries
    return [q for q in queries if q["category"] in categories]


def sample_queries(
    queries: List[dict],
    n: int,
    seed: int = 42,
    all_queries: bool = False,
) -> List[dict]:
    """Return *n* randomly sampled queries (sorted by id); or all if *all_queries*."""
    if all_queries or n >= len(queries):
        return sorted(queries, key=lambda q: q["id"])
    rng = random.Random(seed)
    sampled = rng.sample(queries, n)
    return sorted(sampled, key=lambda q: q["id"])
