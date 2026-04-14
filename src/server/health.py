"""Server health checks and cache statistics."""

from __future__ import annotations

from typing import Dict, Optional

import httpx

from src.core.client import check_server


def wait_for_server(
    base_url: str,
    retries: int = 30,
    interval: float = 2.0,
) -> bool:
    """Poll *base_url* until the server responds or *retries* exhausted."""
    import time

    for _ in range(retries):
        if check_server(base_url):
            return True
        time.sleep(interval)
    return False


def get_cache_stats(base_url: str, timeout: float = 5.0) -> Optional[Dict]:
    """Return SGLang /get_cache_stats payload, or None if unavailable."""
    try:
        r = httpx.get(f"{base_url}/get_cache_stats", timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None
