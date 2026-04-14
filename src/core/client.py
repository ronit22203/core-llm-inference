"""Unified HTTP client for SGLang and vLLM (OpenAI-compatible SSE streaming)."""

from __future__ import annotations

import json
import time
from typing import List, Optional, Tuple

import httpx


def check_server(base_url: str, timeout: float = 5.0) -> bool:
    """Return True if an OpenAI-compatible server is reachable at *base_url*."""
    for path in ("/v1/models", "/health"):
        try:
            r = httpx.get(f"{base_url}{path}", timeout=timeout)
            if r.status_code == 200:
                return True
        except Exception:
            continue
    return False


def list_models(base_url: str, timeout: float = 10.0) -> List[dict]:
    """Return the list of models served at *base_url*/v1/models."""
    r = httpx.get(f"{base_url}/v1/models", timeout=timeout)
    r.raise_for_status()
    return r.json().get("data", [])


def stream_chat(
    base_url: str,
    model: str,
    prompt: str,
    timeout: int = 300,
) -> Tuple[str, List[float], float, int, int, float]:
    """
    POST /v1/chat/completions with SSE streaming.

    Returns
    -------
    (response_text, inter_token_latencies_ms, ttft_ms,
     prompt_tokens, output_tokens, total_time_s)
    """
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    response_parts: List[str] = []
    token_times: List[float] = []
    ttft_ms: Optional[float] = None
    prompt_tokens = 0
    output_tokens = 0
    t0 = time.perf_counter()
    last_token_time = t0

    http_timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=5.0)

    try:
        with httpx.Client(timeout=http_timeout) as client:
            with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if (time.perf_counter() - t0) > timeout:
                        raise TimeoutError(f"Query exceeded {timeout}s")

                    if not line or line == "data: [DONE]":
                        continue
                    if not line.startswith("data: "):
                        continue

                    data = json.loads(line[6:])

                    if data.get("usage"):
                        prompt_tokens = data["usage"].get("prompt_tokens", 0)
                        output_tokens = data["usage"].get("completion_tokens", 0)

                    choices = data.get("choices", [])
                    if choices and "delta" in choices[0]:
                        content = choices[0]["delta"].get("content", "")
                        if content:
                            now = time.perf_counter()
                            if ttft_ms is None:
                                ttft_ms = (now - t0) * 1000
                            else:
                                token_times.append((now - last_token_time) * 1000)
                            last_token_time = now
                            response_parts.append(content)

    except (httpx.TimeoutException, TimeoutError):
        pass  # return partial results on timeout

    total_time = time.perf_counter() - t0
    return (
        "".join(response_parts),
        token_times,
        ttft_ms or 0.0,
        prompt_tokens,
        output_tokens,
        total_time,
    )
