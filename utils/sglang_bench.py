#!/usr/bin/env python3
"""
sglang_bench.py
===============
LLM inference benchmark for SGLang on NVIDIA GPUs (prod Vast.ai RTX 5080).

Metrics
-------
  TTFT   Time to First Token          — user-perceived responsiveness  (ms)
  TPS    Output Tokens Per Second     — generation throughput          (tok/s)
  TPOT   Time Per Output Token        — per-step generation latency    (ms/tok)
  ITL    Inter-Token Latency          — same as TPOT for single stream (ms/tok)
  VRAM   GPU memory usage/headroom    — nvidia-smi                     (GB)
  MBU    Model Bandwidth Utilization  — hardware efficiency            (%)

MBU formula
-----------
Each autoregressive decode step reads all model weights from VRAM.
  bandwidth_needed (GB/s) = model_size_gb × tps
  MBU (%) = bandwidth_needed / peak_bandwidth_gbps × 100

Usage
-----
  # benchmark with 10 random queries (default) — JSONL auto-saved
  python utils/sglang_bench.py --model Qwen/Qwen2.5-7B-Instruct

  # run all 100 queries, specify GPU for MBU calculation
  python utils/sglang_bench.py --model Qwen/Qwen2.5-7B-Instruct --all-queries --gpu rtx-5080

  # filter by category, warmup
  python utils/sglang_bench.py --model Qwen/Qwen2.5-7B-Instruct \\
      --categories Cardiology Oncology --warmup

  # custom SGLang server URL
  python utils/sglang_bench.py --model Qwen/Qwen2.5-7B-Instruct --url http://remote:30000
"""

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import httpx
except ImportError:
    print("❌  httpx is required.  Install with:  pip install httpx")
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────────────

SGLANG_DEFAULT_URL = "http://localhost:30000"
QUERY_FILE         = Path(__file__).parent / "query.json"

# Peak memory bandwidth (GB/s) per NVIDIA GPU
GPU_BANDWIDTH_GBPS: Dict[str, float] = {
    "rtx-5090":  1792.0,
    "rtx-5080":   960.0,
    "rtx-5070ti": 896.0,
    "rtx-5070":   672.0,
    "rtx-4090":  1008.0,
    "rtx-4080":   716.8,
    "rtx-4070ti": 504.0,
    "rtx-4070":   504.0,
    "rtx-3090":   936.2,
    "rtx-3080":   760.3,
    "a100-40":   1555.0,
    "a100-80":   2039.0,
    "h100":      3350.0,
    "l40s":       864.0,
}

# Model size estimates (GB) for common models when API doesn't report it
MODEL_SIZE_ESTIMATES_GB: Dict[str, float] = {
    "Qwen/Qwen2.5-7B-Instruct":    14.0,   # FP16
    "Qwen/Qwen2.5-14B-Instruct":   28.0,
    "Qwen/Qwen2.5-72B-Instruct":  144.0,
    "Qwen/Qwen2.5-3B-Instruct":     6.0,
    "meta-llama/Llama-3.1-8B-Instruct": 16.0,
    "meta-llama/Llama-3.1-70B-Instruct": 140.0,
    "mistralai/Mistral-7B-Instruct-v0.3": 14.0,
}

# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    query_id:         int
    category:         str
    query:            str
    model_response:   str
    query_snippet:    str
    model:            str
    # Core inference metrics
    ttft_ms:          float   # Time to First Token (prefill latency)
    tps:              float   # Output tokens per second
    tpot_ms:          float   # Time Per Output Token
    itl_ms:           float   # Inter-Token Latency (= TPOT, single stream)
    # Token counts
    prompt_tokens:    int
    output_tokens:    int
    # Timing breakdown (ms)
    total_duration_ms:   float
    generation_ms:       float
    # GPU VRAM (nvidia-smi)
    vram_used_gb:     float
    vram_free_gb:     float
    vram_total_gb:    float
    # Efficiency
    mbu_pct:          float
    model_size_gb:    float
    # GPU metadata
    gpu_name:         str = ""
    gpu_temp_c:       int = 0
    gpu_power_w:      float = 0.0
    gpu_utilization:  int = 0
    error:            Optional[str] = None

# ── NVIDIA GPU memory & stats ──────────────────────────────────────────────────

def get_gpu_stats() -> Dict[str, object]:
    """Read GPU stats via nvidia-smi."""
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.free,memory.total,"
                "temperature.gpu,power.draw,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        # Take first GPU line
        parts = [p.strip() for p in raw.splitlines()[0].split(",")]
        used_mb  = float(parts[1])
        free_mb  = float(parts[2])
        total_mb = float(parts[3])
        return {
            "gpu_name":        parts[0],
            "vram_used_gb":    round(used_mb  / 1024, 2),
            "vram_free_gb":    round(free_mb  / 1024, 2),
            "vram_total_gb":   round(total_mb / 1024, 2),
            "gpu_temp_c":      int(float(parts[4])),
            "gpu_power_w":     round(float(parts[5]), 1),
            "gpu_utilization": int(float(parts[6])),
        }
    except Exception as exc:
        return {
            "gpu_name": "unknown", "vram_used_gb": 0.0, "vram_free_gb": 0.0,
            "vram_total_gb": 0.0, "gpu_temp_c": 0, "gpu_power_w": 0.0,
            "gpu_utilization": 0, "error": str(exc),
        }

# ── SGLang API helpers ─────────────────────────────────────────────────────────

def check_sglang(base_url: str) -> bool:
    """Check if SGLang server is reachable."""
    try:
        r = httpx.get(f"{base_url}/v1/models", timeout=5)
        return r.status_code == 200
    except Exception:
        # Fallback: try health endpoint
        try:
            r = httpx.get(f"{base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False


def list_models(base_url: str) -> List[dict]:
    """List models from SGLang's OpenAI-compat endpoint."""
    r = httpx.get(f"{base_url}/v1/models", timeout=10)
    r.raise_for_status()
    return r.json().get("data", [])


def get_model_size_gb(model: str) -> float:
    """Estimate model size from known table."""
    if model in MODEL_SIZE_ESTIMATES_GB:
        return MODEL_SIZE_ESTIMATES_GB[model]
    # Rough heuristic: extract param count from name
    name_lower = model.lower()
    for token in name_lower.replace("-", " ").replace("_", " ").split():
        if token.endswith("b"):
            try:
                params = float(token[:-1])
                return round(params * 2.0, 1)  # FP16 ≈ 2 bytes/param
            except ValueError:
                continue
    return 0.0


def stream_chat(
    base_url: str, model: str, prompt: str, timeout: int = 300
) -> Tuple[str, List[float], float, int, int, float]:
    """
    Stream POST /v1/chat/completions (OpenAI SSE format).
    Returns (response_text, inter_token_latencies, wall_ttft_ms,
             prompt_tokens, output_tokens, total_time_s).
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
                    # Hard timeout: 5 minutes per query
                    elapsed = time.perf_counter() - t0
                    if elapsed > timeout:
                        raise TimeoutError(
                            f"Query exceeded {timeout}s limit ({elapsed:.0f}s)"
                        )

                    if not line or line == "data: [DONE]":
                        continue
                    if not line.startswith("data: "):
                        continue

                    data = json.loads(line[6:])

                    # Usage chunk (SGLang sends this at the end)
                    if data.get("usage"):
                        prompt_tokens = data["usage"].get("prompt_tokens", 0)
                        output_tokens = data["usage"].get("completion_tokens", 0)

                    # Content delta
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
        pass  # Return partial results

    total_time = time.perf_counter() - t0
    response_text = "".join(response_parts)
    return (response_text, token_times, ttft_ms or 0.0,
            prompt_tokens, output_tokens, total_time)

# ── Single-query benchmark ─────────────────────────────────────────────────────

def benchmark_query(
    query: dict,
    model: str,
    base_url: str,
    model_size_gb: float,
    bandwidth_gbps: float,
) -> BenchResult:
    """Run one query and return a fully-populated BenchResult."""
    error_result = lambda exc: BenchResult(
        query_id=query["id"], category=query["category"],
        query=query["query"], model_response="",
        query_snippet=query["query"][:70], model=model,
        ttft_ms=0, tps=0, tpot_ms=0, itl_ms=0,
        prompt_tokens=0, output_tokens=0,
        total_duration_ms=0, generation_ms=0,
        vram_used_gb=0, vram_free_gb=0, vram_total_gb=0,
        mbu_pct=0, model_size_gb=model_size_gb, error=str(exc),
    )

    try:
        (response_text, token_times, ttft_ms,
         prompt_tokens, output_tokens, total_time_s) = stream_chat(
            base_url, model, query["query"]
        )
    except Exception as exc:
        return error_result(exc)

    if output_tokens == 0 and not response_text:
        return error_result("Empty response — no tokens generated")

    gpu = get_gpu_stats()
    total_ms = total_time_s * 1000
    generation_ms = total_ms - ttft_ms if ttft_ms > 0 else total_ms
    tps = output_tokens / (generation_ms / 1000) if generation_ms > 0 else 0.0
    tpot_ms = generation_ms / output_tokens if output_tokens > 0 else 0.0
    itl_ms = (statistics.mean(token_times) if token_times
              else tpot_ms)

    # MBU: bandwidth_needed = model_size_gb × tps; MBU% = needed / peak × 100
    mbu_pct = 0.0
    if bandwidth_gbps > 0 and tps > 0 and model_size_gb > 0:
        mbu_pct = min((model_size_gb * tps) / bandwidth_gbps * 100, 100.0)

    snippet = query["query"][:70] + ("…" if len(query["query"]) > 70 else "")

    return BenchResult(
        query_id=query["id"],
        category=query["category"],
        query=query["query"],
        model_response=response_text,
        query_snippet=snippet,
        model=model,
        ttft_ms=round(ttft_ms,  2),
        tps=round(tps,          2),
        tpot_ms=round(tpot_ms,  2),
        itl_ms=round(itl_ms,    2),
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_duration_ms=round(total_ms,       2),
        generation_ms=round(generation_ms,      2),
        vram_used_gb=gpu["vram_used_gb"],
        vram_free_gb=gpu["vram_free_gb"],
        vram_total_gb=gpu["vram_total_gb"],
        mbu_pct=round(mbu_pct, 2),
        model_size_gb=round(model_size_gb, 3),
        gpu_name=gpu["gpu_name"],
        gpu_temp_c=gpu["gpu_temp_c"],
        gpu_power_w=gpu["gpu_power_w"],
        gpu_utilization=gpu["gpu_utilization"],
    )

# ── Statistics helpers ─────────────────────────────────────────────────────────

def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k  = (len(s) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def compute_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return dict(mean=0, median=0, p95=0, min=0, max=0, stdev=0)
    return {
        "mean":   round(statistics.mean(values),   2),
        "median": round(statistics.median(values),  2),
        "p95":    round(percentile(values, 95),     2),
        "min":    round(min(values),                2),
        "max":    round(max(values),                2),
        "stdev":  round(statistics.stdev(values) if len(values) > 1 else 0.0, 2),
    }

# ── Terminal reporting ─────────────────────────────────────────────────────────

W = 110   # total table width

def sep(char="─"):
    print(char * W)


def print_progress(i: int, total: int, r: BenchResult) -> None:
    tag = "✓" if not r.error else "✗"
    cat = r.category[:20]
    if r.error:
        print(f"  [{i:3d}/{total}] {tag} Q{r.query_id:3d} [{cat:<20}]  ERROR: {r.error[:55]}")
    else:
        print(
            f"  [{i:3d}/{total}] {tag} Q{r.query_id:3d} [{cat:<20}]"
            f"  TTFT={r.ttft_ms:7.0f}ms"
            f"  TPS={r.tps:6.1f}"
            f"  TPOT={r.tpot_ms:6.1f}ms"
            f"  MBU={r.mbu_pct:5.1f}%"
            f"  VRAM={r.vram_used_gb:.1f}GB"
            f"  {r.gpu_temp_c}°C"
        )


def print_summary(results: List[BenchResult], bandwidth_gbps: float) -> None:
    good = [r for r in results if not r.error]
    if not good:
        print("  No successful runs to summarise.")
        return

    sep("═")
    print(
        f"  BENCHMARK SUMMARY  │  {good[0].model}"
        f"  │  {len(good)} queries"
        f"  │  GPU peak: {bandwidth_gbps} GB/s"
    )
    sep("═")

    hdr_cols = ["Metric", "Mean", "Median", "P95", "Min", "Max", "Stdev"]
    col_w    = [28, 12, 12, 12, 12, 12, 12]
    print("  " + "".join(h.ljust(w) for h, w in zip(hdr_cols, col_w)))
    sep()

    def row(label: str, values: List[float], unit: str = "") -> None:
        s = compute_stats(values)
        cols = [
            label,
            f"{s['mean']}{unit}",
            f"{s['median']}{unit}",
            f"{s['p95']}{unit}",
            f"{s['min']}{unit}",
            f"{s['max']}{unit}",
            f"{s['stdev']}{unit}",
        ]
        print("  " + "".join(c.ljust(w) for c, w in zip(cols, col_w)))

    row("TTFT   (ms)",      [r.ttft_ms for r in good],          " ms")
    row("TPS    (tok/s)",   [r.tps     for r in good],          " t/s")
    row("TPOT   (ms/tok)",  [r.tpot_ms for r in good],          " ms")
    row("ITL    (ms/tok)",  [r.itl_ms  for r in good],          " ms")
    row("Prompt tokens",    [float(r.prompt_tokens) for r in good])
    row("Output tokens",    [float(r.output_tokens) for r in good])
    row("Total time (ms)",  [r.total_duration_ms for r in good], " ms")

    sep()
    vram_vals = [r.vram_used_gb for r in good]
    free_vals = [r.vram_free_gb for r in good]
    mbu_vals  = [r.mbu_pct     for r in good]
    temp_vals = [float(r.gpu_temp_c)  for r in good]
    power_vals = [r.gpu_power_w for r in good]

    print(f"\n  GPU  │  {good[0].gpu_name}  │  Total VRAM: {good[0].vram_total_gb:.0f} GB")
    print(f"    VRAM Used  avg {statistics.mean(vram_vals):.2f} GB   peak {max(vram_vals):.2f} GB")
    print(f"    Headroom   avg {statistics.mean(free_vals):.2f} GB   min  {min(free_vals):.2f} GB")
    print(f"    MBU        avg {statistics.mean(mbu_vals):.1f}%     peak {max(mbu_vals):.1f}%")
    print(f"    Temp       avg {statistics.mean(temp_vals):.0f}°C    peak {max(temp_vals):.0f}°C")
    print(f"    Power      avg {statistics.mean(power_vals):.0f}W     peak {max(power_vals):.0f}W")

    # Per-category breakdown
    cats: Dict[str, List[BenchResult]] = {}
    for r in good:
        cats.setdefault(r.category, []).append(r)

    if len(cats) > 1:
        print()
        sep()
        print(
            f"  {'Category':<30}"
            f"  {'N':>4}"
            f"  {'Avg TTFT':>10}"
            f"  {'Avg TPS':>9}"
            f"  {'Avg TPOT':>10}"
            f"  {'Avg MBU':>8}"
        )
        sep()
        for cat, rs in sorted(cats.items()):
            print(
                f"  {cat:<30}"
                f"  {len(rs):>4}"
                f"  {statistics.mean(r.ttft_ms for r in rs):>8.0f}ms"
                f"  {statistics.mean(r.tps     for r in rs):>7.1f}t/s"
                f"  {statistics.mean(r.tpot_ms for r in rs):>8.1f}ms"
                f"  {statistics.mean(r.mbu_pct for r in rs):>6.1f}%"
            )

    sep("═")

    failed = [r for r in results if r.error]
    if failed:
        print(f"\n  ⚠  {len(failed)} queries failed:")
        for r in failed:
            print(f"     Q{r.query_id}: {r.error}")
    print()

# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SGLang LLM inference benchmark — NVIDIA GPU (Vast.ai RTX 5080)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", "-m", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name served by SGLang (default: Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--url", default=SGLANG_DEFAULT_URL,
                        help=f"SGLang server base URL (default: {SGLANG_DEFAULT_URL})")
    parser.add_argument("--num-queries", "-n", type=int, default=10,
                        help="Queries to sample (default: 10)")
    parser.add_argument("--all-queries", action="store_true",
                        help="Run all 100 queries (overrides --num-queries)")
    parser.add_argument("--gpu", default="rtx-5080", choices=list(GPU_BANDWIDTH_GBPS),
                        help="NVIDIA GPU variant for MBU calculation (default: rtx-5080)")
    parser.add_argument("--model-size-gb", type=float, default=None,
                        help="Override model size in GB for MBU calculation")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for query sampling (default: 42)")
    parser.add_argument("--warmup", action="store_true",
                        help="Send one warmup request before benchmarking")
    parser.add_argument("--categories", nargs="+", metavar="CAT",
                        help="Restrict to these categories (case-sensitive)")
    parser.add_argument("--jsonl-output", action="store_true", default=True,
                        help="Save results to a JSONL file (one per line, default: enabled)")
    parser.add_argument("--no-jsonl-output", dest="jsonl_output", action="store_false",
                        help="Disable JSONL output")
    parser.add_argument("--list-models", action="store_true",
                        help="Print models served by SGLang and exit")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    # ── Preflight ──────────────────────────────────────────────────────────────
    if not check_sglang(base_url):
        print(f"❌  SGLang server is not reachable at {base_url}")
        print("    Start it with:  python -m sglang.launch_server "
              "--model Qwen/Qwen2.5-7B-Instruct --port 30000")
        sys.exit(1)

    if args.list_models:
        models = list_models(base_url)
        print(f"\n  {'MODEL':<50}  {'OWNED BY'}")
        print("  " + "─" * 62)
        for m in models:
            print(f"  {m.get('id', 'unknown'):<50}  {m.get('owned_by', '')}")
        print()
        sys.exit(0)

    # ── Load & filter queries ──────────────────────────────────────────────────
    with open(QUERY_FILE) as f:
        all_queries = json.load(f)["medical_queries"]

    if args.categories:
        all_queries = [q for q in all_queries if q["category"] in args.categories]
        if not all_queries:
            print(f"❌  No queries found for categories: {args.categories}")
            sys.exit(1)

    if args.all_queries:
        queries = all_queries
    else:
        rng     = random.Random(args.seed)
        queries = rng.sample(all_queries, min(args.num_queries, len(all_queries)))
        queries.sort(key=lambda q: q["id"])

    # ── Model metadata ─────────────────────────────────────────────────────────
    bandwidth_gbps = GPU_BANDWIDTH_GBPS[args.gpu]
    model_size_gb  = args.model_size_gb or get_model_size_gb(args.model)
    gpu0           = get_gpu_stats()

    print()
    print("  ╔════════════════════════════════════════════════════════════╗")
    print(f"  ║  SGLang Benchmark  │  {args.model:<36}  ║")
    print("  ╚════════════════════════════════════════════════════════════╝")
    print(f"  GPU:       {gpu0['gpu_name']}  →  {bandwidth_gbps} GB/s peak bandwidth")
    print(f"  Model:     {args.model}  (~{model_size_gb:.1f} GB)")
    print(f"  Server:    {base_url}")
    print(f"  Queries:   {len(queries)}")
    print(f"  VRAM:      {gpu0['vram_used_gb']:.1f} GB used / {gpu0['vram_total_gb']:.0f} GB total "
          f"({gpu0['vram_free_gb']:.1f} GB headroom)")
    print(f"  GPU:       {gpu0['gpu_temp_c']}°C  {gpu0['gpu_power_w']}W  "
          f"{gpu0['gpu_utilization']}% util")
    print()

    # ── Optional warmup ────────────────────────────────────────────────────────
    if args.warmup:
        print("  ⏳  Warming up model …")
        try:
            stream_chat(base_url, args.model, "Hello", timeout=60)
            print("  ✓   Warmup complete\n")
        except Exception as exc:
            print(f"  ⚠   Warmup failed: {exc}\n")

    # ── Benchmark loop ─────────────────────────────────────────────────────────
    sep()
    print(
        f"  {'#':>6}  S  {'QID':>4}  {'Category':<22}"
        f"  {'TTFT':>8}  {'TPS':>7}  {'TPOT':>8}  {'MBU':>6}  {'VRAM':>6}  {'Temp':>5}"
    )
    sep()

    results: List[BenchResult] = []
    for i, query in enumerate(queries, 1):
        result = benchmark_query(query, args.model, base_url,
                                 model_size_gb, bandwidth_gbps)
        results.append(result)
        print_progress(i, len(queries), result)

    sep()
    print()

    # ── Summary ────────────────────────────────────────────────────────────────
    print_summary(results, bandwidth_gbps)

    # ── JSONL export (default on) ──────────────────────────────────────────────
    if args.jsonl_output:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        safe_model = args.model.replace("/", "_").replace(":", "_")
        jsonl_path = (Path(__file__).parent
                      / f"benchmark_results_sglang_{safe_model}_{timestamp}.jsonl")

        with open(jsonl_path, "w") as f:
            for result in results:
                record = asdict(result)
                record["_meta"] = {
                    "engine":         "sglang",
                    "model":          args.model,
                    "gpu":            args.gpu,
                    "gpu_name":       gpu0["gpu_name"],
                    "bandwidth_gbps": bandwidth_gbps,
                    "model_size_gb":  model_size_gb,
                    "server_url":     base_url,
                    "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                f.write(json.dumps(record) + "\n")

        print(f"  💾  JSONL results saved → {jsonl_path.resolve()}")
        print()


if __name__ == "__main__":
    main()