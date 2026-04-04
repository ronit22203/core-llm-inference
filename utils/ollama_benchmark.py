#!/usr/bin/env python3
"""
ollama_benchmark.py
===================
LLM inference benchmark for Ollama on Apple Silicon (M4 16 GB, MLX backend).

Metrics
-------
  TTFT   Time to First Token          — user-perceived responsiveness  (ms)
  TPS    Output Tokens Per Second     — generation throughput          (tok/s)
  TPOT   Time Per Output Token        — per-step generation latency    (ms/tok)
  ITL    Inter-Token Latency          — same as TPOT for single stream (ms/tok)
  VRAM   Unified memory usage/headroom — Apple unified memory          (GB)
  MBU    Model Bandwidth Utilization  — hardware efficiency            (%)

MBU formula
-----------
Each autoregressive decode step reads all model weights from unified memory.
  bandwidth_needed (GB/s) = model_size_gb × tps
  MBU (%) = bandwidth_needed / peak_bandwidth_gbps × 100

Usage
-----
  # benchmark with 10 random queries (default)
  python utils/ollama_benchmark.py --model qwen3:8b

  # run all 100 queries, specify chip for MBU calculation
  python utils/ollama_benchmark.py --model qwen3:8b --all-queries --chip m4

  # filter by category, warmup, save JSON
  python utils/ollama_benchmark.py --model qwen3:8b \\
      --categories Cardiology Oncology --warmup --output results.json

  # list available models
  python utils/ollama_benchmark.py --list-models

  # stream live metrics to Prometheus Pushgateway (start monitoring/ stack first)
  python utils/ollama_benchmark.py --model qwen3:8b --prometheus
  python utils/ollama_benchmark.py --model qwen3:8b --prometheus --pushgateway-url http://localhost:9091
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

try:
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False

# ── Constants ──────────────────────────────────────────────────────────────────

OLLAMA_BASE = "http://localhost:11434"
QUERY_FILE  = Path(__file__).parent / "query.json"

# Peak memory bandwidth (GB/s) per Apple Silicon chip variant
CHIP_BANDWIDTH_GBPS: Dict[str, float] = {
    "m1":       68.0,
    "m1-pro":  200.0,
    "m1-max":  400.0,
    "m1-ultra":800.0,
    "m2":      100.0,
    "m2-pro":  200.0,
    "m2-max":  400.0,
    "m2-ultra":800.0,
    "m3":      100.0,
    "m3-pro":  150.0,
    "m3-max":  300.0,
    "m3-ultra":600.0,
    "m4":      120.0,
    "m4-pro":  273.0,
    "m4-max":  546.0,
    "m4-ultra":1092.0,
}

# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    query_id:         int
    category:         str
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
    load_duration_ms:    float
    prompt_eval_ms:      float
    generate_ms:         float
    # Apple unified memory
    mem_used_gb:      float
    mem_free_gb:      float
    mem_total_gb:     float
    # Efficiency
    mbu_pct:          float
    model_size_gb:    float
    error:            Optional[str] = None

# ── Apple Silicon memory ───────────────────────────────────────────────────────

def get_memory_stats() -> Dict[str, float]:
    """Read unified memory via vm_stat + sysctl."""
    try:
        total_bytes = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        )
        vm = subprocess.check_output(["vm_stat"], text=True)

        page_size = 4096
        pages: Dict[str, int] = {}
        for line in vm.splitlines():
            if "page size of" in line:
                parts = line.split()
                page_size = int(parts[parts.index("of") + 1])
            elif ":" in line:
                k, v = line.split(":", 1)
                try:
                    pages[k.strip()] = int(v.strip().rstrip("."))
                except ValueError:
                    pass

        used_pages = (
            pages.get("Pages wired down", 0)
            + pages.get("Pages active", 0)
            + pages.get("Pages speculative", 0)
            + pages.get("Pages occupied by compressor", 0)
        )
        used_bytes = used_pages * page_size
        free_bytes = max(total_bytes - used_bytes, 0)

        return {
            "used_gb":   used_bytes  / 1e9,
            "free_gb":   free_bytes  / 1e9,
            "total_gb":  total_bytes / 1e9,
            "usage_pct": used_bytes  / total_bytes * 100,
        }
    except Exception as exc:
        return {"used_gb": 0.0, "free_gb": 0.0, "total_gb": 16.0,
                "usage_pct": 0.0, "error": str(exc)}

# ── Ollama API helpers ─────────────────────────────────────────────────────────

def check_ollama() -> bool:
    try:
        r = httpx.get(f"{OLLAMA_BASE}/", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def list_models() -> List[dict]:
    r = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
    r.raise_for_status()
    return r.json().get("models", [])


def get_model_size_gb(model: str) -> float:
    try:
        target_name = model.split(":")[0]
        for m in list_models():
            if m["name"] == model or m["name"].split(":")[0] == target_name:
                return m["size"] / 1e9
    except Exception:
        pass
    return 0.0


def stream_generate(
    model: str, prompt: str, timeout: int = 180
) -> Tuple[List[dict], float]:
    """
    Stream POST /api/generate.
    Returns (list_of_chunks, wall_clock_ttft_ms).
    TTFT is measured as wall-clock time from request-sent to first non-empty token.
    """
    chunks: List[dict] = []
    ttft_ms: Optional[float] = None
    t0 = time.perf_counter()

    # Per-operation timeouts + total timeout prevent hangs.
    # read=15s: No token for 15s → timeout (catches stalls between tokens)
    # pool timeout for connection acquisition
    http_timeout = httpx.Timeout(connect=10.0, read=15.0, write=10.0, pool=5.0)

    try:
        with httpx.Client(timeout=http_timeout) as client:
            with client.stream(
                "POST",
                f"{OLLAMA_BASE}/api/generate",
                json={
                    "model":   model,
                    "prompt":  prompt,
                    "stream":  True,
                    "options": {"temperature": 0.0, "seed": 42},
                },
            ) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines():
                    # Enforce total request timeout: fail if any single request exceeds 5 minutes
                    elapsed = (time.perf_counter() - t0) * 1000
                    if elapsed > 300_000:  # 300 seconds = 5 minutes max per query
                        raise TimeoutError(f"Query exceeded 5 minute limit ({elapsed/1000:.0f}s)")
                    
                    raw = raw.strip()
                    if not raw:
                        continue
                    chunk = json.loads(raw)
                    chunks.append(chunk)
                    if ttft_ms is None and chunk.get("response"):
                        ttft_ms = (time.perf_counter() - t0) * 1000
    except (httpx.TimeoutException, TimeoutError) as exc:
        return chunks, ttft_ms or 0.0  # Return partial results; benchmark_query will handle error

    return chunks, ttft_ms or 0.0

# ── Prometheus helpers ─────────────────────────────────────────────────────────

def push_metrics(result: BenchResult, pushgateway_url: str) -> None:
    """Push per-query metrics to a Prometheus Pushgateway."""
    if not _PROMETHEUS_AVAILABLE:
        return
    registry = CollectorRegistry()
    label_names = ["model", "category", "query_id"]
    label_values = [result.model, result.category, str(result.query_id)]

    metrics = [
        ("ollama_benchmark_ttft_ms",       "Time to First Token (ms)",           result.ttft_ms),
        ("ollama_benchmark_tps",            "Output tokens per second (tok/s)",   result.tps),
        ("ollama_benchmark_tpot_ms",        "Time Per Output Token (ms/tok)",     result.tpot_ms),
        ("ollama_benchmark_itl_ms",         "Inter-Token Latency (ms/tok)",       result.itl_ms),
        ("ollama_benchmark_mbu_pct",        "Model Bandwidth Utilization (%)",    result.mbu_pct),
        ("ollama_benchmark_mem_used_gb",    "Unified memory used (GB)",           result.mem_used_gb),
        ("ollama_benchmark_mem_free_gb",    "Unified memory free (GB)",           result.mem_free_gb),
        ("ollama_benchmark_output_tokens",  "Output token count",                 float(result.output_tokens)),
        ("ollama_benchmark_prompt_tokens",  "Prompt token count",                 float(result.prompt_tokens)),
        ("ollama_benchmark_total_ms",       "Total request duration (ms)",        result.total_duration_ms),
    ]
    for name, helptext, value in metrics:
        g = Gauge(name, helptext, label_names, registry=registry)
        g.labels(*label_values).set(value)

    try:
        push_to_gateway(
            pushgateway_url,
            job="ollama_benchmark",
            grouping_key={"query_id": str(result.query_id)},
            registry=registry,
        )
    except Exception as exc:
        print(f"  ⚠   Prometheus push failed: {exc}")


# ── Single-query benchmark ─────────────────────────────────────────────────────

def benchmark_query(
    query: dict,
    model: str,
    model_size_gb: float,
    bandwidth_gbps: float,
) -> BenchResult:
    """Run one query and return a fully-populated BenchResult."""
    try:
        chunks, _wall_ttft_ms = stream_generate(model, query["query"])
    except Exception as exc:
        return BenchResult(
            query_id=query["id"], category=query["category"],
            query_snippet=query["query"][:70], model=model,
            ttft_ms=0, tps=0, tpot_ms=0, itl_ms=0,
            prompt_tokens=0, output_tokens=0,
            total_duration_ms=0, load_duration_ms=0,
            prompt_eval_ms=0, generate_ms=0,
            mem_used_gb=0, mem_free_gb=0, mem_total_gb=0,
            mbu_pct=0, model_size_gb=model_size_gb, error=str(exc),
        )

    mem = get_memory_stats()
    final = next((c for c in reversed(chunks) if c.get("done")), {})

    # Ollama reports all durations in nanoseconds; convert to ms
    NS_TO_MS          = 1e6
    prompt_eval_ms    = final.get("prompt_eval_duration", 0) / NS_TO_MS
    generate_ms       = final.get("eval_duration",        0) / NS_TO_MS
    total_ms          = final.get("total_duration",        0) / NS_TO_MS
    load_ms           = final.get("load_duration",         0) / NS_TO_MS
    prompt_tokens     = final.get("prompt_eval_count", 0)
    output_tokens     = final.get("eval_count",        0)

    # TTFT = Ollama's internal prompt-eval duration (prefill, most accurate)
    ttft_ms  = prompt_eval_ms if prompt_eval_ms > 0 else _wall_ttft_ms
    tps      = output_tokens / (generate_ms / 1000) if generate_ms > 0 else 0.0
    tpot_ms  = generate_ms / output_tokens           if output_tokens > 0 else 0.0

    # MBU: bandwidth_needed = model_size_gb × tps; MBU% = needed / peak × 100
    mbu_pct = 0.0
    if bandwidth_gbps > 0 and tps > 0 and model_size_gb > 0:
        mbu_pct = min((model_size_gb * tps) / bandwidth_gbps * 100, 100.0)

    snippet = query["query"][:70] + ("…" if len(query["query"]) > 70 else "")

    return BenchResult(
        query_id=query["id"],
        category=query["category"],
        query_snippet=snippet,
        model=model,
        ttft_ms=round(ttft_ms,  2),
        tps=round(tps,          2),
        tpot_ms=round(tpot_ms,  2),
        itl_ms=round(tpot_ms,   2),   # ITL ≡ TPOT for a single-stream request
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_duration_ms=round(total_ms,       2),
        load_duration_ms=round(load_ms,         2),
        prompt_eval_ms=round(prompt_eval_ms,    2),
        generate_ms=round(generate_ms,          2),
        mem_used_gb=round(mem["used_gb"],       2),
        mem_free_gb=round(mem["free_gb"],       2),
        mem_total_gb=round(mem["total_gb"],     2),
        mbu_pct=round(mbu_pct,                  2),
        model_size_gb=round(model_size_gb,      3),
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

W = 104   # total table width

def sep(char="─"):
    print(char * W)


def print_progress(i: int, total: int, r: BenchResult) -> None:
    tag  = "✓" if not r.error else "✗"
    cat  = r.category[:20]
    if r.error:
        print(f"  [{i:3d}/{total}] {tag} Q{r.query_id:3d} [{cat:<20}]  ERROR: {r.error[:55]}")
    else:
        print(
            f"  [{i:3d}/{total}] {tag} Q{r.query_id:3d} [{cat:<20}]"
            f"  TTFT={r.ttft_ms:7.0f}ms"
            f"  TPS={r.tps:6.1f}"
            f"  TPOT={r.tpot_ms:6.1f}ms"
            f"  MBU={r.mbu_pct:5.1f}%"
            f"  Mem↑={r.mem_used_gb:.1f}GB"
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
        f"  │  Chip peak: {bandwidth_gbps} GB/s"
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
    row("Load time  (ms)",  [r.load_duration_ms  for r in good], " ms")

    sep()
    mem_vals  = [r.mem_used_gb for r in good]
    free_vals = [r.mem_free_gb for r in good]
    mbu_vals  = [r.mbu_pct     for r in good]
    print(f"\n  MEMORY  (Apple Unified)  │  Total: {good[0].mem_total_gb:.0f} GB")
    print(f"    Used     avg {statistics.mean(mem_vals):.2f} GB   peak {max(mem_vals):.2f} GB")
    print(f"    Headroom avg {statistics.mean(free_vals):.2f} GB   min  {min(free_vals):.2f} GB")
    print(f"    MBU      avg {statistics.mean(mbu_vals):.1f}%     peak {max(mbu_vals):.1f}%")

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
                f"  {statistics.mean(r.ttft_ms  for r in rs):>8.0f}ms"
                f"  {statistics.mean(r.tps       for r in rs):>7.1f}t/s"
                f"  {statistics.mean(r.tpot_ms  for r in rs):>8.1f}ms"
                f"  {statistics.mean(r.mbu_pct  for r in rs):>6.1f}%"
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
        description="Ollama LLM inference benchmark — Apple Silicon (M4 + MLX)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model",  "-m", help="Ollama model tag  (e.g. qwen3:8b)")
    parser.add_argument("--num-queries", "-n", type=int, default=10,
                        help="Queries to sample (default: 10)")
    parser.add_argument("--all-queries", action="store_true",
                        help="Run all 100 queries (overrides --num-queries)")
    parser.add_argument("--chip", default="m4", choices=list(CHIP_BANDWIDTH_GBPS),
                        help="Apple Silicon variant for MBU calculation (default: m4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for query sampling (default: 42)")
    parser.add_argument("--warmup", action="store_true",
                        help="Send one warmup request before benchmarking")
    parser.add_argument("--categories", nargs="+", metavar="CAT",
                        help="Restrict to these categories (case-sensitive)")
    parser.add_argument("--output", "-o", metavar="FILE",
                        help="Save full results to a JSON file")
    parser.add_argument("--jsonl-output", action="store_true",
                        help="Save results to a JSONL file in the script directory (one result per line)")
    parser.add_argument("--prometheus", action="store_true",
                        help="Push per-query metrics to a Prometheus Pushgateway")
    parser.add_argument("--pushgateway-url", default="http://localhost:9091",
                        metavar="URL",
                        help="Pushgateway address (default: http://localhost:9091)")
    parser.add_argument("--list-models", action="store_true",
                        help="Print available Ollama models and exit")
    args = parser.parse_args()

    # ── Preflight ──────────────────────────────────────────────────────────────
    if args.prometheus and not _PROMETHEUS_AVAILABLE:
        print("❌  prometheus_client is required for --prometheus.  Install with:  pip install prometheus_client")
        sys.exit(1)

    if not check_ollama():
        print("❌  Ollama is not running.  Start it with:  ollama serve")
        sys.exit(1)

    if args.list_models:
        models = list_models()
        print(f"\n  {'MODEL':<40} {'SIZE':>8}  MODIFIED")
        print("  " + "─" * 62)
        for m in models:
            print(f"  {m['name']:<40} {m['size']/1e9:>7.2f}G  "
                  f"{m.get('modified_at','')[:10]}")
        print()
        sys.exit(0)

    if not args.model:
        parser.error("--model is required (use --list-models to see available models)")

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
    bandwidth_gbps = CHIP_BANDWIDTH_GBPS[args.chip]
    model_size_gb  = get_model_size_gb(args.model)
    mem0           = get_memory_stats()

    print()
    print("  ╔════════════════════════════════════════════════════════╗")
    print(f"  ║  Ollama Benchmark  │  {args.model:<32}  ║")
    print("  ╚════════════════════════════════════════════════════════╝")
    print(f"  Chip:      Apple {args.chip.upper()}  →  {bandwidth_gbps} GB/s peak bandwidth")
    print(f"  Model:     {args.model}  ({model_size_gb:.2f} GB on disk)")
    print(f"  Queries:   {len(queries)}")
    print(f"  Memory:    {mem0['used_gb']:.1f} GB used / {mem0['total_gb']:.0f} GB total "
          f"({mem0['free_gb']:.1f} GB headroom before benchmark)")
    print()

    # ── Optional warmup ────────────────────────────────────────────────────────
    if args.warmup:
        print("  ⏳  Warming up model …")
        try:
            stream_generate(args.model, "Hello", timeout=60)
            print("  ✓   Warmup complete\n")
        except Exception as exc:
            print(f"  ⚠   Warmup failed: {exc}\n")

    # ── Benchmark loop ─────────────────────────────────────────────────────────
    sep()
    print(
        f"  {'#':>6}  S  {'QID':>4}  {'Category':<22}"
        f"  {'TTFT':>8}  {'TPS':>7}  {'TPOT':>8}  {'MBU':>6}  {'Mem':>6}"
    )
    sep()

    if args.prometheus:
        print(f"  📊  Prometheus mode — pushing metrics to {args.pushgateway_url}")
        print()

    results: List[BenchResult] = []
    for i, query in enumerate(queries, 1):
        result = benchmark_query(query, args.model, model_size_gb, bandwidth_gbps)
        results.append(result)
        print_progress(i, len(queries), result)
        if args.prometheus and not result.error:
            push_metrics(result, args.pushgateway_url)

    sep()
    print()

    # ── Summary ────────────────────────────────────────────────────────────────
    print_summary(results, bandwidth_gbps)

    # ── Optional JSON export ───────────────────────────────────────────────────
    if args.output:
        good = [r for r in results if not r.error]
        payload = {
            "model":          args.model,
            "chip":           args.chip,
            "bandwidth_gbps": bandwidth_gbps,
            "model_size_gb":  model_size_gb,
            "num_queries":    len(results),
            "results":        [asdict(r) for r in results],
            "summary": {
                "ttft_ms":  compute_stats([r.ttft_ms  for r in good]),
                "tps":      compute_stats([r.tps      for r in good]),
                "tpot_ms":  compute_stats([r.tpot_ms  for r in good]),
                "mbu_pct":  compute_stats([r.mbu_pct  for r in good]),
                "mem_used_gb": compute_stats([r.mem_used_gb for r in good]),
            },
        }
        out = Path(args.output)
        out.write_text(json.dumps(payload, indent=2))
        print(f"  💾  Results saved → {out.resolve()}")
        print()

    # ── Optional JSONL export ──────────────────────────────────────────────────
    if args.jsonl_output:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        jsonl_path = Path(__file__).parent / f"benchmark_results_{args.model.replace(':', '_')}_{timestamp}.jsonl"
        
        with open(jsonl_path, "w") as f:
            for result in results:
                result_dict = asdict(result)
                result_dict["model"] = args.model
                result_dict["chip"] = args.chip
                result_dict["bandwidth_gbps"] = bandwidth_gbps
                result_dict["model_size_gb"] = model_size_gb
                f.write(json.dumps(result_dict) + "\n")
        
        print(f"  💾  JSONL results saved → {jsonl_path.resolve()}")
        print()


if __name__ == "__main__":
    main()
