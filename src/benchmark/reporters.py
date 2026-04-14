"""Per-category breakdown and tabular summary printing."""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING, Dict, List

from src.core.metrics import compute_stats

if TYPE_CHECKING:
    from src.core.models import BenchResult

W = 110  # table width


def _sep(char: str = "─") -> None:
    print(char * W)


def print_progress(i: int, total: int, r: "BenchResult") -> None:
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


def print_summary(results: List["BenchResult"], bandwidth_gbps: float) -> None:
    good = [r for r in results if not r.error]
    if not good:
        print("  No successful runs to summarise.")
        return

    _sep("═")
    print(
        f"  BENCHMARK SUMMARY  │  {good[0].model}"
        f"  │  {len(good)} queries"
        f"  │  GPU peak: {bandwidth_gbps} GB/s"
    )
    _sep("═")

    hdr_cols = ["Metric", "Mean", "Median", "P95", "Min", "Max", "Stdev"]
    col_w    = [28, 12, 12, 12, 12, 12, 12]
    print("  " + "".join(h.ljust(w) for h, w in zip(hdr_cols, col_w)))
    _sep()

    def _row(label: str, values: List[float], unit: str = "") -> None:
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

    _row("TTFT   (ms)",     [r.ttft_ms for r in good],          " ms")
    _row("TPS    (tok/s)",  [r.tps     for r in good],          " t/s")
    _row("TPOT   (ms/tok)", [r.tpot_ms for r in good],          " ms")
    _row("ITL    (ms/tok)", [r.itl_ms  for r in good],          " ms")
    _row("Prompt tokens",   [float(r.prompt_tokens) for r in good])
    _row("Output tokens",   [float(r.output_tokens) for r in good])
    _row("Total time (ms)", [r.total_duration_ms for r in good], " ms")

    _sep()
    vram_vals  = [r.vram_used_gb  for r in good]
    free_vals  = [r.vram_free_gb  for r in good]
    mbu_vals   = [r.mbu_pct       for r in good]
    temp_vals  = [float(r.gpu_temp_c) for r in good]
    power_vals = [r.gpu_power_w   for r in good]

    print(f"\n  GPU  │  {good[0].gpu_name}  │  Total VRAM: {good[0].vram_total_gb:.0f} GB")
    print(f"    VRAM Used  avg {statistics.mean(vram_vals):.2f} GB   peak {max(vram_vals):.2f} GB")
    print(f"    Headroom   avg {statistics.mean(free_vals):.2f} GB   min  {min(free_vals):.2f} GB")
    print(f"    MBU        avg {statistics.mean(mbu_vals):.1f}%     peak {max(mbu_vals):.1f}%")
    print(f"    Temp       avg {statistics.mean(temp_vals):.0f}°C    peak {max(temp_vals):.0f}°C")
    print(f"    Power      avg {statistics.mean(power_vals):.0f}W     peak {max(power_vals):.0f}W")

    # Per-category breakdown
    cats: Dict[str, List["BenchResult"]] = {}
    for r in good:
        cats.setdefault(r.category, []).append(r)

    if len(cats) > 1:
        print()
        _sep()
        print(
            f"  {'Category':<30}"
            f"  {'N':>4}"
            f"  {'Avg TTFT':>10}"
            f"  {'Avg TPS':>9}"
            f"  {'Avg TPOT':>10}"
            f"  {'Avg MBU':>8}"
        )
        _sep()
        for cat, rs in sorted(cats.items()):
            print(
                f"  {cat:<30}"
                f"  {len(rs):>4}"
                f"  {statistics.mean(r.ttft_ms for r in rs):>8.0f}ms"
                f"  {statistics.mean(r.tps     for r in rs):>7.1f}t/s"
                f"  {statistics.mean(r.tpot_ms for r in rs):>8.1f}ms"
                f"  {statistics.mean(r.mbu_pct for r in rs):>6.1f}%"
            )

    _sep("═")

    failed = [r for r in results if r.error]
    if failed:
        print(f"\n  ⚠  {len(failed)} queries failed:")
        for r in failed:
            print(f"     Q{r.query_id}: {r.error}")
    print()
