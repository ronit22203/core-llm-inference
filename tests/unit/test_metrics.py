"""Unit tests for src/core/metrics.py"""

import pytest
from src.core.metrics import (
    compute_mbu,
    compute_stats,
    get_model_size_gb,
    percentile,
    GPU_BANDWIDTH_GBPS,
)


class TestComputeMbu:
    def test_known_values(self):
        # 14 GB model × 62.5 tok/s / 960 GB/s = 91.15%
        mbu = compute_mbu(14.0, 62.5, 960.0)
        assert abs(mbu - 91.15) < 0.1

    def test_capped_at_100(self):
        assert compute_mbu(100.0, 1000.0, 100.0) == 100.0

    def test_zero_bandwidth(self):
        assert compute_mbu(14.0, 62.5, 0.0) == 0.0

    def test_zero_tps(self):
        assert compute_mbu(14.0, 0.0, 960.0) == 0.0


class TestGetModelSizeGb:
    def test_known_model(self):
        assert get_model_size_gb("Qwen/Qwen2.5-7B-Instruct") == 14.0

    def test_param_heuristic(self):
        # "13b" in name → 13 × 2 = 26 GB
        assert get_model_size_gb("SomeOrg/MyModel-13B-Chat") == 26.0

    def test_unknown_returns_zero(self):
        assert get_model_size_gb("unknown/model-no-size") == 0.0


class TestPercentile:
    def test_empty(self):
        assert percentile([], 50) == 0.0

    def test_single(self):
        assert percentile([5.0], 50) == 5.0

    def test_p50(self):
        assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0

    def test_p95(self):
        data = list(range(1, 101))
        assert percentile([float(x) for x in data], 95) == pytest.approx(95.05, abs=0.1)


class TestComputeStats:
    def test_empty(self):
        s = compute_stats([])
        assert s["mean"] == 0

    def test_basic(self):
        s = compute_stats([10.0, 20.0, 30.0])
        assert s["mean"] == 20.0
        assert s["min"] == 10.0
        assert s["max"] == 30.0

    def test_gpu_bandwidth_table_populated(self):
        assert "rtx-5080" in GPU_BANDWIDTH_GBPS
        assert GPU_BANDWIDTH_GBPS["rtx-5080"] == 960.0
