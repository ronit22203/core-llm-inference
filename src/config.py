"""Load and merge YAML config with environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    raise ImportError("pyyaml is required: pip install pyyaml")


_DEFAULT_CONFIG = Path(__file__).parents[2] / "config" / "default.yaml"


def load_config(config_file: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load YAML config from *config_file* (defaults to config/default.yaml)
    and apply environment variable overrides.
    """
    path = Path(config_file) if config_file else _DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    # Env overrides (prefix-free for readability)
    _override(cfg, "engine",                   "INFERENCE_ENGINE")
    _override(cfg, "model",                    "INFERENCE_MODEL")
    _override(cfg, "model_size_gb",            "INFERENCE_MODEL_SIZE_GB", float)
    _override(cfg, ["server", "host"],         "SERVER_HOST")
    _override(cfg, ["server", "port"],         "SERVER_PORT", int)
    _override(cfg, ["server", "timeout_seconds"], "SERVER_TIMEOUT", int)
    _override(cfg, ["gpu", "type"],            "GPU_TYPE")
    _override(cfg, ["gpu", "peak_bandwidth_gbps"], "GPU_PEAK_BANDWIDTH_GBPS", float)
    _override(cfg, ["benchmark", "num_queries"],   "BENCHMARK_NUM_QUERIES", int)
    _override(cfg, ["benchmark", "concurrency"],   "BENCHMARK_CONCURRENCY", int)
    _override(cfg, ["benchmark", "warmup"],        "BENCHMARK_WARMUP", _bool)
    _override(cfg, ["benchmark", "output_dir"],    "BENCHMARK_OUTPUT_DIR")
    _override(cfg, ["benchmark", "seed"],          "BENCHMARK_SEED", int)
    _override(cfg, ["monitoring", "prometheus_pushgateway"], "PROMETHEUS_PUSHGATEWAY")
    _override(cfg, ["monitoring", "enable_gpu_metrics"],     "ENABLE_GPU_METRICS", _bool)
    _override(cfg, ["monitoring", "log_jsonl"],              "LOG_JSONL", _bool)

    # BENCHMARK_CATEGORIES is space/comma-separated
    cats_env = os.getenv("BENCHMARK_CATEGORIES", "").strip()
    if cats_env:
        cats = [c.strip() for c in cats_env.replace(",", " ").split() if c.strip()]
        _set_nested(cfg, ["benchmark", "categories"], cats)

    return cfg


def _bool(value: str) -> bool:
    return value.lower() in ("1", "true", "yes")


def _set_nested(d: dict, keys: list, value: Any) -> None:
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _override(cfg: dict, key, env_var: str, cast=str) -> None:
    val = os.getenv(env_var)
    if val is None:
        return
    if isinstance(key, list):
        _set_nested(cfg, key, cast(val))
    else:
        cfg[key] = cast(val)
