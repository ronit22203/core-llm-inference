# core-llm-inference

Modular LLM inference benchmarking and server management for **SGLang** and **vLLM**.

Measure real-world throughput, latency, and hardware efficiency of your inference stack with a single CLI command.

---

## Features

- **`benchmark`** — Run inference workloads against a live server and report TTFT, TPS, TPOT, ITL, and MBU
- **`serve`** — Launch an SGLang inference server (foreground or detached)
- **`status`** — Inspect server health, loaded models, GPU stats, and KV-cache utilisation
- **JSONL export** — Timestamped result files written to `results/` after every run
- **Prometheus / Grafana** — Push metrics to a Pushgateway; pre-built monitoring stack via Docker Compose
- **GPU coverage** — NVIDIA RTX 30/40/50-series, A100, H100, L40S; Apple Silicon M1–M4
- **Configurable** — Single YAML file with full environment-variable overrides

---

## Metrics

| Metric | Description |
|---|---|
| **TTFT** | Time to first token (ms) |
| **TPS** | Tokens per second (output throughput) |
| **TPOT** | Time per output token (ms) |
| **ITL** | Inter-token latency (ms/tok) |
| **MBU** | Model Bandwidth Utilisation (%) — how much of the GPU's peak memory bandwidth the model consumes |

---

## Installation

Requires **Python ≥ 3.12**. Install with [uv](https://github.com/astral-sh/uv) (recommended) or pip:

```bash
# with uv
uv pip install -e .

# with pip
pip install -e .
```

Copy and customise the environment file:

```bash
cp .env.example .env
```

---

## Quick Start

### 1. Start an inference server

```bash
# foreground (Ctrl-C to stop)
core-llm-inference serve --model Qwen/Qwen2.5-7B-Instruct

# detached background process
core-llm-inference serve --model Qwen/Qwen2.5-7B-Instruct --detach
```

### 2. Run a benchmark

```bash
# 10 queries (default), all categories
core-llm-inference benchmark

# 50 queries, specific category, warmup enabled
core-llm-inference benchmark --queries 50 --category diagnosis --warmup

# run all queries and push metrics to Prometheus
core-llm-inference benchmark --all-queries --prometheus
```

### 3. Check server status

```bash
core-llm-inference status
core-llm-inference status --cache      # include KV-cache stats
core-llm-inference status --no-gpu     # skip GPU readout
```

---

## CLI Reference

### `serve`

```
core-llm-inference serve [OPTIONS]

Options:
  -c, --config FILE     YAML config file
  -m, --model TEXT      Model name (HuggingFace path)
  --host TEXT           Bind host  [default: 0.0.0.0]
  --port INTEGER        Bind port  [default: 30000]
  --detach              Start server and exit immediately
```

### `benchmark`

```
core-llm-inference benchmark [OPTIONS]

Options:
  -c, --config FILE          YAML config file
  -m, --model TEXT           Model name
  --url TEXT                 Server base URL
  -n, --queries INTEGER      Number of queries to sample
  --all-queries              Run every query in the dataset
  --gpu TEXT                 GPU type key for MBU calculation
  --model-size-gb FLOAT      Model size override (GB, FP16)
  -C, --category TEXT        Filter by category (repeatable)
  --warmup / --no-warmup     Run a warmup request first
  --seed INTEGER             Random seed for query sampling
  -o, --output-dir DIR       Directory for JSONL results
  --no-jsonl                 Disable JSONL export
  --prometheus               Push metrics to Pushgateway
  --pushgateway TEXT         Pushgateway URL
```

### `status`

```
core-llm-inference status [OPTIONS]

Options:
  -c, --config FILE    YAML config file
  --url TEXT           Server URL  [default: http://localhost:30000]
  --cache              Show KV-cache statistics
  --gpu / --no-gpu     Show GPU stats  [default: --gpu]
```

---

## Configuration

Default values live in `config/default.yaml`. Every key can be overridden via environment variables (see `.env.example`).

```yaml
engine: "sglang"                    # sglang | vllm
model: "Qwen/Qwen2.5-7B-Instruct"
model_size_gb: 14.0                 # null = auto-detect from model name

server:
  host: "0.0.0.0"
  port: 30000
  timeout_seconds: 300

gpu:
  type: "rtx-5080"                  # key into bandwidth table
  peak_bandwidth_gbps: 960.0

benchmark:
  num_queries: 10
  concurrency: 1
  warmup: false
  categories: []                    # empty = all categories
  output_dir: "results"
  seed: 42

monitoring:
  prometheus_pushgateway: "http://localhost:9091"
  enable_gpu_metrics: true
  log_jsonl: true
```

**Supported GPU keys** (for MBU calculation):

| NVIDIA | Apple Silicon |
|---|---|
| `rtx-5090`, `rtx-5080`, `rtx-5070ti`, `rtx-5070` | `m4-max`, `m4-pro`, `m4` |
| `rtx-4090`, `rtx-4080`, `rtx-4070ti`, `rtx-4070` | `m3-max`, `m3-pro`, `m3` |
| `rtx-3090`, `rtx-3080` | `m2-ultra`, `m2-max`, `m2-pro`, `m2` |
| `h100`, `a100-80`, `a100-40`, `l40s` | `m1-ultra`, `m1-max`, `m1-pro`, `m1` |

---

## Project Structure

```
core-llm-inference/
├── config/
│   └── default.yaml          # Default configuration
├── src/
│   ├── cli/                  # Typer CLI commands
│   │   ├── main.py
│   │   ├── benchmark_cmd.py
│   │   ├── serve_cmd.py
│   │   └── status_cmd.py
│   ├── core/                 # Inference client, metrics, data models
│   │   ├── client.py
│   │   ├── metrics.py
│   │   └── models.py
│   ├── server/               # SGLang server launcher & health checks
│   ├── benchmark/            # Query loading, runner, reporters, JSONL export
│   └── monitoring/           # GPU collector, Prometheus push
├── monitoring/               # Docker Compose stack (Prometheus + Grafana)
├── utils/
│   └── query.json            # Benchmark query dataset
├── tests/
│   ├── unit/
│   └── integration/
├── results/                  # Benchmark outputs (git-ignored)
└── .env.example
```

---

## Monitoring Stack

Start a local Prometheus + Pushgateway + Grafana stack:

```bash
cd monitoring
docker compose up -d
```

| Service | URL |
|---|---|
| Grafana | http://localhost:3000 (admin / admin) |
| Prometheus | http://localhost:9090 |
| Pushgateway | http://localhost:9091 |

Then run benchmarks with `--prometheus` to push metrics automatically.

---

## Development

```bash
# install with dev extras
uv pip install -e ".[dev]"

# run unit tests
python -m pytest tests/unit/

# run all tests
python -m pytest
```

---

## Results

Each benchmark run writes a timestamped JSONL file to `results/`:

```
results/benchmark_results_sglang_Qwen_Qwen2.5-7B-Instruct_20250415_123456.jsonl
```

Each line is a JSON object with all per-query metrics plus a `_meta` block containing engine, model, GPU, and server details.
