# syntax=docker/dockerfile:1

# ──────────────────────────────────────────────────────────────────────────────
# core-llm-inference — pre-baked SGLang production image
#
# Base: lmsysorg/sglang:latest (CUDA 12.4, torch, sglang[all], flashinfer)
# Adds: the core-llm-inference CLI package (httpx, typer, prometheus-client, …)
#
# Build:  docker build -t ghcr.io/ronit22203/clinical-trials-inference:latest .
# Run:    docker run --gpus all -p 30000:30000 \
#                    -v /workspace/hf-cache:/root/.cache/huggingface \
#                    -e MODEL_PATH=Qwen/Qwen2.5-7B-Instruct \
#                    ghcr.io/ronit22203/clinical-trials-inference:latest
# ──────────────────────────────────────────────────────────────────────────────
FROM lmsysorg/sglang:latest

WORKDIR /app/core-llm-inference

# Install package deps first — separate layer for better cache reuse
COPY pyproject.toml ./
RUN pip install --no-cache-dir httpx>=0.27 typer>=0.12 pyyaml>=6.0 \
        prometheus-client>=0.20 python-dotenv>=1.0

# Copy source and install the CLI
COPY . .
RUN pip install --no-cache-dir -e .

# ── Runtime configuration (all overridable via docker run -e) ─────────────────
ENV MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
ENV HOST="0.0.0.0"
ENV PORT="30000"

EXPOSE 30000

# ── HuggingFace model cache — mount volume here to persist across restarts ────
VOLUME ["/root/.cache/huggingface"]

CMD python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}"
