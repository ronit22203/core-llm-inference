# GPU Hardware Selection for LLM Inference

A reference guide for evaluating cloud GPU instances when running LLM inference workloads with SGLang or vLLM.

---

## Why Memory Bandwidth is the Primary Constraint

LLM inference (autoregressive decoding) is **memory-bandwidth bound**, not compute bound. Each generated token requires reading the entire model weight matrix from VRAM once. As a result:

- **Doubling memory bandwidth doubles tokens-per-second** (at batch size 1)
- TFLOPS matter far less than GB/s for single-sequence decode
- Model Bandwidth Utilisation (MBU) is the key efficiency KPI

**Theoretical maximum TPS** at batch size 1:

```
max TPS = peak_bandwidth_gbps / model_size_gb
```

_Example: RTX 3080 Ti (808 GB/s) running Qwen 2.5 7B (~14 GB FP16) → ~57.7 tok/s theoretical max._  
_SGLang at 91% MBU → ~52.5 tok/s actual._

---

## Key Hardware Dimensions

### VRAM Capacity

VRAM must hold model weights plus the KV cache for all in-flight sequences.

| VRAM  | Fits (FP16)                                     |
| ----- | ----------------------------------------------- |
| 8 GB  | 3B models; 7B with heavy quantisation (INT4)    |
| 12 GB | 7–8B models with moderate KV cache              |
| 16 GB | 7–8B models with headroom; 13B (INT4)           |
| 24 GB | 13B models (FP16); 70B (INT4 + KV compression)  |
| 80 GB | 70B models (FP16); 405B (multi-GPU TP)          |

**Rule of thumb:** FP16 ≈ 2 GB per billion parameters. FP8 KV cache reduces effective KV footprint by ~50%, increasing available headroom for larger contexts or higher concurrency.

### Memory Bandwidth

| GPU          | Peak Bandwidth | Theoretical TPS (7B FP16) |
| ------------ | -------------- | ------------------------- |
| RTX 3080     | 760 GB/s       | ~54 tok/s                 |
| RTX 3080 Ti  | 808 GB/s       | ~57 tok/s                 |
| RTX 3090     | 936 GB/s       | ~67 tok/s                 |
| RTX 4070     | 504 GB/s       | ~36 tok/s                 |
| RTX 4070 Ti  | 504 GB/s       | ~36 tok/s                 |
| RTX 4080     | 717 GB/s       | ~51 tok/s                 |
| RTX 4090     | 1008 GB/s      | ~72 tok/s                 |
| RTX 5080     | 960 GB/s       | ~68 tok/s                 |
| RTX 5090     | 1792 GB/s      | ~128 tok/s                |
| L40S         | 864 GB/s       | ~62 tok/s                 |
| A100 40 GB   | 1555 GB/s      | ~111 tok/s                |
| A100 80 GB   | 2039 GB/s      | ~145 tok/s                |
| H100         | 3350 GB/s      | ~239 tok/s                |

_Theoretical values assume 100% MBU. Real-world SGLang results typically fall at 85–95% of theoretical._

### PCIe Bandwidth

PCIe governs CPU-to-GPU transfers: model loading on startup and KV cache eviction under memory pressure.

| PCIe Generation | x16 Bandwidth |
| --------------- | ------------- |
| PCIe 3.0 x16    | ~15.8 GB/s    |
| PCIe 4.0 x16    | ~31.5 GB/s    |
| PCIe 5.0 x16    | ~63.0 GB/s    |

For single-sequence inference, PCIe generation rarely affects decode throughput once the model is fully resident in VRAM. At high concurrency with frequent KV cache eviction, PCIe 4.0 x16 is recommended.

### CPU Considerations

- **Prefill latency** is partly CPU-bound for short sequences; 8+ high-frequency cores keep the CPU from being the bottleneck above 4 concurrent requests.
- **Batch scheduling**: SGLang and vLLM use the CPU to manage continuous batching and KV cache allocation. Insufficient cores create scheduling overhead above 4 concurrent in-flight sequences.
- **System RAM**: Maintain at least 1.5× model size in system RAM. 32 GB is the practical minimum for 7B–13B workloads.

---

## GPU Tier Comparison

| GPU          | VRAM   | Bandwidth  | Approx. Spot $/hr | Best For                           |
| ------------ | ------ | ---------- | ------------------ | ---------------------------------- |
| RTX 3080 Ti  | 12 GB  | 808 GB/s   | $0.01–0.02         | 7B FP16, cost-optimised research   |
| RTX 3090     | 24 GB  | 936 GB/s   | $0.02–0.05         | 7B–13B FP16, larger KV headroom    |
| RTX 4090     | 24 GB  | 1008 GB/s  | $0.35–0.60         | Best consumer bandwidth; 13B FP16  |
| RTX 5080     | 16 GB  | 960 GB/s   | $0.40–0.70         | High-bandwidth, mid-VRAM           |
| RTX 5090     | 32 GB  | 1792 GB/s  | $0.80–1.50         | Top consumer throughput            |
| L40S         | 48 GB  | 864 GB/s   | $0.80–1.50         | High concurrency, large KV cache   |
| A100 40 GB   | 40 GB  | 1555 GB/s  | $1.50–2.50         | 13B–70B INT4, high concurrency     |
| A100 80 GB   | 80 GB  | 2039 GB/s  | $2.50–4.00         | 70B FP16, production workloads     |
| H100         | 80 GB  | 3350 GB/s  | $3.50–6.00         | Maximum throughput, SLA-critical   |

_Pricing is indicative of spot-market rates on providers such as Vast.ai and RunPod. Dedicated and reserved rates are higher._

---

## Selecting Hardware for Your Workload

### Single-Sequence / Low-Concurrency (1–2 requests)

The bottleneck is memory bandwidth. Prioritise bandwidth per dollar.

- **RTX 3080 Ti / RTX 3090**: Strong value for 7B–8B FP16 workloads
- **RTX 4090 / RTX 5080**: Higher bandwidth; accommodates 13B FP16
- A100 and H100 are cost-inefficient here — excess VRAM yields no throughput gain at batch size 1

### High-Concurrency (8+ simultaneous requests)

The bottleneck shifts from bandwidth to KV cache capacity and scheduling overhead.

- VRAM must hold a separate KV cache per in-flight sequence; 24+ GB recommended
- **L40S (48 GB)** or **A100 80 GB**: Large VRAM reduces eviction pressure
- Pair with a CPU offering 8+ high-frequency cores to avoid batch-scheduler bottlenecks

### Cost-Optimised Research / Benchmarking

- RTX 30-series rigs on spot-market providers offer the best bandwidth-per-dollar ratio
- 12 GB VRAM is sufficient for Qwen 2.5 7B FP16 with FP8 KV cache enabled
- A modest compute budget supports full benchmark suites including concurrency sweeps, quantisation comparisons, and parameter grid searches

### Production / SLA-Sensitive

- **H100 or A100 80 GB**: Predictable latency, ECC memory, large VRAM headroom
- Multi-GPU tensor parallelism (`--tp 2` or `--tp 4`) for 70B+ models
- Prefer dedicated or reserved instances over spot for uptime guarantees

---

## SGLang Optimisation by VRAM Tier

| VRAM Tier  | Recommended Configuration                                   |
| ---------- | ----------------------------------------------------------- |
| 12 GB      | FP8 KV cache + RadixAttention prefix sharing; limit context |
| 16–24 GB   | FP16 weights + FP8 KV cache; moderate concurrency           |
| 40–80 GB   | FP16 throughout; large KV cache; high-concurrency batching  |
| Multi-GPU  | Tensor parallelism (`--tp`); pipeline parallelism for 70B+  |

---

## Key Takeaways

1. **Bandwidth determines single-sequence throughput.** The RTX 3090 has twice the VRAM of the RTX 3080 Ti but only ~16% more bandwidth. For decode-bound workloads, bandwidth is the deciding factor.

2. **MBU is the efficiency KPI.** Values below 60% indicate the inference stack is not saturating the hardware. SGLang with RadixAttention typically achieves 85–95% MBU.

3. **12 GB is a viable production floor** for 7B–8B models with FP8 KV cache. 24 GB provides headroom for 13B models and larger context windows without eviction.

4. **PCIe generation rarely determines inference throughput.** It affects model load time and KV eviction throughput, not steady-state decode speed.

5. **CPU core count matters at high concurrency.** 8+ cores on the SGLang/vLLM batch scheduler prevents CPU-side bottlenecks above 4 concurrent in-flight requests.
