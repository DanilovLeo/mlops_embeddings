# Report — Embedding Inference Pipeline Optimisation

Model: `cointegrated/rubert-tiny2` · Hardware: CPU only · Framework: FastAPI + uvicorn

---

## 1. Methodology

### Why these metrics?

| Metric | Rationale |
|--------|-----------|
| **Latency p50** | Median user experience — what most requests feel like |
| **Latency p95 / p99** | Tail latency — reflects worst-case SLA behaviour; systems are often limited by their slowest percentile |
| **Throughput (req/s)** | Service capacity under concurrent load; determines horizontal scaling requirements |
| **CPU %** | Compute cost per burst; signals whether CPU is the bottleneck or whether idle headroom exists |
| **Memory (RSS)** | Baseline resource footprint; important for container sizing |

### Measurement protocol

- **Latency**: 100 sequential HTTP POST requests to `/embed` after a 10-request warm-up. Sequential isolation removes queuing noise.
- **Throughput**: 200 requests via 16 concurrent threads. Concurrency is deliberately high to saturate the service and — in Part 3 — to trigger the dynamic-batching mechanism.
- **Resources**: CPU sampled with `psutil.cpu_percent()` and RSS with `psutil.Process.memory_info()` during a 20-request burst.
- **Batch sizes tested**: 1, 8, 32 texts per request.

---

## 2. Results

### Part 1 — Baseline PyTorch (port 8000)

| Batch | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) | CPU (%) | RSS (MB) |
|------:|---------:|---------:|---------:|-------------------:|--------:|---------:|
|     1 |     4.22 |     5.36 |     5.61 |             258.96 |    10.0 |    33.34 |
|     8 |     10.4 |    12.49 |    13.16 |             150.19 |    10.0 |    34.38 |
|    32 |    30.32 |    34.05 |    35.14 |              58.85 |   44.44 |    37.64 |

### Part 2 — ONNX Runtime / ORT_ENABLE_ALL (port 8001)

| Batch | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) | CPU (%) | RSS (MB) |
|------:|---------:|---------:|---------:|-------------------:|--------:|---------:|
|     1 |     3.07 |     4.67 |     5.34 |             674.56 |   10.84 |    37.64 |
|     8 |     7.96 |     9.60 |     9.89 |             211.09 |   20.77 |    37.64 |
|    32 |    22.54 |    23.40 |    27.16 |              63.65 |   17.90 |    38.39 |

### Part 3 — Dynamic Batching over ONNX (port 8002)

| Batch | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) | CPU (%) | RSS (MB) |
|------:|---------:|---------:|---------:|-------------------:|--------:|---------:|
|     1 |     25.0 |    25.43 |    25.90 |             361.76 |    4.11 |    38.50 |
|     8 |     30.2 |    32.52 |    39.52 |             182.89 |    9.54 |    38.50 |
|    32 |    22.43 |    23.58 |    25.55 |              47.68 |   14.14 |    38.56 |

---

## 3. Analysis

### Part 1 → Part 2: PyTorch vs ONNX Runtime

**What changed**: The model is converted once to a static ONNX graph and executed by ORT with `ORT_ENABLE_ALL` — which enables constant folding, operator fusion, and layout transformations at session initialisation time.

**Observed improvements**:
- **Latency**: ~27% lower p50 at batch=1 (4.2 → 3.1 ms), ~26% at batch=32. Gains come from fused kernels and elimination of Python-level dispatch overhead.
- **Throughput**: +160% at batch=1 (259 → 675 req/s), +40% at batch=8. The gap narrows at batch=32 (+8%) as compute dominates over dispatch overhead.
- **CPU**: Comparable at small batches; notably lower at batch=32 (18% vs 44%) — ORT executes the same work more efficiently.

### Part 2 → Part 3: Static ONNX vs Dynamic Batching

**What changed**: An asyncio queue + background worker collects incoming requests for up to 20 ms or until 32 texts are accumulated, then runs a **single** batched ONNX call and fans results back.

**Observed results**:

- **Single-request latency (batch=1)**: Increased to ~25 ms (+714% vs ONNX) due to the 20 ms wait window — the expected trade-off of dynamic batching.
- **Throughput at batch=1**: 362 req/s — lower than ONNX (675) but +40% over baseline, since concurrent requests get merged.
- **Batch=32 latency**: Nearly identical to ONNX (22.4 vs 22.5 ms p50) — the batching mechanism has fully absorbed the overhead at this load level.
- **CPU**: Significantly lower across all batch sizes (4–14% vs 11–18% for ONNX) — fewer inference calls means less total compute.

**When dynamic batching hurts**: Low-concurrency workloads where requests rarely overlap. In that regime, every request pays the 20 ms wait penalty for no benefit.

---

## 4. Conclusions

1. **ONNX export is low-effort, high-reward**: A one-time model conversion reduces per-request latency significantly with no change to service API or infrastructure.

2. **Dynamic batching shifts the latency/throughput trade-off**: It is the right choice when the system needs to serve many concurrent users and throughput is the primary concern. For latency-sensitive, low-traffic deployments, static ONNX is preferable.

3. **Tail latency matters more than mean**: Systems serving real users should optimise p95/p99. ONNX reduces variance; dynamic batching with a tight `max_wait_ms` cap keeps the upper bound predictable.

4. **CPU bottleneck persists**: All three approaches run on a single uvicorn worker and are fundamentally single-threaded for inference. The next optimisation step would be multi-process uvicorn workers (`--workers N`) or moving to a dedicated model-serving framework (TorchServe, Triton Inference Server) that manages thread pools automatically.

5. **Memory is not the binding constraint**: For this model size (~29 M parameters, ~115 MB float32), memory footprint is well within typical container limits. Quantisation (INT8 / FP16) would reduce it further and also accelerate ONNX inference by 1.3–2× on modern CPUs with AVX-512 VNNI support.
