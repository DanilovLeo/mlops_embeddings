# MLOps Homework — Embedding Inference Pipeline

Benchmarks three progressive optimisation levels for CPU inference of
[cointegrated/rubert-tiny2](https://huggingface.co/cointegrated/rubert-tiny2):

| Part | Technique | Port |
|------|-----------|------|
| 1 | Baseline PyTorch | 8000 |
| 2 | ONNX Runtime (`ORT_ENABLE_ALL`) | 8001 |
| 3 | Dynamic batching over ONNX | 8002 |

## Requirements

- Python **3.10+**
- CPU only (no CUDA needed)

## Setup

```bash
# Clone / enter the project root
cd mlops_hw

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies for all parts + the benchmarker
pip install -r part1_baseline/requirements.txt
pip install -r part2_onnx/requirements.txt          # superset; installs onnxruntime too
pip install -r benchmark/requirements.txt
pip install onnx                                     # required for ONNX export in part 2
```

## Step-by-step Reproduction

### 1 — Baseline (PyTorch)

**Terminal 1** — start the service:
```bash
cd mlops_hw
uvicorn part1_baseline.service:app --host 0.0.0.0 --port 8000
```

**Terminal 2** — run the benchmark:
```bash
cd mlops_hw
python benchmark/run_benchmark.py --part 1
# results saved to results/part1_baseline.json
```

### 2 — ONNX

**Step 2a** — export the model to ONNX (one-time, ~1-2 min):
```bash
cd mlops_hw
python part2_onnx/convert_to_onnx.py
# saves models/rubert_mini.onnx
```

**Terminal 1** — start the ONNX service:
```bash
uvicorn part2_onnx.service:app --host 0.0.0.0 --port 8001
```

**Terminal 2** — benchmark:
```bash
python benchmark/run_benchmark.py --part 2
# results saved to results/part2_onnx.json
```

### 3 — Dynamic Batching

The ONNX model must already exist (run step 2a above).

**Terminal 1** — start the dynamic-batching service:
```bash
uvicorn part3_dynamic_batching.service:app --host 0.0.0.0 --port 8002
```

**Terminal 2** — benchmark (sends concurrent requests to trigger batching):
```bash
python benchmark/run_benchmark.py --part 3
# results saved to results/part3_dynamic_batching.json
```

### All Parts at Once

Start all three services in separate terminals, then:
```bash
python benchmark/run_benchmark.py --part all
```
The script prints a side-by-side comparison after each run.

## Quick Smoke Test

```bash
curl -s -X POST http://localhost:8000/embed \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Привет, мир!", "Hello world"]}' | python -m json.tool
```

## Project Layout

```
mlops_hw/
├── part1_baseline/
│   ├── service.py          FastAPI + PyTorch, port 8000
│   └── requirements.txt
├── part2_onnx/
│   ├── convert_to_onnx.py  Exports model → models/rubert_mini.onnx
│   ├── service.py          FastAPI + OnnxRuntime, port 8001
│   └── requirements.txt
├── part3_dynamic_batching/
│   ├── service.py          FastAPI + OnnxRuntime + async queue, port 8002
│   └── requirements.txt
├── benchmark/
│   ├── run_benchmark.py    Measures latency / throughput / resources
│   └── requirements.txt
├── models/                 ONNX model stored here
├── results/                JSON benchmark outputs 
├── REPORT.md
└── README.md
```

## Benchmark Metrics Explained

| Metric | Why |
|--------|-----|
| Latency p50/p95/p99 | User-facing response-time percentiles |
| Throughput (req/s) | Service capacity under concurrent load |
| System CPU % | Compute cost per request burst |
| Client RSS MB | Memory footprint of the benchmark process |

Latency is measured sequentially (one request at a time) to isolate true
service processing time from queuing effects.  
Throughput is measured with 16 concurrent workers to saturate the service and for Part 3 to
actually trigger the batching mechanism.
