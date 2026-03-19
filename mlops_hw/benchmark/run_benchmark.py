#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import psutil
import requests

# Constants

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

PARTS = {
    1: {"name": "baseline",          "port": 8000, "result": "part1_baseline.json"},
    2: {"name": "onnx",              "port": 8001, "result": "part2_onnx.json"},
    3: {"name": "dynamic_batching",  "port": 8002, "result": "part3_dynamic_batching.json"},
}

BATCH_SIZES = [1, 8, 32]
N_WARMUP = 10
N_LATENCY = 100        # sequential requests for latency measurement
N_THROUGHPUT = 200     # total requests for throughput measurement
CONCURRENCY = 16       # parallel workers for throughput / Part-3 batching trigger

SAMPLE_TEXT = (
    "Это тестовый текст для измерения производительности модели встраивания слов. "
    "Он намеренно сделан достаточно длинным, чтобы отражать реальные условия."
)


# Helpers

def _url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def _wait_for_service(port: int, timeout: float = 120.0) -> bool:
    url = f"{_url(port)}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _single_request(url: str, texts: list[str]) -> float:
    """Returns round-trip latency in seconds."""
    t0 = time.perf_counter()
    resp = requests.post(f"{url}/embed", json={"texts": texts}, timeout=120)
    resp.raise_for_status()
    return time.perf_counter() - t0


def _percentile(data: list[float], p: float) -> float:
    """p in [0, 100]."""
    if len(data) == 0:
        return 0.0
    sorted_data = sorted(data)
    idx = (p / 100) * (len(sorted_data) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_data) - 1)
    frac = idx - lo
    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])


# Measurement routines

def measure_latency(url: str, texts: list[str]) -> dict:
    """Sequential requests; returns latency stats in ms."""
    # Warmup
    for _ in range(N_WARMUP):
        _single_request(url, texts)

    latencies_s: list[float] = []
    for _ in range(N_LATENCY):
        latencies_s.append(_single_request(url, texts))

    ms = [x * 1000 for x in latencies_s]
    return {
        "p50_ms":  round(_percentile(ms, 50),  2),
        "p95_ms":  round(_percentile(ms, 95),  2),
        "p99_ms":  round(_percentile(ms, 99),  2),
        "mean_ms": round(statistics.mean(ms),   2),
        "min_ms":  round(min(ms),               2),
        "max_ms":  round(max(ms),               2),
    }


def measure_throughput(url: str, texts: list[str], workers: int = CONCURRENCY) -> dict:
    # Short warmup
    for _ in range(5):
        _single_request(url, texts)

    t_start = time.perf_counter()
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_single_request, url, texts) for _ in range(N_THROUGHPUT)]
        for fut in concurrent.futures.as_completed(futures):
            try:
                fut.result()
            except Exception:
                errors += 1
    elapsed = time.perf_counter() - t_start

    rps = (N_THROUGHPUT - errors) / elapsed
    return {
        "requests_per_sec": round(rps, 2),
        "total_requests": N_THROUGHPUT,
        "errors": errors,
        "elapsed_s": round(elapsed, 3),
        "concurrency": workers,
    }


def measure_resources(url: str, texts: list[str]) -> dict:
    proc = psutil.Process(os.getpid())

    # Baseline snapshot
    psutil.cpu_percent(interval=None)  # discard first call (always 0.0)

    cpu_samples: list[float] = []
    mem_mb_samples: list[float] = []

    for _ in range(20):
        _single_request(url, texts)
        cpu_samples.append(psutil.cpu_percent(interval=None))
        mem_mb_samples.append(proc.memory_info().rss / 1024 / 1024)

    return {
        "system_cpu_mean_pct": round(statistics.mean(cpu_samples),    2),
        "system_cpu_max_pct":  round(max(cpu_samples),                2),
        "client_rss_mb_mean":  round(statistics.mean(mem_mb_samples), 2),
    }


# Per-part benchmark

def benchmark_part(part: int) -> dict:
    meta = PARTS[part]
    port = meta["port"]
    url = _url(port)

    print(f"\n{'='*60}")
    print(f"  Part {part} — {meta['name']} (port {port})")
    print(f"{'='*60}")

    if not _wait_for_service(port, timeout=5):
        print(f"  [ERROR] Service not reachable at {url}. Is it running?")
        sys.exit(1)
    print(f"  Service is up at {url}")

    results: dict = {"part": part, "name": meta["name"], "port": port, "batches": {}}

    for bs in BATCH_SIZES:
        texts = [SAMPLE_TEXT] * bs
        print(f"\n  Batch size = {bs}")

        print(f"    Measuring latency  ({N_LATENCY} sequential requests) …")
        lat = measure_latency(url, texts)
        print(f"      p50={lat['p50_ms']} ms  p95={lat['p95_ms']} ms  p99={lat['p99_ms']} ms")

        print(f"    Measuring throughput ({N_THROUGHPUT} concurrent requests, {CONCURRENCY} workers) …")
        tput = measure_throughput(url, texts, workers=CONCURRENCY)
        print(f"      {tput['requests_per_sec']} req/s")

        print(f"    Sampling resources …")
        res = measure_resources(url, texts)
        print(f"      CPU={res['system_cpu_mean_pct']}%  RSS={res['client_rss_mb_mean']} MB")

        results["batches"][str(bs)] = {
            "batch_size": bs,
            "latency": lat,
            "throughput": tput,
            "resources": res,
        }

    return results


# Comparison helpers

def _load_result(filename: str) -> dict | None:
    path = RESULTS_DIR / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _print_comparison(r1: dict, r2: dict) -> None:
    n1, n2 = r1["name"], r2["name"]
    print(f"\n{'─'*60}")
    print(f"  Comparison: {n1}  vs  {n2}")
    print(f"{'─'*60}")
    header = f"  {'Batch':>6}  {'Metric':>25}  {n1:>12}  {n2:>12}  {'Δ':>8}"
    print(header)
    print(f"  {'-'*len(header.strip())}")

    for bs in [1, 8, 32]:
        b1 = r1["batches"].get(str(bs), {})
        b2 = r2["batches"].get(str(bs), {})
        if not b1 or not b2:
            continue
        for metric, unit in [("p50_ms", "ms"), ("p95_ms", "ms"), ("p99_ms", "ms")]:
            v1 = b1["latency"][metric]
            v2 = b2["latency"][metric]
            pct = ((v2 - v1) / v1 * 100) if v1 else 0
            sign = "+" if pct > 0 else ""
            print(f"  {bs:>6}  {metric + ' latency':>25}  {v1:>11.1f}  {v2:>11.1f}  {sign}{pct:>6.1f}%")

        rps1 = b1["throughput"]["requests_per_sec"]
        rps2 = b2["throughput"]["requests_per_sec"]
        pct = ((rps2 - rps1) / rps1 * 100) if rps1 else 0
        sign = "+" if pct > 0 else ""
        print(f"  {bs:>6}  {'throughput req/s':>25}  {rps1:>11.1f}  {rps2:>11.1f}  {sign}{pct:>6.1f}%")


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding service benchmarker")
    parser.add_argument(
        "--part",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which part to benchmark (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for JSON results (default: ./results)",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    parts_to_run = [1, 2, 3] if args.part == "all" else [int(args.part)]
    completed: dict[int, dict] = {}

    for part in parts_to_run:
        result = benchmark_part(part)
        completed[part] = result

        out_file = output_dir / PARTS[part]["result"]
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Results saved → {out_file}")

    # Print comparisons if multiple parts were benchmarked.
    if 1 in completed and 2 in completed:
        _print_comparison(completed[1], completed[2])
    if 2 in completed and 3 in completed:
        _print_comparison(completed[2], completed[3])
    if 1 in completed and 3 in completed:
        _print_comparison(completed[1], completed[3])

    print("\nBenchmark complete.\n")


if __name__ == "__main__":
    main()
