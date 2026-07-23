"""Deterministic single-process inference benchmark."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time

import torch

from app.core.data_processor import SequenceProcessor
from app.core.model import DeepSequenceModel


def _percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def run_benchmark(iterations: int = 200, warmup: int = 20) -> dict:
    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative")
    torch.manual_seed(7)
    processor = SequenceProcessor(max_length=50).fit([[f"item_{index}" for index in range(500)]])
    model = DeepSequenceModel(
        num_items=processor.vocab_size,
        embedding_dim=32,
        hidden_dim=64,
        num_layers=1,
    ).eval()
    tensor = processor.to_tensor([f"item_{index}" for index in range(20)])
    for _ in range(warmup):
        model.recommend(tensor, top_k=10)
    latencies = []
    started = time.perf_counter()
    for _ in range(iterations):
        call_started = time.perf_counter()
        model.recommend(tensor, top_k=10)
        latencies.append((time.perf_counter() - call_started) * 1_000)
    duration = time.perf_counter() - started
    return {
        "scope": "single-process CPU PyTorch inference; batch=1; no HTTP or network",
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor() or "not reported",
        },
        "workload": {
            "iterations": iterations,
            "warmup": warmup,
            "catalogue_size": processor.vocab_size,
            "sequence_length": 50,
            "top_k": 10,
        },
        "results": {
            "latency_ms_mean": round(statistics.fmean(latencies), 3),
            "latency_ms_p50": round(statistics.median(latencies), 3),
            "latency_ms_p95": round(_percentile(latencies, 0.95), 3),
            "latency_ms_p99": round(_percentile(latencies, 0.99), 3),
            "sequential_inferences_per_second": round(iterations / duration, 3),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.iterations, args.warmup), indent=2))


if __name__ == "__main__":
    main()
