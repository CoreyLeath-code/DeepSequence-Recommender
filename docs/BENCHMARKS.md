# Benchmark methodology

The inference benchmark measures one CPU PyTorch forward/top-k path. It uses a seeded model,
fixed workload, warmup, `time.perf_counter`, and nearest-rank percentiles. CI uploads the complete
JSON result so future comparisons retain environment and workload metadata.

```bash
python -m benchmarks.inference --iterations 200 --warmup 20
```

Reference observation collected July 22, 2026:

| Scope | Value |
|---|---|
| Python / PyTorch | 3.12.13 / 2.13.0+cpu |
| Platform | Windows 11 build 26200, AMD64 Family 25 Model 97 |
| Workload | batch 1, 500 items, sequence 50, top-k 10 |
| Samples | 20 warmup, 200 measured |
| p50 / p95 / p99 | 0.556 / 0.755 / 1.118 ms |
| Sequential rate | 1,743.0 inferences/s |

Do not compare results across hardware as if they were model improvements. For release decisions,
run both commits on the same idle host, publish raw JSON, and add HTTP/concurrency/load testing in
the target environment. Ranking metrics require a representative chronological dataset and must
be reported separately from systems latency.
