from benchmarks.inference import run_benchmark


def test_benchmark_emits_reproducible_contract() -> None:
    report = run_benchmark(iterations=3, warmup=1)
    assert report["workload"]["iterations"] == 3
    assert report["results"]["latency_ms_p95"] > 0
