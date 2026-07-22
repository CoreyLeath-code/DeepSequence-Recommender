"""Prometheus metrics for the DeepSequence Recommender API."""

from prometheus_client import Counter, Gauge, Histogram

active_requests = Gauge(
    "deepseq_active_requests",
    "Currently active recommendation requests",
)

recommendations_total = Counter(
    "deepseq_recommendations_total",
    "Total number of recommendation requests served",
    ["status"],
)

recommendation_latency = Histogram(
    "deepseq_recommendation_latency_seconds",
    "End-to-end latency for recommendation requests",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

model_inference_latency = Histogram(
    "deepseq_model_inference_latency_seconds",
    "Latency of the neural model forward-pass",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

cache_hits_total = Counter(
    "deepseq_cache_hits_total",
    "Total number of replica-local recommendation cache hits",
)

cache_misses_total = Counter(
    "deepseq_cache_misses_total",
    "Total number of replica-local recommendation cache misses",
)

feedback_events_total = Counter(
    "deepseq_feedback_events_total",
    "Feedback events accepted for offline learning",
    ["event_type"],
)
