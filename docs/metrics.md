# Metrics

Prometheus metrics are exposed at `GET /metrics`.

| Metric | Meaning |
|---|---|
| `deepseq_active_requests` | In-flight model requests admitted by this replica |
| `deepseq_recommendations_total{status}` | Success, cache, fallback, overload, and error outcomes |
| `deepseq_recommendation_latency_seconds` | End-to-end admitted request latency |
| `deepseq_model_inference_latency_seconds` | PyTorch inference duration |
| `deepseq_cache_hits_total` / `deepseq_cache_misses_total` | Replica-local cache behavior |
| `deepseq_feedback_events_total{event_type}` | Accepted learning-feedback events |

No availability SLO is claimed by the repository. Operators should establish objectives only after
target-environment load and failure testing. Recommended alert dimensions include model version,
error/fallback rate, saturation, p95/p99 latency, unknown-item rate, drift, delayed-label ranking
quality, coverage, and experiment guardrails.
