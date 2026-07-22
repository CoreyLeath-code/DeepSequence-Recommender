# Architecture

The system separates four responsibilities:

1. **Learning:** validate events, split chronologically, create causal examples, train, and evaluate.
2. **Artifacts:** bind weights, vocabulary, architecture, lineage, metrics, and checksums.
3. **Serving:** verify the bundle, enforce request policy, infer, cache, and fall back under pressure.
4. **Operations:** expose health/metrics, collect minimized feedback, and roll out immutable pairs.

The model is never loaded independently from its vocabulary. Production readiness depends on the
bundle, not merely on the process being alive. Liveness uses `/health`; model readiness uses
`/recommendations/health`.

The in-process cache, rate limiter, and admission controller protect one replica. A multi-replica
deployment should enforce distributed policy at the gateway and use a governed event/cache system.
Hard inference cancellation requires process isolation or a dedicated model server.
