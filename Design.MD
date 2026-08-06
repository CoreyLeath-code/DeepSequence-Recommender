# Design

## Purpose and boundary

DeepSequence Recommender ranks known catalogue items from a user's recent timestamped interactions. It is an offline-training and API-serving example, not a deployment with a trained production model or a claim of production availability. It is not intended for high-impact decisions, including credit, employment, health, eligibility, or safety-critical use.

## Learning path

The training side validates interaction events, creates chronological splits, produces causal next-item examples, trains a sequence model, and evaluates it against a popularity baseline. The documented model is a bidirectional LSTM sequence encoder with padding-aware attention and a next-item classification head. Candidate evaluation must use a versioned dataset and report ranking measures such as Recall@K, NDCG@K, MRR@K, catalogue coverage, and baseline comparison; offline ranking results do not establish causal user or business impact.

## Artifact contract

Model weights are bundled with the item vocabulary, architecture parameters, dataset identifier, metrics, version, creation time, and checksums. The vocabulary is an inseparable part of the model contract: the service does not load weights independently of it. Production readiness is therefore based on verification of the complete bundle, not on process liveness alone.

## Serving and operations

The service verifies the bundle, enforces request policy, performs ranking, and exposes separate process liveness and model-readiness endpoints. It includes feedback capture and operational protections such as caching, rate limiting, admission control, and fallback behavior. These controls are in-process; multi-replica deployments require distributed policy enforcement at the gateway and governed cache/event infrastructure.

## Limits and required production work

The repository uses a deterministic untrained model for development integration and fails closed in production unless a verified bundle is mounted. Before a real launch, the system needs a versioned dataset, reproducible configuration, target-environment evidence, bias/calibration/novelty/diversity and cold-start analysis, a rollout and rollback plan, and protection against exposure and popularity bias.

## Evidence in this repository

- `src/training/train.py` is the training implementation.
- `app/core/artifacts.py` and `app/core/model.py` implement bundle and serving-side model handling.
- `docs/architecture.md` and `docs/MODEL_CARD.md` document the deployed-contract and evaluation boundaries.
