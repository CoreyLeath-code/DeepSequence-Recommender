# Model card

## Model

Bidirectional LSTM sequence encoder with padding-aware attention and a next-item classification
head. The production artifact includes architecture parameters, item vocabulary, dataset ID,
ranking metrics, version, creation time, and checksums.

## Intended use

Ranking known catalogue items for users with at least one known recent interaction. It is not
intended for safety-critical decisions, eligibility, employment, credit, health, or other
high-impact domains.

## Required evaluation

Report Recall@K, NDCG@K, MRR@K, catalogue coverage, and the same metrics for a popularity baseline.
Add novelty, diversity, calibration, popularity bias, and cold-start cohorts before a real launch.

## Limitations

The dense output head scales linearly with catalogue size. Unknown items are rejected from the
sequence. Historical interactions may encode exposure and popularity bias. Offline ranking
metrics do not prove causal user or business impact.

## Promotion policy

A candidate needs a versioned dataset, reproducible configuration, compatible bundle, quality no
worse than the declared baseline, target-environment evidence, and an approved rollback plan.
