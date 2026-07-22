"""Ranking metrics with explicit definitions and no framework dependency."""

from __future__ import annotations

import math


def recall_at_k(predictions: list[list[str]], targets: list[str]) -> float:
    return sum(target in predicted for predicted, target in zip(predictions, targets)) / len(targets)


def mrr_at_k(predictions: list[list[str]], targets: list[str]) -> float:
    reciprocal_ranks = []
    for predicted, target in zip(predictions, targets):
        reciprocal_ranks.append(
            1 / (predicted.index(target) + 1) if target in predicted else 0.0
        )
    return sum(reciprocal_ranks) / len(targets)


def ndcg_at_k(predictions: list[list[str]], targets: list[str]) -> float:
    gains = []
    for predicted, target in zip(predictions, targets):
        gains.append(
            1 / math.log2(predicted.index(target) + 2) if target in predicted else 0.0
        )
    return sum(gains) / len(targets)


def catalogue_coverage(predictions: list[list[str]], catalogue: set[str]) -> float:
    recommended = {item for prediction in predictions for item in prediction}
    return len(recommended & catalogue) / len(catalogue) if catalogue else 0.0


def evaluate_ranking(
    predictions: list[list[str]], targets: list[str], catalogue: set[str]
) -> dict[str, float]:
    if not targets or len(predictions) != len(targets):
        raise ValueError("Predictions and non-empty targets must have equal lengths")
    return {
        "recall_at_k": recall_at_k(predictions, targets),
        "ndcg_at_k": ndcg_at_k(predictions, targets),
        "mrr_at_k": mrr_at_k(predictions, targets),
        "catalogue_coverage": catalogue_coverage(predictions, catalogue),
    }
