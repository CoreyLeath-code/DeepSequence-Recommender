"""Auditable non-neural baselines used as model promotion gates."""

from __future__ import annotations

from collections import Counter


class PopularityBaseline:
    def __init__(self) -> None:
        self.ranking: list[str] = []

    def fit(self, sequences: list[list[str]]) -> PopularityBaseline:
        counts = Counter(item for sequence in sequences for item in sequence)
        self.ranking = [item for item, _ in counts.most_common()]
        return self

    def recommend(self, seen: list[str], top_k: int) -> list[str]:
        excluded = set(seen)
        return [item for item in self.ranking if item not in excluded][:top_k]
