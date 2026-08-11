"""Candidate retrieval for the two-stage recommendation serving path."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from app.core.model import DeepSequenceModel


@dataclass(frozen=True)
class RetrievalResult:
    """Candidate IDs emitted by a retriever before sequence-aware ranking."""

    candidate_ids: list[int]


class ExactEmbeddingRetriever:
    """Retrieve a bounded candidate pool with normalized item-embedding similarity.

    This is an exact in-memory retriever, not an ANN or FAISS implementation.  It
    establishes a replaceable retrieval boundary while keeping the default bundle
    self-contained and deterministic for the current catalogue scale.
    """

    def __init__(self, candidate_pool_size: int) -> None:
        if candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be at least one")
        self.candidate_pool_size = candidate_pool_size

    @torch.no_grad()
    def retrieve(
        self,
        model: DeepSequenceModel,
        item_sequence: torch.Tensor,
        *,
        top_k: int,
        exclude_ids: list[int] | None = None,
    ) -> RetrievalResult:
        """Return eligible IDs ordered by embedding-similarity score."""

        if item_sequence.ndim != 2 or item_sequence.shape[0] != 1:
            raise ValueError("Retrieval requires one padded recommendation sequence")
        if not 1 <= top_k <= model.num_items:
            raise ValueError(f"top_k must be between 1 and {model.num_items}")

        history_ids = item_sequence[0]
        known_mask = history_ids.ne(model.padding_idx)
        if not known_mask.any():
            raise ValueError("Retrieval requires at least one known item")

        excluded = {
            item_id
            for item_id in (exclude_ids or [])
            if 1 <= item_id <= model.num_items
        }
        eligible_count = model.num_items - len(excluded)
        if top_k > eligible_count:
            raise ValueError("top_k exceeds the remaining eligible catalogue")

        query = model.embedding(history_ids[known_mask]).mean(dim=0)
        query = F.normalize(query, dim=0, eps=1e-12)
        catalogue = F.normalize(
            model.embedding.weight[1 : model.num_items + 1],
            dim=1,
            eps=1e-12,
        )
        scores = torch.matmul(catalogue, query)
        if excluded:
            scores[[item_id - 1 for item_id in excluded]] = float("-inf")

        candidate_count = min(
            max(top_k, self.candidate_pool_size),
            eligible_count,
        )
        retrieved = torch.topk(scores, k=candidate_count).indices.add(1).tolist()
        return RetrievalResult(candidate_ids=retrieved)
