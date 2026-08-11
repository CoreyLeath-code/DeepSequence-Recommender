import pytest
import torch

from app.core.model import DeepSequenceModel
from app.core.retrieval import ExactEmbeddingRetriever


def _model() -> DeepSequenceModel:
    model = DeepSequenceModel(
        num_items=4,
        embedding_dim=2,
        hidden_dim=2,
        num_layers=1,
        dropout=0.0,
    ).eval()
    with torch.no_grad():
        model.embedding.weight.copy_(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                ]
            )
        )
        model.output_proj.weight.zero_()
        model.output_proj.bias.copy_(
            torch.tensor([float("-inf"), 0.1, 0.9, 0.2, 0.8])
        )
    return model


def test_embedding_retriever_excludes_history_and_bounds_candidate_pool() -> None:
    retriever = ExactEmbeddingRetriever(candidate_pool_size=3)

    result = retriever.retrieve(
        _model(),
        torch.tensor([[0, 0, 1]]),
        top_k=1,
        exclude_ids=[1],
    )

    assert result.candidate_ids == [2, 3, 4]


def test_ranker_only_orders_retrieved_candidates() -> None:
    ranked = _model().rank_candidates(
        torch.tensor([[0, 0, 1]]),
        [1, 3, 4],
        top_k=2,
    )

    assert ranked == [4, 3]


def test_retriever_rejects_an_impossible_request() -> None:
    retriever = ExactEmbeddingRetriever(candidate_pool_size=2)

    with pytest.raises(ValueError, match="remaining eligible"):
        retriever.retrieve(
            _model(),
            torch.tensor([[0, 0, 1]]),
            top_k=4,
            exclude_ids=[1],
        )
