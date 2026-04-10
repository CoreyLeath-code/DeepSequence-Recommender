"""Unit tests for DeepSequence Recommender core components."""

from __future__ import annotations

import pytest

from app.core.data_processor import SequenceProcessor
from app.core.model import DeepSequenceModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def processor() -> SequenceProcessor:
    proc = SequenceProcessor(max_length=10)
    sequences = [["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]]
    proc.fit(sequences)
    return proc


@pytest.fixture()
def model(processor: SequenceProcessor) -> DeepSequenceModel:
    return DeepSequenceModel(
        num_items=processor.vocab_size,
        embedding_dim=16,
        hidden_dim=32,
        num_layers=1,
    )


# ---------------------------------------------------------------------------
# SequenceProcessor tests
# ---------------------------------------------------------------------------


class TestSequenceProcessor:
    def test_vocab_size(self, processor: SequenceProcessor) -> None:
        assert processor.vocab_size == 10

    def test_item_to_idx_known(self, processor: SequenceProcessor) -> None:
        idx = processor.item_to_idx("a")
        assert idx > 0

    def test_item_to_idx_unknown(self, processor: SequenceProcessor) -> None:
        assert processor.item_to_idx("zzz") == 0

    def test_idx_to_item_roundtrip(self, processor: SequenceProcessor) -> None:
        idx = processor.item_to_idx("b")
        assert processor.idx_to_item(idx) == "b"

    def test_pad_sequence_short(self, processor: SequenceProcessor) -> None:
        padded = processor.pad_sequence([1, 2, 3])
        assert len(padded) == 10
        assert padded[:7] == [0] * 7

    def test_pad_sequence_long(self, processor: SequenceProcessor) -> None:
        padded = processor.pad_sequence(list(range(15)))
        assert len(padded) == 10

    def test_to_tensor_shape(self, processor: SequenceProcessor) -> None:
        tensor = processor.to_tensor(["a", "b", "c"])
        assert tensor.shape == (1, 10)

    def test_decode_recommendations(self, processor: SequenceProcessor) -> None:
        idx = processor.item_to_idx("a")
        decoded = processor.decode_recommendations([idx])
        assert decoded == ["a"]

    def test_empty_sequence_encodes(self, processor: SequenceProcessor) -> None:
        tensor = processor.to_tensor([])
        assert tensor.shape == (1, 10)
        assert tensor.sum().item() == 0  # all padding zeros

    def test_single_item_sequence(self, processor: SequenceProcessor) -> None:
        tensor = processor.to_tensor(["a"])
        assert tensor.shape == (1, 10)
        assert tensor[0, -1].item() == processor.item_to_idx("a")

    def test_sequence_with_duplicates(self, processor: SequenceProcessor) -> None:
        tensor = processor.to_tensor(["a", "a", "b"])
        # duplicates are allowed in the sequence
        assert tensor.shape == (1, 10)

    def test_sequence_exceeding_max_length(self, processor: SequenceProcessor) -> None:
        long_seq = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "a", "b", "c"]
        tensor = processor.to_tensor(long_seq)
        assert tensor.shape == (1, 10)


# ---------------------------------------------------------------------------
# DeepSequenceModel tests
# ---------------------------------------------------------------------------


class TestDeepSequenceModel:
    def test_forward_shape(self, model: DeepSequenceModel, processor: SequenceProcessor) -> None:
        tensor = processor.to_tensor(["a", "b", "c"])
        logits = model(tensor)
        assert logits.shape == (1, processor.vocab_size)

    def test_recommend_returns_top_k(
        self, model: DeepSequenceModel, processor: SequenceProcessor
    ) -> None:
        tensor = processor.to_tensor(["a", "b", "c"])
        recs = model.recommend(tensor, top_k=3)
        assert len(recs) == 3

    def test_recommend_excludes_ids(
        self, model: DeepSequenceModel, processor: SequenceProcessor
    ) -> None:
        tensor = processor.to_tensor(["a", "b", "c"])
        exclude = [processor.item_to_idx("a")]
        recs = model.recommend(tensor, top_k=5, exclude_ids=exclude)
        assert exclude[0] not in recs

    def test_recommend_no_duplicates(
        self, model: DeepSequenceModel, processor: SequenceProcessor
    ) -> None:
        tensor = processor.to_tensor(["a", "b"])
        recs = model.recommend(tensor, top_k=5)
        assert len(recs) == len(set(recs))
