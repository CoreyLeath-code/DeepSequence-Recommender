"""Model-bundle integrity and ranking-evaluation tests."""

from datetime import datetime, timezone

import pytest
import onnx

from app.core.artifacts import ModelManifest, load_bundle, save_bundle
from app.core.data_processor import SequenceProcessor
from app.core.model import DeepSequenceModel
from src.training.metrics import evaluate_ranking
from src.serving.onnx_exporter import export_bundle_to_onnx


def test_model_bundle_roundtrip_and_checksum(tmp_path) -> None:
    processor = SequenceProcessor(max_length=5).fit([["a", "b", "c"]])
    model = DeepSequenceModel(
        num_items=processor.vocab_size,
        embedding_dim=8,
        hidden_dim=8,
        num_layers=1,
    )
    manifest = ModelManifest(
        model_version="test-v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        architecture_config={
            "embedding_dim": 8,
            "hidden_dim": 8,
            "num_layers": 1,
            "dropout": 0.3,
            "padding_idx": 0,
            "max_sequence_length": 5,
        },
        dataset_id="dataset-test",
        weights_sha256="pending",
        vocabulary_sha256="pending",
    )
    save_bundle(tmp_path, model, processor, manifest)

    restored_processor, restored_model, restored_manifest = load_bundle(tmp_path)
    assert restored_manifest.model_version == "test-v1"
    assert restored_processor.export_vocabulary() == processor.export_vocabulary()
    assert restored_model.recommend(restored_processor.to_tensor(["a"]), top_k=2)

    onnx_path = export_bundle_to_onnx(tmp_path, tmp_path / "model.onnx")
    onnx.checker.check_model(onnx.load(onnx_path))

    (tmp_path / "vocabulary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path)


def test_ranking_metrics_have_known_values() -> None:
    metrics = evaluate_ranking(
        [["a", "b"], ["c", "d"]], ["a", "d"], {"a", "b", "c", "d"}
    )
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr_at_k"] == 0.75
    assert metrics["catalogue_coverage"] == 1.0
