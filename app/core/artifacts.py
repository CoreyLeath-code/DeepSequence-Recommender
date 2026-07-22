"""Versioned, checksummed model bundles shared by training and serving."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field

from app.core.data_processor import SequenceProcessor
from app.core.model import DeepSequenceModel


class ModelManifest(BaseModel):
    model_version: str
    created_at: str
    architecture: str = "bidirectional-lstm-attention"
    architecture_config: dict[str, Any]
    metrics: dict[str, float] = Field(default_factory=dict)
    dataset_id: str
    weights_sha256: str
    vocabulary_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_bundle(
    directory: str | Path,
    model: DeepSequenceModel,
    processor: SequenceProcessor,
    manifest: ModelManifest,
) -> ModelManifest:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    weights_path = target / "model.pt"
    vocabulary_path = target / "vocabulary.json"
    torch.save(model.state_dict(), weights_path)
    vocabulary_path.write_text(
        json.dumps(processor.export_vocabulary(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    completed = manifest.model_copy(
        update={
            "weights_sha256": _sha256(weights_path),
            "vocabulary_sha256": _sha256(vocabulary_path),
        }
    )
    (target / "manifest.json").write_text(
        completed.model_dump_json(indent=2), encoding="utf-8"
    )
    return completed


def load_bundle(
    directory: str | Path,
) -> tuple[SequenceProcessor, DeepSequenceModel, ModelManifest]:
    target = Path(directory)
    manifest = ModelManifest.model_validate_json(
        (target / "manifest.json").read_text(encoding="utf-8")
    )
    weights_path = target / "model.pt"
    vocabulary_path = target / "vocabulary.json"
    if _sha256(weights_path) != manifest.weights_sha256:
        raise ValueError("Model weights checksum does not match the manifest")
    if _sha256(vocabulary_path) != manifest.vocabulary_sha256:
        raise ValueError("Vocabulary checksum does not match the manifest")

    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    config = dict(manifest.architecture_config)
    max_length = int(config.pop("max_sequence_length"))
    processor = SequenceProcessor.from_vocabulary(vocabulary, max_length=max_length)
    model = DeepSequenceModel(num_items=processor.vocab_size, **config)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return processor, model, manifest
