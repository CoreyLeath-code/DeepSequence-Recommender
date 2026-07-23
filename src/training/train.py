"""Train, evaluate, and package the production sequence recommender."""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from app.core.artifacts import ModelManifest, save_bundle
from app.core.data_processor import SequenceProcessor
from app.core.model import DeepSequenceModel
from src.training.baselines import PopularityBaseline
from src.training.data import (
    dataset_id,
    load_jsonl,
    next_item_examples,
    session_sequences,
    temporal_split,
)
from src.training.metrics import evaluate_ranking


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    train_sequences: list[list[str]],
    *,
    embedding_dim: int = 32,
    hidden_dim: int = 64,
    num_layers: int = 1,
    max_sequence_length: int = 50,
    epochs: int = 3,
    learning_rate: float = 1e-3,
    seed: int = 7,
) -> tuple[SequenceProcessor, DeepSequenceModel]:
    seed_everything(seed)
    processor = SequenceProcessor(max_length=max_sequence_length).fit(train_sequences)
    if processor.vocab_size < 2:
        raise ValueError("Training requires at least two catalogue items")
    examples = next_item_examples(train_sequences)
    model = DeepSequenceModel(
        num_items=processor.vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        for history, target in examples:
            target_index = processor.item_to_idx(target)
            if target_index == 0:
                continue
            optimizer.zero_grad()
            logits = model(processor.to_tensor(history))
            loss = F.cross_entropy(logits, torch.tensor([target_index]))
            loss.backward()
            optimizer.step()
    model.eval()
    return processor, model


def model_predictions(
    model: DeepSequenceModel,
    processor: SequenceProcessor,
    examples: list[tuple[list[str], str]],
    top_k: int,
) -> tuple[list[list[str]], list[str]]:
    predictions: list[list[str]] = []
    targets: list[str] = []
    for history, target in examples:
        if processor.item_to_idx(target) == 0:
            continue
        known_history = [item for item in history if processor.item_to_idx(item) != 0]
        if not known_history:
            continue
        eligible_top_k = min(top_k, processor.vocab_size - len(set(known_history)))
        if eligible_top_k < 1:
            continue
        indices = model.recommend(
            processor.to_tensor(known_history),
            top_k=eligible_top_k,
            exclude_ids=[processor.item_to_idx(item) for item in known_history],
        )
        predictions.append(
            [item for item in processor.decode_recommendations(indices) if item is not None]
        )
        targets.append(target)
    return predictions, targets


def run_training(
    dataset_path: str | Path,
    output_directory: str | Path,
    *,
    epochs: int = 3,
    top_k: int = 10,
    seed: int = 7,
) -> dict[str, object]:
    events = load_jsonl(dataset_path)
    train_events, validation_events, test_events = temporal_split(events)
    train_sequences = session_sequences(train_events)
    validation_examples = next_item_examples(session_sequences(validation_events))
    test_examples = next_item_examples(session_sequences(test_events))
    processor, model = train_model(train_sequences, epochs=epochs, seed=seed)

    baseline = PopularityBaseline().fit(train_sequences)
    catalogue = set(processor.export_vocabulary())
    eligible_validation = [
        (history, target)
        for history, target in validation_examples
        if processor.item_to_idx(target) != 0
        and any(processor.item_to_idx(item) != 0 for item in history)
    ]
    validation_predictions, validation_targets = model_predictions(
        model, processor, eligible_validation, top_k
    )
    if not validation_targets:
        raise ValueError("Validation split has no evaluable known-item examples")
    neural_metrics = evaluate_ranking(validation_predictions, validation_targets, catalogue)
    baseline_predictions = [
        baseline.recommend(history, top_k) for history, _target in eligible_validation
    ]
    baseline_metrics = evaluate_ranking(baseline_predictions, validation_targets, catalogue)

    test_predictions, test_targets = model_predictions(model, processor, test_examples, top_k)
    test_metrics = (
        evaluate_ranking(test_predictions, test_targets, catalogue) if test_targets else {}
    )
    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    config = {
        "embedding_dim": 32,
        "hidden_dim": 64,
        "num_layers": 1,
        "dropout": 0.3,
        "padding_idx": 0,
        "max_sequence_length": 50,
    }
    manifest = ModelManifest(
        model_version=version,
        created_at=datetime.now(UTC).isoformat(),
        architecture_config=config,
        metrics={f"validation_{key}": value for key, value in neural_metrics.items()},
        dataset_id=dataset_id(events),
        weights_sha256="pending",
        vocabulary_sha256="pending",
    )
    completed = save_bundle(output_directory, model, processor, manifest)
    report = {
        "model_version": version,
        "dataset_id": completed.dataset_id,
        "validation": neural_metrics,
        "popularity_baseline": baseline_metrics,
        "test": test_metrics,
        "promotion_eligible": neural_metrics["ndcg_at_k"] >= baseline_metrics["ndcg_at_k"],
    }
    (Path(output_directory) / "evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="models/candidate")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(
        json.dumps(
            run_training(
                args.dataset, args.output, epochs=args.epochs, top_k=args.top_k, seed=args.seed
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
