"""Versioned interaction contracts and leakage-resistant sequence datasets."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class InteractionEvent(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    event_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    item_id: str = Field(min_length=1, max_length=256)
    event_type: Literal["view", "click", "cart", "purchase"]
    timestamp: datetime


def load_jsonl(path: str | Path) -> list[InteractionEvent]:
    events: list[InteractionEvent] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    events.append(InteractionEvent.model_validate_json(line))
                except ValueError as exc:
                    raise ValueError(f"Invalid event at line {line_number}") from exc
    if not events:
        raise ValueError("Dataset contains no interaction events")
    event_ids = [event.event_id for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("event_id values must be unique")
    return sorted(events, key=lambda event: event.timestamp)


def dataset_id(events: list[InteractionEvent]) -> str:
    import hashlib

    canonical = "\n".join(event.model_dump_json() for event in events)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def temporal_split(
    events: list[InteractionEvent], train_fraction: float = 0.7, validation_fraction: float = 0.15
) -> tuple[list[InteractionEvent], list[InteractionEvent], list[InteractionEvent]]:
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("Split fractions are outside valid bounds")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Training and validation fractions must sum to less than one")
    ordered = sorted(events, key=lambda event: event.timestamp)
    train_end = max(1, int(len(ordered) * train_fraction))
    validation_end = max(train_end + 1, int(len(ordered) * (train_fraction + validation_fraction)))
    return ordered[:train_end], ordered[train_end:validation_end], ordered[validation_end:]


def session_sequences(events: list[InteractionEvent]) -> list[list[str]]:
    sessions: dict[str, list[InteractionEvent]] = defaultdict(list)
    for event in events:
        sessions[event.session_id].append(event)
    return [
        [event.item_id for event in sorted(session, key=lambda item: item.timestamp)]
        for session in sessions.values()
        if len(session) >= 2
    ]


def next_item_examples(sequences: list[list[str]]) -> list[tuple[list[str], str]]:
    return [
        (sequence[:index], sequence[index])
        for sequence in sequences
        for index in range(1, len(sequence))
    ]


def write_jsonl(path: str | Path, events: list[InteractionEvent]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(event.model_dump_json() for event in events) + "\n",
        encoding="utf-8",
    )
