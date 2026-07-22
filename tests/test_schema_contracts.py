"""Versioned event and temporal-split contract tests."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from src.training.data import InteractionEvent, next_item_examples, session_sequences, temporal_split


def event(index: int, session: str = "s1") -> InteractionEvent:
    return InteractionEvent(
        event_id=f"event-{index}",
        user_id="user-1",
        session_id=session,
        item_id=f"item-{index % 3}",
        event_type="click",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index),
    )


def test_event_schema_rejects_unknown_event_type() -> None:
    with pytest.raises(ValidationError):
        InteractionEvent(
            event_id="e",
            user_id="u",
            session_id="s",
            item_id="i",
            event_type="share",
            timestamp=datetime.now(timezone.utc),
        )


def test_temporal_split_preserves_order() -> None:
    train, validation, test = temporal_split([event(index) for index in range(20)])
    assert max(item.timestamp for item in train) < min(item.timestamp for item in validation)
    assert max(item.timestamp for item in validation) < min(item.timestamp for item in test)


def test_next_item_examples_are_causal() -> None:
    examples = next_item_examples(session_sequences([event(index) for index in range(4)]))
    assert examples[0] == (["item-0"], "item-1")
