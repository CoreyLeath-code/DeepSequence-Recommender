"""Generate a deterministic interaction dataset for local pipeline validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from src.training.data import InteractionEvent, write_jsonl


def generate_events(sessions: int = 20, events_per_session: int = 6) -> list[InteractionEvent]:
    events = []
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    index = 0
    for session_index in range(sessions):
        for position in range(events_per_session):
            item_index = (session_index + position) % 12
            events.append(
                InteractionEvent(
                    event_id=f"event-{index}",
                    user_id=f"user-{session_index % 5}",
                    session_id=f"session-{session_index}",
                    item_id=f"item-{item_index}",
                    event_type="purchase" if position == events_per_session - 1 else "click",
                    timestamp=started + timedelta(minutes=index),
                )
            )
            index += 1
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/demo-events.jsonl")
    args = parser.parse_args()
    events = generate_events()
    write_jsonl(args.output, events)
    print(f"Wrote {len(events)} events to {args.output}")


if __name__ == "__main__":
    main()
