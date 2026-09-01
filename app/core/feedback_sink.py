"""Feedback delivery adapters for local logging and durable AWS SQS ingestion."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class FeedbackPublishError(RuntimeError):
    """Raised when a configured durable feedback sink cannot accept an event."""


class FeedbackSink(Protocol):
    name: str

    def publish(self, event: Mapping[str, Any]) -> None:
        """Publish one privacy-minimized feedback event."""


class LogFeedbackSink:
    name = "structured-log"

    def publish(self, event: Mapping[str, Any]) -> None:
        logger.info("recommendation_feedback=%s", json.dumps(dict(event), sort_keys=True))


class SqsFeedbackSink:
    name = "aws-sqs"

    def __init__(self, queue_url: str, *, region: str, client: Any | None = None) -> None:
        if not queue_url.strip():
            raise ValueError("queue_url must not be empty")
        if not region.strip():
            raise ValueError("region must not be empty")
        self.queue_url = queue_url
        self.region = region
        self._client = client

    def _sqs_client(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client("sqs", region_name=self.region)
        return self._client

    def publish(self, event: Mapping[str, Any]) -> None:
        payload = json.dumps(dict(event), separators=(",", ":"), sort_keys=True)
        try:
            self._sqs_client().send_message(
                QueueUrl=self.queue_url,
                MessageBody=payload,
                MessageAttributes={
                    "schema_version": {"DataType": "Number", "StringValue": "1"},
                    "event_type": {
                        "DataType": "String",
                        "StringValue": str(event.get("event_type", "unknown")),
                    },
                },
            )
        except Exception as exc:
            raise FeedbackPublishError("AWS SQS feedback publish failed") from exc


def build_feedback_sink(*, queue_url: str | None, region: str) -> FeedbackSink:
    if queue_url and queue_url.strip():
        return SqsFeedbackSink(queue_url, region=region)
    return LogFeedbackSink()
