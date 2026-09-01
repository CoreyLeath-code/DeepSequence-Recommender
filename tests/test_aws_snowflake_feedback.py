"""Credential-free contract tests for the AWS Lambda + Snowflake feedback path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.feedback_sink import LogFeedbackSink, SqsFeedbackSink, build_feedback_sink
from serverless.feedback_ingestion import handler

ROOT = Path(__file__).resolve().parents[1]


class FakeSqsClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def send_message(self, **kwargs: Any) -> dict[str, str]:
        self.calls.append(kwargs)
        return {"MessageId": "sqs-1"}


class FakeS3Client:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def put_object(self, **kwargs: Any) -> dict[str, str]:
        self.calls.append(kwargs)
        return {"ETag": "etag"}


def valid_feedback_event() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event_id": "evt-1",
        "impression_id": "imp-1",
        "anonymous_user_id": "f1e2d3c4b5a69788",
        "item_id": "item-42",
        "event_type": "click",
        "position": 2,
        "model_version": "model-v1",
        "occurred_at": "2026-09-01T15:30:00+00:00",
    }


def test_feedback_sink_defaults_to_local_structured_logging() -> None:
    sink = build_feedback_sink(queue_url=None, region="us-east-1")
    assert isinstance(sink, LogFeedbackSink)
    assert sink.name == "structured-log"


def test_sqs_feedback_sink_publishes_versioned_event() -> None:
    client = FakeSqsClient()
    sink = SqsFeedbackSink(
        "https://sqs.us-east-1.amazonaws.com/123456789012/deepsequence-feedback",
        region="us-east-1",
        client=client,
    )
    event = valid_feedback_event()

    sink.publish(event)

    assert len(client.calls) == 1
    call = client.calls[0]
    body = json.loads(call["MessageBody"])
    assert body == event
    assert "user_id" not in body
    assert call["MessageAttributes"]["schema_version"]["StringValue"] == "1"
    assert call["MessageAttributes"]["event_type"]["StringValue"] == "click"


def test_lambda_lands_valid_event_in_partitioned_s3(monkeypatch) -> None:
    client = FakeS3Client()
    monkeypatch.setenv("FEEDBACK_BUCKET", "deepsequence-feedback-test")
    monkeypatch.setattr(handler, "_default_s3_client", lambda: client)
    event = {
        "Records": [
            {
                "messageId": "message-123",
                "body": json.dumps(valid_feedback_event()),
            }
        ]
    }

    result = handler.lambda_handler(event, object())

    assert result == {"batchItemFailures": []}
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["Bucket"] == "deepsequence-feedback-test"
    assert (
        call["Key"]
        == "feedback/event_date=2026-09-01/hour=15/message_id=message-123.json"
    )
    landed = json.loads(call["Body"].decode("utf-8"))
    assert landed["event_id"] == "evt-1"
    assert "user_id" not in landed
    assert call["ServerSideEncryption"] == "AES256"


def test_lambda_rejects_direct_pii_without_writing(monkeypatch) -> None:
    client = FakeS3Client()
    monkeypatch.setenv("FEEDBACK_BUCKET", "deepsequence-feedback-test")
    monkeypatch.setattr(handler, "_default_s3_client", lambda: client)
    unsafe = valid_feedback_event() | {"user_id": "raw-private-id"}
    event = {"Records": [{"messageId": "unsafe-1", "body": json.dumps(unsafe)}]}

    result = handler.lambda_handler(event, object())

    assert result == {"batchItemFailures": [{"itemIdentifier": "unsafe-1"}]}
    assert client.calls == []


def test_snowflake_assets_define_raw_and_curated_contracts() -> None:
    schema = (ROOT / "snowflake" / "sql" / "001_feedback_schema.sql").read_text(
        encoding="utf-8"
    )
    assert "PAYLOAD VARIANT NOT NULL" in schema
    assert "RECOMMENDATION_FEEDBACK" in schema
    assert "MODEL_FEEDBACK_DAILY" in schema
    assert "ANONYMOUS_USER_ID" in schema

    pipe = (ROOT / "snowflake" / "sql" / "002_snowpipe.template.sql").read_text(
        encoding="utf-8"
    )
    assert "CREATE STORAGE INTEGRATION" in pipe
    assert "CREATE PIPE" in pipe
    assert "AUTO_INGEST = TRUE" in pipe
    assert "METADATA$FILENAME" in pipe


def test_cloud_evidence_starts_not_run() -> None:
    evidence = json.loads(
        (ROOT / "evidence" / "aws-snowflake-feedback-results.json").read_text(encoding="utf-8")
    )
    assert evidence["status"] == "not_run"
    assert evidence["aws"]["deployment_status"] == "not_run"
    assert evidence["snowflake"]["snowpipe_status"] == "not_run"
