"""SQS-triggered Lambda that lands privacy-minimized feedback in an S3 Snowpipe prefix."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import UTC, datetime
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

ALLOWED_EVENT_TYPES = {"impression", "click", "skip", "cart", "purchase", "dislike"}
REQUIRED_FIELDS = {
    "schema_version",
    "event_id",
    "impression_id",
    "anonymous_user_id",
    "item_id",
    "event_type",
    "model_version",
    "occurred_at",
}
PROHIBITED_DIRECT_PII_FIELDS = {
    "user_id",
    "email",
    "phone",
    "phone_number",
    "full_name",
    "first_name",
    "last_name",
    "address",
}


def _default_s3_client() -> Any:
    import boto3

    return boto3.client("s3")


def parse_feedback_message(body: str) -> dict[str, Any]:
    """Validate the versioned event contract and reject direct PII fields."""
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise TypeError("feedback message must decode to a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported feedback schema_version")

    missing = REQUIRED_FIELDS.difference(payload)
    if missing:
        raise ValueError(f"feedback message missing required fields: {sorted(missing)}")

    prohibited = PROHIBITED_DIRECT_PII_FIELDS.intersection(payload)
    if prohibited:
        raise ValueError(f"feedback message contains prohibited direct PII fields: {sorted(prohibited)}")

    if payload["event_type"] not in ALLOWED_EVENT_TYPES:
        raise ValueError("unsupported feedback event_type")
    if not isinstance(payload["anonymous_user_id"], str) or not payload["anonymous_user_id"]:
        raise TypeError("anonymous_user_id must be a non-empty string")
    if not isinstance(payload["event_id"], str) or not payload["event_id"]:
        raise TypeError("event_id must be a non-empty string")

    parsed = datetime.fromisoformat(str(payload["occurred_at"]))
    if parsed.tzinfo is None:
        raise ValueError("occurred_at must include a timezone")
    return payload


def _storage_key(payload: dict[str, Any], message_id: str) -> str:
    occurred = datetime.fromisoformat(str(payload["occurred_at"])).astimezone(UTC)
    safe_message_id = re.sub(r"[^A-Za-z0-9_.-]", "_", message_id)
    return (
        f"feedback/event_date={occurred:%Y-%m-%d}/hour={occurred:%H}/"
        f"message_id={safe_message_id}.json"
    )


def _process_record(record: dict[str, Any], *, bucket: str, s3_client: Any) -> str:
    message_id = record.get("messageId")
    body = record.get("body")
    if not isinstance(message_id, str) or not message_id:
        raise TypeError("SQS record requires messageId")
    if not isinstance(body, str):
        raise TypeError("SQS record body must be a string")

    payload = parse_feedback_message(body)
    key = _storage_key(payload, message_id)
    normalized = json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=normalized.encode("utf-8"),
        ContentType="application/x-ndjson",
        ServerSideEncryption="AES256",
        Metadata={
            "schema-version": "1",
            "event-type": str(payload["event_type"]),
        },
    )
    return key


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, list[dict[str, str]]]:
    """Process SQS records with partial-batch failure semantics for safe retries."""
    del context
    bucket = os.getenv("FEEDBACK_BUCKET", "").strip()
    if not bucket:
        raise RuntimeError("FEEDBACK_BUCKET is required")

    records = event.get("Records", [])
    if not isinstance(records, list):
        raise TypeError("Records must be a list")

    s3_client = _default_s3_client()
    failures: list[dict[str, str]] = []
    for record in records:
        message_id = record.get("messageId", "unknown") if isinstance(record, dict) else "unknown"
        try:
            if not isinstance(record, dict):
                raise TypeError("SQS record must be an object")
            key = _process_record(record, bucket=bucket, s3_client=s3_client)
            logger.info("feedback_landed message_id=%s key=%s", message_id, key)
        except (TypeError, ValueError, BotoCoreError, ClientError) as exc:
            logger.warning(
                "feedback_ingestion_failed message_id=%s error_type=%s",
                message_id,
                type(exc).__name__,
            )
            failures.append({"itemIdentifier": str(message_id)})

    return {"batchItemFailures": failures}
