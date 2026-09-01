"""Credential-free structural validation for the AWS/Snowflake integration assets."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(text: str, *needles: str) -> None:
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise SystemExit(f"Missing required integration markers: {missing}")


def main() -> None:
    terraform = (ROOT / "infra" / "aws" / "main.tf").read_text(encoding="utf-8")
    require(
        terraform,
        'resource "aws_sqs_queue" "feedback"',
        'resource "aws_sqs_queue" "feedback_dlq"',
        'resource "aws_lambda_function" "feedback_ingestion"',
        'resource "aws_lambda_event_source_mapping" "feedback"',
        'resource "aws_s3_bucket" "feedback"',
        'function_response_types            = ["ReportBatchItemFailures"]',
        'Action   = ["sqs:SendMessage"]',
    )

    schema = (ROOT / "snowflake" / "sql" / "001_feedback_schema.sql").read_text(
        encoding="utf-8"
    )
    require(
        schema,
        "RAW_FEEDBACK",
        "PAYLOAD VARIANT",
        "RECOMMENDATION_FEEDBACK",
        "MODEL_FEEDBACK_DAILY",
    )

    pipe = (ROOT / "snowflake" / "sql" / "002_snowpipe.template.sql").read_text(
        encoding="utf-8"
    )
    require(
        pipe,
        "CREATE STORAGE INTEGRATION",
        "CREATE STAGE",
        "CREATE PIPE",
        "AUTO_INGEST = TRUE",
        "<AWS_ROLE_ARN>",
        "<S3_BUCKET>",
    )

    evidence = json.loads(
        (ROOT / "evidence" / "aws-snowflake-feedback-results.json").read_text(encoding="utf-8")
    )
    if evidence.get("status") != "not_run":
        raise SystemExit("Cloud evidence must remain not_run until an authorized measured run exists")

    print("AWS Lambda + Snowflake integration assets validated without cloud credentials")


if __name__ == "__main__":
    main()
