terraform {
  required_version = ">= 1.6.0"

  required_providers {
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

locals {
  name              = "${var.name_prefix}-feedback"
  lambda_source_dir = "${path.module}/../../serverless/feedback_ingestion"
}

data "archive_file" "feedback_lambda" {
  type        = "zip"
  source_dir  = local.lambda_source_dir
  output_path = "${path.module}/feedback_ingestion.zip"
}

resource "aws_s3_bucket" "feedback" {
  bucket_prefix = "${var.name_prefix}-feedback-"
  force_destroy = var.force_destroy_bucket
}

resource "aws_s3_bucket_public_access_block" "feedback" {
  bucket = aws_s3_bucket.feedback.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "feedback" {
  bucket = aws_s3_bucket.feedback.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "feedback" {
  bucket = aws_s3_bucket.feedback.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_sqs_queue" "feedback_dlq" {
  name                      = "${local.name}-dlq"
  message_retention_seconds = 1209600
  sqs_managed_sse_enabled   = true
}

resource "aws_sqs_queue" "feedback" {
  name                       = local.name
  message_retention_seconds  = 345600
  visibility_timeout_seconds = 180
  sqs_managed_sse_enabled    = true

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.feedback_dlq.arn
    maxReceiveCount     = var.max_receive_count
  })
}

resource "aws_iam_role" "feedback_lambda" {
  name = "${local.name}-lambda"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.feedback_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "feedback_lambda" {
  name = "${local.name}-runtime"
  role = aws_iam_role.feedback_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes"
        ]
        Resource = aws_sqs_queue.feedback.arn
      },
      {
        Effect = "Allow"
        Action = ["s3:PutObject"]
        Resource = "${aws_s3_bucket.feedback.arn}/feedback/*"
      }
    ]
  })
}

resource "aws_cloudwatch_log_group" "feedback_lambda" {
  name              = "/aws/lambda/${local.name}-ingestion"
  retention_in_days = var.log_retention_days
}

resource "aws_lambda_function" "feedback_ingestion" {
  function_name = "${local.name}-ingestion"
  role          = aws_iam_role.feedback_lambda.arn
  handler       = "handler.lambda_handler"
  runtime       = "python3.11"
  timeout       = 30
  memory_size   = 256

  filename         = data.archive_file.feedback_lambda.output_path
  source_code_hash = data.archive_file.feedback_lambda.output_base64sha256

  environment {
    variables = {
      FEEDBACK_BUCKET = aws_s3_bucket.feedback.bucket
    }
  }

  depends_on = [
    aws_cloudwatch_log_group.feedback_lambda,
    aws_iam_role_policy.feedback_lambda,
    aws_iam_role_policy_attachment.lambda_basic,
  ]
}

resource "aws_lambda_event_source_mapping" "feedback" {
  event_source_arn                   = aws_sqs_queue.feedback.arn
  function_name                      = aws_lambda_function.feedback_ingestion.arn
  batch_size                         = 10
  maximum_batching_window_in_seconds = 5
  function_response_types            = ["ReportBatchItemFailures"]
}

resource "aws_iam_policy" "feedback_publisher" {
  name        = "${local.name}-publisher"
  description = "Least-privilege policy for the recommender API to publish feedback to SQS."

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["sqs:SendMessage"]
      Resource = aws_sqs_queue.feedback.arn
    }]
  })
}
