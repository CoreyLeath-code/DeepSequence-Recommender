output "feedback_queue_url" {
  description = "Set this value as FEEDBACK_QUEUE_URL on the recommender API."
  value       = aws_sqs_queue.feedback.url
}

output "feedback_queue_arn" {
  description = "ARN of the durable feedback queue."
  value       = aws_sqs_queue.feedback.arn
}

output "feedback_dlq_arn" {
  description = "ARN of the dead-letter queue for failed feedback records."
  value       = aws_sqs_queue.feedback_dlq.arn
}

output "feedback_bucket" {
  description = "S3 bucket that Snowflake/Snowpipe should ingest from."
  value       = aws_s3_bucket.feedback.bucket
}

output "feedback_publisher_policy_arn" {
  description = "Attach this least-privilege policy to the API workload identity."
  value       = aws_iam_policy.feedback_publisher.arn
}

output "feedback_lambda_name" {
  description = "Feedback ingestion Lambda function name."
  value       = aws_lambda_function.feedback_ingestion.function_name
}
