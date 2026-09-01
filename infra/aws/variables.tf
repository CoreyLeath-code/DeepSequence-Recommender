variable "aws_region" {
  description = "AWS region for the feedback pipeline."
  type        = string
  default     = "us-east-1"
}

variable "name_prefix" {
  description = "Resource-name prefix."
  type        = string
  default     = "deepsequence"
}

variable "max_receive_count" {
  description = "Number of SQS delivery attempts before a message moves to the DLQ."
  type        = number
  default     = 5

  validation {
    condition     = var.max_receive_count >= 2 && var.max_receive_count <= 20
    error_message = "max_receive_count must be between 2 and 20."
  }
}

variable "log_retention_days" {
  description = "CloudWatch log retention for the ingestion Lambda."
  type        = number
  default     = 14
}

variable "force_destroy_bucket" {
  description = "Allow Terraform destroy to remove non-empty feedback buckets. Keep false outside disposable development environments."
  type        = bool
  default     = false
}
