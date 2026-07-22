# Operations runbook

## Readiness failure

1. Inspect startup logs for a missing manifest, checksum mismatch, vocabulary validation, or strict
   state-dict errors.
2. Verify the complete immutable bundle is mounted at `MODEL_BUNDLE_PATH`.
3. Roll back to the previous image and bundle pair; never mix bundle files.

## Elevated latency or fallback rate

1. Check active requests, inference latency, cache hit rate, CPU, memory, and throttling.
2. Stop rollout growth and reduce admission capacity if saturation is increasing.
3. Roll back when the canary exceeds its error, fallback, or latency guardrail.

## Quality regression

Segment by model version, known-item rate, cold-start cohort, and catalogue segment. Validate
feature/vocabulary compatibility and input drift, then restore the previous bundle while retaining
impression and outcome evidence.

## Corrupted artifact

Checksum failures are non-retryable deployment errors. Quarantine the candidate, regenerate it
from the recorded dataset/configuration, and investigate storage or promotion integrity.
