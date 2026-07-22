# Privacy and feedback policy

The API accepts user behavior that may be personal data. Raw user IDs are not emitted by the
feedback logger; a truncated SHA-256 pseudonym is used. Hashing is minimization, not anonymization.

A production deployment must define lawful purpose, consent, access controls, encryption,
retention, deletion, regional processing, and prohibited sensitive attributes. Keep raw identifiers
in a controlled identity layer, use short-lived pseudonyms for modeling, and propagate deletion
requests to event, feature, training, cache, and model-lineage stores.

Do not log raw click histories or API keys. Collect only fields required for evaluation and
learning, audit downstream exports, and do not reuse feedback for unrelated high-impact decisions.
