"""Minimal service authentication primitives."""

from __future__ import annotations

import secrets


def api_key_is_valid(provided: str | None, configured: str | None) -> bool:
    """Allow open development when unconfigured; compare keys in constant time otherwise."""
    if configured is None:
        return True
    return provided is not None and secrets.compare_digest(provided, configured)
