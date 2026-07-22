"""Serving controls for admission, caching, and deterministic fallback."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass


class AdmissionController:
    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("Concurrency limit must be positive")
        self._semaphore = threading.BoundedSemaphore(limit)

    def acquire(self) -> bool:
        return self._semaphore.acquire(blocking=False)

    def release(self) -> None:
        self._semaphore.release()


@dataclass
class CacheEntry:
    recommendations: list[str]
    expires_at: float


class RecommendationCache:
    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        self._entries: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()

    @staticmethod
    def key(model_version: str, sequence: list[str], top_k: int) -> str:
        payload = json.dumps([model_version, sequence, top_k], separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> list[str] | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.expires_at <= time.monotonic():
                self._entries.pop(key, None)
                return None
            return list(entry.recommendations)

    def put(self, key: str, recommendations: list[str]) -> None:
        with self._lock:
            self._entries[key] = CacheEntry(
                recommendations=list(recommendations),
                expires_at=time.monotonic() + self.ttl_seconds,
            )


class RateLimiter:
    """Per-process fixed-window limiter; use a shared gateway for distributed limits."""

    def __init__(self, requests_per_minute: int) -> None:
        self.limit = requests_per_minute
        self._windows: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()

    def allow(self, identity: str) -> bool:
        now = time.monotonic()
        with self._lock:
            count, window_started = self._windows.get(identity, (0, now))
            if now - window_started >= 60:
                count, window_started = 0, now
            if count >= self.limit:
                return False
            self._windows[identity] = (count + 1, window_started)
            return True
