import pytest
from fastapi import HTTPException

from app.api.routes import _authorize, _rate_limit_identity
from app.core.config import settings
from app.core.security import api_key_is_valid
from app.core.serving import RateLimiter, RecommendationCache


def test_rate_limiter_fails_closed_after_budget() -> None:
    limiter = RateLimiter(2)
    assert limiter.allow("user") is True
    assert limiter.allow("user") is True
    assert limiter.allow("user") is False


def test_rate_limit_identity_is_credential_scoped_and_non_secret() -> None:
    limiter = RateLimiter(1)
    identity = _rate_limit_identity("shared-credential")

    assert identity != "shared-credential"
    assert identity == _rate_limit_identity("shared-credential")
    assert limiter.allow(identity) is True
    # A caller can change user_id, but cannot change this credential-derived bucket.
    assert limiter.allow(_rate_limit_identity("shared-credential")) is False


def test_cache_is_model_version_aware() -> None:
    cache = RecommendationCache(ttl_seconds=30)
    first = cache.key("v1", ["a"], 2)
    second = cache.key("v2", ["a"], 2)
    cache.put(first, ["b"])
    assert cache.get(first) == ["b"]
    assert cache.get(second) is None


def test_api_key_validation() -> None:
    assert api_key_is_valid(None, None) is True
    assert api_key_is_valid("correct", "correct") is True
    assert api_key_is_valid("wrong", "correct") is False


def test_production_without_api_key_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "api_key", None)

    with pytest.raises(HTTPException) as error:
        _authorize(None)

    assert error.value.status_code == 503


def test_development_without_api_key_remains_open(monkeypatch) -> None:
    monkeypatch.setattr(settings, "environment", "development")
    monkeypatch.setattr(settings, "api_key", None)

    _authorize(None)
