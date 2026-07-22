from app.core.security import api_key_is_valid
from app.core.serving import RateLimiter, RecommendationCache


def test_rate_limiter_fails_closed_after_budget() -> None:
    limiter = RateLimiter(2)
    assert limiter.allow("user") is True
    assert limiter.allow("user") is True
    assert limiter.allow("user") is False


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
