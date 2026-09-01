"""Validated recommendation routes with bounded inference and fallback behavior."""

from __future__ import annotations

import hashlib
import hmac
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.data_processor import SequenceProcessor
from app.core.feedback_sink import FeedbackPublishError, build_feedback_sink
from app.core.metrics import (
    active_requests,
    cache_hits_total,
    cache_misses_total,
    feedback_events_total,
    model_inference_latency,
    recommendation_latency,
    recommendations_total,
)
from app.core.model import DeepSequenceModel
from app.core.retrieval import ExactEmbeddingRetriever
from app.core.security import api_key_is_valid
from app.core.serving import AdmissionController, RateLimiter, RecommendationCache

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@dataclass
class ModelRuntime:
    processor: SequenceProcessor
    model: DeepSequenceModel
    model_version: str
    trained: bool
    popular_items: list[str]
    retriever: ExactEmbeddingRetriever


_runtime: ModelRuntime | None = None
_admission = AdmissionController(settings.max_concurrent_inferences)
_cache = RecommendationCache(settings.cache_ttl_seconds)
_rate_limiter = RateLimiter(settings.requests_per_minute)
_feedback_sink = build_feedback_sink(
    queue_url=settings.feedback_queue_url,
    region=settings.aws_region,
)


def init_model(
    processor: SequenceProcessor,
    model: DeepSequenceModel,
    *,
    model_version: str,
    trained: bool,
    popular_items: list[str] | None = None,
) -> None:
    global _runtime
    _runtime = ModelRuntime(
        processor=processor,
        model=model,
        model_version=model_version,
        trained=trained,
        popular_items=popular_items or list(processor.export_vocabulary())[: settings.max_top_k],
        retriever=ExactEmbeddingRetriever(settings.retrieval_candidate_pool_size),
    )


class RecommendRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    item_sequence: list[str] = Field(min_length=1, max_length=500)
    top_k: int = Field(default=settings.top_k, ge=1, le=settings.max_top_k)


class RecommendResponse(BaseModel):
    user_id: str
    recommendations: list[str]
    latency_ms: float
    model_version: str
    fallback: bool = False
    cache_hit: bool = False


class FeedbackRequest(BaseModel):
    impression_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    item_id: str = Field(min_length=1, max_length=256)
    event_type: Literal["impression", "click", "skip", "cart", "purchase", "dislike"]
    position: int | None = Field(default=None, ge=0, le=10_000)
    model_version: str = Field(min_length=1, max_length=128)


def _authorize(api_key: str | None) -> None:
    """Enforce configured authentication for production recommendation traffic."""

    if settings.environment.lower() == "production" and not settings.api_key:
        raise HTTPException(status_code=503, detail="Production API key is not configured")
    if not api_key_is_valid(api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")


def _rate_limit_identity(api_key: str | None) -> str:
    """Return a non-secret identity that callers cannot control through request fields."""

    if api_key is None:
        return "development-anonymous"
    return hmac.new(
        api_key.encode("utf-8"), b"deepsequence-rate-limit-identity:v1", hashlib.sha256
    ).hexdigest()


def _response(
    request: RecommendRequest,
    recommendations: list[str],
    started: float,
    *,
    fallback: bool = False,
    cache_hit: bool = False,
) -> RecommendResponse:
    assert _runtime is not None
    return RecommendResponse(
        user_id=request.user_id,
        recommendations=recommendations,
        latency_ms=(time.perf_counter() - started) * 1_000,
        model_version=_runtime.model_version,
        fallback=fallback,
        cache_hit=cache_hit,
    )


@router.post("/", response_model=RecommendResponse, summary="Generate recommendations")
def recommend(
    req: RecommendRequest, x_api_key: str | None = Header(default=None)
) -> RecommendResponse:
    _authorize(x_api_key)
    if not _rate_limiter.allow(_rate_limit_identity(x_api_key)):
        raise HTTPException(status_code=429, detail="Recommendation rate limit exceeded")
    if _runtime is None:
        raise HTTPException(status_code=503, detail="Model not initialised")
    if req.top_k > _runtime.processor.vocab_size:
        raise HTTPException(status_code=422, detail="top_k exceeds catalogue size")

    known_items = [item for item in req.item_sequence if _runtime.processor.item_to_idx(item) != 0]
    if not known_items:
        raise HTTPException(status_code=422, detail="Sequence contains no known catalogue items")
    remaining_items = _runtime.processor.vocab_size - len(set(known_items))
    if req.top_k > remaining_items:
        raise HTTPException(status_code=422, detail="top_k exceeds remaining eligible items")

    started = time.perf_counter()
    cache_key = _cache.key(_runtime.model_version, known_items, req.top_k)
    cached = _cache.get(cache_key)
    if cached is not None:
        cache_hits_total.inc()
        recommendations_total.labels(status="cache_hit").inc()
        return _response(req, cached, started, cache_hit=True)
    cache_misses_total.inc()

    if not _admission.acquire():
        recommendations_total.labels(status="fallback_overload").inc()
        return _response(req, _runtime.popular_items[: req.top_k], started, fallback=True)

    active_requests.inc()
    try:
        tensor = _runtime.processor.to_tensor(known_items)
        infer_started = time.perf_counter()
        excluded_ids = [_runtime.processor.item_to_idx(item) for item in known_items]
        candidates = _runtime.retriever.retrieve(
            _runtime.model,
            tensor,
            top_k=req.top_k,
            exclude_ids=excluded_ids,
        )
        indices = _runtime.model.rank_candidates(
            tensor,
            candidates.candidate_ids,
            top_k=req.top_k,
        )
        inference_ms = (time.perf_counter() - infer_started) * 1_000
        model_inference_latency.observe(inference_ms / 1_000)
        if inference_ms > settings.max_inference_ms:
            recommendations_total.labels(status="fallback_latency").inc()
            return _response(req, _runtime.popular_items[: req.top_k], started, fallback=True)

        decoded = _runtime.processor.decode_recommendations(indices)
        recommendations = [item for item in decoded if item is not None]
        _cache.put(cache_key, recommendations)
        recommendations_total.labels(status="success").inc()
        return _response(req, recommendations, started)
    except (ValueError, RuntimeError, KeyError) as exc:
        recommendations_total.labels(status="error").inc()
        raise HTTPException(status_code=500, detail="Recommendation inference failed") from exc
    finally:
        recommendation_latency.observe(time.perf_counter() - started)
        active_requests.dec()
        _admission.release()


@router.post("/feedback", status_code=202, summary="Capture recommendation feedback")
def feedback(req: FeedbackRequest, x_api_key: str | None = Header(default=None)) -> dict:
    """Emit a privacy-minimized event to SQS when configured, otherwise structured logs."""
    _authorize(x_api_key)
    anonymized_user = hashlib.sha256(req.user_id.encode()).hexdigest()[:16]
    event = req.model_dump(exclude={"user_id"}) | {
        "schema_version": 1,
        "event_id": str(uuid.uuid4()),
        "anonymous_user_id": anonymized_user,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        _feedback_sink.publish(event)
    except FeedbackPublishError as exc:
        raise HTTPException(status_code=503, detail="Feedback pipeline unavailable") from exc
    feedback_events_total.labels(event_type=req.event_type).inc()
    return {
        "accepted": True,
        "impression_id": req.impression_id,
        "delivery": _feedback_sink.name,
    }


@router.get("/health", summary="Model readiness check")
def health() -> dict:
    return {
        "status": "ready" if _runtime is not None else "not_ready",
        "model_loaded": _runtime is not None,
        "trained_model": _runtime.trained if _runtime else False,
        "model_version": _runtime.model_version if _runtime else None,
        "vocab_size": _runtime.processor.vocab_size if _runtime else 0,
    }
