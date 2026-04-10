"""FastAPI routes for recommendation endpoints."""

from __future__ import annotations

import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.core.data_processor import SequenceProcessor
from app.core.metrics import (
    active_requests,
    model_inference_latency,
    recommendation_latency,
    recommendations_total,
)
from app.core.model import DeepSequenceModel

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# ---------------------------------------------------------------------------
# Shared in-memory model / processor (initialised at startup via app.main)
# ---------------------------------------------------------------------------
_processor: Optional[SequenceProcessor] = None
_model: Optional[DeepSequenceModel] = None


def init_model(processor: SequenceProcessor, model: DeepSequenceModel) -> None:
    global _processor, _model
    _processor = processor
    _model = model


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class RecommendRequest(BaseModel):
    user_id: str
    item_sequence: List[str]
    top_k: int = settings.top_k


class RecommendResponse(BaseModel):
    user_id: str
    recommendations: List[str]
    latency_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/", response_model=RecommendResponse, summary="Generate recommendations")
def recommend(req: RecommendRequest) -> RecommendResponse:
    """Return top-k next-item recommendations for a user interaction sequence."""
    if _model is None or _processor is None:
        raise HTTPException(status_code=503, detail="Model not initialised")

    active_requests.inc()
    wall_start = time.perf_counter()
    try:
        tensor = _processor.to_tensor(req.item_sequence)

        infer_start = time.perf_counter()
        raw_indices = _model.recommend(
            tensor,
            top_k=req.top_k,
            exclude_ids=[_processor.item_to_idx(i) for i in req.item_sequence],
        )
        model_inference_latency.observe(time.perf_counter() - infer_start)

        decoded = _processor.decode_recommendations(raw_indices)
        recommendations = [r for r in decoded if r is not None]

        recommendations_total.labels(status="success").inc()
        return RecommendResponse(
            user_id=req.user_id,
            recommendations=recommendations,
            latency_ms=(time.perf_counter() - wall_start) * 1000,
        )
    except (ValueError, RuntimeError, KeyError) as exc:
        recommendations_total.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        recommendation_latency.observe(time.perf_counter() - wall_start)
        active_requests.dec()


@router.get("/health", summary="Health check")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "vocab_size": _processor.vocab_size if _processor else 0,
    }
