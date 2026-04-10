"""DeepSequence Recommender – FastAPI application entry-point."""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI
from prometheus_client import make_asgi_app

from app.api.routes import init_model, router
from app.core.config import settings
from app.core.data_processor import SequenceProcessor
from app.core.model import DeepSequenceModel

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

_startup_time: float = 0.0

app = FastAPI(
    title="DeepSequence Recommender",
    description="Production-grade deep learning sequence recommender service.",
    version="1.0.0",
)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.include_router(router)


@app.on_event("startup")
def startup_event() -> None:
    """Initialise a default in-memory model for demonstration purposes."""
    global _startup_time
    logger.info("Initialising DeepSequence model…")

    processor = SequenceProcessor(max_length=settings.max_sequence_length)
    # Seed processor with a small catalogue so the model can start immediately.
    demo_items = [[f"item_{i}" for i in range(200)]]
    processor.fit(demo_items)

    model = DeepSequenceModel(
        num_items=processor.vocab_size,
        embedding_dim=settings.embedding_dim,
        hidden_dim=settings.hidden_dim,
        num_layers=settings.num_layers,
    )

    init_model(processor, model)
    _startup_time = time.time()
    logger.info(
        "Model ready. vocab_size=%d embedding_dim=%d hidden_dim=%d",
        processor.vocab_size,
        settings.embedding_dim,
        settings.hidden_dim,
    )


@app.get("/", summary="Root")
def root() -> dict:
    return {"service": "DeepSequence Recommender", "version": "1.0.0"}


@app.get("/health", summary="Application health check", tags=["health"])
def health() -> dict:
    """Return service liveness and basic diagnostics."""
    uptime = round(time.time() - _startup_time, 1) if _startup_time else 0.0
    return {
        "status": "ok",
        "version": "1.0.0",
        "environment": settings.environment,
        "uptime_seconds": uptime,
    }
