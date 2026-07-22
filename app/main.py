"""DeepSequence Recommender FastAPI application entry point."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from prometheus_client import make_asgi_app

from app.api.routes import init_model, router
from app.core.artifacts import load_bundle
from app.core.config import settings
from app.core.data_processor import SequenceProcessor
from app.core.model import DeepSequenceModel

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)
_startup_time: float = 0.0


def initialize_model() -> None:
    """Load a verified model bundle or an explicit development-only fallback."""
    global _startup_time
    bundle_path = Path(settings.model_bundle_path)
    if (bundle_path / "manifest.json").is_file():
        processor, model, manifest = load_bundle(bundle_path)
        init_model(
            processor,
            model,
            model_version=manifest.model_version,
            trained=True,
        )
    elif settings.environment.lower() == "production":
        raise RuntimeError(f"Production requires a verified model bundle at {bundle_path}")
    else:
        logger.warning("No trained bundle found; using deterministic development model")
        torch.manual_seed(7)
        processor = SequenceProcessor(max_length=settings.max_sequence_length)
        processor.fit([[f"item_{index}" for index in range(200)]])
        model = DeepSequenceModel(
            num_items=processor.vocab_size,
            embedding_dim=settings.embedding_dim,
            hidden_dim=settings.hidden_dim,
            num_layers=settings.num_layers,
        ).eval()
        init_model(
            processor,
            model,
            model_version="development-untrained",
            trained=False,
        )
    _startup_time = time.time()
    logger.info(
        "Model ready. vocab_size=%d environment=%s bundle=%s",
        processor.vocab_size,
        settings.environment,
        bundle_path,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    initialize_model()
    yield


app = FastAPI(
    title="DeepSequence Recommender",
    description="Versioned, evaluated deep sequence recommendation service.",
    version="1.1.0",
    lifespan=lifespan,
)
app.mount("/metrics", make_asgi_app())
app.include_router(router)


@app.get("/", summary="Root")
def root() -> dict:
    return {"service": "DeepSequence Recommender", "version": "1.1.0"}


@app.get("/health", summary="Application liveness check", tags=["health"])
def health() -> dict:
    uptime = round(time.time() - _startup_time, 1) if _startup_time else 0.0
    return {
        "status": "ok",
        "version": "1.1.0",
        "environment": settings.environment,
        "uptime_seconds": uptime,
    }
