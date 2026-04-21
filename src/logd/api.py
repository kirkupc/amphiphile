"""FastAPI service wrapper.

Exposes `/predict` and `/health`. Model is loaded once at startup (lifespan
event); every request reuses the same in-memory ensemble. This matters for
throughput — loading the baseline ensemble + reliability takes ~1 s, so per-
request loading would destroy batch-1 latency.

Run with: `uv run uvicorn logd.api:app --host 0.0.0.0 --port 8000`
Or via Docker: `docker run -p 8000:8000 amphiphile`
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import os

from logd.inference import LoadedModel, load_model, predict
from logd.utils import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

LOG = get_logger(__name__)

MAX_SMILES_PER_REQUEST = 10_000
MODEL_TYPE = os.environ.get("LOGD_MODEL", "baseline")
MODEL_VERSION = f"{MODEL_TYPE}-v0.1.0"


class PredictRequest(BaseModel):
    smiles: list[str] = Field(..., min_length=1, description="SMILES strings")


class PredictionOut(BaseModel):
    smiles: str
    predicted_logd: float | None
    uncertainty: float | None
    reliable: bool
    error: str | None


class PredictResponse(BaseModel):
    predictions: list[PredictionOut]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded: bool


_model: LoadedModel | None = None


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> "AsyncIterator[None]":
    global _model
    LOG.info("Loading model at startup (type=%s)", MODEL_TYPE)
    try:
        _model = load_model(model_type=MODEL_TYPE)
        LOG.info("Model loaded (type=%s)", MODEL_TYPE)
    except FileNotFoundError as e:
        LOG.warning("Model artifacts missing at startup: %s", e)
        _model = None
    yield
    _model = None


app = FastAPI(
    title="amphiphile",
    description="logD prediction service",
    version="0.1.0",
    lifespan=_lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _model is not None else "degraded",
        model_version=MODEL_VERSION,
        model_loaded=_model is not None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `logd train` to produce artifacts, then restart.",
        )
    if len(req.smiles) > MAX_SMILES_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"Too many SMILES ({len(req.smiles)} > {MAX_SMILES_PER_REQUEST}). "
            "Split into smaller batches.",
        )
    results = predict(req.smiles, model=_model)
    return PredictResponse(
        predictions=[
            PredictionOut(
                smiles=r.smiles,
                predicted_logd=r.predicted_logd,
                uncertainty=r.uncertainty,
                reliable=r.reliable,
                error=r.error,
            )
            for r in results
        ],
        model_version=MODEL_VERSION,
    )
