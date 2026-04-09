"""FastAPI application for valve condition prediction."""

import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

from src.models.predict import predict_cycle
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODELS_DIR = Path("models")
TRAIN_METRICS_PATH = MODELS_DIR / "train_metrics.json"
EVAL_METRICS_PATH = MODELS_DIR / "eval_metrics.json"

app = FastAPI(
    title="Valve Condition Predictor",
    description="Predictive maintenance API for hydraulic system valve condition",
    version="1.0.0",
)

Instrumentator().instrument(app).expose(app)


class PredictionResponse(BaseModel):
    cycle_id: int
    prediction: str
    probability: float


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    version: str
    best_model: str
    cv_f1_macro: float | None
    test_f1_macro: float | None
    test_roc_auc: float | None


_start_time = time.time()


@app.on_event("startup")
async def startup_event() -> None:
    """Pre-load model and features at startup to avoid cold-start latency."""
    logger.info("Starting up: pre-loading model and feature data...")
    from src.models.predict import _load_model, _load_features
    _load_model()
    _load_features()
    logger.info("Startup complete.")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return API health status and uptime."""
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/predict", response_model=PredictionResponse)
def predict(
    cycle_id: int = Query(..., ge=0, description="Zero-based cycle index"),
) -> PredictionResponse:
    """Predict valve condition for a given cycle.

    Args:
        cycle_id: Zero-based index of the production cycle.

    Returns:
        Prediction result with label and probability.
    """
    try:
        result = predict_cycle(cycle_id)
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Prediction failed for cycle {cycle_id}")
        raise HTTPException(status_code=500, detail="Prediction failed")

    return PredictionResponse(**result)


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Return model version and training/evaluation metrics."""
    train_metrics: dict = {}
    eval_metrics: dict = {}

    if TRAIN_METRICS_PATH.exists():
        with open(TRAIN_METRICS_PATH) as f:
            train_metrics = json.load(f)

    if EVAL_METRICS_PATH.exists():
        with open(EVAL_METRICS_PATH) as f:
            eval_metrics = json.load(f)

    return ModelInfoResponse(
        version="v1",
        best_model=train_metrics.get("best_model", "unknown"),
        cv_f1_macro=train_metrics.get("cv_f1_macro"),
        test_f1_macro=eval_metrics.get("f1_macro"),
        test_roc_auc=eval_metrics.get("roc_auc"),
    )
