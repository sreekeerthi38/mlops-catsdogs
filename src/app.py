"""M2 + M5: FastAPI inference service.

Endpoints
---------
GET  /            -> service metadata
GET  /health      -> liveness/readiness probe (M2, M4)
POST /predict     -> multipart image upload -> {label, confidence, probabilities} (M2)
GET  /metrics     -> Prometheus metrics (M5), added by the instrumentator

Also implements request/response logging (excluding bodies) and in-app request
count + latency counters (M5).

Run locally:
    uvicorn src.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io
import logging
import time

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from .config import CLASSES, MODEL_PATH
from .data import preprocess_image
from . import inference

# ---- logging (structured-ish, no request/response bodies) -------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
)
logger = logging.getLogger("inference")

# ---- in-app metrics (M5: request count + latency) ---------------------------
REQUEST_COUNT = Counter(
    "app_requests_total", "Total requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Request latency (s)", ["endpoint"]
)
PREDICTIONS = Counter(
    "app_predictions_total", "Predictions by label", ["label"]
)

app = FastAPI(title="Cats vs Dogs Classifier", version="1.0.0")


@app.on_event("startup")
def _load_model() -> None:
    """Load the model once at startup. Service still starts if the model is
    missing so /health stays green; /predict then returns 503 until a model exists."""
    try:
        app.state.model = inference.load_model(MODEL_PATH)
        app.state.model_loaded = True
        logger.info("model loaded from %s", MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        app.state.model = None
        app.state.model_loaded = False
        logger.warning("model NOT loaded (%s): %s", MODEL_PATH, exc)


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    endpoint = request.url.path
    REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
    REQUEST_LATENCY.labels(endpoint).observe(elapsed)
    # log metadata only -- never the uploaded image or response body
    logger.info("%s %s -> %s (%.1f ms)",
                request.method, endpoint, response.status_code, elapsed * 1000)
    return response


@app.get("/")
def root() -> dict:
    return {"service": "cats-vs-dogs", "version": app.version,
            "model_loaded": getattr(app.state, "model_loaded", False),
            "classes": CLASSES}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": getattr(app.state, "model_loaded", False)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not getattr(app.state, "model_loaded", False):
        raise HTTPException(status_code=503, detail="model not loaded")
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")
    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw))
        arr = preprocess_image(image)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}")

    result = inference.predict(app.state.model, arr)
    PREDICTIONS.labels(result["label"]).inc()
    return result


# expose GET /metrics for Prometheus (M5)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")
