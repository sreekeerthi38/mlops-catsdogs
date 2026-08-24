"""M2 + M5: FastAPI inference service.

Endpoints
---------
GET  /            -> service metadata
GET  /health      -> LIVENESS: 200 while the process is up (M2, M4)
GET  /ready       -> READINESS: 503 until the model is loaded (M4)
POST /predict     -> multipart image upload -> {label, confidence, probabilities} (M2)
GET  /metrics     -> Prometheus metrics (M5), added by the instrumentator

Why two probes: a single /health that returns 200 with no model loaded makes a
Kubernetes readinessProbe pass on a pod that can only answer 503, so the
Service happily routes traffic into a dead replica. Liveness answers "is the
process alive", readiness answers "can it serve predictions".

Also implements request/response logging (metadata only, never image bytes) and
in-app request count + latency counters (M5).

Run locally:
    uvicorn src.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup (replaces the deprecated @on_event).

    The service still starts if the model is missing so /health stays green and
    the container is debuggable; /ready and /predict report 503 until a model
    exists.
    """
    try:
        app.state.model = inference.load_model(MODEL_PATH)
        app.state.model_loaded = True
        app.state.classes = getattr(app.state.model, "classes", CLASSES)
        app.state.arch = getattr(app.state.model, "arch", "unknown")
        logger.info("model loaded from %s (arch=%s)", MODEL_PATH, app.state.arch)
    except Exception as exc:  # noqa: BLE001
        app.state.model = None
        app.state.model_loaded = False
        app.state.classes = CLASSES
        app.state.arch = "unknown"
        logger.warning("model NOT loaded (%s): %s", MODEL_PATH, exc)
    yield


app = FastAPI(title="Cats vs Dogs Classifier", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    endpoint = request.url.path
    REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
    REQUEST_LATENCY.labels(endpoint).observe(elapsed)
    # log metadata only -- never the uploaded image or the response body
    logger.info("%s %s -> %s (%.1f ms)",
                request.method, endpoint, response.status_code, elapsed * 1000)
    return response


@app.get("/")
def root() -> dict:
    return {"service": "cats-vs-dogs", "version": app.version,
            "model_loaded": getattr(app.state, "model_loaded", False),
            "arch": getattr(app.state, "arch", "unknown"),
            "classes": getattr(app.state, "classes", CLASSES)}


@app.get("/health")
def health() -> dict:
    """Liveness: the process is serving."""
    return {"status": "ok", "model_loaded": getattr(app.state, "model_loaded", False)}


@app.get("/ready")
def ready():
    """Readiness: 200 only when predictions can actually be served."""
    if not getattr(app.state, "model_loaded", False):
        return JSONResponse(status_code=503,
                            content={"status": "not_ready", "reason": "model not loaded"})
    return {"status": "ready", "arch": getattr(app.state, "arch", "unknown")}


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
