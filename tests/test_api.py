"""M3: API-level tests for the inference service (uses the httpx-backed TestClient).

Runs against an app instance with NO model on disk, which is exactly the state
that used to slip past the old single /health probe.
"""
import pytest

pytest.importorskip("torch")
from fastapi.testclient import TestClient  # noqa: E402

from src.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module=None):
    # point MODEL_PATH at a nonexistent file so startup takes the "no model" branch
    import src.config as config
    missing = tmp_path_factory.mktemp("nomodel") / "model.pt"
    original, config.MODEL_PATH = config.MODEL_PATH, missing
    import src.app as app_module
    app_module.MODEL_PATH = missing
    with TestClient(app) as c:
        yield c
    config.MODEL_PATH = original


def test_health_is_live_even_without_a_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_reports_503_without_a_model(client):
    r = client.get("/ready")
    assert r.status_code == 503


def test_predict_rejects_when_model_missing(client):
    r = client.post("/predict", files={"file": ("x.jpg", b"not-an-image", "image/jpeg")})
    assert r.status_code == 503


def test_metrics_endpoint_exposed(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "app_requests_total" in r.text
