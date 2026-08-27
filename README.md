# Cats vs Dogs End-to-End MLOps Pipeline

**BITS Pilani Â· MLOps (S1-25_AIMLCZG523) Â· Assignment 2**
Sreenivasulu Remuri

Repository: `https://github.com/sreekeerthi38/mlops-catsdogs`
Container image: `ghcr.io/sreekeerthi38/mlops-catsdogs:latest`

An end-to-end pipeline covering model building and experiment tracking (M1),
packaging and containerization (M2), CI (M3), CD and deployment (M4), and
monitoring plus post-deployment performance tracking (M5).


## Results

| Architecture | Trainable params | Epochs | Test images | Test accuracy |
|---|---|---|---|---|
| `simple_cnn` (from scratch) | 98,178 | 3 | 200 | 0.64 |
| `mobilenet_v2` (frozen backbone) | 2,562 | 3 | 200 | **0.97** |

| Post-deployment (M5) | Value |
|---|---|
| Model | `mobilenet_v2` |
| Samples | 200 |
| Accuracy | 0.97 |
| Mean latency | 29.6 ms |
| p95 latency | 36.8 ms |

Source artifacts: `models/confusion_matrix.png`, `models/loss_curve.png`,
`models/labels.json`, `models/perf_report.json`, and the MLflow run store.

## Stack

| Concern | Choice | Why |
|---|---|---|
| Framework | PyTorch â€” `SimpleCNN` baseline + frozen MobileNetV2 | Both train on CPU in minutes over a 2000-image subset; two runs give MLflow a real comparison |
| Experiment tracking | MLflow (local file store) | No server to run; `mlflow ui` browses runs, params, metrics, artifacts |
| Data versioning | DVC with a local remote | Versions `data/processed` without cloud storage |
| Model versioning | Git-LFS on `models/*.pt` | The Docker build copies the model from the checkout, so it must be in the repo |
| API | FastAPI | Swagger at `/docs`, Pydantic validation, async multipart upload |
| CI | GitHub Actions | Builds and tests on every push and PR; publishes from `main` |
| Registry | GitHub Container Registry | Free, authenticated by the automatic `GITHUB_TOKEN` |
| Deploy target | Docker Compose (Kubernetes manifests in `k8s/` as an alternative) | Lowest-friction target that still exercises a genuine registry pull |
| CD | Watchtower polling GHCR (Actions variant retained in `cd.yml`) | Redeploys automatically without a self-hosted runner on the deploy host |
| Monitoring | Prometheus + in-app counters | Request count, latency histogram, per-label prediction counter at `/metrics` |

## Layout

```
src/           config, data (preprocess + split), model (2 archs), train, inference, app
scripts/       prepare_data.py, smoke_test.sh, perf_tracking.py
tests/         test_data.py, test_inference.py, test_api.py
.github/       workflows/ci.yml, workflows/cd.yml
k8s/           deployment.yaml (readinessProbe -> /ready), service.yaml
monitoring/    prometheus.yml
models/        model.pt (LFS), confusion_matrix.png, loss_curve.png, labels.json, perf_report.json
samples/       cat.jpg used by the smoke test and the demo
```

## M1 â€” Model development & experiment tracking

**Data.** Kaggle Dogs-vs-Cats, preprocessed to 224Ã—224 RGB, split 80/10/10 with
per-class stratification (`src/data.py::split_dataset`). Training augmentation:
horizontal flip, Â±15Â° rotation, colour jitter. Validation and test see resize
and ImageNet normalization only.

```bash
kaggle competitions download -c dogs-vs-cats -p data/
unzip -q data/dogs-vs-cats.zip -d data/raw && unzip -q data/raw/train.zip -d data/raw
python scripts/prepare_data.py --raw-dir data/raw/train
```

Split ratios, subset size and image size come from `params.yaml`, so a run is
reproducible from the committed config alone.

**Versioning.** Git for code. DVC for the preprocessed dataset:

```bash
dvc init && dvc remote add -d localremote /tmp/dvcstore
dvc add data/processed && dvc push
```

Git-LFS carries `models/*.pt` â€” the Docker build copies the model out of the
checkout, so an ignored or pointer-only model produces an image that answers
503 to every prediction.

**Training and tracking.**

```bash
python -m src.train --arch simple_cnn   --out-name model_simple_cnn.pt
python -m src.train --arch mobilenet_v2 --out-name model.pt
mlflow ui   # params, metrics per epoch, confusion matrix, loss curve, model artifact
```

Each run logs architecture, epochs, batch size, learning rate, optimizer,
device, trainable parameter count and dataset sizes; per-epoch train loss, val
loss and val accuracy; final test loss and accuracy; and the confusion matrix,
loss curve and serialized `.pt` as artifacts.

## M2 â€” Packaging & containerization

`/health` (liveness), `/ready` (readiness â€” 503 until the model loads),
`/predict` (multipart image â†’ label, confidence, per-class probabilities),
`/metrics`, and `/` for service metadata.

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
curl -F "file=@samples/cat.jpg;type=image/jpeg" http://localhost:8000/predict

docker build -t catsdogs-api:local .
docker run --rm -p 8000:8000 catsdogs-api:local
```

`requirements.txt` pins every key library. `torch`/`torchvision` install from
the PyTorch CPU index (see the file header, the Dockerfile, CI and `make setup`)
to avoid pulling CUDA wheels into a CPU-only image.

## M3 â€” CI

`.github/workflows/ci.yml`, on **every push and pull request**: checkout with
LFS â†’ install pinned dependencies â†’ verify `models/model.pt` is a real
checkpoint rather than an LFS pointer â†’ `pytest` â†’ build the Docker image â†’
run the container and smoke-test it. On `main` only, a second job logs into
GHCR and pushes `:latest` and `:<sha>`.

Tests: `tests/test_data.py` (preprocessing â€” shape, dtype, greyscale handling,
normalization range, type rejection), `tests/test_inference.py` (model forward
pass and the `predict` utility), `tests/test_api.py` (liveness, readiness under
a missing model, 503 on predict, metrics exposure).

## M4 â€” CD & deployment

**Target:** Docker Compose â€” `api` (from GHCR) plus `prometheus`.
`.github/workflows/cd.yml` runs on a self-hosted runner after CI succeeds on
`main`: checkout with LFS â†’ verify the model is not a pointer â†’
`docker compose pull && docker compose up -d` â†’ poll `/ready` â†’
`scripts/smoke_test.sh`. A failing smoke test fails the pipeline.

Kubernetes alternative in `k8s/`: Deployment with `readinessProbe` on `/ready`
and `livenessProbe` on `/health`, plus a NodePort Service on 30080.

## M5 â€” Monitoring, logs & post-deployment performance

Middleware logs method, path, status and latency for every request and **never**
logs image bytes or response bodies. Metrics at `/metrics`:
`app_requests_total{method,endpoint,status}`,
`app_request_latency_seconds{endpoint}`, `app_predictions_total{label}`, plus
the standard HTTP metrics from `prometheus-fastapi-instrumentator`. Prometheus
scrapes the `api` service every 15s (`monitoring/prometheus.yml`,
http://localhost:9090).

```bash
python scripts/perf_tracking.py --base-url http://localhost:8000
```

Replays the held-out test split against the deployed endpoint and writes
`models/perf_report.json` with generation time, data source, served
architecture, accuracy, mean and p95 latency, a confusion matrix, and
per-request records.

## Reproducing from scratch

```bash
make setup            # CPU torch + pinned deps
make data             # RAW=/path/to/data/raw/train
make train            # writes models/model.pt + MLflow run
make test             # pytest
make up               # docker compose up
make smoke            # post-deploy smoke test
make perf             # post-deploy performance report
```



## CD mechanism

Automatic redeployment is handled by **Watchtower**, which polls GHCR every 30
seconds and recreates `catsdogs-api` whenever `:latest` changes. A merge to
`main` therefore deploys unattended: CI publishes the image, Watchtower picks
it up within 30s.

`.github/workflows/cd.yml` implements the GitHub Actions variant (LFS checkout,
model-pointer verification, `docker compose pull`, readiness wait, smoke test
with pipeline failure on error) and is retained for reference. It requires a
self-hosted runner on the deploy host, since GitHub-hosted runners cannot reach a
service on localhost; it is set to `workflow_dispatch` only.

Note: the upstream `containrrr/watchtower` image is unmaintained and fails
against current Docker API versions (`client version 1.25 is too old`). The
maintained fork `nickfedor/watchtower` is used instead.
