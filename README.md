# Cats vs Dogs — End-to-End MLOps Pipeline

BITS Pilani · MLOps (S1-25_AIMLCZG523) · Assignment 2

An end-to-end pipeline: model building + experiment tracking (M1), packaging +
containerization (M2), CI (M3), CD + deployment (M4), and monitoring + post-deploy
performance (M5).

## Chosen stack (and why)

| Concern | Choice | Why |
|---|---|---|
| Framework | PyTorch + a small custom CNN | Trains on CPU in minutes over a subset; the point is the pipeline, not SOTA accuracy |
| Experiment tracking | MLflow (local file store) | No server to run; `mlflow ui` to browse runs |
| Data versioning | DVC + local remote | Meets M1 without cloud storage |
| API | FastAPI | Auto Swagger docs at `/docs` (great for the recording), Pydantic validation |
| CI | GitHub Actions | Zero infra to host, integrates with GHCR |
| Registry | GitHub Container Registry (GHCR) | Free, `GITHUB_TOKEN`-authenticated, no extra account |
| Deploy target | Docker Compose | Lowest-friction target that still shows real CD |
| CD | Self-hosted runner **or** Watchtower | Both auto-update on new images on `main` |
| Monitoring | Prometheus + in-app counters | Request count + latency, `/metrics` endpoint |

> Kubernetes manifests are included under `k8s/` as an alternative M4 target if
> you prefer kind/minikube over Compose.

## Prerequisites

- Python 3.11, Docker (Desktop), Git, and a GitHub account.
- Optional: a Kaggle account + API token to download the real dataset.
- Windows users: run commands in Git Bash / WSL, or use the raw commands the
  `Makefile` wraps.

---

## Quickstart (synthetic data — proves the whole pipeline in ~2 minutes)

You can exercise every stage without downloading anything. Swap in real data later.

```bash
make setup            # install CPU torch + all deps
make data-synth       # generate a tiny labelled dataset
make train            # M1: trains + logs to MLflow, writes models/model.pt
make test             # M3: unit tests
make up               # M4: docker compose up (builds image, serves on :8000)
make smoke            # M4: post-deploy smoke test
make perf             # M5: post-deploy performance report
```

Open the API docs at http://localhost:8000/docs, MLflow at `mlflow ui`
(http://localhost:5000), Prometheus at http://localhost:9090.

---

## Full walkthrough by milestone

### 0. Git init (required for grading — a real history matters)

```bash
git init && git add . && git commit -m "chore: project scaffold"
# create an EMPTY repo on GitHub, then:
git remote add origin https://github.com/<you>/mlops-catsdogs.git
git push -u origin main
```

### M1 — Model development & experiment tracking

1. **Get the data.** Real Kaggle Dogs-vs-Cats:
   ```bash
   pip install kaggle
   # place kaggle.json in ~/.kaggle/ (Account -> Create API Token)
   kaggle competitions download -c dogs-vs-cats -p data/
   unzip -q data/dogs-vs-cats.zip -d data/raw
   unzip -q data/raw/train.zip -d data/raw       # -> data/raw/train/cat.0.jpg ...
   ```
   Then split (subset keeps CPU training fast):
   ```bash
   python scripts/prepare_data.py --raw-dir data/raw/train --subset 2000
   ```
   > No Kaggle? Use `make data-synth` to prove the flow, then substitute real data.

2. **Version data with DVC:**
   ```bash
   dvc init
   mkdir -p /tmp/dvcstore
   dvc remote add -d localremote /tmp/dvcstore
   dvc add data/processed
   dvc push
   git add data/processed.dvc .dvc/config .gitignore
   git commit -m "data: version processed dataset with DVC"
   ```

3. **Train + track:**
   ```bash
   python -m src.train --epochs 3 --batch-size 32
   mlflow ui        # inspect params, metrics, confusion_matrix.png, loss_curve.png
   ```
   Produces `models/model.pt` (`.pt` = the required serialized artifact).

### M2 — Packaging & containerization

```bash
# run the API directly
uvicorn src.app:app --host 0.0.0.0 --port 8000
# verify (health + a prediction)
curl http://localhost:8000/health
curl -F "file=@some_cat.jpg;type=image/jpeg" http://localhost:8000/predict

# containerize
docker build -t catsdogs-api:local .
docker run --rm -p 8000:8000 -v "$PWD/models:/app/models:ro" catsdogs-api:local
```

`requirements.txt` pins all key libraries. Endpoints: `/health` and `/predict`
(returns `label`, `confidence`, per-class `probabilities`).

### M3 — CI (test + build + publish)

- `.github/workflows/ci.yml` runs on every push/PR: checkout → install → `pytest`
  → build image; on `main` it logs into GHCR and **pushes** `:latest` and `:<sha>`.
- Unit tests: `tests/test_data.py` (preprocessing) and `tests/test_inference.py`
  (inference utility). Run locally with `pytest -q`.
- GHCR needs no secrets beyond the automatic `GITHUB_TOKEN`. After the first push,
  make the package public: GitHub → your profile → Packages → the image → Settings.

### M4 — CD & deployment

**Target:** Docker Compose (`docker-compose.yml`) — `api` + `prometheus` (+ optional
`watchtower`). Kubernetes alternative in `k8s/`.

Pick ONE CD mechanism:

- **Self-hosted runner (explicit pipeline):** `.github/workflows/cd.yml` triggers
  after CI succeeds on `main`, runs `docker compose pull && up -d`, waits for
  `/health`, then runs the smoke test (pipeline fails if the smoke test fails).
  Register a runner: repo → Settings → Actions → Runners → New self-hosted runner.
  Switch `api` in `docker-compose.yml` from `build:` to the GHCR `image:`.

- **Watchtower (zero-runner):** set `api` to the GHCR `image:`, keep the
  `watchtower` service; it polls the registry every 30s and auto-restarts `api`
  when a new image is pushed. Run `make smoke` to validate.

**Smoke test:** `scripts/smoke_test.sh` calls `/health` and one `/predict`, exiting
non-zero on failure.

### M5 — Monitoring, logs & post-deploy performance

- Request/response **logging** (metadata only, no image bytes) via middleware in
  `src/app.py`.
- **Metrics:** in-app counters (`app_requests_total`, `app_request_latency_seconds`,
  `app_predictions_total`) plus HTTP metrics from the instrumentator, all at
  `GET /metrics`. Prometheus scrapes it (`monitoring/prometheus.yml`).
- **Post-deploy performance:**
  ```bash
  python scripts/perf_tracking.py --base-url http://localhost:8000
  # or --synthetic --n 20 if you have no test split handy
  ```
  Writes `models/perf_report.json` with accuracy, latency p95, confusion matrix.

---

## Deliverables checklist

- [ ] **Train first** so `models/model.pt` (+ `confusion_matrix.png`,
      `loss_curve.png`, `labels.json`) exist. The PDF requires the trained model
      artifact in the submission.
- [ ] **Zip the working folder** (source, DVC/CI/CD/Docker/deploy configs, AND
      `models/`). Do **NOT** use `git archive` — `models/*.pt` is git-ignored and
      would be dropped. Use:
      ```bash
      zip -r submission.zip mlops-catsdogs \
        -x '*/.venv/*' '*/venv/*' '*/mlruns/*' '*/mlartifacts/*' \
           '*/data/processed/*' '*/data/_synthetic_raw/*' \
           '*/.pytest_cache/*' '*/__pycache__/*' '*/.git/*'
      ```
      Then open the zip and confirm `models/model.pt` is inside before submitting.
- [ ] **Screen recording (<5 min)** showing code-change → deployed prediction.

### Suggested recording shot list (~4 min)

1. (0:00) Repo tour: `src/`, `Dockerfile`, `.github/workflows`, `docker-compose.yml`.
2. (0:30) `mlflow ui` — show a run: params, metrics, confusion matrix.
3. (1:00) Make a **visible code change** (e.g., bump `version` in `src/app.py` or
   tweak a log line) and `git push`.
4. (1:20) GitHub → Actions: CI runs tests, builds, pushes to GHCR (show the package).
5. (2:30) CD updates the running container (self-hosted runner job **or** Watchtower
   log pulling the new image).
6. (3:10) `curl /health` then `curl -F file=@cat.jpg /predict` → live prediction.
7. (3:40) `/metrics` + `perf_tracking.py` output. Done.

---

## What you must supply (cannot be pre-baked)

- The real dataset (Kaggle token) — or accept synthetic for the demo.
- Your GitHub repo + (for the self-hosted CD path) a runner on your machine.
- The screen recording.
- Compute to train (CPU is fine on the 2k subset; a few minutes per epoch).

## Notes

- Dependency versions are pinned for reproducibility. If pip's resolver conflicts
  on your platform, loosen the offending patch pin.
- `torch`/`torchvision` are installed from the PyTorch CPU index (see
  `requirements.txt` header, Dockerfile, CI, and `make setup`).
