# Fixes applied to the Assignment 2 submission

Every item below was a defect in the reviewed zip. Grouped by why it mattered.

## 1. Contradictory / stale evidence (this was the worst one)

* **Deleted `models/perf_report.json`.** It reported `accuracy: 1.0` over 12
  samples and was written at 14:41; `model.pt` and `confusion_matrix.png` were
  written at 15:11. The 12 samples (`cats_00000`–`cats_00005`, `dogs_00000`–
  `dogs_00005`) are exactly what `--synthetic --per-class 60` produces at an
  80/10/10 split, so the report measured a throwaway synthetic model while the
  confusion matrix in the same folder showed 200 real test images at 63%.
  **You must regenerate this file** — see the run order below.
* **`scripts/perf_tracking.py`** now stamps `generated_at`, `source`,
  `served_arch` and `service_version` into the report, and prints a loud warning
  in `--synthetic` mode. A stale or synthetic report is now self-identifying.
* **Deleted repo-root `test.jpg`** — 224×224 of random RGB noise. Replaced with
  `samples/` and instructions to drop in a real held-out cat photo.

## 2. Deployment would have failed even after the config fixes

* **`.gitignore` no longer ignores `models/`.** It previously ignored
  `models/*.pt` while `.gitattributes` LFS-tracked the same pattern — Git skips
  ignored files before LFS ever sees them, so the model was never committed, the
  CI `COPY models/ ./models/` baked an empty directory, and `/predict` would
  have returned 503 forever.
* **`lfs: true` added to every `actions/checkout@v4`** in both workflows.
  Checkout does *not* fetch LFS objects by default; it writes a ~130-byte
  pointer file. `torch.load()` on a pointer dies with an unpickling error.
* **Pointer guards added** to CI and CD: both fail fast with a readable message
  if `models/model.pt` is missing or is an unresolved LFS pointer.
* **`docker-compose.yml`** keeps the `./models:/app/models:ro` mount but now
  documents that the mount *shadows* the model baked into the image, so the host
  directory must be LFS-resolved.

## 3. Rubric-literal gaps

* **M1 — dataset versioning.** `.dvcignore` restored and DVC init documented.
  Git-LFS covers the *model*; M1.1 asks for DVC/Git-LFS on the *dataset*. Run
  the `dvc init` block below — it is five minutes of work for marks you
  currently score zero on.
* **M3.2 — build on every push *and* merge request.** CI now builds the image in
  the `test` job on PRs (`push: false, load: true`) and additionally runs the
  container and smoke-tests it. Publishing to GHCR still only happens on `main`.
* **M4 — readiness.** `k8s/deployment.yaml` points its `readinessProbe` at the
  new `/ready` endpoint, uses the real GHCR image instead of the
  `ghcr.io/OWNER/REPO` placeholder, adds `imagePullSecrets` (delete it once the
  package is public) and Prometheus scrape annotations.

## 4. Model quality

* **`src/model.py`** gains `build_mobilenet()` and a `build_model(arch=...)`
  factory. The frozen-backbone MobileNetV2 trains ~2.5k parameters, so a CPU
  epoch costs about the same as the baseline CNN. `SimpleCNN` is kept —
  training both gives MLflow a genuine comparison, which is stronger M1
  evidence than a single run.
* **`src/train.py`** takes `--arch`, logs it plus `trainable_params`,
  `train_images` and `test_images`, names the MLflow run after the architecture,
  and writes `arch` into both the checkpoint and `labels.json`.
* **`src/inference.py`** reads `arch` and `classes` out of the checkpoint and
  rebuilds the right network with `pretrained=False` (no ImageNet download
  inside the container). Legacy checkpoints saving `arch: "SimpleCNN"` still
  load via an alias map.

## 5. Correctness bugs

* **`scripts/prepare_data.py` → `make_synthetic()`**: the tint was added to a
  `uint8` array, wrapping at 256 *before* `np.clip` could act — a pixel of 58
  became 2, not 255. Now done in `int16` then clipped.
* **`src/data.py` → `split_dataset()`**: added `clean=True`, which wipes the
  output directory first. Without it, re-running with a smaller `--subset` left
  orphaned files from the previous run (filenames are index-based, so only the
  first N were overwritten) and silently mixed two datasets. This is the most
  likely cause of the 12-image test split.
* **`params.yaml` is now actually read.** `prepare_data.py` takes `split`,
  `subset` and `img_size` from it; `train.py` takes `arch`.
* **`src/inference.py`**: `torch.load(..., weights_only=True)` — kills the
  `FutureWarning` and the arbitrary-code-execution surface.
* **`src/app.py`**: `@app.on_event("startup")` (deprecated in FastAPI 0.115)
  replaced with a `lifespan` context manager.
* **`src/app.py`**: split `/health` (liveness, always 200) from `/ready`
  (readiness, 503 with no model). A single `/health` returning 200 with
  `model_loaded: false` made the k8s readinessProbe pass on a pod that could
  only answer 503.
* **`tests/test_api.py`** added — four API-level tests that finally use the
  `httpx` dev dependency, including the no-model case that the old probe missed.

## Run order (do these in sequence)

```bash
# 0. Resolve LFS locally so models/model.pt is a real checkpoint
git lfs install
git lfs track "models/*.pt"
git add -f models/model.pt models/confusion_matrix.png models/loss_curve.png models/labels.json

# 1. Real Kaggle data (NOT --synthetic)
python scripts/prepare_data.py --raw-dir data/raw/train    # subset comes from params.yaml

# 2. DVC — the M1 marks you are currently not scoring
dvc init
mkdir -p /tmp/dvcstore
dvc remote add -d localremote /tmp/dvcstore
dvc add data/processed
dvc push
git add data/processed.dvc .dvc/config .dvcignore

# 3. Train both architectures so MLflow holds a comparison
python -m src.train --arch simple_cnn   --out-name model_simple_cnn.pt
python -m src.train --arch mobilenet_v2 --out-name model.pt

# 4. Real demo image
cp data/processed/test/cats/cats_00000.jpg samples/cat.jpg

# 5. Verify locally before pushing
pytest -q
docker compose build && docker compose up -d
BASE_URL=http://localhost:8000 bash scripts/smoke_test.sh
python scripts/perf_tracking.py --base-url http://localhost:8000   # regenerates perf_report.json

# 6. Push — this is the part I cannot do for you
git add -A
git commit -m "fix: real dataset versioning, LFS-resolved model, readiness probe, PR-time image build"
git push origin main
```

After the push: GitHub → Actions → confirm CI is green, then Packages → make
`mlops-catsdogs` **public** (otherwise the grader cannot pull it and your k8s
manifest needs the `imagePullSecrets` block).
