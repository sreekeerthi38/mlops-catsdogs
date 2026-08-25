# Runbook — Windows / PowerShell

Every command is PowerShell. Each phase ends with a **CHECK** — do not move on
until it passes. If a CHECK fails, the failure is local to that phase, which is
the whole point of the checkpoints.

**Global rule: always type `curl.exe`, never `curl`.** In Windows PowerShell,
`curl` is an alias for `Invoke-WebRequest` and takes different flags entirely.

---

## Phase 0 — Preflight

```powershell
cd C:\path\to\mlops-catsdogs

git --version
git lfs version
python --version          # expect 3.11.x
docker --version
docker compose version
curl.exe --version        # note the .exe

git status --short
git ls-files models/      # EMPTY OUTPUT = the model was never committed
```

**CHECK:** all six tools report a version, and Docker Desktop is running
(whale icon in the system tray, not just installed).

If `python --version` reports 3.12 or newer, create a 3.11 environment —
`torch==2.5.1` has no 3.13 wheels and pip will fail with a confusing
"no matching distribution" error:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If `.\.venv\Scripts\Activate.ps1` is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

That's scoped to the current window only and reverts when you close it.

---

## Phase 1 — Dependencies

```powershell
python -m pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install dvc
```

The CPU index URL is not optional — without it pip pulls the CUDA build, which
is ~2.5GB and bloats your Docker image for no benefit on a CPU-only box.

**CHECK:**

```powershell
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
# expect: 2.5.1+cpu 0.20.1+cpu
```

---

## Phase 2 — Model artifact and Git-LFS

This is the phase that silently breaks deployment. Do it before anything else
touches git.

```powershell
git lfs install
git lfs track "models/*.pt"
git add .gitattributes
git add -f models/model.pt models/confusion_matrix.png models/loss_curve.png models/labels.json
git status --short
```

The `-f` is required only if the OLD `.gitignore` is still in place. The
corrected `.gitignore` no longer ignores `models/`, so plain `git add` works
once you've copied the fixed files in.

**CHECK:**

```powershell
git ls-files models/
# must list model.pt, confusion_matrix.png, loss_curve.png, labels.json

git lfs ls-files
# must show model.pt with an object ID, not "(pointer)"
```

If `git ls-files models/` is still empty, `.gitignore` is winning — confirm you
replaced it with the corrected version.

---

## Phase 3 — Data

**If `data\raw` is intact**, skip to the prepare step.

**If you need to re-download** (~850MB — start this in a second window now):

```powershell
pip install kaggle
# Kaggle -> Account -> Create New API Token -> downloads kaggle.json
mkdir $env:USERPROFILE\.kaggle -Force
Move-Item $env:USERPROFILE\Downloads\kaggle.json $env:USERPROFILE\.kaggle\ -Force

# You MUST accept the competition rules on the website first, or the CLI
# returns a 403 that reads like an auth failure.
# https://www.kaggle.com/c/dogs-vs-cats/rules

kaggle competitions download -c dogs-vs-cats -p data\
Expand-Archive data\dogs-vs-cats.zip -DestinationPath data\raw -Force
Expand-Archive data\raw\train.zip -DestinationPath data\raw -Force
```

Then, whichever branch you took:

```powershell
python scripts\prepare_data.py --raw-dir data\raw\train
```

Split ratios and subset size come from `params.yaml` (2000 images, 80/10/10).

**CHECK:**

```powershell
(Get-ChildItem data\processed\train\cats).Count    # ~800
(Get-ChildItem data\processed\test\cats).Count     # ~100
(Get-ChildItem data\processed\test\dogs).Count     # ~100
```

If test shows 6 per class you are on synthetic data — you passed `--synthetic`
by mistake. Re-run without it.

---

## Phase 4 — DVC (the M1 marks you currently score zero on)

```powershell
dvc init
mkdir C:\dvcstore -Force
dvc remote add -d localremote C:\dvcstore
dvc add data\processed
dvc push
git add data\processed.dvc .dvc\config .dvcignore .gitignore
```

**CHECK:**

```powershell
Test-Path data\processed.dvc      # True
dvc status                        # "Data and pipelines are up to date."
(Get-ChildItem C:\dvcstore -Recurse -File).Count   # > 0
```

`dvc init` fails outside a git repo — if it errors, you're in the wrong folder.

---

## Phase 5 — Train both architectures

```powershell
python -m src.train --arch simple_cnn   --out-name model_simple_cnn.pt
python -m src.train --arch mobilenet_v2 --out-name model.pt
```

The MobileNetV2 run downloads ImageNet weights once (~14MB) and then trains
only the classifier head. Expect it to beat the baseline substantially in the
same 3 epochs.

**CHECK:** the second run prints `test_acc` well above the baseline's 0.63.
Then:

```powershell
mlflow ui
# open http://localhost:5000 - two runs, named simple_cnn and mobilenet_v2
```

Screenshot that comparison view. It's your M1 evidence, and `mlruns/` may not
survive into the submission zip.

Record the numbers in the README results table now, while you have them.

---

## Phase 6 — Demo image and local service

```powershell
Copy-Item data\processed\test\cats\cats_00000.jpg samples\cat.jpg
Copy-Item data\processed\test\dogs\dogs_00000.jpg samples\dog.jpg

pytest -q
docker compose build
docker compose up -d
docker compose ps
```

**CHECK:**

```powershell
curl.exe -fsS http://localhost:8000/health
curl.exe -fsS http://localhost:8000/ready
.\scripts\smoke_test.ps1
```

`/ready` returning 503 means the model didn't load — check with
`docker compose logs api`. The usual cause is an unresolved LFS pointer in
`models\model.pt`; the mount shadows the copy baked into the image.

---

## Phase 7 — Post-deployment performance (M5)

```powershell
python scripts\perf_tracking.py --base-url http://localhost:8000
Get-Content models\perf_report.json | Select-Object -First 12
```

**CHECK:** `samples` reads ~200 (not 12), `source` reads `data/processed/test`
(not the synthetic warning), and `accuracy` roughly matches the `test_acc` from
Phase 5. If those two accuracy numbers disagree, you are serving a different
model than you trained.

---

## Phase 8 — Push via a pull request

Do not push straight to `main`. A PR is a free rehearsal of your entire M3, and
it gives you a green check to show on camera.

```powershell
git checkout -b fix/mlops-pipeline
git add -A
git commit -m "fix: dataset versioning, LFS-resolved model, readiness probe, PR-time image build"
git push -u origin fix/mlops-pipeline
```

Open the PR on GitHub. CI runs: checkout with LFS, install, verify the model
isn't a pointer, pytest, build the image, run the container, smoke-test it.

**CHECK:** the CI check is green on the PR before you merge.

```powershell
git checkout main
git merge fix/mlops-pipeline
git push origin main
```

Merging to `main` triggers the publish job.

**CHECK:** GitHub → Packages → `mlops-catsdogs` shows a fresh `latest` and a
`:<sha>` tag. Then **make the package public** (Package settings → Change
visibility), otherwise the examiner can't pull it and your k8s manifest needs
the `imagePullSecrets` block.

---

## Phase 9 — Self-hosted runner and automatic deploy

Run PowerShell **as Administrator**:

```powershell
mkdir C:\actions-runner; cd C:\actions-runner
# GitHub -> repo -> Settings -> Actions -> Runners -> New self-hosted runner -> Windows x64
# Use the download + hash commands GitHub shows you, then:
.\config.cmd --url https://github.com/sreekeerthi38/mlops-catsdogs --token <TOKEN>
.\svc.ps1 install
.\svc.ps1 start
```

**CHECK:** repo → Settings → Actions → Runners shows the runner as **Idle**
(green). Docker Desktop must be running before the runner service starts.

Then trigger a deploy: Actions → CD → Run workflow.

**CHECK:** the CD job goes green through all five steps, ending with
`[smoke] PASS`.

---

## Phase 10 — The recording (under 5 minutes)

The assignment wants "code change to deployed model prediction," so make a real
change and let the pipeline carry it:

1. **(0:00)** Repo tour — 20 seconds. Show `.dvc`, `models/model.pt` in
   `git lfs ls-files`, the two workflows.
2. **(0:30)** MLflow UI — the two-run comparison.
3. **(1:00)** Make a visible code change: bump `version` in `src/app.py` from
   `1.0.0` to `1.0.1`. Commit and push to `main`.
4. **(1:30)** Actions tab: CI running. Tests pass, image builds, image pushes.
5. **(3:00)** CD fires automatically. Show the pull, the redeploy, `[smoke] PASS`.
6. **(4:00)** `curl.exe http://localhost:8000/` — the version now reads `1.0.1`,
   proving the deployed container is the one your code change produced.
7. **(4:20)** `curl.exe -F "file=@samples\cat.jpg;type=image/jpeg" http://localhost:8000/predict`
   → a real cat, correctly labelled.
8. **(4:40)** Prometheus at http://localhost:9090, query `app_requests_total`.

The version bump in step 3 is what makes step 6 evidence rather than assertion.
Without it you're showing a container that might have been running all along.

---

## Failure lookup

| Symptom | Cause | Fix |
|---|---|---|
| `curl : parameter cannot be found` | You typed `curl`, not `curl.exe` | Use `curl.exe` |
| `/ready` returns 503 | Model didn't load | `git lfs pull`, then `docker compose up -d --force-recreate` |
| `UnpicklingError` in container logs | `model.pt` is an LFS pointer | `git lfs pull`, rebuild image |
| `no matching distribution for torch==2.5.1` | Python 3.12+ | Use Python 3.11 |
| Kaggle CLI 403 | Competition rules not accepted | Accept on the website first |
| `dvc init` fails | Not inside a git repo | `cd` to the repo root |
| CD job stays queued forever | No runner registered, or offline | Settings → Actions → Runners |
| CD fails at pointer check | Runner checked out without LFS | Confirm `lfs: true` in `cd.yml` |
| `docker compose pull` denied | GHCR package is private | Make it public, or `docker login ghcr.io` |
| `Activate.ps1 cannot be loaded` | Execution policy | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
