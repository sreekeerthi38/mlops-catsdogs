# Convenience targets. On Windows either install `make` (choco install make) or
# run the underlying commands shown in each recipe by hand.
.PHONY: help setup data data-synth train train-baseline mlflow api test smoke perf \
        docker-build docker-run up down dvc-init clean

IMAGE ?= catsdogs-api:local
BASE_URL ?= http://localhost:8000

help:
	@echo "setup        install deps (CPU torch + requirements)"
	@echo "data         split real Kaggle data (RAW=/path SUBSET=2000)"
	@echo "data-synth   generate synthetic data (no download)"
	@echo "train        train model + log to MLflow"
	@echo "mlflow       open MLflow UI (http://localhost:5000)"
	@echo "api          run FastAPI locally"
	@echo "test         run pytest"
	@echo "docker-build build the Docker image"
	@echo "docker-run   run the image on :8000"
	@echo "up / down    docker compose up/down"
	@echo "smoke        run post-deploy smoke test"
	@echo "perf         run post-deploy performance tracking"

setup:
	python -m pip install --upgrade pip
	pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt -r requirements-dev.txt

data:
	python scripts/prepare_data.py --raw-dir "$(RAW)"

data-synth:
	python scripts/prepare_data.py --synthetic --per-class 60

train:
	python -m src.train --arch mobilenet_v2 --out-name model.pt

train-baseline:
	python -m src.train --arch simple_cnn --out-name model_simple_cnn.pt

mlflow:
	mlflow ui

api:
	uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 -v $(PWD)/models:/app/models:ro $(IMAGE)

up:
	docker compose up -d --build

down:
	docker compose down

smoke:
	chmod +x scripts/smoke_test.sh && BASE_URL=$(BASE_URL) ./scripts/smoke_test.sh

perf:
	python scripts/perf_tracking.py --base-url $(BASE_URL)

dvc-init:
	dvc init
	mkdir -p /tmp/dvcstore
	dvc remote add -d localremote /tmp/dvcstore
	dvc add data/processed
	git add data/processed.dvc .gitignore .dvc/config

clean:
	rm -rf __pycache__ .pytest_cache mlruns mlartifacts data/_synthetic_raw
