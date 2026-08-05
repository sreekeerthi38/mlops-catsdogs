#!/usr/bin/env bash
# M4 smoke test: verify /health and one /predict call against a running service.
# Exits non-zero on any failure so a CI/CD pipeline can fail the deploy.
#
# Usage: BASE_URL=http://localhost:8000 ./scripts/smoke_test.sh
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
echo "[smoke] target = ${BASE_URL}"

# --- 1) health check ---------------------------------------------------------
echo "[smoke] GET /health"
health="$(curl -fsS "${BASE_URL}/health")"
echo "  -> ${health}"
echo "${health}" | grep -q '"status":"ok"' || { echo "[smoke] FAIL: health not ok"; exit 1; }

# --- 2) generate a tiny test image (no binary asset committed) ---------------
tmp_img="$(mktemp --suffix=.jpg)"
python - "$tmp_img" <<'PY'
import sys
import numpy as np
from PIL import Image
Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)).save(sys.argv[1])
PY

# --- 3) prediction -----------------------------------------------------------
echo "[smoke] POST /predict"
pred="$(curl -fsS -F "file=@${tmp_img};type=image/jpeg" "${BASE_URL}/predict")"
echo "  -> ${pred}"
echo "${pred}" | grep -q '"label"' || { echo "[smoke] FAIL: no label in response"; exit 1; }

rm -f "${tmp_img}"
echo "[smoke] PASS"
