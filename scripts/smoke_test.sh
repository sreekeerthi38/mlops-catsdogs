#!/usr/bin/env bash
# M4 smoke test: verify /health, /ready and one real /predict call.
# Exits non-zero on any failure so a CI/CD pipeline can fail the deploy.
#
# Usage: BASE_URL=http://localhost:8000 ./scripts/smoke_test.sh
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
SAMPLE="${SAMPLE:-samples/cat.jpg}"
echo "[smoke] target = ${BASE_URL}"

# --- 1) liveness -------------------------------------------------------------
echo "[smoke] GET /health"
health="$(curl -fsS "${BASE_URL}/health")"
echo "  -> ${health}"
echo "${health}" | grep -q '"status":"ok"' || { echo "[smoke] FAIL: health not ok"; exit 1; }

# --- 2) readiness (503 until the model is loaded) ----------------------------
echo "[smoke] GET /ready"
ready="$(curl -fsS "${BASE_URL}/ready")" || { echo "[smoke] FAIL: service not ready (model missing?)"; exit 1; }
echo "  -> ${ready}"

# --- 3) prediction on a real image where available ---------------------------
# A real cat/dog photo proves the model works. Random noise only proves the
# HTTP wiring works -- fine as a fallback in CI, useless as a demo.
if [[ -f "${SAMPLE}" ]]; then
  img="${SAMPLE}"
  cleanup=false
  echo "[smoke] using sample image ${SAMPLE}"
else
  echo "[smoke] WARNING: ${SAMPLE} not found -- falling back to a generated image"
  img="$(mktemp --suffix=.jpg)"
  cleanup=true
  python - "$img" <<'PY'
import sys
import numpy as np
from PIL import Image
Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)).save(sys.argv[1])
PY
fi

echo "[smoke] POST /predict"
pred="$(curl -fsS -F "file=@${img};type=image/jpeg" "${BASE_URL}/predict")"
echo "  -> ${pred}"
echo "${pred}" | grep -q '"label"' || { echo "[smoke] FAIL: no label in response"; exit 1; }

[[ "${cleanup}" == "true" ]] && rm -f "${img}"
echo "[smoke] PASS"
