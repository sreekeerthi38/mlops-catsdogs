"""M5: post-deployment model performance tracking.

Sends a batch of labelled images (real from data/processed/test, or synthetic) to
the deployed /predict endpoint, compares predictions to the true labels, and
reports accuracy + a confusion matrix. Saves a JSON report.

Usage:
    python scripts/perf_tracking.py --base-url http://localhost:8000
    python scripts/perf_tracking.py --synthetic --n 20
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

SINGULAR = {"cats": "cat", "dogs": "dog"}


def iter_test_images(test_dir: Path):
    for cls_dir in sorted(test_dir.glob("*")):
        if cls_dir.is_dir():
            true = SINGULAR.get(cls_dir.name, cls_dir.name)
            for img in sorted(cls_dir.glob("*")):
                yield img.read_bytes(), true, img.name


def iter_synthetic(n: int):
    rng = np.random.default_rng(0)
    for i in range(n):
        true = "cat" if i % 2 == 0 else "dog"
        arr = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG")
        yield buf.getvalue(), true, f"synthetic_{i}.jpg"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--test-dir", default="data/processed/test")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n", type=int, default=20, help="synthetic sample size")
    ap.add_argument("--out", default="models/perf_report.json")
    args = ap.parse_args()

    source = iter_synthetic(args.n) if args.synthetic else iter_test_images(Path(args.test_dir))

    total, correct, latencies = 0, 0, []
    confusion = {"cat": {"cat": 0, "dog": 0}, "dog": {"cat": 0, "dog": 0}}
    records = []

    for raw, true, name in source:
        t0 = time.perf_counter()
        resp = requests.post(f"{args.base_url}/predict",
                             files={"file": (name, raw, "image/jpeg")}, timeout=30)
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)
        resp.raise_for_status()
        pred = resp.json()["label"]
        total += 1
        correct += int(pred == true)
        if true in confusion and pred in confusion[true]:
            confusion[true][pred] += 1
        records.append({"file": name, "true": true, "pred": pred, "latency_ms": round(dt, 1)})

    report = {
        "samples": total,
        "accuracy": round(correct / total, 4) if total else None,
        "avg_latency_ms": round(float(np.mean(latencies)), 1) if latencies else None,
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 1) if latencies else None,
        "confusion_matrix": confusion,
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    print(f"[done] full report -> {args.out}")


if __name__ == "__main__":
    main()
