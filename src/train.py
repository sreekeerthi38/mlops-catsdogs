"""M1: train the baseline CNN, log everything to MLflow, and serialize the model.

Run:
    python -m src.train --epochs 3 --batch-size 32

MLflow writes to a local ./mlruns store by default (no server needed). View with:
    mlflow ui
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend for CI / servers
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch  # noqa: F401  (registers the mlflow.pytorch flavor)
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from .config import CLASSES, MODELS_DIR, PROCESSED_DIR, load_params
from .data import make_dataloaders
from .model import SimpleCNN


def evaluate(model, loader, device) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * x.size(0)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    return loss_sum / max(total, 1), correct / max(total, 1), np.array(y_true), np.array(y_pred)


def _save_confusion_matrix(y_true, y_pred, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def _save_loss_curve(history: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(history["train_loss"], label="train_loss", marker="o")
    ax.plot(history["val_loss"], label="val_loss", marker="o")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Loss Curves"); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def main() -> None:
    params = load_params().get("train", {})
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=params.get("epochs", 3))
    ap.add_argument("--batch-size", type=int, default=params.get("batch_size", 32))
    ap.add_argument("--lr", type=float, default=params.get("lr", 1e-3))
    ap.add_argument("--data-dir", type=str, default=str(PROCESSED_DIR))
    ap.add_argument("--experiment", type=str, default=params.get("experiment", "cats-vs-dogs"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    train_loader, val_loader, test_loader = make_dataloaders(
        Path(args.data_dir), batch_size=args.batch_size, num_workers=0
    )

    model = SimpleCNN(num_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run() as run:
        mlflow.log_params({
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "model": "SimpleCNN", "img_size": 224,
            "optimizer": "Adam", "device": str(device),
        })

        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        for epoch in range(args.epochs):
            model.train()
            running = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                running += loss.item() * x.size(0)
            train_loss = running / len(train_loader.dataset)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, device)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            print(f"[epoch {epoch+1}/{args.epochs}] train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # ---- final test evaluation ----
        test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, device)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_acc", test_acc)
        print(f"[test] loss={test_loss:.4f} acc={test_acc:.4f}")

        # ---- artifacts: plots + serialized model ----
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cm_path = MODELS_DIR / "confusion_matrix.png"
        loss_path = MODELS_DIR / "loss_curve.png"
        _save_confusion_matrix(y_true, y_pred, cm_path)
        _save_loss_curve(history, loss_path)

        model_path = MODELS_DIR / "model.pt"
        torch.save({"state_dict": model.state_dict(), "classes": CLASSES,
                    "arch": "SimpleCNN"}, model_path)
        (MODELS_DIR / "labels.json").write_text(json.dumps({"classes": CLASSES}, indent=2))

        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(loss_path))
        mlflow.log_artifact(str(model_path))
        # log the model in MLflow's native format too (nice for the model registry)
        try:
            mlflow.pytorch.log_model(model, artifact_path="model")
        except Exception as exc:  # non-fatal; local artifact above is the source of truth
            print(f"[warn] mlflow.pytorch.log_model skipped: {exc}")

        print(f"[done] run_id={run.info.run_id}  model saved -> {model_path}")


if __name__ == "__main__":
    main()
