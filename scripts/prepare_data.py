"""Prepare the dataset into an ImageFolder train/val/test split.

Defaults (split ratios, subset size, image size) come from params.yaml so the
run is reproducible from a single tracked config file; CLI flags override.

Two modes:

1) Real Kaggle data (Dogs vs Cats) -- what the assignment requires:
     python scripts/prepare_data.py --raw-dir data/raw/train --subset 2000
   Handles both common layouts:
     (a) flat folder of files named  cat.0.jpg / dog.0.jpg
     (b) class subfolders            cats/*.jpg , dogs/*.jpg  (or Cat/ Dog/)

2) Synthetic data -- smoke-testing the plumbing only. NEVER report metrics
   from a synthetic run as assignment results:
     python scripts/prepare_data.py --synthetic --per-class 60

Output: data/processed/{train,val,test}/{cats,dogs}
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PROCESSED_DIR, load_params  # noqa: E402
from src.data import split_dataset                 # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _label_from_name(path: Path) -> str | None:
    name = path.stem.lower()
    if name.startswith("cat"):
        return "cats"
    if name.startswith("dog"):
        return "dogs"
    return None


def collect_pairs(raw_dir: Path) -> list[tuple[Path, str]]:
    """Return list of (image_path, class_folder_name)."""
    raw_dir = Path(raw_dir)
    pairs: list[tuple[Path, str]] = []

    # layout (b): class subfolders
    for cls_dir in raw_dir.iterdir() if raw_dir.exists() else []:
        if cls_dir.is_dir():
            low = cls_dir.name.lower()
            cls = "cats" if low.startswith("cat") else "dogs" if low.startswith("dog") else None
            if cls:
                for f in cls_dir.rglob("*"):
                    if f.suffix.lower() in IMG_EXT:
                        pairs.append((f, cls))

    # layout (a): flat files named cat.* / dog.*
    if not pairs:
        for f in raw_dir.rglob("*"):
            if f.suffix.lower() in IMG_EXT:
                cls = _label_from_name(f)
                if cls:
                    pairs.append((f, cls))
    return pairs


def make_synthetic(out_root: Path, per_class: int, size: int = 224, seed: int = 0) -> Path:
    """Create tiny colored-noise images so the pipeline runs without Kaggle.

    The addition is done in int16 before clipping. Adding a Python int to a
    uint8 array wraps at 256 *before* np.clip can act, which turned bright
    tinted pixels into near-black ones and destroyed the class signal.
    """
    rng = np.random.default_rng(seed)
    tmp = out_root / "_synthetic_raw"
    # Same orphan trap as split_dataset: filenames are index-based, so a rerun
    # with a smaller --per-class leaves the previous run's extra images behind
    # and the "new" dataset is silently the old, larger one.
    if tmp.exists():
        shutil.rmtree(tmp)
    for cls, tint in (("cats", (200, 60, 60)), ("dogs", (60, 60, 200))):
        d = tmp / cls
        d.mkdir(parents=True, exist_ok=True)
        tint_arr = np.array(tint, dtype=np.int16)
        for i in range(per_class):
            base = rng.integers(0, 60, (size, size, 3), dtype=np.uint8).astype(np.int16)
            img = np.clip(base + tint_arr, 0, 255).astype(np.uint8)
            Image.fromarray(img).save(d / f"{cls[:-1]}.{i}.jpg")
    return tmp


def main() -> None:
    params = load_params().get("data", {})
    split = tuple(params.get("split", [0.8, 0.1, 0.1]))
    img_size = int(params.get("img_size", 224))

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=str, help="folder with extracted Kaggle images")
    ap.add_argument("--out-dir", type=str, default=str(PROCESSED_DIR))
    ap.add_argument("--subset", type=int, default=int(params.get("subset", 0)),
                    help="cap total images (0 = all); default from params.yaml")
    ap.add_argument("--synthetic", action="store_true", help="generate synthetic data")
    ap.add_argument("--per-class", type=int, default=60, help="synthetic images per class")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    if args.synthetic:
        raw_dir = make_synthetic(out_dir.parent, args.per_class, size=img_size)
        print(f"[synthetic] generated -> {raw_dir}")
        print("[synthetic] WARNING: plumbing check only -- do not report these metrics.")
    elif args.raw_dir:
        raw_dir = Path(args.raw_dir)
    else:
        ap.error("provide --raw-dir <path> or --synthetic")

    pairs = collect_pairs(raw_dir)
    if not pairs:
        raise SystemExit(f"No labelled images found under {raw_dir}. "
                         "Check the folder layout (see docstring).")

    if args.subset and len(pairs) > args.subset:
        # keep class balance when subsetting
        cats = [p for p in pairs if p[1] == "cats"][: args.subset // 2]
        dogs = [p for p in pairs if p[1] == "dogs"][: args.subset // 2]
        pairs = cats + dogs

    counts = split_dataset(pairs, out_dir, ratios=split, seed=args.seed, clean=True)
    print(f"[done] {sum(counts.values())} images -> {out_dir}  "
          f"splits={counts}  ratios={split}")


if __name__ == "__main__":
    main()
