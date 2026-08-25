# samples/

Put ONE real cat photo here as `cat.jpg` (and optionally `dog.jpg`) before
recording the demo. Copy one straight out of the held-out test split so the
image is genuinely unseen by the model:

```bash
cp data/processed/test/cats/cats_00000.jpg samples/cat.jpg
cp data/processed/test/dogs/dogs_00000.jpg samples/dog.jpg
```

`scripts/smoke_test.sh` uses `samples/cat.jpg` when present and falls back to a
generated noise image only so CI can run without committed binaries. The old
repo-root `test.jpg` was 224x224 of random RGB noise — predicting on it during
the screen recording demonstrates nothing about the model.
