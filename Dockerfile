# M2: containerize the FastAPI inference service.
FROM python:3.11-slim

# System deps kept minimal; slim image + CPU-only torch keeps this reasonable.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/model.pt

WORKDIR /app

# Install CPU-only torch first (from the PyTorch index), then the rest.
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# App code + trained model artifact
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

# Basic container healthcheck hitting the app's /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
