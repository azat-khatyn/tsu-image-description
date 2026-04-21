FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/src \
    UPLOAD_DIR=/tmp/uploads \
    HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.docker.txt .
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r requirements.docker.txt

COPY . .

RUN mkdir -p /tmp/uploads /mounted_data /models/huggingface

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
