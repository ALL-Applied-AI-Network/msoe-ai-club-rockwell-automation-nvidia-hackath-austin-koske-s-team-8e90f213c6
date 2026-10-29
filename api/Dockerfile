FROM ghcr.io/astral-sh/uv:0.12.21 AS uv
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app/api
COPY api/pyproject.toml api/uv.lock ./
RUN uv sync --frozen --no-dev --python /usr/local/bin/python
# requirements-model.txt (CPU) or requirements-model-cuda.txt (GB10/NVIDIA GPU, see compose.gpu.yaml).
ARG MODEL_REQUIREMENTS=requirements-model.txt
COPY api/requirements-model.txt api/requirements-model-cuda.txt ./
RUN uv pip install --python /app/api/.venv/bin/python -r ${MODEL_REQUIREMENTS}
COPY api/ /app/api/
COPY replay/__init__.py replay/config.py replay/geometry.py replay/measurement.py replay/diagnostics.py /app/replay/
COPY integration/runtime.py integration/measurement_qa.py /app/integration/
COPY setup/vision/temporal_pool.py setup/vision/train_pair_masks.py /app/setup/vision/
ENV PYTHONPATH=/app:/app/api PYTHONUNBUFFERED=1 APP__DATA_DIR=/data APP__MODEL_ASSETS=/models APP__DEVICE=cpu YOLO_CONFIG_DIR=/tmp/ultralytics
RUN useradd --create-home --uid 10001 appuser && mkdir -p /data && chown appuser /data
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s CMD ["/app/api/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]
CMD ["/app/api/.venv/bin/python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
