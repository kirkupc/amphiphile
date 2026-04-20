# Single-stage image for the amphiphile logD prediction service.
#
# Build:   docker build -t amphiphile .
# Run:     docker run -p 8000:8000 -v $(pwd)/models:/app/models amphiphile
# Health:  curl http://localhost:8000/health
#
# Note: the container ships without pre-trained model artifacts. Mount a local
# `models/` directory containing `baseline.joblib` + `reliability.joblib`, or
# let the CI pipeline pre-bake them into the image.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libxrender1 libxext6 libsm6 libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies in a separate layer for caching.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
RUN uv sync --frozen --no-dev

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uv", "run", "uvicorn", "logd.api:app", "--host", "0.0.0.0", "--port", "8000"]
