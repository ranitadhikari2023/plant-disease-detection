# =============================================================================
# Stage 1: Build & Dependency Resolution
# =============================================================================
FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Install uv for ultra-fast dependency compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /bin/uv

# Create isolated production virtual environment without TensorFlow
ENV UV_PROJECT_ENVIRONMENT="/app/.venv"
RUN uv venv /app/.venv --python 3.12

# Install production serving dependencies only
RUN uv pip install --python /app/.venv/bin/python --no-cache \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "starlette>=0.38.0" \
    "python-multipart>=0.0.12" \
    "pydantic>=2.8.0" \
    "pydantic-settings>=2.4.0" \
    "pillow>=10.4.0" \
    "numpy>=1.26.0" \
    "onnxruntime>=1.19.0"

# =============================================================================
# Stage 2: Minimal Non-Root Production Runtime
# =============================================================================
FROM python:3.12-slim AS runtime

LABEL maintainer="LeafScan Engineering Team" \
    description="Production Plant Pathology Screening API (LeafScan V3)" \
    version="3.0.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    LEAFSCAN_APP_ENV="production" \
    LEAFSCAN_HOST="0.0.0.0" \
    LEAFSCAN_PORT=8000 \
    LEAFSCAN_MODEL_BUNDLE_PATH="/app/models/bundles/v3"

WORKDIR /app

# Security: Create non-root unprivileged service user (UID 10001)
RUN groupadd -g 10001 leafscan && \
    useradd -u 10001 -g leafscan -s /usr/sbin/nologin -M leafscan

# Copy virtual environment from builder stage
COPY --from=builder --chown=leafscan:leafscan /app/.venv /app/.venv

# Copy application source, frontend, and production model bundle
COPY --chown=leafscan:leafscan src/ /app/src/
COPY --chown=leafscan:leafscan frontend/ /app/frontend/
COPY --chown=leafscan:leafscan models/bundles/v3/ /app/models/bundles/v3/
COPY --chown=leafscan:leafscan pyproject.toml README.md /app/

# Switch to non-root user
USER leafscan:leafscan

EXPOSE 8000

# Container Healthcheck verifying readiness probe
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2)" || exit 1

ENTRYPOINT ["uvicorn", "leafscan.api.app:app"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--no-access-log"]
