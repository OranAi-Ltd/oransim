# ============================================================================
# Oransim — Causal Digital Twin for Marketing at Scale
# ----------------------------------------------------------------------------
# Multi-stage Python image. Stage 1 builds wheels for the runtime deps; stage
# 2 produces a slim image with only what's needed to run the FastAPI backend.
# The LightGBM baseline world model is built-in; install the [ml] extra for
# the PyTorch-dependent Causal Transformer + Causal Neural Hawkes models.
# ============================================================================

ARG PYTHON_VERSION=3.12

# -------- stage 1: builder --------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Build deps needed to compile some wheels (lightgbm / numpy on non-x86)
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY backend backend

# Install into a prefix we can copy to stage 2
RUN pip install --prefix=/install .

# -------- stage 2: runtime --------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend \
    POP_SIZE=100000 \
    SOUL_POOL_N=100 \
    LLM_MODE=mock \
    PORT=8001

# libgomp is needed by LightGBM
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 1000 oransim

COPY --from=builder /install /usr/local

WORKDIR /app
COPY --chown=oransim:oransim backend backend
COPY --chown=oransim:oransim data data
COPY --chown=oransim:oransim frontend frontend
COPY --chown=oransim:oransim pyproject.toml README.md LICENSE NOTICE ./

USER oransim

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:${PORT}/api/health || exit 1

CMD ["python", "-m", "uvicorn", "oransim.api:app", "--host", "0.0.0.0", "--port", "8001"]
