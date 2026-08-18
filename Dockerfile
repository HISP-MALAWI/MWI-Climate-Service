# syntax=docker/dockerfile:1

# ---- Build stage -----------------------------------------------------------
FROM python:3.13-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

COPY pyproject.toml uv.lock ./

# Pass --all-extras so server dependencies like uvicorn get installed
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --all-extras

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-extras

# ---- Runtime stage ---------------------------------------------------------
FROM python:3.13-slim-bookworm AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal32 \
    libgeos-c1v5 \
    libproj25 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 ocs

WORKDIR /app

# Copy everything including .venv
COPY --from=builder /app /app

RUN mkdir -p /app/data && chown -R ocs:ocs /app/data

# Set up the virtual environment
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    CLIMATE_SERVICE_CONFIG=/app/climate-service.yaml \
    PYTHONUNBUFFERED=1

USER ocs

EXPOSE 8002

CMD ["python", "-m", "uvicorn", "open_climate_service.main:app", "--host", "0.0.0.0", "--port", "8002"]