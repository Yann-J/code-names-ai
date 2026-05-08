# syntax=docker/dockerfile:1.7

# ---------- pwa-builder: produce the Vite PWA bundle -----------------------
FROM node:22-alpine AS pwa-builder

WORKDIR /build/web-ui

COPY web-ui/package.json web-ui/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --no-audit --no-fund

COPY web-ui/ ./
ENV NODE_ENV=production
# vite.config.ts writes to ../src/codenames_ai/web/static/pwa, which under
# /build/web-ui resolves to /build/src/codenames_ai/web/static/pwa.
RUN npx tsc -b && npx vite build


# ---------- py-builder: install python deps with uv ------------------------
FROM python:3.12-slim-bookworm AS py-builder

COPY --from=ghcr.io/astral-sh/uv:0.5.14 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-install-project --no-dev

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev


# ---------- runtime: slim image with venv, source, API, and PWA ------------
FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CODENAMES_AI_CACHE_DIR=/cache \
    CODENAMES_AI_CONFIG=

RUN useradd -r -u 10001 -m -d /home/app -s /usr/sbin/nologin app \
    && mkdir -p /cache \
    && chown -R app:app /cache

WORKDIR /app

COPY --from=py-builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src ./src
COPY --chown=app:app config ./config
COPY --chown=app:app data ./data
# Drop in the freshly-built PWA. Build context excludes the committed copy
# (see .dockerignore) so this is the only thing landing in static/pwa/.
COPY --from=pwa-builder --chown=app:app \
     /build/src/codenames_ai/web/static/pwa/ \
     /app/src/codenames_ai/web/static/pwa/

USER app

EXPOSE 8000
VOLUME ["/cache"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/', timeout=3).status == 200 else 1)"

# Shell form so ${CODENAMES_AI_CONFIG} expands at container start; when unset
# or empty, no --config flag is appended and the server uses its built-in default.
CMD codenames-ai serve --host 0.0.0.0 --port 8000 ${CODENAMES_AI_CONFIG:+--config "$CODENAMES_AI_CONFIG"}
