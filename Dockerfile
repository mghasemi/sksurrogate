# Targets:
#   dev  (default for compose profile "dev"): Python + Node, Vite HMR, bind-mounted source
#   prod: Python only; FastAPI serves the pre-built UI on one port (8013), non-root

# ---- Node deps for the dev server (kept outside the mounted tree) ----
FROM node:22-bookworm AS webdeps
WORKDIR /build/webui
COPY webui/package.json webui/package-lock.json ./
RUN npm ci

# ---- Production UI build ----
FROM webdeps AS webbuild
COPY webui/ ./
RUN npm run build

# ---- Shared Python base ----
FROM python:3.12-slim AS base
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
# Optional extras, e.g. --build-arg EXTRAS="jupyter nbformat"
ARG EXTRAS=""
RUN if [ -n "$EXTRAS" ]; then pip install --no-cache-dir $EXTRAS; fi
WORKDIR /app

# ---- Development image ----
FROM base AS dev
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*
# Baked-in copies are only placeholders; the bind mount is the source of truth.
COPY api/ ./api/
COPY SKSurrogate/ ./SKSurrogate/
COPY webui/ ./webui/
# mltrace reads its version via importlib.metadata, so distribution metadata must
# exist; --no-deps because requirements.txt already installed the dependencies.
COPY setup.py readme.rst License.txt ./
RUN pip install --no-cache-dir --no-deps -e .
COPY --from=webdeps /build/webui/node_modules /opt/webui-node-modules
COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENV SKSURROGATE_MODE=dev
EXPOSE 5173 8013
ENTRYPOINT ["/entrypoint.sh"]

# ---- Production image ----
FROM base AS prod
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 sksurrogate
COPY api/ ./api/
COPY SKSurrogate/ ./SKSurrogate/
COPY setup.py readme.rst License.txt ./
RUN pip install --no-cache-dir --no-deps .
COPY --from=webbuild /build/webui/dist ./webui/dist
COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && mkdir -p /data \
    && chown -R sksurrogate:sksurrogate /data /app
ENV SKSURROGATE_MODE=prod \
    SKSURROGATE_API_HOME=/data \
    SKSURROGATE_WEBUI_DIST=/app/webui/dist
USER sksurrogate
VOLUME /data
EXPOSE 8013
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8013/api/health || exit 1
ENTRYPOINT ["/entrypoint.sh"]
