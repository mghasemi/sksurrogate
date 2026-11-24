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

# ---- Shared TeX and Python runtime ----
FROM python:3.12-slim-bookworm AS texbase
# The branded style uses LuaLaTeX, TikZ, tabularray and system OpenType fonts.
# Fira Code and Latin Modern Math are lightweight fallbacks for Fira Mono and
# Libertinus Math when the latter are not installed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       texlive-luatex texlive-latex-extra fonts-inter fonts-firacode fontconfig \
    && rm -rf /var/lib/apt/lists/*

# ---- Shared Python base ----
FROM texbase AS base
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
# LightGBM (and XGBoost's optional libgomp path) links `libgomp.so.1` at import
# time via ctypes; python:3.12-slim does not ship it, so any estimator config
# containing LGBM crashed with OSError on import (aml.types() imports every
# module in the config). Install the GNU OpenMP runtime in the shared base so
# both dev and prod targets get it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# requirements.txt is the *dev/CI umbrella* (core + optional extras + API stack)
# and references its sibling tier files with -r, so copy them all into /tmp —
# pip resolves -r paths relative to the referencing file's directory.
# It is deliberately NOT what `pip install sksurrogate` pulls in: the wheel's
# install_requires is requirements-core.txt only, and every backend the image
# needs here is an extra in setup.py.
COPY requirements*.txt /tmp/
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
COPY sksurrogate_mcp/ ./sksurrogate_mcp/
COPY webui/ ./webui/
# mltrace reads its version via importlib.metadata, so distribution metadata must
# exist; --no-deps because requirements.txt already installed the dependencies.
COPY setup.py readme.rst LICENSE ./
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
# setup.py lists sksurrogate_mcp as a package, so `pip install .` below fails
# without it. Its runtime dependencies stay optional ([mcp] extra).
COPY sksurrogate_mcp/ ./sksurrogate_mcp/
COPY setup.py readme.rst LICENSE ./
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
