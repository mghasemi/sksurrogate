# ---- Stage 1: install Node deps once, keep them outside the mounted tree ----
FROM node:22-bookworm AS webdeps
WORKDIR /build/webui
COPY webui/package.json ./
RUN npm install

# ---- Stage 2: runtime image (Python + Node) ----
FROM python:3.12-slim

# Node runtime for the Vite dev server
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Python deps in a venv outside the mounted tree
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
# Baked-in copies are only placeholders; the bind mount is the source of truth.
COPY api/ ./api/
COPY SKSurrogate/ ./SKSurrogate/
COPY webui/ ./webui/
# Stash node_modules where the bind mount can't shadow it
COPY --from=webdeps /build/webui/node_modules /opt/webui-node-modules

COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5173 8013
ENTRYPOINT ["/entrypoint.sh"]
