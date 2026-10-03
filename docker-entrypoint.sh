#!/bin/sh
set -e

# The repo bind mount shadows the image's webui/node_modules, so re-link it
# from the stashed copy (only if the host doesn't have its own).
if [ ! -x /app/webui/node_modules/.bin/vite ] && [ -d /opt/webui-node-modules ]; then
  ln -sfn /opt/webui-node-modules /app/webui/node_modules
fi

# Backend: FastAPI with auto-reload, watching only source dirs (not var/ data)
cd /app
uvicorn api.main:app --host 0.0.0.0 --port 8013 \
  --reload --reload-dir /app/api --reload-dir /app/SKSurrogate &
BACKEND_PID=$!

# Frontend: Vite dev server with HMR, bound to all interfaces so the
# published port is reachable from the host browser.
cd /app/webui
./node_modules/.bin/vite --host 0.0.0.0 --port 5173 &
FRONTEND_PID=$!

trap 'kill "$BACKEND_PID" "$FRONTEND_PID"' INT TERM EXIT
wait
