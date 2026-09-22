"""FastAPI app factory for the SKSurrogate control-plane API.

Run locally with:

    uvicorn api.main:app --reload

See docs/ui-plan.md for the overall architecture and phased delivery plan.
No authentication and SQLite-only storage are intentional v1 decisions.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import (
    bundles,
    datasets,
    experiments,
    inference,
    jobs,
    lineage,
    monitoring,
    quality_gates,
    registry,
    retraining,
    sensitivity,
)

app = FastAPI(
    title="SKSurrogate API",
    description="Control-plane API for the SKSurrogate MLOps toolkit.",
    version="0.1.0",
)

# The UI is a separate SPA client (see docs/ui-plan.md); allow any local dev origin for now.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router)
app.include_router(bundles.router)
app.include_router(quality_gates.router)
app.include_router(registry.router)
app.include_router(inference.router)
app.include_router(monitoring.router)
app.include_router(sensitivity.router)
app.include_router(experiments.router)
app.include_router(jobs.router)
app.include_router(retraining.router)
app.include_router(lineage.router)


@app.get("/api/health", tags=["health"])
def health():
    return {"status": "ok"}
