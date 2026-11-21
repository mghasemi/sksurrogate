"""Model-serving control plane: declare which model a task serves, then score through it.

``api.routers.inference`` resolves the model from *every* request body. This
router adds the deployment view on top of it:

* an operator declares once which trained/registered model a task serves
  (``POST /api/serving/{task_name}``);
* clients then score through that declaration without naming a model
  (``POST /api/serving/{task_name}/predict``);
* and the UI can *declare* the state instead of inferring it
  (``GET /api/serving``, ``GET /api/serving/{task_name}``), which is what the
  top-bar serving indicator reads.

The pin is persisted per task (see :mod:`api.serving`). In ``alias`` mode it
follows the registry, so a later promote/rollback changes what the serving
endpoint scores; the status payload always reports the version it currently
resolves to. Prediction itself is the shared core in :mod:`api.inference_service`,
so the served path and the per-request path validate, score and record
identically — they differ only in how the bundle was chosen.
"""

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import acl, serving
from ..deps import bad_request, current_principal, resolve_bundle, task_access
from ..inference_service import resolve_input_frame, score, score_forecast

router = APIRouter(prefix="/api/serving", tags=["serving"], dependencies=[Depends(task_access)])


class StartRequest(BaseModel):
    """Name the model to serve: exactly one of ``model_version`` or ``alias``."""

    model_version: str | None = None
    alias: str | None = None
    note: str | None = None


class ServedPredictRequest(BaseModel):
    """Prediction payload for the served model — deliberately has no model field."""

    partition: str | None = None
    rows: list[dict] | None = None
    request_id: str | None = None
    preflight: bool = False


def _resolve_pin(task_name, state):
    """Load the bundle the pin points at.

    ``alias`` mode re-resolves the registry pointer on every call (that is the
    point of the mode); ``version`` mode loads the exact version frozen at pin
    time, so a later promotion cannot change what is being served.
    """
    if state["resolve_mode"] == "alias":
        return resolve_bundle(task_name, alias=state["requested"])
    return resolve_bundle(task_name, model_version=state["model_version"])


def _view(task_name, state):
    """Public serving payload: the stored pin plus the model it resolves to now."""
    if state is None:
        return {
            "task_name": task_name,
            "active": False,
            "resolve_mode": None,
            "requested": None,
            "model_version": None,
            "note": None,
            "started_at": None,
            "started_by": None,
            "updated_at": None,
            "resolve_error": None,
            "history": [],
        }
    view = {
        "task_name": task_name,
        "active": bool(state.get("active")),
        "resolve_mode": state.get("resolve_mode"),
        "requested": state.get("requested"),
        "model_version": state.get("model_version"),
        "note": state.get("note"),
        "started_at": state.get("started_at"),
        "started_by": state.get("started_by"),
        "updated_at": state.get("updated_at"),
        "resolve_error": None,
        "history": list(state.get("history") or []),
    }
    if view["active"]:
        try:
            view["model_version"] = _resolve_pin(task_name, state).model_version
        except HTTPException as exc:
            # The pin outlived its target (alias retargeted at a removed version,
            # version unregistered): report why instead of pretending it is fine.
            view["resolve_error"] = str(exc.detail)
    return view


def _not_serving(task_name):
    return HTTPException(status_code=409, detail="Task %r is not serving a model" % task_name)


@router.get("")
def list_serving():
    """Tasks currently serving a model — summaries only, for the UI indicator.

    A list surface, so it filters through the ACL (the router-level
    ``task_access`` dependency cannot: this route has no task in its path).
    """
    tasks = acl.visible_ids(acl.TASK, serving.list_tasks())
    active = []
    for task_name in tasks:
        state = serving.read_state(task_name)
        if state and state.get("active"):
            active.append(_view(task_name, state))
    return {"serving": active, "count": len(active)}


@router.get("/{task_name}")
def serving_status(task_name: str):
    """The serving pin for one task; ``active: false`` when nothing is pinned.

    Answering 200 for an unconfigured task (instead of 404) keeps the UI's
    "is this task serving?" question a data question rather than an error path.
    """
    return _view(task_name, serving.read_state(task_name))


@router.post("/{task_name}")
def start_serving(task_name: str, body: StartRequest):
    """Declare which model this task serves, replacing any previous pin.

    The model must resolve (unregistered bundle file, exact registry version, or
    registry alias), so a typo answers 404 at declaration time rather than
    failing later on the serving endpoint.
    """
    if (body.model_version is None) == (body.alias is None):
        raise bad_request("Exactly one of 'model_version' or 'alias' must be provided")
    if body.model_version is not None:
        resolve_mode, identifier = "version", body.model_version
    else:
        resolve_mode, identifier = "alias", body.alias
    bundle = resolve_bundle(task_name, model_version=body.model_version, alias=body.alias)
    state = serving.start_serving(
        task_name,
        resolve_mode=resolve_mode,
        identifier=identifier,
        model_version=bundle.model_version,
        note=body.note,
        actor=current_principal().username,
    )
    return _view(task_name, state)


@router.delete("/{task_name}")
def stop_serving(task_name: str):
    """Stop serving the pinned model; the pin stays in the history (409 if idle)."""
    state = serving.stop_serving(task_name, actor=current_principal().username)
    if state is None:
        raise _not_serving(task_name)
    return _view(task_name, state)


@router.post("/{task_name}/predict")
def predict_served(task_name: str, body: ServedPredictRequest):
    """Score through the served model — the caller does not name one.

    This is the endpoint a deployment points its clients at. Responses carry the
    resolved ``model_version`` and the pin's ``resolve_mode``/``requested``
    alongside the predictions, so a client can tell which model answered even
    when an alias moved underneath it.
    """
    state = serving.read_state(task_name)
    if state is None or not state.get("active"):
        raise _not_serving(task_name)
    bundle = _resolve_pin(task_name, state)
    frame, ignored_columns = resolve_input_frame(
        task_name, bundle=bundle, rows=body.rows, partition=body.partition
    )
    result = score(
        task_name, frame, bundle, request_id=body.request_id, preflight=body.preflight
    )
    return {
        "task_name": task_name,
        "model_version": bundle.model_version,
        "resolve_mode": state.get("resolve_mode"),
        "requested": state.get("requested"),
        "request_id": result["request_id"],
        "predictions": result["predictions"],
        "metrics": result["metrics"],
        "ignored_columns": ignored_columns,
    }


class ServedForecastRequest(BaseModel):
    """Forecast payload for the served model — deliberately has no model field.

    ``history`` holds one series of past observations (the temporal design
    validates it); tabular pins answer 400 on this route (plan phase P6).
    """

    history: list[dict]
    future_exogenous: list[dict] | None = None
    horizon: int | None = None
    request_id: str | None = None


@router.post("/{task_name}/forecast")
def forecast_served(task_name: str, body: ServedForecastRequest):
    """Serve a multi-step forecast through the pinned model (plan phase P6).

    Mirrors :func:`predict_served` exactly — same pin resolution (an ``alias``
    pin re-resolves on every call), same monitoring record — but scores with
    :func:`api.inference_service.score_forecast`, so a temporal design and its
    history rows are what the request carries.
    """
    state = serving.read_state(task_name)
    if state is None or not state.get("active"):
        raise _not_serving(task_name)
    bundle = _resolve_pin(task_name, state)
    future_frame = (
        pd.DataFrame(body.future_exogenous) if body.future_exogenous is not None else None
    )
    result = score_forecast(
        task_name,
        pd.DataFrame(body.history),
        bundle,
        horizon=body.horizon,
        future_exogenous=future_frame,
        request_id=body.request_id,
    )
    return {
        "task_name": task_name,
        "model_version": bundle.model_version,
        "resolve_mode": state.get("resolve_mode"),
        "requested": state.get("requested"),
        **result,
    }
