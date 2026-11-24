"""Serving-state store: which model each task is currently serving.

The per-request inference path lets every caller name a model in the body
(``/api/inference/{task}/predict``). This module holds the *deployment*
counterpart: an operator declares once which trained/registered model a task
serves, and that declaration is what ``/api/serving/{task}/predict`` scores
through and what the UI's serving indicator displays.

State is one small JSON document per task under
``<API home>/serving/<task>.json`` — the control plane has no database and keeps
every other piece of deployment state (registry index, job records, run
settings) in a file too. Beside the current pin the document carries an
append-only ``history`` of start/replace/stop events, matching the registry's
audit surfaces.

Two resolution modes (see ``docs/inference.rst``):

``version``
    The pin freezes an *exact* registered model version. Later promotions or
    rollbacks in the registry do not change what is served.
``alias``
    The pin names a registry pointer (a lifecycle alias such as ``production``
    or a user-chosen alias). Every serving request re-resolves it, so a later
    ``promote``/``rollback`` changes what is served — the operator follows the
    registry instead of a snapshot.
"""

import json
from datetime import datetime, timezone

from SKSurrogate.artifacts import read_artifact, stamp

from .config import settings

#: The two ways a serving pin can name its model (see the module docstring).
RESOLVE_MODES = ("version", "alias")


def _now():
    return datetime.now(timezone.utc).isoformat()


def read_state(task_name):
    """Return the stored serving document for ``task_name``, or ``None``.

    A corrupted or non-object file is treated as "never configured" rather than
    raising: the worst case is that an operator re-pins the model.
    """
    path = settings.task_serving_path(task_name)
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict):
        return None
    try:
        return read_artifact("serving_pin", document)
    except ValueError:
        # A newer writer's document is not safer to guess about; report it as
        # unconfigured rather than serving through fields this code cannot read.
        return None


def write_state(state):
    """Persist (atomically) one serving document and return it."""
    state = stamp("serving_pin", state)
    path = settings.task_serving_path(state["task_name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return state


def _record(state, event, *, actor, note, resolved_model_version):
    history = state.setdefault("history", [])
    history.append(
        {
            "event": event,
            "resolve_mode": state["resolve_mode"],
            "requested": state["requested"],
            "model_version": resolved_model_version,
            "note": note,
            "actor": actor,
            "at": _now(),
        }
    )


def start_serving(task_name, *, resolve_mode, identifier, model_version, note=None, actor=None):
    """Pin ``task_name`` to a model, replacing any previous pin.

    ``identifier`` is what the operator selected (a version id or an alias
    name); ``model_version`` is the exact registered version it resolved to at
    this moment — for ``alias`` mode it is a snapshot for display only, since
    later requests re-resolve the alias. ``started_at``/``started_by`` describe
    the pin being installed now, so replacing a live pin resets them while
    ``history`` keeps the whole sequence.
    """
    if resolve_mode not in RESOLVE_MODES:
        raise ValueError("resolve_mode must be one of %s" % ", ".join(RESOLVE_MODES))
    previous = read_state(task_name)
    was_active = bool((previous or {}).get("active"))
    timestamp = _now()
    state = {
        "task_name": task_name,
        "active": True,
        "resolve_mode": resolve_mode,
        "requested": identifier,
        "model_version": model_version,
        "note": note,
        "started_at": timestamp,
        "started_by": actor,
        "updated_at": timestamp,
        "history": list((previous or {}).get("history") or []),
    }
    _record(
        state,
        "replace" if was_active else "start",
        actor=actor,
        note=note,
        resolved_model_version=model_version,
    )
    return write_state(state)


def stop_serving(task_name, *, actor=None, note=None):
    """Stop serving ``task_name``; the pin is kept in history, ``active`` goes false.

    Returns ``None`` when nothing was being served (the caller answers 409).
    """
    state = read_state(task_name)
    if state is None or not state.get("active"):
        return None
    state["active"] = False
    state["note"] = note if note is not None else state.get("note")
    state["updated_at"] = _now()
    _record(
        state,
        "stop",
        actor=actor,
        note=note,
        resolved_model_version=state.get("model_version"),
    )
    return write_state(state)


def list_tasks():
    """Every task with a stored serving document, whatever its ``active`` flag."""
    directory = settings.serving_dir
    if not directory.exists():
        return []
    return sorted(path.stem for path in directory.glob("*.json"))
