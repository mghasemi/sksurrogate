"""Synthetic data generation endpoints (Phase 5.3).

Wraps ``SKSurrogate.synthdat.SynthData``: builds a generator from one of the
task's registered partitions, applies per-column type overrides, and persists
each generated sample as a CSV under the task's dataset directory so it shows
up in the registry lineage (Phase 3.4) and can be re-listed/previewed later.

The ``where``/``filter`` constraint DSL is intentionally not exposed in v1.
"""

import csv
import re
import time
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Query, Depends
from pydantic import BaseModel
from starlette.exceptions import HTTPException
from starlette.responses import FileResponse

from SKSurrogate import DataPreprocess, ModelRegistry, SynthData

from ..config import settings
from ..columns import drop_ignored, read_ignored_columns
from ..safety import safe_segment
from ..deps import bad_request, not_found, open_tracker, task_access

router = APIRouter(prefix="/api/datasets", tags=["synthetic-data"], dependencies=[Depends(task_access)])

_SYNTH_TYPES = {"bin", "int", "real", "cat", "date"}


def _validate_partition_name(partition):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", partition or ""):
        raise bad_request("partition must contain only letters, numbers, underscores, or hyphens")


class GenerateSyntheticRequest(BaseModel):
    """Parameters for one synthetic-sample generation run."""

    num: int = 100
    partition: str | None = None
    distribution_type: str = "marginal"
    default_rv: str = "uniform"
    # column -> synth type ('bin'|'int'|'real'|'cat'|'date'); columns left out
    # fall back to the DataPreprocess-deduced type of that column.
    type_overrides: dict[str, str] | None = None


def _is_numeric_column(frame, column):
    """True when a column can feed ``bin``/``int``/``real`` SynthData types.

    SynthBin/SynthInt/SynthReal compare and bound values numerically, so a
    text column (e.g. ``Gender`` = "Female"/"Male") handed to them fails with
    ``'<' not supported between instances of 'str' and 'float'``. Datetimes are
    excluded too — they have their own ``date`` type.
    """
    series = frame[column]
    return bool(
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_datetime64_any_dtype(series)
    )


def _deduced_synth_types(frame):
    """Map every column to a SynthData type string using the deducer's types.

    ``binary`` is only a *cardinality* observation: two distinct string values
    are binary too. SynthData's ``bin`` type expects numeric 0/1 values, so a
    text binary column is generated as ``cat`` instead.
    """
    preview = DataPreprocess(frame)
    preview.deduce_types()
    types = {}
    for column, typ in preview.pivot_types.items():
        if typ == "float64":
            types[column] = "real"
        elif typ == "int64":
            types[column] = "int"
        elif typ == "binary":
            types[column] = "bin" if _is_numeric_column(frame, column) else "cat"
        elif typ == "datetime64":
            types[column] = "date"
        else:
            types[column] = "cat"
    return types


def _load_partition_frame(task_name, partition):
    """Return the registered frame for ``partition`` or raise 404.

    Ignored columns are dropped so synthetic samples never contain features the
    dataset has excluded from modeling.
    """
    if not settings.mltrace_db_path(task_name).exists():
        raise not_found("No dataset registered for task %r" % task_name)
    ignored = read_ignored_columns(task_name)
    with open_tracker(task_name) as tracker:
        try:
            return drop_ignored(tracker.get_dataframe(partition), ignored)
        except ValueError:
            pass
    csv_path = settings.task_dataset_dir(task_name) / (safe_segment(partition, "partition") + ".csv")
    if csv_path.exists():
        return drop_ignored(pd.read_csv(csv_path), ignored)
    raise not_found("No dataset partition %r stored for task %r" % (partition, task_name))


@router.post("/{task_name}/synthetic")
def generate_synthetic(task_name: str, body: GenerateSyntheticRequest):
    """Generate ``num`` synthetic rows from a registered partition and persist them."""
    if body.num <= 0:
        raise bad_request("num must be a positive integer")
    if body.distribution_type not in ("marginal", "joint"):
        raise bad_request("distribution_type must be 'marginal' or 'joint'")
    if body.default_rv not in ("uniform", "normal"):
        raise bad_request("default_rv must be 'uniform' or 'normal'")

    partition = body.partition or "train"
    _validate_partition_name(partition)
    frame = _load_partition_frame(task_name, partition)

    synth = SynthData(
        frame, default_rv=body.default_rv, distribution_type=body.distribution_type
    )
    overrides = dict(body.type_overrides or {})
    # Columns the dataset excludes are not in the frame; a stale client override
    # for one must not fail the request (they never reach generation).
    for column in list(overrides):
        if column not in frame.columns:
            overrides.pop(column)
    for column, typ in overrides.items():
        if typ not in _SYNTH_TYPES:
            raise bad_request(
                "synthetic type must be one of %s (got %r)" % (", ".join(sorted(_SYNTH_TYPES)), typ)
            )
        if typ in {"bin", "int", "real"} and not _is_numeric_column(frame, column):
            raise bad_request(
                "Synthetic type %r for column %r requires numeric values, but the column contains "
                "%s data; use 'cat' for text or encode the column first"
                % (typ, column, frame[column].dtype)
            )
        if typ == "date" and pd.api.types.is_numeric_dtype(frame[column]):
            raise bad_request(
                "Synthetic type 'date' for column %r requires date-formatted strings" % column
            )
    # Untyped columns default to SynthCat inside transform(); map them from the
    # deducer so numeric columns stay numeric unless the user overrides.
    for column, typ in _deduced_synth_types(frame).items():
        if column not in overrides:
            synth.set_type([column], typ)
    for column, typ in overrides.items():
        synth.set_type([column], typ)

    try:
        synth.transform()
        generated = synth.generate(body.num)
        if generated is None:
            raise ValueError("generator did not produce a data frame")
    except Exception as exc:
        raise bad_request("Synthetic generation failed: %s" % exc)

    dataset_dir = settings.task_dataset_dir(task_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    out_path = dataset_dir / ("synthetic-%d.csv" % time.time_ns())
    generated.to_csv(out_path, index=False)

    # Register the generated frame under its own fingerprint so multiple samples
    # remain distinct lineage artifacts.
    try:
        with open_tracker(task_name) as tracker:
            fingerprint = tracker._compute_dataset_fingerprint(generated)
        ModelRegistry(settings.registry_dir).register_dataset(
            task_name,
            fingerprint,
            out_path,
            metadata={
                "partition": partition,
                "source": "synthetic",
                "rows": int(len(generated)),
                "distribution_type": body.distribution_type,
            },
        )
    except Exception as exc:
        out_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Synthetic CSV was generated but could not be registered: %s" % exc,
        ) from exc

    return {
        "task_name": task_name,
        "path": str(out_path),
        "partition": partition,
        "rows": int(len(generated)),
        "columns": list(generated.columns),
        "preview": generated.head(10).to_dict(orient="records"),
    }


@router.get("/{task_name}/synthetic")
def list_synthetic(task_name: str, limit: Annotated[int, Query(ge=1)] = 50):
    """List previously generated synthetic samples (newest first) with a preview."""
    dataset_dir = settings.task_dataset_dir(task_name)
    if not dataset_dir.exists():
        raise not_found("No datasets stored for task %r" % task_name)

    files = []
    for path in sorted(dataset_dir.glob("synthetic-*.csv"), reverse=True)[: max(limit, 1)]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            row_count = sum(1 for _ in csv.reader(handle)) - 1
        files.append({"name": path.name, "path": str(path), "rows": max(row_count, 0)})

    preview = None
    if files:
        latest = pd.read_csv(files[0]["path"], nrows=10)
        preview = {"file": files[0]["name"], "columns": list(latest.columns),
                   "rows": latest.to_dict(orient="records")}
    return {"task_name": task_name, "files": files, "preview": preview}


@router.get("/{task_name}/synthetic/{filename}")
def download_synthetic(task_name: str, filename: str):
    """Download a previously generated synthetic CSV."""
    if Path(filename).name != filename or not filename.startswith("synthetic-") or not filename.endswith(".csv"):
        raise bad_request("Invalid synthetic dataset filename")
    path = settings.task_dataset_dir(task_name) / filename
    if not path.is_file():
        raise not_found("No synthetic dataset %r stored for task %r" % (filename, task_name))
    return FileResponse(path, media_type="text/csv", filename=filename)
