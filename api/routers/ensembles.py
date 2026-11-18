"""Ensemble endpoints: combine several trained bundles into one model.

An ensemble here is an ordinary ``ModelBundle`` whose estimator is a scikit-learn
meta-estimator over the *selected models* of the task, so everything that already
consumes a bundle applies to it unchanged: registration, promotion, serving,
learning curves, diagnostic curves, nested-CV evaluation and the run report. This
router therefore only adds the combiners themselves plus the fit job.

Endpoints:

- ``/methods``    → which combiners apply to the task's problem family, with their
                    parameter schema and capability requirements (cheap)
- ``/candidates`` → the task's models that can be selected as members (cheap)
- ``/fit``        → submit the combining job (a background job: stacking fits every
                    member out-of-fold, and bagging refits each member per replica)
- ``/`` (GET)     → list the task's ensembles with their member-versus-ensemble scores
- ``/{version}``  → one ensemble's detail: members, parameters, scores, caveats

Two orthogonal capabilities extend every combiner, and both are recorded with the
result rather than left implicit:

- **per-member calibration** (``calibration``) wraps each member's *outermost* form in a
  ``CalibratedClassifierCV``, so the probabilities that enter the merge are calibrated.
  The per-member Brier score and log loss before/after are stored as the evidence.
- **weight optimization** (``weight_optimization``) fits the member weights on
  out-of-fold member output, and ``honest_cv`` then prices that search by refitting the
  members, the calibration and the weights inside every outer fold.

The member-versus-ensemble comparison is scoped by an explicit caveat carried in the
response and stored with the model (see ``api.training.ENSEMBLE_CAVEATS``): the
members were fitted and tuned on this same data, so their out-of-fold scores and the
ensemble's are jointly optimistic, and the honest estimate is the nested-CV
evaluation. A weight search adds a second layer of optimism, which is why the
weight-optimization caveat quotes the measured gap and ``honest_cv`` exists.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from SKSurrogate import load_bundle, method_catalogue, problem_family_of

from ..config import settings
from ..safety import safe_segment
from ..deps import bad_request, bundle_summary, not_found, task_access
from ..jobs import job_manager
from ..training import (
    ENSEMBLE_CAVEATS,
    MAX_ENSEMBLE_MEMBERS,
    ensemble_plan,
    fit_ensemble_bundle,
    get_registered_target,
    load_partition,
)

router = APIRouter(prefix="/api/ensembles", tags=["ensembles"], dependencies=[Depends(task_access)])

#: How many stored bundles are inspected when listing members. Each candidate costs
#: one ``joblib`` load, so an unbounded sweep would make the page slower than the job
#: it starts; ``MAX_ENSEMBLE_MEMBERS`` bounds what can actually be combined.
_CANDIDATE_PROBE_LIMIT = 40


class FitEnsembleRequest(BaseModel):
    """A combine request: which models, which combiner, and that combiner's parameters.

    ``params`` is deliberately free-form and validated against the method's schema on
    the server (``SKSurrogate.ensembling.validate_params``), so the UI and the API
    cannot drift: an unknown key is refused rather than ignored.

    ``calibration`` and ``weight_optimization`` are orthogonal to ``method`` — they post-
    process every member and the merge itself — and are validated by the same core
    functions the library uses, so a request rejected here would be rejected there too.
    """

    members: list[str] = Field(default_factory=list)
    method: str
    params: dict | None = None
    train_partition: str = "train"
    scoring: str | None = None  # defaults to the family's metric
    random_state: int | None = None
    owner: str | None = None
    run_id: str | None = None
    label: str | None = None  # optional human name for the job and its bundle
    register: bool = False  # additionally register the result as a candidate version
    #: ``{"method": "sigmoid"|"isotonic", "cv": int, "ensemble": "auto"|"per_fold"|"single"}``
    #: applied to each member's outermost form; classification members only.
    calibration: dict | None = None
    #: ``{"strategy": "least_squares"|"dirichlet"|"greedy", "samples": int, "rounds": int}``
    #: fitted on out-of-fold member output; unavailable for stacking.
    weight_optimization: dict | None = None
    #: ``{"folds": int, "inner_folds": int}`` — the honest estimate that refits members,
    #: calibration and weights inside every outer fold. Costs outer-folds x member fits.
    honest_cv: dict | None = None


def _stored_bundle_files(task_name):
    """Unregistered then registered bundle files for a task, newest first.

    Sorted by section rather than globally so a model that exists in both places is
    reported once, from the unregistered directory — the file every other consuming
    endpoint resolves first.
    """
    bundles_dir = settings.task_bundles_dir(task_name)
    registry_dir = settings.registry_dir / safe_segment(task_name, "task name")
    files = []
    for directory in (bundles_dir, registry_dir):
        if directory.exists():
            files.extend(
                sorted(directory.glob("*.bundle"), key=lambda path: path.stat().st_mtime, reverse=True)
            )
    return files


def _candidate_entries(task_name, limit=_CANDIDATE_PROBE_LIMIT):
    """The task's models that can be selected as members, newest first.

    Unregistered and registered bundles are both offered (a model does not have to be
    promoted before it can be combined), each entry carrying the family, metrics and —
    for ensembles — the recorded combination, so the member picker and the ensemble
    list need one sweep instead of one load per row. Unreadable files are skipped
    rather than failing the page; ``load_bundle`` never exposes the fitted object.
    """
    entries = []
    seen = set()
    for path in _stored_bundle_files(task_name):
        if path.stem in seen:
            continue
        seen.add(path.stem)
        try:
            bundle = load_bundle(path, strict_dependencies=False)
        except Exception:  # noqa: BLE001 - a corrupt file must not break the listing
            continue
        metadata = (bundle.metadata or {}).get("ensemble") or {}
        entries.append(
            {
                "model_version": bundle.model_version,
                "label": (bundle.metadata or {}).get("label"),
                "family": problem_family_of(bundle.model),
                "metrics": bundle.metrics,
                "created_at": bundle.created_at,
                "is_ensemble": bool(metadata),
                "source": "bundle" if path.parent == settings.task_bundles_dir(task_name) else "registry",
                "ensemble": (
                    {
                        "method": metadata.get("method"),
                        "family": metadata.get("family"),
                        "aggregation": metadata.get("aggregation"),
                        "member_count": len(metadata.get("members") or []),
                        "scores": metadata.get("scores"),
                        "calibrated": bool(metadata.get("calibration")),
                        "weighted": bool(metadata.get("weight_optimization")),
                        "nested_score": ((metadata.get("scores") or {}).get("nested") or {}).get("mean"),
                    }
                    if metadata
                    else None
                ),
            }
        )
        if len(entries) >= limit:
            break
    return entries


@router.get("/{task_name}/methods")
def ensemble_methods(task_name: str):
    """Combiners available for the task, with their parameter schema and requirements."""
    candidates = _candidate_entries(task_name)
    family = next(
        (entry["family"] for entry in candidates if entry["family"] is not None), None
    )
    return {
        "task_name": task_name,
        "max_members": MAX_ENSEMBLE_MEMBERS,
        "caveats": list(ENSEMBLE_CAVEATS),
        **method_catalogue(family),
    }


@router.get("/{task_name}/candidates")
def ensemble_candidates(task_name: str):
    """Models of the task that can be selected as members, newest first."""
    return {"task_name": task_name, "candidates": _candidate_entries(task_name)}


def _validate_combination_options(body):
    """Reject malformed calibration / weight-optimization / honest-CV payloads at submit.

    Field-level shape only (the value sets and the cross-field conflicts are checked by
    the core functions the job also calls); a bad shape here would otherwise fail at the
    first ``.get`` inside the worker.
    """
    if body.calibration is not None and not isinstance(body.calibration, dict):
        raise bad_request("'calibration' must be an object of method/cv/ensemble.")
    if body.weight_optimization is not None and not isinstance(body.weight_optimization, dict):
        raise bad_request("'weight_optimization' must be an object of strategy/samples/rounds.")
    if body.honest_cv is not None and not isinstance(body.honest_cv, dict):
        raise bad_request("'honest_cv' must be an object of folds/inner_folds.")
    if body.honest_cv:
        folds = body.honest_cv.get("folds", 5)
        inner = body.honest_cv.get("inner_folds", 3)
        for name, value in (("folds", folds), ("inner_folds", inner)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise bad_request("honest_cv.%s must be an integer of at least 2." % name)
    if body.calibration and body.weight_optimization and body.method == "stacking":
        raise bad_request(
            "Stacking learns its own combination in the final estimator, so member "
            "weights do not apply; optimize weights with voting, averaging, bagging or "
            "boosting instead."
        )


def _submit_ensemble(task_name, body):
    """Validate the request now, then run the combining job in the background.

    Validation happens in the request (so a bad member id answers 404 and a bad
    parameter or incompatible member set answers 400 immediately); the job then
    repeats it inside the worker, where a failure is recorded on the job rather than
    lost with the request context.
    """
    if not body.method:
        raise bad_request("An ensemble method is required.")
    _validate_combination_options(body)
    target, _ = get_registered_target(task_name)
    frame = load_partition(task_name, body.train_partition)
    feature_columns = [column for column in frame.columns if column != target]
    # Fail fast: resolves members, checks the family, the capability requirements, every
    # parameter value, the calibration eligibility and the weight-optimization conflicts
    # before the job is queued.
    ensemble_plan(
        task_name,
        members=body.members,
        method=body.method,
        params=body.params,
        calibration=body.calibration,
        weight_optimization=body.weight_optimization,
        columns=feature_columns,
        random_state=body.random_state,
    )

    def make_job(job_id):
        def _run():
            _bundle, summary = fit_ensemble_bundle(
                task_name,
                members=body.members,
                method=body.method,
                params=body.params,
                train_partition=body.train_partition,
                scoring=body.scoring,
                random_state=body.random_state,
                owner=body.owner,
                run_id=body.run_id or job_id,
                label=body.label,
                register=body.register,
                calibration=body.calibration,
                weight_optimization=body.weight_optimization,
                honest_cv=body.honest_cv,
            )
            return summary

        return _run

    def metadata_for_job(job_id):
        request = body.model_dump(mode="json")
        request["run_id"] = body.run_id or job_id
        return {"ensemble": {"request": request}}

    job_id = job_manager.submit(
        task_name,
        "ensemble",
        make_job,
        metadata_factory=metadata_for_job,
        label=body.label,
    )
    record = job_manager.get(job_id) or {}
    return {"job_id": job_id, "display_name": record.get("display_name"), "status": "queued"}


@router.post("/{task_name}/fit")
def fit_ensemble(task_name: str, body: FitEnsembleRequest):
    """Submit a job that combines the selected models into one ensemble bundle."""
    return _submit_ensemble(task_name, body)


@router.get("/{task_name}")
def list_ensembles(task_name: str):
    """List the task's ensembles with their method and stored scores.

    Membership and scores come from each bundle's own metadata, so a listed row always
    matches the artifact on disk rather than a separately maintained index.
    """
    ensembles = []
    for entry in _candidate_entries(task_name):
        if not entry["is_ensemble"]:
            continue
        ensembles.append(
            {
                "model_version": entry["model_version"],
                "label": entry["label"],
                "created_at": entry["created_at"],
                "metrics": entry["metrics"],
                "source": entry["source"],
                **(entry["ensemble"] or {}),
            }
        )
    return {"task_name": task_name, "ensembles": ensembles}


def _find_bundle_file(task_name, model_version):
    """Locate a stored bundle file by version id (unregistered directory first)."""
    version = safe_segment(model_version, "model version")
    for directory in (
        settings.task_bundles_dir(task_name),
        settings.registry_dir / safe_segment(task_name, "task name"),
    ):
        candidate = directory / (version + ".bundle")
        if candidate.exists():
            return candidate
    return None


@router.get("/{task_name}/{model_version}")
def ensemble_detail(task_name: str, model_version: str):
    """One ensemble's members, parameters, same-fold scores and caveats.

    Declared last: this two-segment pattern would otherwise swallow the static
    suffixes above it (``methods``, ``candidates``).
    """
    path = _find_bundle_file(task_name, model_version)
    if path is None:
        raise not_found("No bundle %r found for task %r" % (model_version, task_name))
    bundle = load_bundle(path, strict_dependencies=False)
    metadata = (bundle.metadata or {}).get("ensemble")
    if not metadata:
        raise bad_request(
            "Bundle %r is not an ensemble (it carries no ensemble record); "
            "see /api/bundles for the bundle's own metadata" % model_version
        )
    return {**bundle_summary(bundle), "ensemble": metadata}
