"""Model registry & deployment lifecycle endpoints.

Wraps ``SKSurrogate.ModelRegistry`` and ``SKSurrogate.DeploymentApprovalGate``.
No authentication in v1 (see docs/ui-plan.md section 9): ``approvers`` are
free-text names, not verified identities.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from SKSurrogate import DeploymentApprovalGate, ModelRegistry, load_bundle

from ..config import settings
from ..deps import bad_request, bundle_summary, not_found

router = APIRouter(prefix="/api/registry", tags=["registry"])


def _registry():
    return ModelRegistry(settings.registry_dir)


class RegisterRequest(BaseModel):
    model_version: str


class PromoteRequest(BaseModel):
    model_version: str
    state: str
    approvers: list[str]
    quality_report: dict | None = None
    required_approvals: int = 1


class RollbackRequest(BaseModel):
    state: str
    model_version: str | None = None


@router.post("/{task_name}/register")
def register_bundle(task_name: str, body: RegisterRequest):
    """Move a bundle produced by /api/bundles into the registry as a candidate."""
    bundle_path = settings.task_bundles_dir(task_name) / (body.model_version + ".bundle")
    if not bundle_path.exists():
        raise not_found("No bundle %r found for task %r" % (body.model_version, task_name))
    bundle = load_bundle(bundle_path, strict_dependencies=False)
    model_version = _registry().register(bundle)
    return {"task_name": task_name, "model_version": model_version, "state": "candidate"}


@router.post("/{task_name}/promote")
def promote_bundle(task_name: str, body: PromoteRequest):
    """Promote a registered model version, requiring N unique named approvers."""
    gate = DeploymentApprovalGate(_registry(), required_approvals=body.required_approvals)
    try:
        result = gate.promote(
            task_name,
            body.model_version,
            body.state,
            approvers=body.approvers,
            quality_report=body.quality_report,
        )
    except KeyError as exc:
        raise not_found(str(exc))
    except (PermissionError, ValueError) as exc:
        raise bad_request(str(exc))
    return result


@router.post("/{task_name}/rollback")
def rollback_bundle(task_name: str, body: RollbackRequest):
    """Point a lifecycle alias back at a prior model version."""
    gate = DeploymentApprovalGate(_registry())
    try:
        return gate.rollback(task_name, body.state, model_version=body.model_version)
    except KeyError as exc:
        raise not_found(str(exc))


@router.get("/{task_name}/summary")
def registry_summary(task_name: str):
    """All lifecycle aliases + per-version states in one round-trip."""
    try:
        return {"task_name": task_name, **_registry().summary(task_name)}
    except KeyError as exc:
        raise not_found(str(exc))


@router.get("/{task_name}/history")
def registry_history(task_name: str):
    try:
        return {"task_name": task_name, "history": _registry().history(task_name)}
    except KeyError as exc:
        raise not_found(str(exc))


@router.get("/{task_name}/audit")
def registry_audit(task_name: str):
    try:
        return {"task_name": task_name, "audit": _registry().audit_log(task_name)}
    except KeyError as exc:
        raise not_found(str(exc))


@router.get("/{task_name}/load")
def load_registered_bundle(task_name: str, alias: str = "production"):
    """Return metadata for the version currently pointed at by a lifecycle alias."""
    try:
        bundle = _registry().load(task_name, alias=alias, strict_dependencies=False)
    except KeyError as exc:
        raise not_found(str(exc))
    return {"alias": alias, **bundle_summary(bundle)}
