"""Approval-gated model promotion and rollback helpers."""


class DeploymentApprovalGate:
    """Require approvals and passing quality checks before promotion."""

    def __init__(self, registry, required_approvals=1):
        if required_approvals < 1:
            raise ValueError("required_approvals must be at least 1")
        self.registry = registry
        self.required_approvals = required_approvals

    def promote(self, task_name, model_version, state, *, approvers, quality_report=None):
        unique_approvers = sorted({str(approver) for approver in approvers if str(approver).strip()})
        if len(unique_approvers) < self.required_approvals:
            raise PermissionError(
                "Deployment requires %d unique approvals" % self.required_approvals
            )
        if quality_report is not None and not quality_report.get("passed", False):
            raise ValueError("Deployment quality gates did not pass")
        for approver in unique_approvers:
            self.registry.record_approval(task_name, model_version, approver, state=state)
        self.registry.promote(task_name, model_version, state)
        return {"status": "promoted", "task_name": task_name, "model_version": model_version, "state": state}

    def rollback(self, task_name, state, model_version=None):
        self.registry.rollback(task_name, state, model_version=model_version)
        return {"status": "rolled_back", "task_name": task_name, "state": state, "model_version": model_version}


__all__ = ["DeploymentApprovalGate"]