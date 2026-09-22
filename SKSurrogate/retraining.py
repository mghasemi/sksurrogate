"""Scheduler-agnostic retraining entry points."""

from .modelbundle import ModelBundle


class RetrainingJob:
    """Run a trainer when a scheduled or data-driven trigger is satisfied."""

    def __init__(self, trainer, registry, task_name, promotion_state=None):
        if not callable(trainer):
            raise TypeError("trainer must be callable")
        self.trainer = trainer
        self.registry = registry
        self.task_name = task_name
        self.promotion_state = promotion_state

    def run(self, *, trigger=None, context=None):
        context = {} if context is None else dict(context)
        if trigger is not None and not trigger(context):
            return {"status": "skipped", "reason": "trigger_not_satisfied", "context": context}

        bundle = self.trainer(context)
        if not isinstance(bundle, ModelBundle):
            raise TypeError("trainer must return a ModelBundle")
        if bundle.task_name is None:
            bundle.task_name = self.task_name
        elif bundle.task_name != self.task_name:
            raise ValueError("trainer returned a bundle for a different task")
        model_version = self.registry.register(bundle)
        if self.promotion_state is not None:
            self.registry.promote(self.task_name, model_version, self.promotion_state)
        return {
            "status": "completed",
            "model_version": model_version,
            "task_name": self.task_name,
            "promotion_state": self.promotion_state,
            "context": context,
        }

    def run_scheduled(self, is_due, *, context=None):
        """Run when a scheduler-specific due predicate returns true."""
        return self.run(trigger=is_due, context=context)

    def run_on_data(self, has_new_data, *, context=None):
        """Run when a data watcher-specific predicate returns true."""
        return self.run(trigger=has_new_data, context=context)


__all__ = ["RetrainingJob"]