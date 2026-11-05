"""Experiment-design endpoints: stateless wrappers over ``SKSurrogate.design``.

The API only proposes candidate points; it never evaluates the real objective.
Bounded continuous numeric domains only (see the library docstrings).
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from SKSurrogate.design import latin_hypercube, suggest_next_point

from ..deps import bad_request, finite_or_none

router = APIRouter(prefix="/api/design", tags=["experiment-design"])


class Variable(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    low: float
    high: float


def _bounds(variables):
    names = [variable.name for variable in variables]
    if len(set(names)) != len(names):
        raise bad_request("Variable names must be unique")
    return names, [[variable.low, variable.high] for variable in variables]


class LatinHypercubeRequest(BaseModel):
    variables: list[Variable] = Field(min_length=1, max_length=100)
    n_samples: int = Field(default=10, ge=1, le=10000)
    random_state: int | None = 0


class SuggestNextRequest(BaseModel):
    variables: list[Variable] = Field(min_length=1, max_length=100)
    samples: list[list[float]] = Field(min_length=2)
    values: list[float] = Field(min_length=2)
    n_candidates: int = Field(default=1000, ge=1, le=10000)
    beta: float = Field(default=2.0, gt=0.0)
    random_state: int | None = 0


@router.post("/latin-hypercube")
def lhs(body: LatinHypercubeRequest):
    names, bounds = _bounds(body.variables)
    try:
        design = latin_hypercube(bounds, body.n_samples, random_state=body.random_state)
    except ValueError as exc:
        raise bad_request(str(exc))
    return {"columns": names, "points": [[finite_or_none(v) for v in row] for row in design.tolist()]}


@router.post("/suggest-next")
def suggest_next(body: SuggestNextRequest):
    """Recommend the next point to try, minimizing the observed scalar objective."""
    names, bounds = _bounds(body.variables)
    try:
        result = suggest_next_point(
            body.samples,
            body.values,
            bounds,
            n_candidates=body.n_candidates,
            beta=body.beta,
            random_state=body.random_state,
        )
    except ValueError as exc:
        raise bad_request(str(exc))
    return {"columns": names, "result": _jsonable(result)}


def _jsonable(value):
    """Convert numpy scalars/arrays in the library's result dict to plain JSON values."""
    import numpy as np

    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return finite_or_none(value)
    if isinstance(value, np.integer):
        return int(value)
    return value
