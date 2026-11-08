"""Archive-reader tolerance tests (Stage A, AI_LAYER02_REMAINING_PLAN.md).

Since record schema v2 every per-evaluation ``records[]`` entry carries the full
candidate configuration under ``params``, and each run JSON is stamped with a
top-level ``schema_version``. Legacy v1 archives (pre-2026-11) carry neither;
the archive readers in ``benchmarks/plot.py`` must keep working on both without
modification — that contract is what acquisition-function comparisons against
existing archives rely on (R7).

Fixtures below mirror the exact field layout of a real v1 file
(``benchmarks/results/20261018T051214Z/*.json``) and its v2 successor, so the
tests stay hermetic: raw result directories are git-ignored (raw data goes to
Zenodo), so no committed archive can be referenced.
"""

from __future__ import annotations

import json

import numpy as np

from benchmarks.methods import RECORD_SCHEMA_VERSION, eval_params
from benchmarks.plot import _best_sofar_curve, _load_records, build_target_table, evals_to_target

# One shared score trajectory (incl. an unscored evaluation) so v1/v2 fixtures differ
# ONLY in the schema-v2 additions.
_SCORES = [0.5, None, 0.7, 0.6]


def _base_record(method: str, seed: int) -> dict:
    return {
        "best_params": {"max_depth": 8, "min_samples_leaf": 0.01},
        "budget": len(_SCORES),
        "dataset": "synthetic",
        "error": None,
        "extra": {},
        "final_cv_best": max(s for s in _SCORES if s is not None),
        "final_test_accuracy": 0.88,
        "final_test_balanced": 0.87,
        "method": method,
        "n_estimators": 50,
        "n_evals": len(_SCORES),
        "seed": seed,
        "status": "ok",
        "total_wall_s": 12.3,
    }


def _v1_record(seed: int = 0) -> dict:
    """Legacy layout, exactly as archived before per-evaluation params existed."""
    record = _base_record("srs-no-surrogate", seed)
    record["records"] = [
        {"cv_score": s, "k": k, "status": "complete", "wall_s": 0.5 * (k + 1)}
        for k, s in enumerate(_SCORES)
    ]
    return record


def _v2_record(seed: int = 1) -> dict:
    """Schema-v2 layout: same scores plus full candidate configs and schema stamp."""
    record = _base_record("srs", seed)
    record["schema_version"] = RECORD_SCHEMA_VERSION
    params_pool = [
        {"max_depth": d, "min_samples_leaf": m} for d in (4, 8, 10) for m in (0.05, 0.01)
    ]
    record["records"] = [
        {
            "cv_score": s,
            "k": k,
            "status": "complete",
            "wall_s": 0.5 * (k + 1),
            "params": dict(params_pool[k]),
        }
        for k, s in enumerate(_SCORES)
    ]
    return record


def _write_archive(tmp_path) -> None:
    (tmp_path / "srs-no-surrogate__synthetic__seed0.json").write_text(
        json.dumps(_v1_record(), sort_keys=True), encoding="utf-8"
    )
    (tmp_path / "srs__synthetic__seed1.json").write_text(
        json.dumps(_v2_record(), sort_keys=True), encoding="utf-8"
    )
    # Error records are present in real archives and must be filtered out by readers.
    broken = _base_record("broken", 9)
    broken["status"] = "error"
    broken["error"] = "ValueError: synthetic failure"
    (tmp_path / "broken__synthetic__seed9.json").write_text(
        json.dumps(broken, sort_keys=True), encoding="utf-8"
    )


def test_load_records_tolerates_mixed_v1_and_v2_archive(tmp_path):
    _write_archive(tmp_path)
    records = _load_records(tmp_path)

    assert [r["method"] for r in records] == ["srs-no-surrogate", "srs"]  # error run filtered
    legacy, modern = records
    assert "params" not in legacy["records"][0] and "schema_version" not in legacy
    assert set(modern["records"][0]["params"]) == {"max_depth", "min_samples_leaf"}
    assert modern["schema_version"] == 2


def test_best_sofar_curve_identical_for_v1_and_v2(tmp_path):
    _write_archive(tmp_path)
    legacy, modern = _load_records(tmp_path)

    curve_v1 = _best_sofar_curve(legacy)
    curve_v2 = _best_sofar_curve(modern)
    assert curve_v1 is not None and curve_v2 is not None  # fixture guarantees a finite score
    assert np.array_equal(curve_v1, curve_v2)  # extra fields change nothing
    assert list(curve_v1) == [0.5, 0.5, 0.7, 0.7]


def test_evals_to_target_and_table_on_legacy_records(tmp_path):
    _write_archive(tmp_path)
    legacy, modern = _load_records(tmp_path)

    # target = final_best - gap = 0.6; running best first reaches it at k=3 (score 0.7 at k=2).
    assert evals_to_target(legacy, mode="gap", abs_gap=0.1) == 2.0
    assert evals_to_target(modern, mode="gap", abs_gap=0.1) == 2.0

    table = build_target_table([legacy, modern], mode="gap", abs_gap=0.1)
    assert list(table["final_cv_best"]) == [0.7, 0.7]
    assert list(table["evals_to_target"]) == [2.0, 2.0]
    assert table["converged_within_budget"].tolist() == [True, True]


def test_eval_params_helper_degrades_gracefully():
    """The canonical accessor: full config for v2 entries, {} for legacy/missing."""
    v2_entry = {"k": 0, "cv_score": 0.5, "wall_s": 0.5, "status": "complete",
                "params": {"max_depth": 4, "min_samples_leaf": 0.05}}
    legacy_entry = {"k": 0, "cv_score": 0.5, "wall_s": 0.5, "status": "complete"}

    assert eval_params(v2_entry) == {"max_depth": 4, "min_samples_leaf": 0.05}
    assert eval_params(legacy_entry) == {}
    assert eval_params({}) == {}  # missing key entirely (pre-v1 edge case)


def test_runrecord_v2_serialization_round_trip():
    """RunRecord.to_json emits schema v2 and per-evaluation params survive intact."""
    from benchmarks.methods import RunRecord

    fixture = _v2_record()  # field-for-field compatible with the RunRecord dataclass
    record = RunRecord(**fixture)

    payload = json.loads(record.to_json())
    assert payload["schema_version"] == RECORD_SCHEMA_VERSION
    for original, serialized in zip(fixture["records"], payload["records"]):
        assert eval_params(serialized) == eval_params(original)
