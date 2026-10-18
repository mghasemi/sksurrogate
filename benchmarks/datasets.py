"""Dataset loading for the benchmark suite (DESIGN.md section 3).

OpenML datasets are pinned by data id (``did``) in ``datasets.json``, never
by name, and downloaded once into a local cache
(``$SKSURROGATE_BENCH_CACHE`` or ``~/.cache/sksurrogate-benchmarks``).
Galaxy3 is bundled with the repository under ``data/``.

The last column of every pinned dataset is the target; the nominal features
of the five OpenML datasets are at most a single binary attribute (ionosphere),
which is ordinal-encoded at load time so all methods see identical numeric
inputs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SPEC_PATH = Path(__file__).resolve().with_name("datasets.json")
_CACHE_ENV = "SKSURROGATE_BENCH_CACHE"


def _load_spec() -> dict:
    with open(_SPEC_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def cache_dir() -> Path:
    """Directory where downloaded OpenML datasets are cached."""
    raw = os.environ.get(_CACHE_ENV, "~/.cache/sksurrogate-benchmarks")
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_names() -> list[str]:
    """All benchmark dataset names (OpenML + local), in DESIGN.md order."""
    spec = _load_spec()
    openml_names = list(spec["datasets"].keys())
    local_names = list(spec.get("local", {}).keys())
    return [name for name in ["ionosphere", "credit-g", "arrhythmia", "spambase", "optdigits", "galaxy3"] if name in openml_names + local_names]


def _download_openml(name: str, did: int) -> pd.DataFrame:
    """Fetch an OpenML dataset by id and cache it as CSV (target = last column).

    Uses the openml-py >= 0.15 object-oriented API: ``get_dataset(did)`` returns
    an ``OpenMLDataset`` whose ``get_data(dataset_format=...)`` yields a
    ``(features, target, ignore_flags, attribute_names)`` tuple. The result is
    recombined into one frame with the class column last (the layout
    :func:`load_dataset` expects). Pinned via setup.py extras to openml<0.16.
    """
    try:
        import openml  # noqa: PLC0415 - optional dependency ([benchmark] extra)
    except ImportError as exc:
        raise ImportError(
            f"openml is required to download '{name}' (did={did}). "
            "Install it with `pip install sksurrogate[benchmark]`."
        ) from exc

    csv_path = cache_dir() / f"{name}.csv"
    if not csv_path.exists():
        dataset = openml.datasets.get_dataset(did)
        features, target, _ignore_flags, _names = dataset.get_data(
            dataset_format="dataframe"
        )
        if not isinstance(features, pd.DataFrame):  # pragma: no cover - defensive
            raise TypeError(f"openml get_data(did={did}) did not return a DataFrame")

        feats = features.reset_index(drop=True)
        if target is None:
            # openml-py leaves the class attribute as the last column of
            # `features` (the legacy single-frame layout load_dataset expects).
            df = feats
        else:
            t = target.reset_index(drop=True)
            if isinstance(t, pd.DataFrame):
                t = t.iloc[:, 0]
            # The class attribute may also appear inside `features`; drop the dup.
            t_name = getattr(target, "name", None) or _names[-1]
            if t_name in feats.columns:
                feats = feats.drop(columns=[t_name])
            df = pd.concat([feats, t.rename(t_name)], axis=1)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)
    return df


def load_dataset(name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a benchmark dataset.

    :param name: one of :func:`dataset_names`.
    :return: ``(X, y)`` — feature DataFrame (numeric only) and target Series.
    :raises ValueError: if the name is unknown or the loaded data does not
        match the pinned metadata in ``datasets.json`` (guards against
        column-order surprises when OpenML file layouts change).
    """
    spec = _load_spec()

    if name in spec.get("local", {}):
        entry = spec["local"][name]
        raw_path = _REPO_ROOT / entry["path"]
        source_desc = str(raw_path)
        df = pd.read_csv(raw_path, sep="\t")
        target_column = entry["target_column"]
    elif name in spec["datasets"]:
        entry = spec["datasets"][name]
        source_desc = f"OpenML dataset did={entry['did']}"
        df = _download_openml(name, int(entry["did"]))
        if not isinstance(df, pd.DataFrame):  # pragma: no cover - defensive
            raise TypeError(f"openml.get_data(did={entry['did']}) did not return a DataFrame")
        target_column = None
    else:
        known = ", ".join(dataset_names())
        raise ValueError(f"Unknown benchmark dataset {name!r}. Known datasets: {known}")

    if target_column is None:
        # Pinned OpenML datasets carry the class attribute in the last column.
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1].copy()
    else:
        if target_column not in df.columns:
            raise ValueError(f"Target column {target_column!r} missing from {source_desc}")
        y = df[target_column]
        X = df.drop(columns=[target_column]).copy()

    # Ordinal-encode any nominal feature columns (only ionosphere has one).
    for column in X.columns:
        if X[column].dtype == object:
            codes, _ = pd.factorize(X[column])
            X[column] = codes.astype(int)

    # Guard against layout drift vs the pinned metadata.
    expected_n = int(entry["n"])
    if len(df) != expected_n:
        raise ValueError(f"{name}: expected {expected_n} rows, got {len(df)}")
    expected_classes = int(entry["classes"])
    if y.nunique() != expected_classes:
        raise ValueError(
            f"{name}: expected target with {expected_classes} classes "
            f"(last column), found {y.nunique()} — check datasets.json pinning"
        )

    return X.reset_index(drop=True), y.reset_index(drop=True)
