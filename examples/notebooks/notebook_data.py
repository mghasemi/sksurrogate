"""Seeded datasets shared by the three end-to-end example notebooks.

The notebooks are meant to run offline and to be reproducible, so the default
source is a *synthetic* generator per task. Real OpenML frames are available as
an explicit opt-in.

Source selection
----------------
``SKSURROGATE_NOTEBOOK_DATA`` (environment variable, default ``"synthetic"``)::

    synthetic   seeded generators below; mixed dtypes and injected missingness
    openml      OpenML ``credit-g`` (classification) / ``wine_quality`` (regression)

The OpenML path needs network access and (for most datasets) an API key. Put the
key in ``SKSURROGATE_OPENML_KEY`` or in ``~/.openml/config``; openml-py stores a
*display-masked* token in its config for some versions, so the key is also set
explicitly on ``openml.config`` before the download starts.

Why synthetic by default
------------------------
The generators are not toys: each one contains the structure that makes a
workflow *necessary* — non-linear effects, an interaction, a categorical
feature, a redundant column, heteroscedastic noise, imbalance, and missing
values with a known mechanism. That way the notebooks exercise profiling,
imputation, search, evaluation and explanation in the same run while staying
deterministic and network-free.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

#: One seed for every generator and every notebook cell.
SEED = 7

#: ``"synthetic"`` (default) or ``"openml"``.
SOURCE = os.environ.get("SKSURROGATE_NOTEBOOK_DATA", "synthetic").strip().lower()

#: OpenML pins used when ``SOURCE == "openml"``.
OPENML_IDS = {
    "classification": 31,  # credit-g: 1000 rows, 20 features (13 nominal), binary class
    "regression": 287,  # wine_quality: 1599 rows, 11 numeric features, sensory score
}


@dataclass(frozen=True)
class Dataset:
    """A modelling frame plus the provenance a notebook should print.

    :param frame: the full frame, target column included
    :param target: name of the target column
    :param source: how the frame was obtained (``synthetic`` / ``openml``)
    :param description: one-line human description for the provenance table
    :param notes: extra JSON-safe facts (missing mechanism, injected gaps, ...)
    """

    frame: pd.DataFrame
    target: str
    source: str
    description: str
    notes: dict = field(default_factory=dict)

    @property
    def features(self) -> list[str]:
        """Feature column names, in frame order."""
        return [column for column in self.frame.columns if column != self.target]


# --------------------------------------------------------------------------- #
# Classification: 3-tier risk, imbalanced, mixed dtypes, MAR + MCAR gaps
# --------------------------------------------------------------------------- #


def _classification_synthetic(n_samples: int, seed: int) -> Dataset:
    """A rare-event churn flag from tenure, usage, support load and contract type.

    The label is the top 12 % of a latent risk score, so the positive class is
    rare and the ranking metrics (lift, cumulative gain, calibration) carry the
    story. ``nps_score`` is generated as a noisy
    function of the other drivers *and* then made missing with a mechanism that
    depends on ``late_payments`` — i.e. MAR, not MCAR. That gives the imputation
    comparison in the notebook something real to find.
    """
    rng = np.random.default_rng(seed)

    tenure = rng.gamma(2.0, 9.0, n_samples)  # months with the provider
    spend = rng.lognormal(4.0, 0.45, n_samples)  # mean monthly bill
    support = rng.poisson(1.1, n_samples)
    late = rng.poisson(0.5, n_samples)
    session = rng.gamma(3.0, 6.0, n_samples)  # mean session length, minutes
    logins = rng.poisson(8.0, n_samples)
    discounts = rng.binomial(12, 0.25, n_samples)
    contract = rng.choice(["monthly", "annual", "biennial"], n_samples, p=[0.55, 0.30, 0.15])
    region = rng.choice(["north", "south", "east", "west"], n_samples, p=[0.42, 0.25, 0.20, 0.13])

    # nps is recoverable from the observable drivers -> knn/iterative imputation
    # can beat a marginal median, which is the point of the comparison.
    nps = (
        34.0
        - 6.0 * support
        - 5.0 * late
        + 0.12 * session
        - 2.0 * np.log1p(tenure)
        + rng.normal(0.0, 8.0, n_samples)
    )
    nps = np.clip(nps, -100.0, 100.0)

    risk = (
        1.15 * late
        + 0.55 * support
        - 0.85 * np.log1p(tenure)
        - 0.35 * np.log(spend)
        - 0.020 * nps
        + 0.35 * np.sin(tenure / 9.0)  # non-linear tenure effect
        + 0.90 * (contract == "monthly")
        + 0.45 * (contract == "annual")
        + 0.25 * (region == "east")
        + 0.004 * spend * (contract == "monthly")  # spend x contract interaction
        + rng.normal(0.0, 0.62, n_samples)
    )
    # A rare-event flag, not a balanced three-way split: the top 12 % of the risk
    # score is "high risk". Severe imbalance is what makes the ranking plots
    # (lift / cumulative gain), calibration and threshold analysis meaningful.
    high_cut = np.quantile(risk, 0.88)
    high_risk = (risk > high_cut).astype(int)

    frame = pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_spend": spend,
            "support_calls": support,
            "late_payments": late,
            "nps_score": nps,
            "avg_session_min": session,
            "app_logins": logins,
            "discounts_used": discounts,
            # A redundant copy: perfect collinearity with monthly_spend.
            "spend_ytd": spend * 12.0,
            # Pure noise, kept so pruning has something to remove.
            "billing_code": rng.integers(1000, 9999, n_samples),
            "campaign_id": rng.integers(1, 40, n_samples),
            "region": region,
            "contract": contract,
            "high_risk": high_risk,
        }
    )

    # MAR: the chance of a missing nps grows with the number of late payments.
    # (Masking happens before the duplicate column is derived, so the exact
    # duplication the collinearity demo needs survives the gaps.)
    mar_rate = np.clip(0.06 + 0.11 * late, 0.0, 0.55)
    frame.loc[rng.random(n_samples) < mar_rate, "nps_score"] = np.nan
    # MCAR gaps on three more columns.
    for column, rate in (
        ("monthly_spend", 0.045),
        ("avg_session_min", 0.07),
        ("app_logins", 0.03),
        ("region", 0.025),
    ):
        frame.loc[rng.random(n_samples) < rate, column] = np.nan
    frame["spend_ytd"] = frame["monthly_spend"] * 12.0

    return Dataset(
        frame=frame,
        target="high_risk",
        source="synthetic",
        description="Synthetic subscription churn flag (12 % positive, mixed dtypes)",
        notes={
            "generator": "notebook_data._classification_synthetic",
            "n_samples": int(n_samples),
            "seed": int(seed),
            "missing_mechanism": "nps_score is MAR on late_payments; other columns MCAR",
        },
    )


# --------------------------------------------------------------------------- #
# Regression: energy intensity, non-linear + interaction, heteroscedastic noise
# --------------------------------------------------------------------------- #


def _regression_signal(
    insulation,
    glazing,
    age_z,
    floor_area,
    occupants,
    degree_days,
    thermal_mass,
    setpoint,
    system_offset,
):
    """The noiseless response surface, in physical units.

    Kept as a named function so ``regression_signal`` can expose it: a notebook that
    compares what a model learned against what the generator put there needs one
    definition, not two that drift apart.
    """
    return (
        58.0
        + 9.0 * np.sin(np.pi * insulation * glazing)
        + 18.0 * (age_z - 0.4) ** 2
        + 6.5 * thermal_mass
        + 4.0 * occupants / 2.4
        + 3.2 * degree_days
        + 2.5 * insulation * degree_days  # interaction: insulation pays off in cold years
        + 1.9 * (setpoint - 21.0)
        - 0.35 * floor_area / 180.0
        + system_offset
    )


def regression_signal(frame: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.Series:
    """The synthetic building target *before* its multiplicative noise.

    Verification aid: it answers "what did the generator put there, at these feature
    values?" so a partial-dependence surface can be compared with the truth instead of
    being eyeballed. ``reference`` supplies the age standardisation and the column fills
    for missing cells (pass the full frame when scoring a subset); the coefficients match
    ``_regression_synthetic`` exactly.

    Rows whose ``heating_system`` is missing are treated as ``heat_pump`` (the default
    branch of ``np.select``); score complete cases when the distinction matters.
    """
    ref = frame if reference is None else reference
    age_mean = float(ref["building_age_years"].mean())
    # ddof=0 to match the generator's numpy standardisation.
    age_std = float(ref["building_age_years"].std(ddof=0))
    fills = {
        column: float(ref[column].mean())
        for column in ("thermal_mass", "occupants", "degree_days_z", "setpoint_c")
    }
    work = frame.fillna(fills)
    system_offset = np.select(
        [work["heating_system"] == "gas", work["heating_system"] == "electric"],
        [0.0, 7.0],
        default=-4.0,
    )
    values = _regression_signal(
        work["insulation_score"].to_numpy(dtype=float),
        work["glazing_score"].to_numpy(dtype=float),
        (work["building_age_years"].to_numpy(dtype=float) - age_mean) / age_std,
        work["floor_area_m2"].to_numpy(dtype=float),
        work["occupants"].to_numpy(dtype=float),
        work["degree_days_z"].to_numpy(dtype=float),
        work["thermal_mass"].to_numpy(dtype=float),
        work["setpoint_c"].to_numpy(dtype=float),
        system_offset,
    )
    return pd.Series(values, index=frame.index, name="energy_kwh_m2_signal")


def _regression_synthetic(n_samples: int, seed: int) -> Dataset:
    """Specific energy demand of a building stock, in kWh per m² and year.

    The response is a classical non-linear surface (sine of an interaction,
    quadratic in age, linear in degree days) with *multiplicative* noise, so a
    constant-variance assumption is wrong on purpose. ``floor_area`` has a
    consistent physical effect, while one duplicated column and one noise column
    make feature pruning meaningful.
    """
    rng = np.random.default_rng(seed)

    insulation = rng.uniform(0.05, 1.0, n_samples)
    glazing = rng.uniform(0.05, 1.0, n_samples)
    age = rng.gamma(3.0, 6.0, n_samples)
    floor_area = rng.lognormal(5.2, 0.35, n_samples)
    occupants = rng.poisson(2.4, n_samples)
    degree_days = rng.normal(0.0, 1.0, n_samples)  # standardised climate
    thermal_mass = rng.uniform(0.0, 1.0, n_samples)
    heating_system = rng.choice(
        ["gas", "electric", "heat_pump"], n_samples, p=[0.45, 0.30, 0.25]
    )
    # The setpoint is a *recoverable* column: heat pumps run cooler, gas boilers
    # warmer, and the value drifts with thermal mass. That gives the imputation
    # comparison something to find, and its missingness is MAR on the categorical
    # heating system (heat-pump retrofits were metered less consistently).
    setpoint = (
        21.0
        - 2.4 * (heating_system == "heat_pump")
        + 1.0 * (heating_system == "gas")
        - 0.2 * (heating_system == "electric")
        + 0.020 * (age - age.mean())
        - 1.0 * thermal_mass
        + rng.normal(0.0, 0.45, n_samples)
    )

    age_z = (age - age.mean()) / age.std()
    system_offset = np.select(
        [heating_system == "gas", heating_system == "electric"],
        [0.0, 7.0],
        default=-4.0,
    )
    energy = _regression_signal(
        insulation, glazing, age_z, floor_area, occupants, degree_days, thermal_mass,
        setpoint, system_offset,
    )
    # Multiplicative noise: the spread grows with the level. The noise is drawn from the
    # signal's level, so the heteroscedasticity is a property of the building, not of the
    # drawing order.
    noise = rng.normal(0.0, 1.0, n_samples) * (2.2 + 0.16 * np.maximum(energy - 40.0, 0.0))
    energy = energy + noise

    frame = pd.DataFrame(
        {
            "insulation_score": insulation,
            "glazing_score": glazing,
            "building_age_years": age,
            "floor_area_m2": floor_area,
            "occupants": occupants,
            "degree_days_z": degree_days,
            "thermal_mass": thermal_mass,
            "setpoint_c": setpoint,
            # Perfect duplicate of floor_area_m2: multicollinearity demo.
            "gross_area_m2": floor_area,
            # Pure noise.
            "meter_serial": rng.integers(10_000, 99_999, n_samples),
            "heating_system": heating_system,
            "energy_kwh_m2": energy,
        }
    )

    for column, rate in (
        ("degree_days_z", 0.055),
        ("thermal_mass", 0.04),
        ("occupants", 0.03),
        ("heating_system", 0.02),
    ):
        frame.loc[rng.random(n_samples) < rate, column] = np.nan
    # MAR on a categorical driver: more setpoint readings lost on heat pumps.
    setpoint_rate = np.where(heating_system == "heat_pump", 0.16, 0.04)
    frame.loc[rng.random(n_samples) < setpoint_rate, "setpoint_c"] = np.nan

    return Dataset(
        frame=frame,
        target="energy_kwh_m2",
        source="synthetic",
        description="Synthetic building energy intensity (non-linear, heteroscedastic)",
        notes={
            "generator": "notebook_data._regression_synthetic",
            "n_samples": int(n_samples),
            "seed": int(seed),
            "missing_mechanism": "setpoint_c is MAR on heating_system; other columns MCAR",
        },
    )


# --------------------------------------------------------------------------- #
# Time series: daily demand, two seasonalities, known-at-origin promotion
# --------------------------------------------------------------------------- #


def daily_series_data(
    n_days: int = 1095, start: str = "2022-01-01", seed: int = SEED
) -> Dataset:
    """Daily unit demand with trend, weekly + annual seasonality and promotions.

    ``promo`` is a *known-at-origin* exogenous driver (a planned campaign), which
    is the only kind of exogenous column a forecast may use. A handful of target
    values and promotion flags are blanked out to exercise the temporal
    contract's gap handling; the exact indices are returned in ``notes`` so the
    notebook can report how many rows the design builder drops.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n_days, freq="D")
    step = np.arange(n_days, dtype=float)

    promo = (rng.random(n_days) < 0.12).astype(float)
    holiday = np.isin(dates.strftime("%m-%d"), ["01-01", "07-04", "12-25", "12-26"]).astype(float)

    # AR(1) noise: unexplained autocorrelation, as in real demand.
    innovations = rng.normal(0.0, 0.8, n_days)
    ar = np.empty(n_days)
    ar[0] = innovations[0]
    for index in range(1, n_days):
        ar[index] = 0.55 * ar[index - 1] + innovations[index]

    level = (
        20.0
        + 0.015 * step
        + 4.0 * np.sin(2 * np.pi * step / 7.0)
        + 0.8 * np.cos(4 * np.pi * step / 7.0)
        + 2.5 * np.sin(2 * np.pi * step / 365.25)
        + 3.0 * promo
        + 6.0 * holiday
        + ar
    )

    frame = pd.DataFrame({"date": dates, "demand": level, "promo": promo})

    # A few blanks: 1.2 % of the target, 3 % of the promotion flag.
    target_gaps = np.flatnonzero(rng.random(n_days) < 0.012)
    promo_gaps = np.flatnonzero(rng.random(n_days) < 0.03)
    frame.loc[target_gaps, "demand"] = np.nan
    frame.loc[promo_gaps, "promo"] = np.nan

    return Dataset(
        frame=frame,
        target="demand",
        source="synthetic",
        description="Synthetic daily demand (trend, weekly + annual seasonality, promotions)",
        notes={
            "generator": "notebook_data.daily_series_data",
            "n_days": int(n_days),
            "start": dates[0].strftime("%Y-%m-%d"),
            "end": dates[-1].strftime("%Y-%m-%d"),
            "horizon": 7,
            "seasonality": 7,
            "target_gap_indices": target_gaps.tolist(),
            "promo_gap_indices": promo_gaps.tolist(),
        },
    )


# --------------------------------------------------------------------------- #
# OpenML opt-in
# --------------------------------------------------------------------------- #


def openml_frame(role: str, max_rows: int | None = 1200, seed: int = SEED) -> Dataset:
    """Fetch a real OpenML frame for ``role`` in ``{"classification", "regression"}``.

    The frame is returned with its native dtypes (nominal columns stay strings)
    so the profiling/imputation cells see the same kind of data a user would
    upload. Rows are subsampled (``max_rows``) with a fixed seed to keep the
    notebook runtime bounded; the target column is renamed to ``target``.
    """
    if role not in OPENML_IDS:
        raise ValueError("role must be 'classification' or 'regression'")
    try:
        import openml
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The OpenML data source needs the openml package: pip install 'sksurrogate[benchmark]'"
        ) from exc

    key = os.environ.get("SKSURROGATE_OPENML_KEY", "").strip()
    if key:
        # openml 0.15 substitutes a display token for an env-supplied key; setting
        # it on the config object directly is what actually reaches the server.
        openml.config.apikey = key

    dataset = openml.datasets.get_dataset(int(OPENML_IDS[role]))
    features, target, _ignored, names = dataset.get_data(dataset_format="dataframe")
    frame = pd.DataFrame(features).copy()
    if target is None:  # pragma: no cover - defensive, layout changed upstream
        target_name = names[-1]
    else:
        target_series = target.iloc[:, 0] if isinstance(target, pd.DataFrame) else target
        target_name = getattr(target, "name", None) or names[-1]
        frame[target_name] = target_series

    if max_rows is not None and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=seed)
    frame = frame.reset_index(drop=True)

    if role == "regression" and not pd.api.types.is_numeric_dtype(frame[target_name]):
        raise ValueError(
            "the configured regression dataset has a non-numeric target %r" % target_name
        )
    if role == "classification":
        # String labels are fine for sklearn; keep them (they are informative).
        frame[target_name] = frame[target_name].astype(str)

    renamed = frame.rename(columns={target_name: "target"})
    return Dataset(
        frame=renamed,
        target="target",
        source="openml",
        description="OpenML %r (did=%d), %d rows sampled"
        % (dataset.name, int(OPENML_IDS[role]), len(renamed)),
        notes={
            "openml_dataset": dataset.name,
            "openml_id": int(OPENML_IDS[role]),
            "rows_sampled": int(len(renamed)),
            "seed": int(seed),
        },
    )


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def classification_data(source: str | None = None, n_samples: int = 900, seed: int = SEED) -> Dataset:
    """The classification notebook's frame (see module docstring for source rules)."""
    chosen = (source or SOURCE).lower()
    if chosen == "openml":
        return openml_frame("classification", seed=seed)
    if chosen != "synthetic":
        raise ValueError("source must be 'synthetic' or 'openml', got %r" % chosen)
    return _classification_synthetic(n_samples=n_samples, seed=seed)


def regression_data(source: str | None = None, n_samples: int = 1000, seed: int = SEED) -> Dataset:
    """The regression notebook's frame (see module docstring for source rules).

    1000 rows rather than 800: the ensemble needs enough rows to afford the depth that a
    smooth two-feature interaction requires, and the notebook measures how much of it the
    model recovers (``regression_signal`` gives the surface it is measured against).
    """
    chosen = (source or SOURCE).lower()
    if chosen == "openml":
        return openml_frame("regression", seed=seed)
    if chosen != "synthetic":
        raise ValueError("source must be 'synthetic' or 'openml', got %r" % chosen)
    return _regression_synthetic(n_samples=n_samples, seed=seed)


def artifact_dir(name: str, root: Path | None = None) -> Path:
    """Create and return ``<cwd>/<name>`` for a notebook's report artifacts."""
    path = (Path.cwd() if root is None else root) / name
    path.mkdir(parents=True, exist_ok=True)
    return path
