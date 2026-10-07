"""
src/features.py
===============
Leakage-free feature engineering for the NASA C-MAPSS dataset.

All features at cycle t are computed using ONLY data from cycles 1…t of
the same engine unit.  No future information leaks in.

Public API
----------
add_rolling_features(df, sensors, windows)
    Per-unit rolling mean, std, min, max.

add_slope_features(df, sensors, window)
    Least-squares slope of each sensor over the last N cycles.

add_drift_features(df, sensors, baseline_cycles)
    Difference from each unit's own first-N-cycle mean.

add_cycle_feature(df)
    Normalised (min-max scaled) cycle number.

build_features(df, dataset, processed_dir, normalize_fn)
    Orchestrate all of the above, loading the dropped-sensor list
    from EDA (data/processed/dropped_sensors_<dataset>.json).

save_feature_names(names, dataset, processed_dir)
load_feature_names(dataset, processed_dir)
    Persist the ordered list of feature columns used during training
    so the Streamlit app can reconstruct the same feature matrix.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from src.data import SENSOR_COLS, OP_COLS, _default_processed_dir

# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_WINDOWS: tuple[int, ...] = (5, 10, 20)
DEFAULT_SLOPE_WINDOW: int = 10
DEFAULT_BASELINE_CYCLES: int = 20


# ── Helpers ────────────────────────────────────────────────────────────────────

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_dropped_sensors(dataset: str, processed_dir: Optional[Path] = None) -> list[str]:
    """
    Load the list of near-constant sensors identified during EDA.
    Returns an empty list if the file does not exist yet (safe default).
    """
    processed_dir = processed_dir or _default_processed_dir()
    fpath = Path(processed_dir) / f"dropped_sensors_{dataset}.json"
    if not fpath.exists():
        warnings.warn(
            f"dropped_sensors_{dataset}.json not found in {processed_dir}. "
            "Run notebook 01_eda.ipynb first, or all sensors will be kept.",
            UserWarning,
            stacklevel=3,
        )
        return []
    with open(fpath) as f:
        return json.load(f)


def _sort_by_unit_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by [unit, cycle] in-place (returns a new DataFrame)."""
    return df.sort_values(["unit", "cycle"], kind="mergesort").reset_index(drop=True)


# ── Feature functions ─────────────────────────────────────────────────────────

def add_rolling_features(
    df: pd.DataFrame,
    sensors: Sequence[str],
    windows: Sequence[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """
    Add per-unit rolling statistics for each sensor.

    For each sensor s and window w, adds:
      {s}_mean_{w}  rolling mean
      {s}_std_{w}   rolling std  (NaN at cycle 1 filled with 0)
      {s}_min_{w}   rolling min
      {s}_max_{w}   rolling max

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'unit', 'cycle', and all columns in `sensors`.
    sensors : sequence of str
    windows : sequence of int
        Rolling window sizes in cycles.

    Returns
    -------
    pd.DataFrame  (copy with new columns appended)
    """
    df = _sort_by_unit_cycle(df.copy())
    g = df.groupby("unit", sort=False)

    for s in sensors:
        for w in windows:
            prefix = f"{s}_w{w}"
            df[f"{prefix}_mean"] = g[s].transform(
                lambda x, _w=w: x.rolling(_w, min_periods=1).mean()
            )
            df[f"{prefix}_std"] = g[s].transform(
                lambda x, _w=w: x.rolling(_w, min_periods=1).std().fillna(0)
            )
            df[f"{prefix}_min"] = g[s].transform(
                lambda x, _w=w: x.rolling(_w, min_periods=1).min()
            )
            df[f"{prefix}_max"] = g[s].transform(
                lambda x, _w=w: x.rolling(_w, min_periods=1).max()
            )
    return df


def add_slope_features(
    df: pd.DataFrame,
    sensors: Sequence[str],
    window: int = DEFAULT_SLOPE_WINDOW,
) -> pd.DataFrame:
    """
    Add the least-squares slope of each sensor over the last `window` cycles
    per unit.

    Slope captures the *rate of change* of degradation — a steeper negative
    (or positive) slope signals faster wear.

    Parameters
    ----------
    df : pd.DataFrame
    sensors : sequence of str
    window : int
        Number of cycles to include in the slope calculation.

    Returns
    -------
    pd.DataFrame  (copy with new {sensor}_slope_{window} columns)
    """
    df = _sort_by_unit_cycle(df.copy())

    def _rolling_slope(x: pd.Series, w: int) -> pd.Series:
        """Compute OLS slope over a rolling window."""
        result = np.zeros(len(x), dtype=float)
        vals = x.values
        for i in range(len(vals)):
            start = max(0, i - w + 1)
            seg = vals[start : i + 1]
            n = len(seg)
            if n < 2:
                result[i] = 0.0
            else:
                t = np.arange(n, dtype=float)
                # OLS slope via formula: (n*sum(tx) - sum(t)*sum(x)) / (n*sum(t^2) - sum(t)^2)
                sum_t = t.sum()
                sum_x = seg.sum()
                sum_tx = (t * seg).sum()
                sum_t2 = (t * t).sum()
                denom = n * sum_t2 - sum_t**2
                result[i] = (n * sum_tx - sum_t * sum_x) / denom if denom != 0 else 0.0
        return pd.Series(result, index=x.index)

    for s in sensors:
        df[f"{s}_slope_{window}"] = (
            df.groupby("unit", sort=False)[s]
            .transform(lambda x, _w=window: _rolling_slope(x, _w))
        )
    return df


def add_drift_features(
    df: pd.DataFrame,
    sensors: Sequence[str],
    baseline_cycles: int = DEFAULT_BASELINE_CYCLES,
) -> pd.DataFrame:
    """
    Add the deviation of each sensor from its unit's own early-life baseline.

    Drift = current_value - mean(values in first `baseline_cycles` cycles).

    This captures *relative* degradation: an engine whose s2 is 5 units above
    its own personal baseline is more worn than one whose s2 is 3 units below
    a global mean.

    Parameters
    ----------
    df : pd.DataFrame
    sensors : sequence of str
    baseline_cycles : int
        Number of early cycles to average as the unit baseline.

    Returns
    -------
    pd.DataFrame  (copy with new {sensor}_drift columns)
    """
    df = _sort_by_unit_cycle(df.copy())

    # Compute per-unit baseline (mean of first N cycles)
    baselines = (
        df[df["cycle"] <= baseline_cycles]
        .groupby("unit")[list(sensors)]
        .mean()
        .rename(columns={s: f"{s}_baseline" for s in sensors})
    )
    df = df.merge(baselines, on="unit", how="left")

    for s in sensors:
        df[f"{s}_drift"] = df[s] - df[f"{s}_baseline"]
        df.drop(columns=[f"{s}_baseline"], inplace=True)

    return df


def add_cycle_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a normalised cycle feature: cycle / max_cycle_of_unit.

    This encodes engine age as a fraction [0, 1] and is especially useful
    for test engines (whose max cycle is unknown at inference time, so we
    use the absolute cycle number instead — already provided as a raw column).

    Returns
    -------
    pd.DataFrame  (copy with new 'cycle_norm' column)
    """
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["cycle_norm"] = df["cycle"] / max_cycle
    return df


def add_lag_features(
    df: pd.DataFrame,
    sensors: Sequence[str],
    lags: Sequence[int] = (1, 3, 5),
) -> pd.DataFrame:
    """
    Add lagged values of each sensor (per unit).

    Lag features let the model see the raw trajectory without relying solely
    on rolling statistics.

    Parameters
    ----------
    df : pd.DataFrame
    sensors : sequence of str
    lags : sequence of int
        Number of cycles to look back.

    Returns
    -------
    pd.DataFrame  (copy with new {sensor}_lag_{k} columns, NaN filled with 0)
    """
    df = _sort_by_unit_cycle(df.copy())
    g = df.groupby("unit", sort=False)
    for s in sensors:
        for k in lags:
            df[f"{s}_lag_{k}"] = g[s].transform(lambda x, _k=k: x.shift(_k).fillna(0))
    return df


# ── Orchestrator ──────────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    dataset: str = "FD001",
    processed_dir: Optional[Path] = None,
    normalize_fn: Optional[Callable] = None,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    slope_window: int = DEFAULT_SLOPE_WINDOW,
    baseline_cycles: int = DEFAULT_BASELINE_CYCLES,
    lags: Sequence[int] = (1, 3, 5),
    include_op_cols: bool = True,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Steps:
      1. Optional per-operating-condition normalisation (FD002/FD004).
      2. Drop near-constant sensors from EDA.
      3. Rolling mean/std/min/max (windows 5, 10, 20).
      4. Sensor slopes (window 10).
      5. Sensor drift from unit baseline (first 20 cycles).
      6. Lag features (lags 1, 3, 5).
      7. Normalised cycle number.
      8. Drop raw sensor columns (keep engineered ones).

    Parameters
    ----------
    df : pd.DataFrame
        Training or test DataFrame (may or may not have 'rul' column).
    dataset : str
        Used to look up the dropped-sensors list.
    processed_dir : Path, optional
    normalize_fn : callable, optional
        If provided, called as normalize_fn(df) → df before feature engineering.
        Use for FD002/FD004 operating-condition normalisation.
    windows : sequence of int
    slope_window : int
    baseline_cycles : int
    lags : sequence of int
    include_op_cols : bool
        Whether to keep op1/op2/op3 in the output (useful for multi-condition datasets).

    Returns
    -------
    pd.DataFrame
        Feature matrix with 'unit', 'cycle', optionally 'rul'/'rul_unclipped',
        all engineered columns, and no raw sensor values.
    """
    df = _sort_by_unit_cycle(df.copy())

    # Step 1: optional normalisation
    if normalize_fn is not None:
        df = normalize_fn(df)

    # Step 2: identify active sensors
    dropped = _load_dropped_sensors(dataset, processed_dir)
    active_sensors = [s for s in SENSOR_COLS if s not in dropped]

    # Step 3: rolling features
    df = add_rolling_features(df, active_sensors, windows=windows)

    # Step 4: slope features
    df = add_slope_features(df, active_sensors, window=slope_window)

    # Step 5: drift features
    df = add_drift_features(df, active_sensors, baseline_cycles=baseline_cycles)

    # Step 6: lag features
    df = add_lag_features(df, active_sensors, lags=lags)

    # Step 7: normalised cycle
    df = add_cycle_feature(df)

    # Step 8: drop raw sensor columns (they are now encoded in engineered features)
    drop_cols = [s for s in SENSOR_COLS if s in df.columns]
    df = df.drop(columns=drop_cols)

    # Optionally drop op columns
    if not include_op_cols:
        df = df.drop(columns=[c for c in OP_COLS if c in df.columns])

    # Sanity check: no NaNs in feature columns
    feature_cols = get_feature_columns(df)
    nan_count = df[feature_cols].isnull().sum().sum()
    if nan_count > 0:
        # Fill any residual NaNs with 0 and warn
        warnings.warn(
            f"Found {nan_count} NaN values in feature columns after engineering. "
            "Filling with 0.",
            UserWarning,
        )
        df[feature_cols] = df[feature_cols].fillna(0)

    return df.reset_index(drop=True)


# ── Feature name utilities ────────────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the ordered list of feature columns (everything except meta and label cols).
    """
    exclude = {"unit", "cycle", "rul", "rul_unclipped"} | set(SENSOR_COLS) | set(OP_COLS)
    return [c for c in df.columns if c not in exclude]


def save_feature_names(
    names: list[str],
    dataset: str = "FD001",
    processed_dir: Optional[Path] = None,
) -> Path:
    """Persist the feature column list for reproducibility."""
    processed_dir = processed_dir or _default_processed_dir()
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out = processed_dir / f"feature_names_{dataset}.json"
    with open(out, "w") as f:
        json.dump(names, f, indent=2)
    return out


def load_feature_names(
    dataset: str = "FD001",
    processed_dir: Optional[Path] = None,
) -> list[str]:
    """Load the feature column list saved during training."""
    processed_dir = processed_dir or _default_processed_dir()
    fpath = Path(processed_dir) / f"feature_names_{dataset}.json"
    if not fpath.exists():
        raise FileNotFoundError(
            f"Feature names file not found: {fpath}\n"
            "Run the feature engineering pipeline first."
        )
    with open(fpath) as f:
        return json.load(f)
