"""
src/data.py
===========
Core data loading and preprocessing for the NASA C-MAPSS turbofan dataset.

Public API
----------
load_raw(dataset, data_dir, split)
    Load a raw train or test split as a DataFrame with named columns.

attach_rul_labels(df)
    Add a 'rul' column to a training DataFrame.

load_test_rul(dataset, data_dir)
    Load the ground-truth RUL values for a test set.

save_processed(df, name, processed_dir)
load_processed(name, processed_dir)
    Persist and reload processed DataFrames as Parquet for speed.

normalize_per_operating_condition(df, n_clusters, random_state)
    Cluster operating conditions (for FD002/FD004) and z-score-normalise
    sensor readings within each cluster. Used for multi-dataset experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── Column definitions ─────────────────────────────────────────────────────────

SENSOR_COLS: list[str] = [f"s{i}" for i in range(1, 22)]          # s1 … s21
OP_COLS: list[str] = [f"op{i}" for i in range(1, 4)]              # op1 op2 op3
BASE_COLS: list[str] = ["unit", "cycle"] + OP_COLS + SENSOR_COLS   # 26 columns

# Valid dataset identifiers
VALID_DATASETS = ("FD001", "FD002", "FD003", "FD004")

# ── Helpers ────────────────────────────────────────────────────────────────────

def _project_root() -> Path:
    """Return the project root (two levels above this file)."""
    return Path(__file__).resolve().parent.parent


def _default_raw_dir() -> Path:
    return _project_root() / "data" / "raw"


def _default_processed_dir() -> Path:
    return _project_root() / "data" / "processed"


def _validate_dataset(dataset: str) -> None:
    if dataset not in VALID_DATASETS:
        raise ValueError(
            f"dataset must be one of {VALID_DATASETS}, got {dataset!r}"
        )


# ── Core loaders ──────────────────────────────────────────────────────────────

def load_raw(
    dataset: str = "FD001",
    split: str = "train",
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load a raw C-MAPSS split from disk.

    Parameters
    ----------
    dataset : str
        One of 'FD001', 'FD002', 'FD003', 'FD004'.
    split : str
        'train' or 'test'.
    data_dir : Path, optional
        Directory containing the raw .txt files.
        Defaults to <project_root>/data/raw/.

    Returns
    -------
    pd.DataFrame
        26 columns: unit, cycle, op1–op3, s1–s21.
        dtypes: int for unit/cycle, float64 for the rest.
    """
    _validate_dataset(dataset)
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    data_dir = data_dir or _default_raw_dir()
    fpath = Path(data_dir) / f"{split}_{dataset}.txt"

    if not fpath.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {fpath}\n"
            "Run `python scripts/download_data.py` to fetch the dataset."
        )

    df = pd.read_csv(
        fpath,
        sep=r"\s+",
        header=None,
        names=BASE_COLS,
        dtype={c: "float64" for c in OP_COLS + SENSOR_COLS} | {"unit": int, "cycle": int},
    )

    # Drop any trailing NaN columns that appear in some file versions
    df = df.dropna(axis=1, how="all")

    # Ensure column count is exactly 26
    if len(df.columns) != len(BASE_COLS):
        raise ValueError(
            f"Expected {len(BASE_COLS)} columns, got {len(df.columns)} "
            f"after loading {fpath}. Check the raw file."
        )

    return df.reset_index(drop=True)


def attach_rul_labels(df: pd.DataFrame, clip: Optional[int] = None) -> pd.DataFrame:
    """
    Add a 'rul' (Remaining Useful Life) column to a training DataFrame.

    For each engine unit, RUL at cycle t = max_cycle_of_that_unit - t.
    The last cycle of every engine therefore has rul == 0.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'unit' and 'cycle' columns (i.e. output of load_raw).
    clip : int, optional
        If given, RUL values are clipped at this maximum (e.g. 125).
        Clipping removes the uninformative plateau at the start of engine
        life where degradation is not yet visible.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with a new 'rul' column (and 'rul_unclipped').
    """
    if "unit" not in df.columns or "cycle" not in df.columns:
        raise ValueError("DataFrame must have 'unit' and 'cycle' columns.")

    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = (max_cycle - df["cycle"]).astype(int)

    if clip is not None:
        df["rul_unclipped"] = df["rul"]
        df["rul"] = df["rul"].clip(upper=clip)

    return df


def load_test_rul(
    dataset: str = "FD001",
    data_dir: Optional[Path] = None,
) -> pd.Series:
    """
    Load the ground-truth RUL values for a test set.

    Each value corresponds to one engine in the test split, ordered by
    engine unit number (unit 1, 2, 3, …).

    Parameters
    ----------
    dataset : str
        One of 'FD001', 'FD002', 'FD003', 'FD004'.
    data_dir : Path, optional
        Directory containing the raw .txt files.

    Returns
    -------
    pd.Series
        Integer RUL values indexed 1 … N_engines, named 'rul'.
    """
    _validate_dataset(dataset)
    data_dir = data_dir or _default_raw_dir()
    fpath = Path(data_dir) / f"RUL_{dataset}.txt"

    if not fpath.exists():
        raise FileNotFoundError(
            f"RUL file not found: {fpath}\n"
            "Run `python scripts/download_data.py` to fetch the dataset."
        )

    rul = pd.read_csv(fpath, header=None, names=["rul"], squeeze=False)["rul"]
    rul.index = pd.RangeIndex(start=1, stop=len(rul) + 1, step=1)
    rul.index.name = "unit"
    return rul.astype(int)


# ── Processed data I/O ────────────────────────────────────────────────────────

def save_processed(
    df: pd.DataFrame,
    name: str,
    processed_dir: Optional[Path] = None,
) -> Path:
    """
    Save a DataFrame as Parquet to the processed data directory.

    Parameters
    ----------
    df : pd.DataFrame
    name : str
        Filename without extension, e.g. 'train_FD001_features'.
    processed_dir : Path, optional

    Returns
    -------
    Path
        Full path of the saved file.
    """
    processed_dir = processed_dir or _default_processed_dir()
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out = processed_dir / f"{name}.parquet"
    df.to_parquet(out, index=False, engine="pyarrow")
    return out


def load_processed(
    name: str,
    processed_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load a Parquet file from the processed data directory.

    Parameters
    ----------
    name : str
        Filename without extension, e.g. 'train_FD001_features'.
    processed_dir : Path, optional

    Returns
    -------
    pd.DataFrame
    """
    processed_dir = processed_dir or _default_processed_dir()
    fpath = Path(processed_dir) / f"{name}.parquet"
    if not fpath.exists():
        raise FileNotFoundError(
            f"Processed file not found: {fpath}\n"
            "Run the feature engineering pipeline first."
        )
    return pd.read_parquet(fpath, engine="pyarrow")


# ── Operating-condition normalisation (FD002 / FD004) ─────────────────────────

def normalize_per_operating_condition(
    df: pd.DataFrame,
    n_clusters: int = 6,
    random_state: int = 42,
    scaler_cache: Optional[dict] = None,
) -> tuple[pd.DataFrame, KMeans, dict[int, StandardScaler]]:
    """
    Cluster operating conditions and z-score-normalise sensor readings
    within each cluster.

    FD002 and FD004 have 6 distinct operating conditions defined by
    (op1, op2, op3). Normalising per condition removes the condition-
    dependent offset and makes sensor values comparable across the fleet.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain op1, op2, op3 and s1–s21 columns.
    n_clusters : int
        Number of operating-condition clusters (6 for FD002/FD004, 1 for FD001).
    random_state : int
        Reproducibility seed for KMeans.
    scaler_cache : dict, optional
        If provided, use pre-fitted {cluster_id: StandardScaler} scalers
        (for applying training-set normalisations to the test set).

    Returns
    -------
    df_norm : pd.DataFrame
        DataFrame with sensor columns replaced by their within-cluster
        z-scores.  Original op columns are retained unchanged.
    kmeans : KMeans
        Fitted KMeans model (save and reuse for the test set).
    scalers : dict[int, StandardScaler]
        Mapping from cluster label → fitted StandardScaler.
    """
    df = df.copy()
    op_data = df[OP_COLS].values

    # Fit or reuse KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df["_op_cluster"] = kmeans.fit_predict(op_data)

    scalers: dict[int, StandardScaler] = {}

    for cluster_id in range(n_clusters):
        mask = df["_op_cluster"] == cluster_id
        if mask.sum() == 0:
            continue

        if scaler_cache and cluster_id in scaler_cache:
            scaler = scaler_cache[cluster_id]
            df.loc[mask, SENSOR_COLS] = scaler.transform(df.loc[mask, SENSOR_COLS])
        else:
            scaler = StandardScaler()
            df.loc[mask, SENSOR_COLS] = scaler.fit_transform(
                df.loc[mask, SENSOR_COLS]
            )
        scalers[cluster_id] = scaler

    df = df.drop(columns=["_op_cluster"])
    return df, kmeans, scalers


def get_last_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per engine: the last (highest-cycle) row.

    Used for evaluating models on the test set against the provided RUL file.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'unit' and 'cycle' columns.

    Returns
    -------
    pd.DataFrame
        One row per engine, sorted by unit.
    """
    idx = df.groupby("unit")["cycle"].idxmax()
    return df.loc[idx].sort_values("unit").reset_index(drop=True)


# ── Convenience loader ────────────────────────────────────────────────────────

def load_dataset(
    dataset: str = "FD001",
    clip_rul: Optional[int] = 125,
    data_dir: Optional[Path] = None,
) -> dict:
    """
    Convenience function: load train + test + ground-truth RUL for a dataset.

    Parameters
    ----------
    dataset : str
    clip_rul : int or None
        Clip training RUL at this value (125 recommended). Pass None to skip.
    data_dir : Path, optional

    Returns
    -------
    dict with keys:
        'train'      : pd.DataFrame  (with 'rul' column)
        'test'       : pd.DataFrame  (no RUL column — truncated engines)
        'rul_test'   : pd.Series     (ground-truth RUL for test engines)
        'dataset'    : str
        'clip_rul'   : int or None
    """
    train = attach_rul_labels(
        load_raw(dataset, "train", data_dir), clip=clip_rul
    )
    test = load_raw(dataset, "test", data_dir)
    rul_test = load_test_rul(dataset, data_dir)

    return {
        "train": train,
        "test": test,
        "rul_test": rul_test,
        "dataset": dataset,
        "clip_rul": clip_rul,
    }
