"""
tests/test_data.py
==================
Unit tests for src/data.py.

These tests use a synthetic in-memory DataFrame so they run without the
actual NASA C-MAPSS files present.  Tests that require the raw files
are skipped automatically when the files are absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import (
    BASE_COLS,
    SENSOR_COLS,
    OP_COLS,
    attach_rul_labels,
    get_last_cycle,
    normalize_per_operating_condition,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_synthetic_train(n_engines: int = 3, max_cycles: int = 20) -> pd.DataFrame:
    """
    Build a minimal synthetic training DataFrame with the correct schema.
    Each engine runs for a different number of cycles to mimic FD001.
    """
    rng = np.random.default_rng(0)
    rows = []
    for unit in range(1, n_engines + 1):
        n_cycles = max_cycles - (unit - 1) * 3  # different lengths
        for cycle in range(1, n_cycles + 1):
            row = {"unit": unit, "cycle": cycle}
            for op in OP_COLS:
                row[op] = rng.uniform(0, 1)
            for s in SENSOR_COLS:
                row[s] = rng.uniform(100, 200)
            rows.append(row)
    df = pd.DataFrame(rows)[BASE_COLS]
    return df


# ── Tests: attach_rul_labels ───────────────────────────────────────────────────

class TestAttachRulLabels:

    def test_rul_column_exists(self):
        df = make_synthetic_train()
        out = attach_rul_labels(df)
        assert "rul" in out.columns

    def test_rul_non_negative(self):
        df = make_synthetic_train()
        out = attach_rul_labels(df)
        assert (out["rul"] >= 0).all(), "All RUL values must be >= 0"

    def test_last_cycle_rul_is_zero(self):
        df = make_synthetic_train()
        out = attach_rul_labels(df)
        last = out.loc[out.groupby("unit")["cycle"].idxmax()]
        assert (last["rul"] == 0).all(), (
            "The last cycle of every engine must have rul == 0"
        )

    def test_rul_formula(self):
        df = make_synthetic_train(n_engines=1, max_cycles=10)
        out = attach_rul_labels(df)
        # For a single engine with cycles 1..10, rul at cycle 1 should be 9
        first = out[out["cycle"] == 1].iloc[0]
        assert first["rul"] == 9, f"Expected rul=9 at cycle 1, got {first['rul']}"

    def test_rul_clip(self):
        df = make_synthetic_train(n_engines=1, max_cycles=200)
        out = attach_rul_labels(df, clip=125)
        assert out["rul"].max() <= 125, "Clipped RUL must not exceed 125"
        assert "rul_unclipped" in out.columns, "rul_unclipped column must exist when clipping"
        # unclipped should be larger for early cycles
        assert (out["rul_unclipped"] >= out["rul"]).all()

    def test_no_clip_no_unclipped_column(self):
        df = make_synthetic_train()
        out = attach_rul_labels(df, clip=None)
        assert "rul_unclipped" not in out.columns

    def test_requires_unit_column(self):
        df = make_synthetic_train().drop(columns=["unit"])
        with pytest.raises(ValueError, match="unit"):
            attach_rul_labels(df)

    def test_original_df_not_modified(self):
        df = make_synthetic_train()
        original_cols = list(df.columns)
        _ = attach_rul_labels(df)
        assert list(df.columns) == original_cols, "Input DataFrame should not be mutated"


# ── Tests: get_last_cycle ──────────────────────────────────────────────────────

class TestGetLastCycle:

    def test_one_row_per_engine(self):
        df = attach_rul_labels(make_synthetic_train(n_engines=5))
        last = get_last_cycle(df)
        assert len(last) == 5

    def test_last_cycle_is_maximum(self):
        df = attach_rul_labels(make_synthetic_train(n_engines=3, max_cycles=15))
        last = get_last_cycle(df)
        for _, row in last.iterrows():
            unit_df = df[df["unit"] == row["unit"]]
            assert row["cycle"] == unit_df["cycle"].max()

    def test_sorted_by_unit(self):
        df = attach_rul_labels(make_synthetic_train(n_engines=5))
        last = get_last_cycle(df)
        assert list(last["unit"]) == sorted(last["unit"].tolist())


# ── Tests: normalize_per_operating_condition ───────────────────────────────────

class TestNormalizePerOperatingCondition:

    def _make_multi_condition_df(self) -> pd.DataFrame:
        """Synthetic data with 2 clear operating conditions."""
        rng = np.random.default_rng(42)
        n = 100
        # Condition A: op1=0, op2=0, op3=0 + noise
        # Condition B: op1=100, op2=50, op3=25 + noise
        op_a = pd.DataFrame({
            "op1": rng.normal(0, 0.1, n // 2),
            "op2": rng.normal(0, 0.1, n // 2),
            "op3": rng.normal(0, 0.1, n // 2),
        })
        op_b = pd.DataFrame({
            "op1": rng.normal(100, 0.1, n // 2),
            "op2": rng.normal(50, 0.1, n // 2),
            "op3": rng.normal(25, 0.1, n // 2),
        })
        ops = pd.concat([op_a, op_b], ignore_index=True)
        sensors = pd.DataFrame(
            rng.normal(150, 10, (n, len(SENSOR_COLS))),
            columns=SENSOR_COLS,
        )
        meta = pd.DataFrame({
            "unit": [1] * n,
            "cycle": list(range(1, n + 1)),
        })
        return pd.concat([meta, ops, sensors], axis=1)[BASE_COLS]

    def test_sensor_values_changed(self):
        df = self._make_multi_condition_df()
        df_norm, _, _ = normalize_per_operating_condition(df, n_clusters=2)
        # After normalisation, sensor values should differ from original
        assert not np.allclose(df["s1"].values, df_norm["s1"].values)

    def test_op_columns_unchanged(self):
        df = self._make_multi_condition_df()
        df_norm, _, _ = normalize_per_operating_condition(df, n_clusters=2)
        for op in OP_COLS:
            np.testing.assert_array_almost_equal(df[op].values, df_norm[op].values)

    def test_no_op_cluster_column_in_output(self):
        df = self._make_multi_condition_df()
        df_norm, _, _ = normalize_per_operating_condition(df, n_clusters=2)
        assert "_op_cluster" not in df_norm.columns

    def test_scalers_returned_for_each_cluster(self):
        df = self._make_multi_condition_df()
        _, kmeans, scalers = normalize_per_operating_condition(df, n_clusters=2)
        assert len(scalers) == 2

    def test_reproducible_with_same_seed(self):
        df = self._make_multi_condition_df()
        df1, _, _ = normalize_per_operating_condition(df, n_clusters=2, random_state=0)
        df2, _, _ = normalize_per_operating_condition(df, n_clusters=2, random_state=0)
        pd.testing.assert_frame_equal(df1, df2)


# ── Tests: load_raw / load_test_rul (integration, skipped without files) ───────

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
FILES_PRESENT = (RAW_DIR / "train_FD001.txt").exists()


@pytest.mark.skipif(not FILES_PRESENT, reason="Raw data files not downloaded yet")
class TestLoadRaw:

    def test_column_names(self):
        from src.data import load_raw
        df = load_raw("FD001", "train")
        assert list(df.columns) == BASE_COLS

    def test_shape_train_fd001(self):
        from src.data import load_raw
        df = load_raw("FD001", "train")
        # FD001 training set: ~20 000 rows, 26 columns
        assert df.shape[1] == 26
        assert df.shape[0] > 10000

    def test_no_null_values(self):
        from src.data import load_raw
        df = load_raw("FD001", "train")
        assert df.isnull().sum().sum() == 0, "Raw training data should have no nulls"

    def test_unit_column_positive_integers(self):
        from src.data import load_raw
        df = load_raw("FD001", "train")
        assert (df["unit"] > 0).all()
        assert df["unit"].dtype in (int, np.int64, np.int32)

    def test_rul_test_row_count_matches_test_engines(self):
        from src.data import load_raw, load_test_rul
        test = load_raw("FD001", "test")
        rul = load_test_rul("FD001")
        n_engines = test["unit"].nunique()
        assert len(rul) == n_engines, (
            f"RUL file has {len(rul)} rows but test set has {n_engines} unique engines"
        )

    def test_attach_rul_last_cycle_zero(self):
        from src.data import load_raw, attach_rul_labels
        df = load_raw("FD001", "train")
        out = attach_rul_labels(df)
        last = out.loc[out.groupby("unit")["cycle"].idxmax()]
        assert (last["rul"] == 0).all()
