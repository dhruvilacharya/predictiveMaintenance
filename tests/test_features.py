"""
tests/test_features.py
======================
Unit tests for src/features.py.

All tests use a synthetic in-memory DataFrame so they run without
the actual NASA data files or the EDA notebook output.

Key invariants verified:
  - No future-data leakage in any feature
  - No NaN values after feature building
  - Output shape is correct
  - Dropped sensors are absent from output
  - Slope feature is 0 when sensor is constant
  - Drift feature is 0 for the first baseline_cycles rows
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import SENSOR_COLS, OP_COLS, BASE_COLS
from src.features import (
    add_rolling_features,
    add_slope_features,
    add_drift_features,
    add_cycle_feature,
    add_lag_features,
    build_features,
    get_feature_columns,
    save_feature_names,
    load_feature_names,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_df(n_engines: int = 3, max_cycles: int = 30, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic training DataFrame."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_engines + 1):
        n_cycles = max_cycles - (unit - 1) * 5
        for cycle in range(1, n_cycles + 1):
            row = {"unit": unit, "cycle": cycle}
            for op in OP_COLS:
                row[op] = 0.0
            for s in SENSOR_COLS:
                # Add a slight trend to simulate degradation
                row[s] = 100.0 + cycle * 0.1 + rng.normal(0, 0.5)
            rows.append(row)
    df = pd.DataFrame(rows)[BASE_COLS]
    return df


def make_constant_sensor_df(n_cycles: int = 20) -> pd.DataFrame:
    """Single-engine DataFrame where s1 is perfectly constant."""
    df = make_df(n_engines=1, max_cycles=n_cycles)
    df["s1"] = 42.0  # perfectly constant
    return df


# ── Rolling features ──────────────────────────────────────────────────────────

class TestAddRollingFeatures:

    def test_columns_created(self):
        df = make_df()
        out = add_rolling_features(df, ["s1"], windows=(5,))
        assert "s1_w5_mean" in out.columns
        assert "s1_w5_std" in out.columns
        assert "s1_w5_min" in out.columns
        assert "s1_w5_max" in out.columns

    def test_no_nan_values(self):
        df = make_df()
        out = add_rolling_features(df, SENSOR_COLS, windows=(5, 10))
        feat_cols = [c for c in out.columns if "_w" in c]
        assert out[feat_cols].isnull().sum().sum() == 0

    def test_no_future_leakage(self):
        """
        For a single engine with a strictly increasing sensor,
        the rolling mean at row t must never exceed the raw value at row t+1.
        """
        df = make_df(n_engines=1, max_cycles=30, seed=1)
        # Make s2 strictly increasing
        df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
        df["s2"] = np.arange(1, len(df) + 1, dtype=float)

        out = add_rolling_features(df, ["s2"], windows=(5,))

        for i in range(len(out) - 1):
            # rolling mean at row i should use only values s2[0..i], not s2[i+1..]
            actual_max_used = out.loc[i, "s2_w5_mean"]
            future_value = out.loc[i + 1, "s2"]
            # If future data had leaked in, the rolling mean could exceed
            # the raw value at position i, which is impossible without lookahead
            # (since s2 is strictly increasing, mean <= current value)
            assert actual_max_used <= out.loc[i, "s2"] + 1e-9, (
                f"Leakage detected at row {i}: rolling mean {actual_max_used:.4f} "
                f"> current value {out.loc[i, 's2']:.4f}"
            )

    def test_per_unit_not_cross_unit(self):
        """Rolling stats must reset at unit boundaries."""
        df = make_df(n_engines=2, max_cycles=10)
        out = add_rolling_features(df, ["s1"], windows=(10,))
        # Mean for unit 2, cycle 1 must equal the raw s1 value (window can't see unit 1)
        unit2_cycle1 = out[(out["unit"] == 2) & (out["cycle"] == 1)]
        raw = unit2_cycle1["s1"].values[0]
        mean = unit2_cycle1["s1_w10_mean"].values[0]
        np.testing.assert_allclose(mean, raw, rtol=1e-6)

    def test_original_df_not_modified(self):
        df = make_df()
        cols_before = list(df.columns)
        add_rolling_features(df, SENSOR_COLS, windows=(5,))
        assert list(df.columns) == cols_before


# ── Slope features ────────────────────────────────────────────────────────────

class TestAddSlopeFeatures:

    def test_columns_created(self):
        df = make_df()
        out = add_slope_features(df, ["s1"], window=5)
        assert "s1_slope_5" in out.columns

    def test_zero_slope_for_constant_sensor(self):
        df = make_constant_sensor_df(n_cycles=25)
        out = add_slope_features(df, ["s1"], window=5)
        # Slope of a constant series must be ~0
        np.testing.assert_allclose(out["s1_slope_5"].values, 0.0, atol=1e-9)

    def test_positive_slope_for_increasing_sensor(self):
        df = make_df(n_engines=1, max_cycles=20)
        df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
        df["s2"] = np.arange(1, len(df) + 1, dtype=float)
        out = add_slope_features(df, ["s2"], window=5)
        # After the warm-up, slope should be ~1 (linear increase of 1 per cycle)
        late_rows = out[out["cycle"] >= 6]["s2_slope_5"]
        assert (late_rows > 0.5).all(), "Slope of linearly increasing sensor should be positive"

    def test_no_nan_values(self):
        df = make_df()
        out = add_slope_features(df, SENSOR_COLS, window=5)
        slope_cols = [c for c in out.columns if "_slope_" in c]
        assert out[slope_cols].isnull().sum().sum() == 0


# ── Drift features ────────────────────────────────────────────────────────────

class TestAddDriftFeatures:

    def test_columns_created(self):
        df = make_df()
        out = add_drift_features(df, ["s1"])
        assert "s1_drift" in out.columns

    def test_drift_near_zero_in_baseline_period(self):
        """For the first baseline_cycles rows, drift ≈ 0 by construction."""
        df = make_constant_sensor_df(n_cycles=30)
        out = add_drift_features(df, ["s1"], baseline_cycles=10)
        early = out[out["cycle"] <= 10]["s1_drift"]
        np.testing.assert_allclose(early.values, 0.0, atol=1e-9)

    def test_no_nan_values(self):
        df = make_df()
        out = add_drift_features(df, SENSOR_COLS, baseline_cycles=5)
        drift_cols = [c for c in out.columns if "_drift" in c]
        assert out[drift_cols].isnull().sum().sum() == 0

    def test_no_leakage(self):
        """
        Drift baseline must use only cycles 1…baseline_cycles, not later data.
        Mutating post-baseline data should not change the drift values for early rows.
        """
        df = make_df(n_engines=1, max_cycles=30)
        df1 = add_drift_features(df.copy(), ["s2"], baseline_cycles=5)

        # Perturb s2 values AFTER the baseline period
        df2 = df.copy()
        df2.loc[df2["cycle"] > 5, "s2"] += 1000.0
        df2_drift = add_drift_features(df2, ["s2"], baseline_cycles=5)

        # Early drift values should be identical (baseline is the same)
        early1 = df1[df1["cycle"] <= 5]["s2_drift"].values
        early2 = df2_drift[df2_drift["cycle"] <= 5]["s2_drift"].values
        np.testing.assert_allclose(early1, early2, atol=1e-9)


# ── Cycle feature ─────────────────────────────────────────────────────────────

class TestAddCycleFeature:

    def test_column_created(self):
        df = make_df()
        out = add_cycle_feature(df)
        assert "cycle_norm" in out.columns

    def test_range_zero_to_one(self):
        df = make_df()
        out = add_cycle_feature(df)
        assert out["cycle_norm"].min() >= 0.0
        assert out["cycle_norm"].max() <= 1.0 + 1e-9

    def test_last_cycle_is_one(self):
        df = make_df()
        out = add_cycle_feature(df)
        last = out.loc[out.groupby("unit")["cycle"].idxmax()]
        np.testing.assert_allclose(last["cycle_norm"].values, 1.0, atol=1e-9)


# ── Lag features ─────────────────────────────────────────────────────────────

class TestAddLagFeatures:

    def test_columns_created(self):
        df = make_df()
        out = add_lag_features(df, ["s1"], lags=(1, 3))
        assert "s1_lag_1" in out.columns
        assert "s1_lag_3" in out.columns

    def test_no_cross_unit_lag(self):
        """Lag at cycle 1 of unit 2 must be 0, not a value from unit 1."""
        df = make_df(n_engines=2, max_cycles=15)
        out = add_lag_features(df, ["s1"], lags=(1,))
        unit2_cycle1 = out[(out["unit"] == 2) & (out["cycle"] == 1)]
        assert float(unit2_cycle1["s1_lag_1"].values[0]) == 0.0

    def test_no_nan_values(self):
        df = make_df()
        out = add_lag_features(df, SENSOR_COLS, lags=(1, 3, 5))
        lag_cols = [c for c in out.columns if "_lag_" in c]
        assert out[lag_cols].isnull().sum().sum() == 0


# ── build_features (integration) ─────────────────────────────────────────────

class TestBuildFeatures:

    def _make_dropped_sensors_file(self, tmpdir: Path, sensors: list[str]) -> Path:
        fpath = tmpdir / "dropped_sensors_FD001.json"
        with open(fpath, "w") as f:
            json.dump(sensors, f)
        return fpath

    def test_output_has_no_raw_sensor_columns(self):
        df = make_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_dropped_sensors_file(Path(tmpdir), [])
            out = build_features(df, dataset="FD001", processed_dir=Path(tmpdir))
        for s in SENSOR_COLS:
            assert s not in out.columns, f"Raw sensor {s} should not be in output"

    def test_dropped_sensors_absent(self):
        df = make_df()
        to_drop = ["s1", "s5", "s10"]
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_dropped_sensors_file(Path(tmpdir), to_drop)
            out = build_features(df, dataset="FD001", processed_dir=Path(tmpdir))
        for s in to_drop:
            dropped_feat_cols = [c for c in out.columns if c.startswith(s + "_")]
            assert len(dropped_feat_cols) == 0, (
                f"Features derived from dropped sensor {s} should not exist: "
                f"{dropped_feat_cols}"
            )

    def test_no_nan_values(self):
        df = make_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_dropped_sensors_file(Path(tmpdir), [])
            out = build_features(df, dataset="FD001", processed_dir=Path(tmpdir))
        feat_cols = get_feature_columns(out)
        assert out[feat_cols].isnull().sum().sum() == 0

    def test_output_shape_reasonable(self):
        df = make_df(n_engines=3, max_cycles=30)
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_dropped_sensors_file(Path(tmpdir), [])
            out = build_features(df, dataset="FD001", processed_dir=Path(tmpdir))
        # Should have many more columns than the original 26
        assert len(out.columns) > 50, "Expected 50+ feature columns after engineering"

    def test_unit_and_cycle_preserved(self):
        df = make_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_dropped_sensors_file(Path(tmpdir), [])
            out = build_features(df, dataset="FD001", processed_dir=Path(tmpdir))
        assert "unit" in out.columns
        assert "cycle" in out.columns

    def test_rul_preserved_if_present(self):
        from src.data import attach_rul_labels
        df = attach_rul_labels(make_df())
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_dropped_sensors_file(Path(tmpdir), [])
            out = build_features(df, dataset="FD001", processed_dir=Path(tmpdir))
        assert "rul" in out.columns

    def test_original_df_not_modified(self):
        df = make_df()
        cols_before = list(df.columns)
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_dropped_sensors_file(Path(tmpdir), [])
            build_features(df, dataset="FD001", processed_dir=Path(tmpdir))
        assert list(df.columns) == cols_before


# ── Feature name I/O ──────────────────────────────────────────────────────────

class TestFeatureNameIO:

    def test_save_and_load(self):
        names = ["feat_a", "feat_b", "feat_c"]
        with tempfile.TemporaryDirectory() as tmpdir:
            save_feature_names(names, "FD001", Path(tmpdir))
            loaded = load_feature_names("FD001", Path(tmpdir))
        assert names == loaded

    def test_load_raises_if_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_feature_names("FD001", Path(tmpdir))
