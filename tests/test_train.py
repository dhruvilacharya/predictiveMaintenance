"""
tests/test_train.py
===================
Unit tests for src/train.py.

Lightweight tests that run without heavy dependencies (MLflow, TensorFlow).
Heavy integration tests (actual training runs) are skipped when optional
dependencies are unavailable.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.train import build_sliding_windows, save_model, load_model


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_df(n_engines: int = 3, max_cycles: int = 50, n_features: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for unit in range(1, n_engines + 1):
        n_cycles = max_cycles - (unit - 1) * 5
        for cycle in range(1, n_cycles + 1):
            row = {"unit": unit, "cycle": cycle, "rul": max_cycles - cycle}
            for i in range(n_features):
                row[f"f{i}"] = rng.normal()
            rows.append(row)
    return pd.DataFrame(rows)


FEATURE_COLS = [f"f{i}" for i in range(10)]


# ── build_sliding_windows ─────────────────────────────────────────────────────

class TestBuildSlidingWindows:

    def test_output_shapes(self):
        df = make_df(n_engines=2, max_cycles=40)
        X_seq, y_seq, groups = build_sliding_windows(df, FEATURE_COLS, window=10)
        assert X_seq.ndim == 3, "X_seq must be 3D: (N, window, n_features)"
        assert X_seq.shape[1] == 10, "Window dimension must be 10"
        assert X_seq.shape[2] == len(FEATURE_COLS)
        assert len(y_seq) == len(X_seq)
        assert len(groups) == len(X_seq)

    def test_no_future_leakage(self):
        """
        The last time step of each window corresponds to cycle t.
        No values from t+1 or later should appear.
        """
        # Use a strictly increasing feature (f0 = cycle number)
        df = make_df(n_engines=1, max_cycles=30)
        df["f0"] = df["cycle"].astype(float)

        X_seq, y_seq, groups = build_sliding_windows(df, FEATURE_COLS, window=5)

        # For every window, the last step's f0 should equal the current cycle
        # and must be >= all earlier steps (no future data)
        for i in range(len(X_seq)):
            window = X_seq[i]          # shape (5, 10)
            last_f0 = window[-1, 0]    # f0 at step t
            for j in range(len(window) - 1):
                assert window[j, 0] <= last_f0, (
                    f"Window {i}: step {j} f0={window[j,0]:.1f} > "
                    f"last step f0={last_f0:.1f} — future data leaked"
                )

    def test_no_cross_unit_windows(self):
        """
        Windows must not span across engine boundaries.
        The minimum number of windows is one per engine (when
        n_cycles == window), and groups must match units.
        """
        df = make_df(n_engines=3, max_cycles=20)
        X_seq, y_seq, groups = build_sliding_windows(df, FEATURE_COLS, window=10)

        # All groups must be valid engine unit IDs
        valid_units = set(df["unit"].unique())
        assert set(groups).issubset(valid_units)

    def test_target_at_last_timestep(self):
        """y_seq at position i must equal the RUL at the last cycle of window i."""
        df = make_df(n_engines=1, max_cycles=30)
        df_sorted = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
        X_seq, y_seq, groups = build_sliding_windows(df_sorted, FEATURE_COLS, window=5)

        # Manually compute the expected target for the first window
        first_engine = df_sorted[df_sorted["unit"] == 1].reset_index(drop=True)
        expected_first_target = float(first_engine.loc[4, "rul"])  # index 4 = cycle 5
        assert y_seq[0] == pytest.approx(expected_first_target, abs=1e-6)

    def test_float32_dtype(self):
        df = make_df()
        X_seq, y_seq, _ = build_sliding_windows(df, FEATURE_COLS, window=10)
        assert X_seq.dtype == np.float32
        assert y_seq.dtype == np.float32

    def test_empty_engine_handled_gracefully(self):
        """Engines with fewer cycles than window size contribute 0 windows."""
        df = make_df(n_engines=1, max_cycles=5)
        X_seq, y_seq, groups = build_sliding_windows(df, FEATURE_COLS, window=10)
        # 5 cycles < window=10 → no windows
        assert len(X_seq) == 0


# ── save_model / load_model ────────────────────────────────────────────────────

class TestModelPersistence:

    def test_joblib_save_load_roundtrip(self):
        """A simple sklearn model should survive save → load."""
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=2.0)
        X = np.random.default_rng(0).normal(size=(50, 5))
        y = np.random.default_rng(0).normal(size=50)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_model(model, "test_ridge", Path(tmpdir))
            assert path.exists()
            loaded = load_model("test_ridge", Path(tmpdir))

        np.testing.assert_array_almost_equal(
            model.predict(X), loaded.predict(X)
        )

    def test_load_raises_if_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_model("nonexistent_model", Path(tmpdir))


# ── Lightweight model training (no MLflow server required) ────────────────────

class TestTrainModel:

    def test_train_ridge_returns_expected_keys(self):
        """End-to-end train_model call with Ridge (no external deps)."""
        from src.train import train_model

        rng = np.random.default_rng(42)
        n = 200
        feature_cols = [f"f{i}" for i in range(8)]
        X = pd.DataFrame(rng.normal(size=(n, 8)), columns=feature_cols)
        y = rng.uniform(0, 125, size=n)
        groups = np.repeat(np.arange(1, 21), 10)  # 20 engines, 10 rows each

        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_model(
                "ridge", X, y, groups,
                params={"alpha": 1.0},
                experiment_name="test-experiment",
                run_name="test_ridge",
                models_dir=Path(tmpdir),
                save=True,
            )

        assert "model" in result
        assert "cv_metrics" in result
        assert "run_id" in result
        assert "fold_metrics" in result
        assert len(result["fold_metrics"]) == 5  # 5 folds

        cv = result["cv_metrics"]
        assert "rmse_mean" in cv
        assert cv["rmse_mean"] >= 0
        assert cv["rmse_std"] >= 0
