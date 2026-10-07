"""
tests/test_evaluate.py
======================
Unit tests for src/evaluate.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    rmse,
    mae,
    nasa_score,
    compute_all_metrics,
    evaluate_cv,
    cost_analysis,
    run_to_failure_cost,
    fixed_schedule_cost,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)
Y_TRUE = RNG.integers(0, 200, size=100).astype(float)
Y_PRED_GOOD = Y_TRUE + RNG.normal(0, 10, size=100)
Y_PERFECT = Y_TRUE.copy()


# ── Metric correctness ────────────────────────────────────────────────────────

class TestBasicMetrics:

    def test_rmse_perfect(self):
        assert rmse(Y_TRUE, Y_PERFECT) == pytest.approx(0.0)

    def test_mae_perfect(self):
        assert mae(Y_TRUE, Y_PERFECT) == pytest.approx(0.0)

    def test_rmse_non_negative(self):
        assert rmse(Y_TRUE, Y_PRED_GOOD) >= 0.0

    def test_mae_non_negative(self):
        assert mae(Y_TRUE, Y_PRED_GOOD) >= 0.0

    def test_rmse_scalar(self):
        val = rmse(np.array([3.0]), np.array([4.0]))
        assert val == pytest.approx(1.0)

    def test_mae_scalar(self):
        val = mae(np.array([3.0]), np.array([6.0]))
        assert val == pytest.approx(3.0)

    def test_compute_all_metrics_keys(self):
        m = compute_all_metrics(Y_TRUE, Y_PRED_GOOD)
        assert set(m.keys()) == {"rmse", "mae", "nasa_score"}


# ── NASA score ────────────────────────────────────────────────────────────────

class TestNasaScore:

    def test_zero_for_perfect_prediction(self):
        assert nasa_score(Y_TRUE, Y_PERFECT) == pytest.approx(0.0, abs=1e-9)

    def test_late_prediction_worse_than_early(self):
        """
        The NASA score penalises late predictions (over-estimating RUL)
        more than symmetric early predictions of the same magnitude.
        """
        y_t = np.array([100.0])
        d = 20.0
        late_pred  = y_t + d   # predict 20 more cycles than remain
        early_pred = y_t - d   # predict 20 fewer cycles than remain

        late_score  = nasa_score(y_t, late_pred)
        early_score = nasa_score(y_t, early_pred)
        assert late_score > early_score, (
            f"Late score ({late_score:.4f}) should exceed early score ({early_score:.4f})"
        )

    def test_non_negative(self):
        """NASA score is always >= 0 for any prediction."""
        score = nasa_score(Y_TRUE, Y_PRED_GOOD)
        assert score >= 0.0

    def test_asymmetry_direction(self):
        """d > 0 (late) uses exp(d/10); d < 0 (early) uses exp(-d/13)."""
        d_pos = 10.0   # late
        d_neg = -10.0  # early
        expected_late  = np.exp(d_pos / 10.0) - 1
        expected_early = np.exp(10.0 / 13.0) - 1

        y_t = np.array([50.0])
        assert nasa_score(y_t, y_t + d_pos) == pytest.approx(expected_late, rel=1e-6)
        assert nasa_score(y_t, y_t + d_neg) == pytest.approx(expected_early, rel=1e-6)

    def test_sum_over_all_engines(self):
        """Score should be the sum, not mean, across engines."""
        y_t = np.array([100.0, 100.0])
        y_p = np.array([110.0, 110.0])
        single = nasa_score(np.array([100.0]), np.array([110.0]))
        double = nasa_score(y_t, y_p)
        assert double == pytest.approx(2 * single, rel=1e-6)


# ── evaluate_cv ───────────────────────────────────────────────────────────────

class TestEvaluateCv:

    def _make_fold_results(self):
        return [
            {"rmse": 10.0, "mae": 8.0, "nasa_score": 500.0},
            {"rmse": 12.0, "mae": 9.0, "nasa_score": 600.0},
            {"rmse": 11.0, "mae": 8.5, "nasa_score": 550.0},
        ]

    def test_mean_keys_present(self):
        cv = evaluate_cv(self._make_fold_results())
        assert "rmse_mean" in cv
        assert "mae_mean" in cv
        assert "nasa_score_mean" in cv

    def test_std_keys_present(self):
        cv = evaluate_cv(self._make_fold_results())
        assert "rmse_std" in cv

    def test_correct_mean(self):
        cv = evaluate_cv(self._make_fold_results())
        assert cv["rmse_mean"] == pytest.approx(11.0, rel=1e-6)

    def test_correct_std(self):
        cv = evaluate_cv(self._make_fold_results())
        expected_std = float(np.std([10.0, 12.0, 11.0]))
        assert cv["rmse_std"] == pytest.approx(expected_std, rel=1e-6)


# ── cost_analysis ─────────────────────────────────────────────────────────────

class TestCostAnalysis:

    def _make_scenario(self, n: int = 50):
        rng = np.random.default_rng(0)
        y_true = rng.integers(10, 150, size=n).astype(float)
        y_pred = y_true + rng.normal(0, 15, size=n)
        y_pred = np.clip(y_pred, 0, None)
        return y_true, y_pred

    def test_output_columns(self):
        y_true, y_pred = self._make_scenario()
        df = cost_analysis(y_true, y_pred)
        required = {"threshold", "total_cost", "n_failures", "n_early"}
        assert required.issubset(set(df.columns))

    def test_threshold_zero_equals_run_to_failure(self):
        """
        At T=0 the model never alerts → all engines fail.
        Cost should equal run-to-failure cost.
        """
        y_true = np.array([50.0, 30.0, 80.0, 20.0])
        y_pred = np.array([55.0, 25.0, 90.0, 15.0])
        cost_failure = 10_000.0

        df = cost_analysis(y_true, y_pred,
                           thresholds=np.array([0.0]),
                           cost_failure=cost_failure,
                           cost_maintenance=1_000.0,
                           cost_per_early_cycle=0.0)
        # T=0: y_pred < 0 is never true, so no alerts fired → all engines fail
        assert df.loc[0, "n_failures"] == len(y_true)

    def test_non_negative_costs(self):
        y_true, y_pred = self._make_scenario()
        df = cost_analysis(y_true, y_pred)
        assert (df["total_cost"] >= 0).all()

    def test_output_length_matches_thresholds(self):
        y_true, y_pred = self._make_scenario()
        thresholds = np.arange(0, 50, 5)
        df = cost_analysis(y_true, y_pred, thresholds=thresholds)
        assert len(df) == len(thresholds)

    def test_run_to_failure_cost(self):
        cost = run_to_failure_cost(n_engines=10, cost_failure=10_000.0)
        assert cost == pytest.approx(100_000.0)


# ── evaluate_on_test (lightweight) ────────────────────────────────────────────

class TestEvaluateOnTest:

    def test_returns_expected_keys(self):
        """Use a trivial constant predictor to check return structure."""
        from src.evaluate import evaluate_on_test

        class ConstantModel:
            def predict(self, X):
                return np.full(len(X), 50.0)

        n = 20
        rng = np.random.default_rng(1)
        feature_cols = [f"f{i}" for i in range(5)]
        X_test = pd.DataFrame(rng.normal(size=(n, 5)), columns=feature_cols)
        y_true = rng.integers(0, 150, size=n).astype(float)

        result = evaluate_on_test(ConstantModel(), X_test, y_true, feature_cols)
        assert "rmse" in result
        assert "mae" in result
        assert "nasa_score" in result
        assert "y_pred" in result
        assert len(result["y_pred"]) == n

    def test_predictions_non_negative(self):
        """Predicted RUL should be clipped to >= 0."""
        from src.evaluate import evaluate_on_test

        class NegativeModel:
            def predict(self, X):
                return np.full(len(X), -10.0)

        n = 10
        feature_cols = ["f1"]
        X_test = pd.DataFrame(np.ones((n, 1)), columns=feature_cols)
        y_true = np.ones(n) * 50.0

        result = evaluate_on_test(NegativeModel(), X_test, y_true, feature_cols)
        assert (result["y_pred"] >= 0).all()
