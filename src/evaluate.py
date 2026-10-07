"""
src/evaluate.py
===============
Metrics, evaluation utilities, and business cost analysis for the RUL project.

Public API
----------
rmse(y_true, y_pred)
mae(y_true, y_pred)
nasa_score(y_true, y_pred)
    The asymmetric NASA scoring function.

evaluate_cv(y_true_all, y_pred_all)
    Aggregate metrics across all CV folds.

evaluate_on_test(model, X_test_last_cycle, y_true_rul, feature_cols)
    Official evaluation on the test set (last cycle per engine).

cost_analysis(y_true, y_pred, thresholds, cost_failure,
              cost_maintenance, cost_per_early_cycle)
    Sweep an alert threshold and compute fleet maintenance cost.

plot_predicted_vs_actual(y_true, y_pred, save_path)
plot_residuals(y_true, y_pred, save_path)
plot_rul_trajectories(df_test_feat, model, feature_cols, unit_ids, save_path)
plot_cost_vs_threshold(cost_df, optimal_t, baseline_rtf, baseline_fixed, save_path)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── Figures dir default ────────────────────────────────────────────────────────

def _figures_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "reports" / "figures"


# ── Core metrics ──────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(np.asarray(y_true), np.asarray(y_pred))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(np.asarray(y_true), np.asarray(y_pred)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA asymmetric scoring function.

    Penalises late predictions (predicting more remaining life than actually
    exists) more severely than early predictions.

        s(d) = exp( d / 10) - 1   if d >= 0  (late: model predicts > true)
             = exp(-d / 13) - 1   if d <  0  (early: model predicts < true)

    where d = y_pred - y_true.

    Lower is better.  A perfect predictor scores 0.

    Reference: Saxena et al. (2008), PHM08.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)
    return float(np.sum(scores))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return dict with rmse, mae, nasa_score."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
    }


def evaluate_cv(
    fold_results: list[dict],
) -> dict:
    """
    Aggregate per-fold metric dicts into mean ± std.

    Parameters
    ----------
    fold_results : list of dicts
        Each dict must have keys 'rmse', 'mae', 'nasa_score'.

    Returns
    -------
    dict with keys 'rmse_mean', 'rmse_std', 'mae_mean', 'mae_std',
                   'nasa_score_mean', 'nasa_score_std'
    """
    keys = ["rmse", "mae", "nasa_score"]
    out = {}
    for k in keys:
        vals = [f[k] for f in fold_results]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


# ── Official test-set evaluation ──────────────────────────────────────────────

def evaluate_on_test(
    model,
    X_test_last_cycle: pd.DataFrame,
    y_true_rul: Union[pd.Series, np.ndarray],
    feature_cols: Optional[list[str]] = None,
) -> dict:
    """
    Evaluate a fitted model on the official test set.

    Uses the LAST cycle of each test engine (the only cycle at which
    prediction is meaningful, because that is when we would raise an alert).

    Parameters
    ----------
    model : fitted sklearn-compatible model
    X_test_last_cycle : pd.DataFrame
        One row per test engine (last cycle).  Must contain feature_cols.
    y_true_rul : array-like
        Ground-truth RUL from RUL_FD001.txt, length = n_engines.
    feature_cols : list of str, optional
        If None, uses all columns except 'unit', 'cycle', 'rul', 'rul_unclipped'.

    Returns
    -------
    dict with keys: rmse, mae, nasa_score, y_pred
    """
    if feature_cols is None:
        exclude = {"unit", "cycle", "rul", "rul_unclipped"}
        feature_cols = [c for c in X_test_last_cycle.columns if c not in exclude]

    X = X_test_last_cycle[feature_cols].values
    y_pred = np.asarray(model.predict(X), dtype=float)
    y_pred = np.clip(y_pred, 0, None)  # RUL cannot be negative

    y_true = np.asarray(y_true_rul, dtype=float)
    assert len(y_true) == len(y_pred), (
        f"Length mismatch: y_true has {len(y_true)} values, "
        f"y_pred has {len(y_pred)} values."
    )

    metrics = compute_all_metrics(y_true, y_pred)
    metrics["y_pred"] = y_pred
    return metrics


# ── Visualisations ────────────────────────────────────────────────────────────

def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual RUL (test set)",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Scatter plot: actual RUL on x-axis, predicted RUL on y-axis.
    Points colour-coded by error direction (over/under-prediction).

    Late predictions (model over-estimates RUL) are shown in red because
    they are the dangerous ones — they indicate more life than actually remains.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = np.where(errors >= 0, "tomato", "steelblue")

    ax.scatter(y_true, y_pred, c=colors, alpha=0.7, s=40, edgecolors="none")
    lim = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(title)

    late_patch = mpatches.Patch(color="tomato", label="Late (over-estimate — dangerous)")
    early_patch = mpatches.Patch(color="steelblue", label="Early (under-estimate — safe)")
    ax.legend(handles=[late_patch, early_patch, ax.lines[0]])

    metrics_txt = (
        f"RMSE={rmse(y_true, y_pred):.1f}  "
        f"MAE={mae(y_true, y_pred):.1f}  "
        f"NASA={nasa_score(y_true, y_pred):.0f}"
    )
    ax.text(0.02, 0.98, metrics_txt, transform=ax.transAxes,
            fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual distribution",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Histogram + KDE of prediction errors (y_pred - y_true)."""
    errors = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(errors, bins=30, kde=True, ax=ax, color="steelblue")
    ax.axvline(0, color="black", lw=1.5, ls="--")
    ax.axvline(errors.mean(), color="tomato", lw=2, ls="--",
               label=f"Mean error = {errors.mean():.1f}")
    ax.set_xlabel("Prediction error (y_pred − y_true)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    # Annotate asymmetry
    pct_late = 100 * (errors >= 0).mean()
    ax.text(0.98, 0.95, f"{pct_late:.0f}% late predictions",
            transform=ax.transAxes, ha="right", va="top",
            color="tomato", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_rul_trajectories(
    df_test: pd.DataFrame,
    model,
    feature_cols: list[str],
    unit_ids: Sequence[int],
    y_true_rul: Optional[Union[pd.Series, np.ndarray]] = None,
    title: str = "RUL trajectories — test engines",
    save_path: Optional[Path] = None,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
) -> plt.Figure:
    """
    Plot predicted RUL over time for selected engines.

    If y_true_rul is given, mark the true endpoint.
    If lower/upper are provided for all test rows, also shade uncertainty bands.
    """
    fig, axes = plt.subplots(1, len(unit_ids), figsize=(5 * len(unit_ids), 4),
                             sharey=False)
    if len(unit_ids) == 1:
        axes = [axes]

    for ax, uid in zip(axes, unit_ids):
        unit_df = df_test[df_test["unit"] == uid].sort_values("cycle")
        if len(unit_df) == 0:
            ax.set_title(f"Unit {uid} not found")
            continue

        X_unit = unit_df[feature_cols].values
        preds = np.clip(model.predict(X_unit), 0, None)
        cycles = unit_df["cycle"].values

        ax.plot(cycles, preds, "steelblue", lw=2, label="Predicted RUL")

        # Uncertainty bands
        if lower is not None and upper is not None:
            idx = unit_df.index
            try:
                lo = np.clip(lower[idx], 0, None)
                hi = np.clip(upper[idx], 0, None)
                ax.fill_between(cycles, lo, hi, alpha=0.2, color="steelblue",
                                label="80% interval")
            except (IndexError, TypeError):
                pass

        # True endpoint
        if y_true_rul is not None:
            true_val = float(np.asarray(y_true_rul)[uid - 1])
            ax.axhline(true_val, color="tomato", ls="--", lw=1.5,
                       label=f"True RUL={true_val:.0f}")

        ax.set_xlabel("Cycle")
        ax.set_ylabel("RUL (cycles)")
        ax.set_title(f"Engine {uid}")
        ax.legend(fontsize=7)

    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cost_vs_threshold(
    cost_df: pd.DataFrame,
    optimal_t: float,
    baseline_rtf: float,
    baseline_fixed: float,
    title: str = "Fleet maintenance cost vs alert threshold",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot total fleet cost vs threshold T with baseline comparison lines.

    Parameters
    ----------
    cost_df : pd.DataFrame
        Output of cost_analysis(); must have 'threshold' and 'total_cost'.
    optimal_t : float
        The threshold that minimises cost (annotated on the plot).
    baseline_rtf : float
        Cost of run-to-failure policy.
    baseline_fixed : float
        Cost of fixed-schedule maintenance policy.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(cost_df["threshold"], cost_df["total_cost"],
            "steelblue", lw=2.5, label="Model-guided cost")
    ax.axhline(baseline_rtf, color="tomato", ls="--", lw=2,
               label=f"Run-to-failure: ${baseline_rtf:,.0f}")
    ax.axhline(baseline_fixed, color="orange", ls="--", lw=2,
               label=f"Fixed schedule: ${baseline_fixed:,.0f}")

    # Annotate optimum
    opt_cost = cost_df.loc[
        (cost_df["threshold"] - optimal_t).abs().idxmin(), "total_cost"
    ]
    ax.scatter([optimal_t], [opt_cost], color="green", s=120, zorder=5)
    ax.annotate(
        f"  T={optimal_t:.0f} cycles\n  Cost=${opt_cost:,.0f}",
        xy=(optimal_t, opt_cost),
        fontsize=9,
        color="green",
    )

    ax.set_xlabel("Alert threshold T (cycles)")
    ax.set_ylabel("Total fleet cost ($)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Business cost analysis ────────────────────────────────────────────────────

def cost_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    cost_failure: float = 10_000.0,
    cost_maintenance: float = 1_000.0,
    cost_per_early_cycle: float = 50.0,
) -> pd.DataFrame:
    """
    Sweep an alert threshold T and compute total fleet maintenance cost.

    Alert rule: raise maintenance alert when predicted_RUL < T.

    Cost assumptions (illustrative — state this clearly):
      - Unplanned failure: $10,000  (production stop + emergency repair)
      - Planned maintenance: $1,000 (scheduled downtime)
      - Early replacement: $50 per cycle of useful life wasted

    Decision outcomes per engine:
      - True positive  (TP): predicted < T, true < T  → planned maintenance  + small early cost
      - False negative (FN): predicted >= T, true < T → engine fails  → failure cost
      - False positive (FP): predicted < T, true >= T → unnecessary maintenance + large early cost
      - True negative  (TN): predicted >= T, true >= T → no action, no cost

    Parameters
    ----------
    y_true : array-like
        True RUL at evaluation time (test set last cycle).
    y_pred : array-like
        Predicted RUL.
    thresholds : np.ndarray, optional
        Alert thresholds to sweep.  Defaults to np.arange(0, 151, 1).
    cost_failure : float
    cost_maintenance : float
    cost_per_early_cycle : float

    Returns
    -------
    pd.DataFrame with columns:
        threshold, total_cost, n_tp, n_fn, n_fp, n_tn,
        n_failures, n_early, cost_from_failures, cost_from_maintenance
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if thresholds is None:
        thresholds = np.arange(0, 151, 1, dtype=float)

    records = []
    for t in thresholds:
        alert = y_pred < t      # model triggers maintenance alert
        fail  = y_true < t      # engine would actually fail soon (true RUL < threshold)

        tp = int((alert & fail).sum())
        fn = int((~alert & fail).sum())
        fp = int((alert & ~fail).sum())
        tn = int((~alert & ~fail).sum())

        # TP: planned maintenance, but some early-life wasted if alert too early
        cycles_wasted_tp = float(np.maximum(y_true[alert & fail] - 0, 0).sum())
        cost_tp = tp * cost_maintenance + cycles_wasted_tp * cost_per_early_cycle

        # FN: engine fails unexpectedly
        cost_fn = fn * cost_failure

        # FP: unnecessary maintenance (engine had life left)
        cycles_wasted_fp = float(np.maximum(y_true[alert & ~fail] - 0, 0).sum())
        cost_fp = fp * cost_maintenance + cycles_wasted_fp * cost_per_early_cycle

        total_cost = cost_tp + cost_fn + cost_fp

        records.append({
            "threshold": t,
            "total_cost": total_cost,
            "n_tp": tp,
            "n_fn": fn,
            "n_fp": fp,
            "n_tn": tn,
            "n_failures": fn,          # alias for clarity
            "n_early": fp,
            "cost_from_failures": cost_fn,
            "cost_from_maintenance": cost_tp + cost_fp,
        })

    return pd.DataFrame(records)


def run_to_failure_cost(n_engines: int, cost_failure: float = 10_000.0) -> float:
    """Baseline cost: every engine fails unexpectedly."""
    return n_engines * cost_failure


def fixed_schedule_cost(
    y_true: np.ndarray,
    fixed_threshold: int = 50,
    cost_maintenance: float = 1_000.0,
    cost_failure: float = 10_000.0,
    cost_per_early_cycle: float = 50.0,
) -> float:
    """
    Baseline cost: maintain every engine on a fixed schedule
    (replace when predicted age = fixed_threshold cycles from end).

    Engines whose true RUL < fixed_threshold would have failed before
    the scheduled maintenance — those count as failures.
    """
    y_true = np.asarray(y_true, dtype=float)
    n_failures = int((y_true < fixed_threshold).sum())
    n_scheduled = int((y_true >= fixed_threshold).sum())
    cycles_wasted = float(np.maximum(y_true[y_true >= fixed_threshold] - fixed_threshold, 0).sum())
    return (n_failures * cost_failure
            + n_scheduled * cost_maintenance
            + cycles_wasted * cost_per_early_cycle)
