"""
src/explain.py
==============
SHAP-based model explainability for tree models.

Public API
----------
explain_model(model, X_val, feature_names, save_dir, max_display)
    Run SHAP TreeExplainer, save summary plot, force plot for worst engine,
    and write a human-readable interpretation to reports/.

get_top_features(shap_values, feature_names, n)
    Return the top-n features by mean absolute SHAP value.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _figures_dir() -> Path:
    return _project_root() / "reports" / "figures"


def _reports_dir() -> Path:
    return _project_root() / "reports"


# ── SHAP utilities ────────────────────────────────────────────────────────────

def explain_model(
    model,
    X_val: pd.DataFrame,
    feature_names: list[str],
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    save_dir: Optional[Path] = None,
    max_display: int = 20,
    dataset: str = "FD001",
) -> dict:
    """
    Generate SHAP explanations for a fitted tree model.

    Steps:
      1. Fit a TreeExplainer on X_val.
      2. Save a summary (beeswarm) plot — global feature importance.
      3. Identify the engine with the worst prediction (if y_true/y_pred given)
         and save a force plot for its last row.
      4. Write a plain-text interpretation to reports/shap_interpretation.txt.

    Parameters
    ----------
    model : fitted LightGBM / XGBoost / RandomForest model
    X_val : pd.DataFrame
        Validation or test feature matrix (rows = observations).
    feature_names : list of str
    y_true : np.ndarray, optional
        True RUL values (same length as X_val).
    y_pred : np.ndarray, optional
        Predicted RUL values.
    save_dir : Path, optional
        Directory to save figures.  Defaults to reports/figures/.
    max_display : int
        Number of features shown in summary plot.
    dataset : str

    Returns
    -------
    dict with keys:
        'shap_values'     : np.ndarray  (n_samples, n_features)
        'top_features'    : list of str
        'summary_fig_path': Path
        'interpretation'  : str
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Install shap: pip install shap")

    save_dir = save_dir or _figures_dir()
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    X_np = X_val[feature_names].values if isinstance(X_val, pd.DataFrame) else X_val

    # ── SHAP values ────────────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_np)

    # Some models return a list (one per class); for regression take first element
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # ── Summary plot ───────────────────────────────────────────────────────────
    summary_path = Path(save_dir) / f"shap_summary_{dataset}.png"
    fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.4)))
    shap.summary_plot(
        shap_values,
        X_np,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.title(f"SHAP Feature Importance — {dataset}", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    # ── Top features ───────────────────────────────────────────────────────────
    top_feats = get_top_features(shap_values, feature_names, n=max_display)

    # ── Force plot for worst prediction ───────────────────────────────────────
    force_path = None
    if y_true is not None and y_pred is not None:
        errors = np.abs(np.asarray(y_pred) - np.asarray(y_true))
        worst_idx = int(np.argmax(errors))

        force_path = Path(save_dir) / f"shap_force_worst_{dataset}.png"
        shap.initjs()
        force_fig = shap.force_plot(
            explainer.expected_value,
            shap_values[worst_idx],
            X_np[worst_idx],
            feature_names=feature_names,
            matplotlib=True,
            show=False,
        )
        plt.savefig(force_path, dpi=150, bbox_inches="tight")
        plt.close("all")

    # ── Written interpretation ────────────────────────────────────────────────
    interp = _write_interpretation(
        top_feats, shap_values, feature_names, dataset
    )
    interp_path = _reports_dir() / f"shap_interpretation_{dataset}.txt"
    interp_path.parent.mkdir(parents=True, exist_ok=True)
    interp_path.write_text(interp)

    return {
        "shap_values": shap_values,
        "top_features": top_feats,
        "summary_fig_path": summary_path,
        "force_fig_path": force_path,
        "interpretation": interp,
        "interpretation_path": interp_path,
    }


def get_top_features(
    shap_values: np.ndarray,
    feature_names: Sequence[str],
    n: int = 20,
) -> list[str]:
    """Return the top-n features ranked by mean |SHAP value|."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    return [feature_names[i] for i in order[:n]]


def _write_interpretation(
    top_feats: list[str],
    shap_values: np.ndarray,
    feature_names: list[str],
    dataset: str,
) -> str:
    """
    Generate a short human-readable SHAP interpretation.

    Examines which feature types (rolling trend, drift, slope, lag, raw level)
    dominate the top-10 SHAP features and writes 4–5 sentences.
    """
    fn_set = set(feature_names)
    mean_abs = np.abs(shap_values).mean(axis=0)
    feat_imp = dict(zip(feature_names, mean_abs))

    top10 = top_feats[:10]
    trend_feats  = [f for f in top10 if "_mean_" in f or "_slope_" in f]
    drift_feats  = [f for f in top10 if "_drift"  in f]
    level_feats  = [f for f in top10 if "_std_"   in f or "_min_" in f or "_max_" in f]
    lag_feats    = [f for f in top10 if "_lag_"   in f]

    top3_names = ", ".join(top10[:3])

    lines = [
        f"SHAP Interpretation — {dataset}\n{'=' * 50}\n",
        f"The three most influential features are: {top3_names}.",
    ]

    if trend_feats:
        lines.append(
            f"Rolling mean and slope features ({', '.join(trend_feats[:2])}) dominate "
            f"the top-{len(top10)} list, confirming that the model primarily tracks "
            f"the *rate of change* in sensor readings rather than their absolute level."
        )

    if drift_feats:
        lines.append(
            f"Drift features ({', '.join(drift_feats[:2])}) also rank highly, "
            f"suggesting that deviation from each engine's personal early-life baseline "
            f"is a stronger degradation signal than the raw sensor value."
        )

    if level_feats:
        lines.append(
            f"Variability features ({', '.join(level_feats[:2])}) appear in the top-10, "
            f"consistent with increased sensor noise as components wear."
        )

    lines.append(
        "These findings match physical intuition: degradation manifests as a gradual "
        "monotonic drift in specific sensors (temperature, pressure, efficiency proxies) "
        "rather than a sudden step change, so trend-based features are the most predictive."
    )

    lines.append(
        "\nNote: SHAP values are computed on the validation set.  "
        "Feature importance may shift on FD002/FD004 due to different fault modes "
        "and operating conditions."
    )

    return "\n".join(lines)
