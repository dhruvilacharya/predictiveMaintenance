"""
app/streamlit_app.py
====================
Predictive Maintenance Fleet Dashboard — 3-page Streamlit application.

Pages:
  1. Fleet Overview    — scatter of all test engines colour-coded by risk level
  2. Engine Deep-Dive  — per-engine sensor trends, RUL trajectory, SHAP force plot
  3. Cost Analysis     — interactive cost-vs-threshold chart with live controls

Run:
  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import (
    load_raw, attach_rul_labels, load_test_rul, get_last_cycle,
    normalize_per_operating_condition, SENSOR_COLS, OP_COLS,
)
from src.features import build_features, get_feature_columns, load_feature_names
from src.evaluate import (
    cost_analysis, run_to_failure_cost, fixed_schedule_cost,
    rmse, mae, nasa_score,
)
from src.train import load_model, predict_with_intervals, save_model

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
)

MODELS_DIR    = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR   = ROOT / "reports" / "figures"


# ── Data loading (cached) ─────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset …")
def load_data(dataset: str):
    """Load and feature-engineer test + train data for a given dataset."""
    n_clusters = 6 if dataset in ("FD002", "FD004") else 1

    train_raw = attach_rul_labels(load_raw(dataset, "train", ROOT / "data" / "raw"), clip=125)
    test_raw  = load_raw(dataset, "test", ROOT / "data" / "raw")
    rul_test  = load_test_rul(dataset, ROOT / "data" / "raw")

    kmeans_model = None
    scalers = {}
    if n_clusters > 1:
        train_raw, kmeans_model, scalers = normalize_per_operating_condition(
            train_raw, n_clusters=n_clusters, random_state=42
        )
        test_norm = test_raw.copy()
        test_norm["_op_cluster"] = kmeans_model.predict(test_raw[OP_COLS].values)
        for cid, scaler in scalers.items():
            mask = test_norm["_op_cluster"] == cid
            if mask.sum() > 0:
                test_norm.loc[mask, SENSOR_COLS] = scaler.transform(
                    test_norm.loc[mask, SENSOR_COLS]
                )
        test_raw = test_norm.drop(columns=["_op_cluster"])

    train_feat = build_features(
        train_raw, dataset=dataset, processed_dir=PROCESSED_DIR,
        include_op_cols=(n_clusters > 1),
    )
    test_feat = build_features(
        test_raw, dataset=dataset, processed_dir=PROCESSED_DIR,
        include_op_cols=(n_clusters > 1),
    )

    try:
        feature_cols = load_feature_names(dataset, PROCESSED_DIR)
    except FileNotFoundError:
        feature_cols = get_feature_columns(train_feat)

    return train_feat, test_feat, rul_test, feature_cols


@st.cache_resource(show_spinner="Loading model …")
def load_point_model(dataset: str):
    """Load the pre-trained LightGBM model for a dataset."""
    model_name = f"lgbm_{dataset}"
    try:
        return load_model(model_name, MODELS_DIR)
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner="Loading interval models …")
def load_interval_models(dataset: str):
    """Load quantile lower/upper models."""
    try:
        lower = load_model("lgbm_quantile_lower", MODELS_DIR)
        upper = load_model("lgbm_quantile_upper", MODELS_DIR)
        return lower, upper
    except FileNotFoundError:
        return None, None


def get_predictions(dataset: str, test_feat, rul_test, feature_cols, model):
    """Return predictions + intervals for the last cycle of each test engine."""
    if model is None:
        return None

    test_last = get_last_cycle(test_feat)
    X = test_last[feature_cols].values

    lower_m, upper_m = load_interval_models(dataset)
    if lower_m is not None and upper_m is not None:
        y_hat, lower, upper = predict_with_intervals(model, lower_m, upper_m, X)
    else:
        y_hat = np.clip(model.predict(X), 0, None)
        sigma = y_hat * 0.15  # fallback: 15% uncertainty band
        lower = np.maximum(y_hat - sigma, 0)
        upper = y_hat + sigma

    y_true = rul_test.values.astype(float)

    df = test_last[["unit"]].copy()
    df["y_true"]  = y_true
    df["y_pred"]  = y_hat
    df["lower"]   = lower
    df["upper"]   = upper
    df["error"]   = y_hat - y_true
    df["abs_err"] = np.abs(df["error"])
    return df


# ── Colour-coding helpers ─────────────────────────────────────────────────────

def risk_color(pred_rul: float, threshold: float) -> str:
    if pred_rul < threshold:
        return "red"
    elif pred_rul < threshold + 30:
        return "orange"
    return "green"


def risk_label(pred_rul: float, threshold: float) -> str:
    if pred_rul < threshold:
        return "🔴 ALERT"
    elif pred_rul < threshold + 30:
        return "🟡 WATCH"
    return "🟢 OK"


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Predictive Maintenance")
    st.markdown("---")

    dataset = st.selectbox("Dataset", ["FD001", "FD002", "FD004"], index=0)
    threshold = st.slider("Alert threshold (cycles)", min_value=0, max_value=150,
                          value=30, step=5,
                          help="Raise a maintenance alert when predicted RUL < T")

    st.markdown("---")
    st.markdown("**Cost assumptions** (illustrative)")
    cost_failure = st.number_input("Unplanned failure cost ($)", value=10_000,
                                   step=1_000, min_value=0)
    cost_maint   = st.number_input("Planned maintenance cost ($)", value=1_000,
                                   step=100, min_value=0)
    cost_early   = st.number_input("Early replacement ($/cycle)", value=50,
                                   step=10, min_value=0)

    st.markdown("---")
    page = st.radio("Navigate to", ["Fleet Overview", "Engine Deep-Dive", "Cost Analysis"])
    st.caption("NASA C-MAPSS dataset — proxy for industrial equipment")

# ── Load data and model ───────────────────────────────────────────────────────

try:
    train_feat, test_feat, rul_test, feature_cols = load_data(dataset)
    model = load_point_model(dataset)
    pred_df = get_predictions(dataset, test_feat, rul_test, feature_cols, model)
except FileNotFoundError as e:
    st.error(
        f"❌ Data or model not found.\n\n"
        f"Run `python scripts/download_data.py` to fetch the dataset, "
        f"then run Notebook 03 to train the models.\n\n`{e}`"
    )
    st.stop()

if model is None:
    st.warning(
        f"⚠️ No trained model found for {dataset}. "
        f"Run Notebook 03 first (`notebooks/03_modeling.ipynb`)."
    )
    st.stop()

if pred_df is None:
    st.error("Could not generate predictions.")
    st.stop()

# ── Page 1: Fleet Overview ────────────────────────────────────────────────────

if page == "Fleet Overview":
    st.title("🏭 Fleet Overview")
    st.markdown(
        f"**Dataset:** {dataset} &nbsp;|&nbsp; "
        f"**Alert threshold:** {threshold} cycles &nbsp;|&nbsp; "
        f"**{len(pred_df)} test engines**"
    )

    # Risk classification
    pred_df["risk"]       = pred_df["y_pred"].apply(lambda p: risk_label(p, threshold))
    pred_df["risk_color"] = pred_df["y_pred"].apply(lambda p: risk_color(p, threshold))

    # ── KPI cards ─────────────────────────────────────────────────────────────
    n_alert = (pred_df["y_pred"] < threshold).sum()
    n_watch = ((pred_df["y_pred"] >= threshold) & (pred_df["y_pred"] < threshold + 30)).sum()
    n_ok    = (pred_df["y_pred"] >= threshold + 30).sum()
    test_rmse  = float(np.sqrt(np.mean((pred_df["y_pred"] - pred_df["y_true"]) ** 2)))
    test_nasa  = float(nasa_score(pred_df["y_true"].values, pred_df["y_pred"].values))

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🔴 Alert",   n_alert, help=f"Predicted RUL < {threshold}")
    col2.metric("🟡 Watch",   n_watch, help=f"RUL within {threshold}–{threshold+30}")
    col3.metric("🟢 OK",      n_ok)
    col4.metric("RMSE",       f"{test_rmse:.1f} cycles")
    col5.metric("NASA score", f"{test_nasa:.0f}")

    st.markdown("---")

    # ── Scatter plot ──────────────────────────────────────────────────────────
    color_map = {"🔴 ALERT": "red", "🟡 WATCH": "orange", "🟢 OK": "green"}
    fig_scatter = px.scatter(
        pred_df,
        x="y_true",
        y="y_pred",
        color="risk",
        color_discrete_map=color_map,
        hover_data={"unit": True, "error": ":.1f", "abs_err": ":.1f",
                    "y_true": ":.0f", "y_pred": ":.0f"},
        labels={"y_true": "Actual RUL (cycles)", "y_pred": "Predicted RUL (cycles)"},
        title=f"Predicted vs Actual RUL — {dataset} (colour = alert status at T={threshold})",
        height=500,
    )
    max_rul = max(pred_df["y_true"].max(), pred_df["y_pred"].max()) * 1.05
    fig_scatter.add_shape(
        type="line", x0=0, y0=0, x1=max_rul, y1=max_rul,
        line=dict(dash="dash", color="grey", width=1.5),
    )
    fig_scatter.add_hline(y=threshold, line=dict(dash="dot", color="tomato", width=2),
                          annotation_text=f"Alert threshold (T={threshold})")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Fleet table ───────────────────────────────────────────────────────────
    st.subheader("Engine status table")
    display_df = pred_df[["unit", "y_true", "y_pred", "lower", "upper", "error", "risk"]].copy()
    display_df.columns = ["Engine", "True RUL", "Pred RUL", "Lower 80%", "Upper 80%",
                          "Error", "Status"]
    display_df = display_df.sort_values("Pred RUL")
    for col in ["True RUL", "Pred RUL", "Lower 80%", "Upper 80%", "Error"]:
        display_df[col] = display_df[col].round(1)

    st.dataframe(
        display_df.style.apply(
            lambda row: ["background-color: #ffe5e5" if "ALERT" in row["Status"]
                         else "background-color: #fff9e5" if "WATCH" in row["Status"]
                         else "" for _ in row],
            axis=1,
        ),
        height=350,
        use_container_width=True,
    )


# ── Page 2: Engine Deep-Dive ──────────────────────────────────────────────────

elif page == "Engine Deep-Dive":
    st.title("🔍 Engine Deep-Dive")

    unit_ids = sorted(pred_df["unit"].tolist())
    selected_unit = st.selectbox("Select engine", unit_ids, index=0)

    unit_row = pred_df[pred_df["unit"] == selected_unit].iloc[0]
    pred_rul  = unit_row["y_pred"]
    true_rul  = unit_row["y_true"]
    error     = unit_row["error"]
    status    = risk_label(pred_rul, threshold)

    # ── Status badge ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status",        status)
    col2.metric("Predicted RUL", f"{pred_rul:.0f} cycles")
    col3.metric("True RUL",      f"{true_rul:.0f} cycles")
    col4.metric("Error",         f"{error:+.0f} cycles",
                delta_color="inverse" if error > 0 else "normal")

    st.markdown("---")

    # ── RUL trajectory over time ───────────────────────────────────────────────
    st.subheader("Predicted RUL over time")
    unit_test = test_feat[test_feat["unit"] == selected_unit].sort_values("cycle")

    if len(unit_test) > 0:
        X_unit = unit_test[feature_cols].values
        preds_over_time = np.clip(model.predict(X_unit), 0, None)
        cycles = unit_test["cycle"].values

        lower_m, upper_m = load_interval_models(dataset)
        if lower_m is not None and upper_m is not None:
            _, lo_time, hi_time = predict_with_intervals(
                model, lower_m, upper_m, X_unit
            )
        else:
            sigma = preds_over_time * 0.15
            lo_time = np.maximum(preds_over_time - sigma, 0)
            hi_time = preds_over_time + sigma

        fig_traj = go.Figure()
        fig_traj.add_trace(go.Scatter(
            x=np.concatenate([cycles, cycles[::-1]]),
            y=np.concatenate([hi_time, lo_time[::-1]]),
            fill="toself", fillcolor="rgba(70,130,180,0.2)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% prediction interval",
            hoverinfo="skip",
        ))
        fig_traj.add_trace(go.Scatter(
            x=cycles, y=preds_over_time, mode="lines",
            name="Predicted RUL", line=dict(color="steelblue", width=2.5),
        ))
        fig_traj.add_hline(
            y=threshold, line=dict(dash="dot", color="tomato", width=2),
            annotation_text=f"Alert threshold (T={threshold})",
        )
        fig_traj.add_hline(
            y=true_rul, line=dict(dash="dash", color="green", width=1.5),
            annotation_text=f"True RUL = {true_rul:.0f}",
        )
        fig_traj.update_layout(
            title=f"Engine {selected_unit} — RUL trajectory (test set)",
            xaxis_title="Cycle",
            yaxis_title="RUL (cycles)",
            height=350,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_traj, use_container_width=True)

    # ── Top-5 sensor trends ───────────────────────────────────────────────────
    st.subheader("Top-5 sensor trends (rolling-mean smoothed)")

    try:
        import shap
        point_model = model
        # Use a small sample for SHAP (last cycle only)
        X_last = unit_test[feature_cols].values[-1:] if len(unit_test) > 0 else None
        if X_last is not None:
            explainer = shap.TreeExplainer(point_model)
            sv = explainer.shap_values(X_last)
            if isinstance(sv, list):
                sv = sv[0]
            feat_imp = pd.Series(np.abs(sv[0]), index=feature_cols)
            top5_feats = feat_imp.nlargest(5).index.tolist()
        else:
            top5_feats = []
    except Exception:
        # Fallback: use most variance sensors
        top5_feats = []

    if not top5_feats:
        # Use sensors with highest correlation to RUL in training data
        top5_feats = [c for c in feature_cols if "_mean_" in c][:5]

    if len(unit_test) > 0 and top5_feats:
        n_cols = min(len(top5_feats), 5)
        fig_sensors = go.Figure()
        for feat in top5_feats[:5]:
            if feat in unit_test.columns:
                vals = unit_test[feat].values
                smooth = pd.Series(vals).rolling(5, min_periods=1).mean().values
                fig_sensors.add_trace(go.Scatter(
                    x=cycles, y=smooth, mode="lines", name=feat,
                ))
        fig_sensors.update_layout(
            title=f"Engine {selected_unit} — Top-5 feature trends",
            xaxis_title="Cycle",
            yaxis_title="Feature value",
            height=300,
            legend=dict(orientation="h", y=-0.3),
        )
        st.plotly_chart(fig_sensors, use_container_width=True)

    # ── SHAP force plot ───────────────────────────────────────────────────────
    st.subheader("SHAP feature contributions (last cycle)")
    try:
        import shap
        X_last = unit_test[feature_cols].values[-1:]
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_last)
        if isinstance(sv, list):
            sv = sv[0]
        feat_imp = pd.Series(sv[0], index=feature_cols)
        top_pos = feat_imp.nlargest(8)
        top_neg = feat_imp.nsmallest(8)
        top_all = pd.concat([top_pos, top_neg]).sort_values()

        fig_shap = go.Figure(go.Bar(
            x=top_all.values,
            y=top_all.index,
            orientation="h",
            marker_color=["tomato" if v > 0 else "steelblue" for v in top_all.values],
        ))
        fig_shap.update_layout(
            title=f"SHAP contributions — Engine {selected_unit} (positive = higher RUL prediction)",
            xaxis_title="SHAP value",
            height=450,
        )
        st.plotly_chart(fig_shap, use_container_width=True)
    except Exception as e:
        st.info(f"SHAP not available: {e}. Install shap and retrain to enable this view.")


# ── Page 3: Cost Analysis ─────────────────────────────────────────────────────

elif page == "Cost Analysis":
    st.title("💰 Cost Analysis")
    st.markdown(
        "Sweep the alert threshold and compute total fleet maintenance cost. "
        "Adjust cost assumptions in the sidebar."
    )

    y_true = pred_df["y_true"].values
    y_pred = pred_df["y_pred"].values
    n_engines = len(y_true)

    thresholds = np.arange(0, 151, 1, dtype=float)
    cost_df = cost_analysis(
        y_true, y_pred,
        thresholds=thresholds,
        cost_failure=float(cost_failure),
        cost_maintenance=float(cost_maint),
        cost_per_early_cycle=float(cost_early),
    )

    opt_idx      = cost_df["total_cost"].idxmin()
    optimal_t    = cost_df.loc[opt_idx, "threshold"]
    optimal_cost = cost_df.loc[opt_idx, "total_cost"]
    rtf_cost     = run_to_failure_cost(n_engines, cost_failure)
    fixed_cost   = fixed_schedule_cost(y_true, fixed_threshold=50,
                                        cost_maintenance=cost_maint,
                                        cost_failure=cost_failure,
                                        cost_per_early_cycle=cost_early)
    saving_rtf   = 100 * (rtf_cost - optimal_cost) / rtf_cost
    saving_fixed = 100 * (fixed_cost - optimal_cost) / max(fixed_cost, 1)

    # Current threshold cost
    curr_cost_row = cost_df.loc[(cost_df["threshold"] - threshold).abs().idxmin()]
    curr_cost = curr_cost_row["total_cost"]

    # ── KPI cards ─────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Optimal threshold", f"T = {optimal_t:.0f} cycles")
    col2.metric("Optimal cost",      f"${optimal_cost:,.0f}")
    col3.metric("vs Run-to-failure", f"{saving_rtf:.1f}% saved",
                delta=f"-${rtf_cost-optimal_cost:,.0f}")
    col4.metric("Current T cost",    f"${curr_cost:,.0f}",
                delta=f"{(curr_cost-optimal_cost):+,.0f} vs optimal")

    st.markdown("---")

    # ── Interactive cost chart ─────────────────────────────────────────────────
    fig_cost = go.Figure()

    fig_cost.add_trace(go.Scatter(
        x=cost_df["threshold"], y=cost_df["total_cost"],
        mode="lines", name="Model-guided cost",
        line=dict(color="steelblue", width=2.5),
    ))
    fig_cost.add_hline(y=rtf_cost, line=dict(dash="dash", color="tomato", width=2),
                       annotation_text=f"Run-to-failure: ${rtf_cost:,.0f}")
    fig_cost.add_hline(y=fixed_cost, line=dict(dash="dash", color="orange", width=2),
                       annotation_text=f"Fixed schedule (50 cycles): ${fixed_cost:,.0f}")
    fig_cost.add_vline(x=optimal_t, line=dict(dash="dot", color="green", width=2),
                       annotation_text=f"Optimal T={optimal_t:.0f}")
    fig_cost.add_vline(x=threshold, line=dict(dash="dot", color="grey", width=1.5),
                       annotation_text=f"Current T={threshold}")

    fig_cost.update_layout(
        title=f"Fleet maintenance cost vs alert threshold — {dataset}",
        xaxis_title="Alert threshold T (cycles)",
        yaxis_title="Total fleet cost ($)",
        height=450,
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_cost, use_container_width=True)

    # ── Cost breakdown at current threshold ───────────────────────────────────
    st.subheader(f"Cost breakdown at T = {threshold} cycles")
    col1, col2, col3 = st.columns(3)
    col1.metric("Engines failing (false negatives)", int(curr_cost_row["n_failures"]),
                help="Engines that slip past the threshold and fail unexpectedly")
    col2.metric("Unnecessary maintenances (false positives)", int(curr_cost_row["n_early"]),
                help="Engines replaced with usable life remaining")
    col3.metric("Cost from failures",
                f"${curr_cost_row['cost_from_failures']:,.0f}")

    st.markdown("---")

    # ── Cost assumptions note ─────────────────────────────────────────────────
    with st.expander("📋 Cost model assumptions"):
        st.markdown(f"""
        | Assumption | Value |
        |---|---|
        | Unplanned failure | ${cost_failure:,} per engine |
        | Planned maintenance | ${cost_maint:,} per event |
        | Early replacement | ${cost_early}/cycle of life wasted |
        | Fixed schedule baseline | 50 cycles before predicted end-of-life |

        > **Disclaimer:** These costs are illustrative. Real-world deployment
        > requires domain-expert cost estimates based on actual downtime costs,
        > repair times, and supply-chain constraints.
        """)

    # ── Download cost table ───────────────────────────────────────────────────
    csv = cost_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download cost table (CSV)",
        csv,
        file_name=f"cost_analysis_{dataset}.csv",
        mime="text/csv",
    )
