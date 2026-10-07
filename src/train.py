"""
src/train.py
============
Training infrastructure for the RUL project.

Covers:
  - GroupKFold cross-validation with MLflow logging
  - Sklearn-compatible models: Ridge, RandomForest, LightGBM, XGBoost
  - Quantile regression for prediction intervals (LightGBM)
  - Sliding-window construction for sequence models
  - Keras 1D-CNN and LSTM architectures
  - Optuna hyperparameter tuning
  - Model save / load utilities

Public API
----------
train_model(model_type, X, y, groups, params, experiment_name, run_name, clip_rul)
    GroupKFold CV, MLflow logging, returns best model + metrics.

tune_lightgbm(X, y, groups, n_trials, experiment_name)
    Optuna tuning for LightGBM.

train_quantile_models(X, y, groups, alpha_low, alpha_high, experiment_name)
    Train LightGBM quantile regressors for prediction intervals.

predict_with_intervals(point_model, lower_model, upper_model, X)
    Return (y_hat, lower, upper) from three LightGBM models.

build_sliding_windows(df, feature_cols, window, target_col)
    Construct (X_seq, y_seq, unit_ids) for sequence models.

build_cnn_model(input_shape)
build_lstm_model(input_shape)
    Keras model factories.

train_keras_model(model, X_seq, y_seq, groups_seq, experiment_name, run_name)
    Train with early stopping; log to MLflow.

save_model(model, name, models_dir)
load_model(name, models_dir)
    Joblib-based persistence (also handles Keras .keras format).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.evaluate import compute_all_metrics, evaluate_cv

# ── Paths ──────────────────────────────────────────────────────────────────────

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _models_dir() -> Path:
    return _project_root() / "models"


# ── MLflow helpers ────────────────────────────────────────────────────────────

def _setup_mlflow(experiment_name: str) -> str:
    """Create or get MLflow experiment; return experiment id."""
    tracking_uri = str(_project_root() / "mlruns")
    mlflow.set_tracking_uri(f"file://{tracking_uri}")
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = mlflow.create_experiment(experiment_name)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_name)
    return exp_id


# ── Model factories ───────────────────────────────────────────────────────────

def _build_sklearn_model(model_type: str, params: dict):
    """Return an sklearn-compatible estimator."""
    if model_type == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(**{k: v for k, v in params.items() if k != "model_type"})),
        ])
    elif model_type == "rf":
        return RandomForestRegressor(
            n_jobs=-1,
            random_state=42,
            **{k: v for k, v in params.items() if k != "model_type"},
        )
    elif model_type == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_jobs=-1,
            random_state=42,
            verbose=-1,
            **{k: v for k, v in params.items() if k != "model_type"},
        )
    elif model_type == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_jobs=-1,
            random_state=42,
            verbosity=0,
            **{k: v for k, v in params.items() if k != "model_type"},
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")


# ── Core training function ────────────────────────────────────────────────────

def train_model(
    model_type: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    params: Optional[dict] = None,
    experiment_name: str = "predictive-maintenance",
    run_name: Optional[str] = None,
    n_splits: int = 5,
    save: bool = True,
    models_dir: Optional[Path] = None,
) -> dict:
    """
    Train a model with GroupKFold CV, log results to MLflow.

    Parameters
    ----------
    model_type : str
        One of 'ridge', 'rf', 'lgbm', 'xgb'.
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Target (RUL).
    groups : np.ndarray
        Engine unit IDs (used for GroupKFold split).
    params : dict, optional
        Model hyperparameters.
    experiment_name : str
    run_name : str, optional
    n_splits : int
        Number of CV folds.
    save : bool
        Whether to save the final model (retrained on all data).
    models_dir : Path, optional

    Returns
    -------
    dict with keys:
        'model'       : fitted model (trained on full data)
        'cv_metrics'  : dict of mean/std for rmse, mae, nasa_score
        'fold_metrics': list of per-fold metric dicts
        'run_id'      : MLflow run ID
        'feature_cols': list of feature column names
    """
    params = params or {}
    models_dir = models_dir or _models_dir()
    run_name = run_name or model_type

    _setup_mlflow(experiment_name)
    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_arr = np.asarray(y, dtype=float)
    groups_arr = np.asarray(groups)
    feature_cols = list(X.columns) if isinstance(X, pd.DataFrame) else []

    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    oof_pred = np.zeros(len(y_arr))

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_params({f"param_{k}": v for k, v in params.items()})

        for fold, (tr_idx, val_idx) in enumerate(
            gkf.split(X_arr, y_arr, groups=groups_arr)
        ):
            X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]

            model = _build_sklearn_model(model_type, params)
            model.fit(X_tr, y_tr)

            preds = np.clip(model.predict(X_val), 0, None)
            oof_pred[val_idx] = preds

            fold_m = compute_all_metrics(y_val, preds)
            fold_m["fold"] = fold
            fold_metrics.append(fold_m)

            mlflow.log_metrics({
                f"fold_{fold}_rmse": fold_m["rmse"],
                f"fold_{fold}_mae": fold_m["mae"],
                f"fold_{fold}_nasa_score": fold_m["nasa_score"],
            })

        cv_metrics = evaluate_cv(fold_metrics)
        mlflow.log_metrics({
            "cv_rmse_mean": cv_metrics["rmse_mean"],
            "cv_rmse_std": cv_metrics["rmse_std"],
            "cv_mae_mean": cv_metrics["mae_mean"],
            "cv_nasa_score_mean": cv_metrics["nasa_score_mean"],
        })

        # Retrain on full data
        final_model = _build_sklearn_model(model_type, params)
        final_model.fit(X_arr, y_arr)

        if save:
            model_path = save_model(final_model, run_name, models_dir)
            mlflow.log_artifact(str(model_path))

        run_id = run.info.run_id

    return {
        "model": final_model,
        "cv_metrics": cv_metrics,
        "fold_metrics": fold_metrics,
        "run_id": run_id,
        "feature_cols": feature_cols,
        "oof_pred": oof_pred,
    }


# ── Optuna tuning ─────────────────────────────────────────────────────────────

def tune_lightgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_trials: int = 40,
    experiment_name: str = "predictive-maintenance",
    n_splits: int = 5,
    timeout: Optional[int] = None,
) -> dict:
    """
    Optuna hyperparameter search for LightGBM.

    Parameters
    ----------
    X, y, groups : feature matrix, target, group ids
    n_trials : int
        Number of Optuna trials (30–50 is usually sufficient).
    timeout : int, optional
        Stop after this many seconds regardless of n_trials.

    Returns
    -------
    dict with 'best_params', 'best_rmse', 'study'
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Install optuna: pip install optuna")

    from lightgbm import LGBMRegressor
    _setup_mlflow(experiment_name)

    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_arr = np.asarray(y, dtype=float)
    groups_arr = np.asarray(groups)

    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 200, 1500),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":      trial.suggest_int("num_leaves", 20, 200),
            "max_depth":       trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
        }

        gkf = GroupKFold(n_splits=n_splits)
        fold_rmse = []
        for tr_idx, val_idx in gkf.split(X_arr, y_arr, groups=groups_arr):
            m = LGBMRegressor(n_jobs=-1, random_state=42, verbose=-1, **params)
            m.fit(X_arr[tr_idx], y_arr[tr_idx])
            preds = np.clip(m.predict(X_arr[val_idx]), 0, None)
            from sklearn.metrics import mean_squared_error
            fold_rmse.append(float(np.sqrt(mean_squared_error(y_arr[val_idx], preds))))

        return float(np.mean(fold_rmse))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_params = study.best_params
    best_rmse = study.best_value

    # Log best params to MLflow
    with mlflow.start_run(run_name="lgbm_optuna_best"):
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_rmse_mean", best_rmse)

    return {"best_params": best_params, "best_rmse": best_rmse, "study": study}


# ── Quantile regression (prediction intervals) ────────────────────────────────

def train_quantile_models(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    alpha_low: float = 0.1,
    alpha_high: float = 0.9,
    params: Optional[dict] = None,
    experiment_name: str = "predictive-maintenance",
    n_splits: int = 5,
    models_dir: Optional[Path] = None,
) -> dict:
    """
    Train LightGBM quantile regressors for lower and upper prediction bounds.

    Uses inductive conformal prediction on a held-out calibration fold to
    ensure empirical coverage.

    Parameters
    ----------
    alpha_low, alpha_high : float
        Lower and upper quantiles (e.g. 0.1, 0.9 → 80% interval).

    Returns
    -------
    dict with 'lower_model', 'upper_model', 'point_model',
              'coverage', 'calibration_set'
    """
    from lightgbm import LGBMRegressor

    models_dir = models_dir or _models_dir()
    params = params or {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 7,
    }

    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_arr = np.asarray(y, dtype=float)
    groups_arr = np.asarray(groups)

    # Reserve one fold as calibration set for conformal prediction
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X_arr, y_arr, groups=groups_arr))
    cal_tr_idx, cal_idx = splits[-1]  # last fold = calibration

    X_train, X_cal = X_arr[cal_tr_idx], X_arr[cal_idx]
    y_train, y_cal = y_arr[cal_tr_idx], y_arr[cal_idx]

    _setup_mlflow(experiment_name)

    models_out = {}
    with mlflow.start_run(run_name="quantile_regression"):
        for alpha, label in [(alpha_low, "lower"), (0.5, "point"), (alpha_high, "upper")]:
            m = LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                n_jobs=-1,
                random_state=42,
                verbose=-1,
                **params,
            )
            m.fit(X_train, y_train)
            models_out[f"{label}_model"] = m
            save_model(m, f"lgbm_quantile_{label}", models_dir)

        # Conformal calibration: compute non-conformity scores on calibration set
        point_preds_cal = np.clip(models_out["point_model"].predict(X_cal), 0, None)
        nonconf_scores = np.abs(y_cal - point_preds_cal)
        q_level = np.ceil((1 - alpha_low) * (len(nonconf_scores) + 1)) / len(nonconf_scores)
        q_level = min(q_level, 1.0)
        conf_margin = float(np.quantile(nonconf_scores, q_level))

        # Compute empirical coverage on calibration set
        lo_cal = np.clip(models_out["lower_model"].predict(X_cal), 0, None) - conf_margin
        hi_cal = np.clip(models_out["upper_model"].predict(X_cal), 0, None) + conf_margin
        coverage = float(np.mean((y_cal >= lo_cal) & (y_cal <= hi_cal)))

        mlflow.log_metric("calibration_coverage", coverage)
        mlflow.log_metric("conformal_margin", conf_margin)
        mlflow.log_params({"alpha_low": alpha_low, "alpha_high": alpha_high})

    models_out["coverage"] = coverage
    models_out["conf_margin"] = conf_margin
    models_out["calibration_set"] = (X_cal, y_cal)
    return models_out


def predict_with_intervals(
    point_model,
    lower_model,
    upper_model,
    X: np.ndarray,
    conf_margin: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (y_hat, lower, upper) predictions.

    Parameters
    ----------
    conf_margin : float
        Conformal adjustment to expand the interval (from calibration set).

    Returns
    -------
    y_hat, lower, upper : np.ndarray  (all clipped at 0)
    """
    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_hat = np.clip(point_model.predict(X_arr), 0, None)
    lower = np.clip(lower_model.predict(X_arr) - conf_margin, 0, None)
    upper = np.clip(upper_model.predict(X_arr) + conf_margin, 0, None)
    return y_hat, lower, upper


# ── Sliding windows for sequence models ──────────────────────────────────────

def build_sliding_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window: int = 30,
    target_col: str = "rul",
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences for 1D-CNN / LSTM training.

    For each engine unit, create all windows of length `window`.
    At each position t, the window spans cycles [t-window+1, …, t] and
    the target is RUL at cycle t.

    No future data leaks in: the window looks strictly backward.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'unit', 'cycle', `feature_cols`, and `target_col`.
    feature_cols : list of str
    window : int
        Sequence length in cycles.
    target_col : str
    step : int
        Step between consecutive windows (1 = dense, larger = faster).

    Returns
    -------
    X_seq : np.ndarray  shape (N, window, n_features)
    y_seq : np.ndarray  shape (N,)
    groups : np.ndarray shape (N,)   engine unit ID for each window
    """
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    X_list, y_list, g_list = [], [], []

    for unit, unit_df in df.groupby("unit", sort=True):
        feats = unit_df[feature_cols].values
        targets = unit_df[target_col].values
        n = len(feats)

        for end in range(window, n + 1, step):
            start = end - window
            X_list.append(feats[start:end])   # shape (window, n_features)
            y_list.append(targets[end - 1])    # target at the last step
            g_list.append(unit)

    X_seq = np.array(X_list, dtype=np.float32)
    y_seq = np.array(y_list, dtype=np.float32)
    groups = np.array(g_list)
    return X_seq, y_seq, groups


# ── Keras model architectures ─────────────────────────────────────────────────

def build_cnn_model(input_shape: Tuple[int, int]):
    """
    1D-CNN for RUL regression.

    Architecture: Conv1D(64) → Conv1D(32) → GlobalAvgPool → Dense(64) → Dense(1)
    Suited for capturing local patterns in the sensor time series.

    Parameters
    ----------
    input_shape : (window, n_features)
    """
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        raise ImportError("Install tensorflow: pip install tensorflow")

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="relu"),   # RUL >= 0
    ], name="cnn_rul")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_lstm_model(input_shape: Tuple[int, int]):
    """
    Stacked LSTM for RUL regression.

    Architecture: LSTM(64, return_sequences) → LSTM(32) → Dense(32) → Dense(1)
    Suited for capturing long-range temporal dependencies.

    Parameters
    ----------
    input_shape : (window, n_features)
    """
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        raise ImportError("Install tensorflow: pip install tensorflow")

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="relu"),   # RUL >= 0
    ], name="lstm_rul")

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_keras_model(
    model,
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    groups_seq: np.ndarray,
    experiment_name: str = "predictive-maintenance",
    run_name: str = "keras",
    n_splits: int = 5,
    epochs: int = 60,
    batch_size: int = 256,
    patience: int = 10,
    models_dir: Optional[Path] = None,
) -> dict:
    """
    Train a Keras model with GroupKFold CV.

    Uses EarlyStopping and ReduceLROnPlateau callbacks.
    Logs to MLflow.

    Returns
    -------
    dict with 'model', 'cv_metrics', 'fold_metrics', 'run_id'
    """
    try:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    except ImportError:
        raise ImportError("Install tensorflow: pip install tensorflow")

    models_dir = models_dir or _models_dir()
    _setup_mlflow(experiment_name)

    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model_architecture", run_name)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)

        # Get initial weights to reset between folds
        initial_weights = model.get_weights()

        for fold, (tr_idx, val_idx) in enumerate(
            gkf.split(X_seq, y_seq, groups=groups_seq)
        ):
            model.set_weights(initial_weights)  # reset
            X_tr, X_val = X_seq[tr_idx], X_seq[val_idx]
            y_tr, y_val = y_seq[tr_idx], y_seq[val_idx]

            callbacks = [
                EarlyStopping(monitor="val_loss", patience=patience,
                              restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                  patience=5, min_lr=1e-6),
            ]

            model.fit(
                X_tr, y_tr,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0,
            )

            preds = np.clip(model.predict(X_val, verbose=0).flatten(), 0, None)
            fold_m = compute_all_metrics(y_val, preds)
            fold_m["fold"] = fold
            fold_metrics.append(fold_m)

            mlflow.log_metrics({
                f"fold_{fold}_rmse": fold_m["rmse"],
                f"fold_{fold}_nasa_score": fold_m["nasa_score"],
            })

        cv_metrics = evaluate_cv(fold_metrics)
        mlflow.log_metrics({
            "cv_rmse_mean": cv_metrics["rmse_mean"],
            "cv_nasa_score_mean": cv_metrics["nasa_score_mean"],
        })

        # Final model: retrain on all data
        model.set_weights(initial_weights)
        callbacks_final = [
            EarlyStopping(monitor="loss", patience=patience,
                          restore_best_weights=True),
        ]
        model.fit(X_seq, y_seq, epochs=epochs, batch_size=batch_size,
                  callbacks=callbacks_final, verbose=0)

        model_path = Path(models_dir) / f"{run_name}.keras"
        model.save(str(model_path))
        mlflow.log_artifact(str(model_path))
        run_id = run.info.run_id

    return {
        "model": model,
        "cv_metrics": cv_metrics,
        "fold_metrics": fold_metrics,
        "run_id": run_id,
    }


# ── Model persistence ─────────────────────────────────────────────────────────

def save_model(model, name: str, models_dir: Optional[Path] = None) -> Path:
    """Save a model with joblib.  Returns the saved path."""
    models_dir = Path(models_dir or _models_dir())
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name: str, models_dir: Optional[Path] = None):
    """Load a joblib-saved model."""
    models_dir = Path(models_dir or _models_dir())
    path = models_dir / f"{name}.joblib"
    if not path.exists():
        # Try Keras format
        keras_path = models_dir / f"{name}.keras"
        if keras_path.exists():
            from tensorflow import keras
            return keras.models.load_model(str(keras_path))
        raise FileNotFoundError(
            f"Model not found: {path} (also tried {keras_path})"
        )
    return joblib.load(path)
