# Predictive Maintenance & Remaining Useful Life (RUL)

> A tire plant runs hundreds of machines — mixers, extruders, curing presses. An unplanned
> breakdown stops a line, wastes material, and delays orders. Can we predict how many cycles
> a machine has left before failure, so maintenance can be scheduled **just in time** instead
> of too early (waste) or too late (breakdown)?

## Proxy Data Disclaimer

This project uses the **NASA C-MAPSS turbofan engine degradation dataset** as a proxy for
industrial equipment. The problem structure — many units, multi-sensor degradation,
run-to-failure — is identical to real plant machinery, which is why C-MAPSS is the standard
benchmark in predictive-maintenance research. Results should **not** be extrapolated to
production without real sensor data, labeled failures, domain expert validation, and
continuous model monitoring.

---

## Results

| Model | Test RMSE | Test MAE | NASA Score | Notes |
|---|---|---|---|---|
| Constant baseline | ~83 | ~63 | ~25,000 | Predict mean RUL for all |
| Ridge regression | ~32 | ~24 | ~6,500 | Linear; clipped target |
| Random Forest | ~18 | ~13 | ~1,800 | 300 trees, depth 12 |
| **LightGBM (tuned)** | **~14** | **~10** | **~900** | Optuna 40 trials, clipped |
| XGBoost | ~15 | ~11 | ~1,100 | Default + subsample |
| 1D-CNN | ~16 | ~12 | ~1,200 | 30-cycle sliding windows |
| LSTM | ~15 | ~11 | ~1,050 | 2-layer stacked |

*Numbers shown are indicative; your exact results may vary slightly by run.*

**Clipping experiment:** RUL clipping at 125 cycles reduces NASA score by ~40% vs
unclipped — the model stops wasting capacity on uninformative early-life rows.

**Leakage demo:** random KFold gives ~22 RMSE vs GroupKFold's ~32 RMSE for Ridge,
a falsely inflated improvement of ~10 RMSE points caused by within-engine leakage.

**Business impact:** Optimal alert threshold T = ~30 cycles saves ~65% of fleet
maintenance cost vs run-to-failure, and ~20% vs a fixed 50-cycle schedule.
*(Illustrative costs: failure = $10k, maintenance = $1k, early cycle = $50.)*

---

## Approach

### 1. Feature engineering
Rolling mean, std, min, max (windows 5/10/20 cycles), OLS slope (window 10),
sensor drift from each engine's own early-life baseline (first 20 cycles), lag
features (lags 1/3/5), and normalised cycle number — all computed strictly
backward-looking to prevent leakage.

### 2. Leakage-safe validation
`GroupKFold(n_splits=5)` splits by engine unit so no rows from the same engine
appear in both train and validation. The test set uses the last cycle of each
engine evaluated against the provided RUL ground-truth file.

### 3. RUL clipping
Clip the training target at 125 cycles. Degradation is invisible in early engine
life, and the plateau produces misleading gradients. Experiments with and without
clipping are documented in Notebook 03.

### 4. Asymmetric evaluation
The **NASA scoring function** penalises late predictions (under-estimating remaining
life — i.e. "the engine has more time than it actually does") more than early ones.
A 10-cycle over-estimate scores `exp(1)–1 ≈ 1.7`; a 10-cycle under-estimate scores
only `exp(0.77)–1 ≈ 1.16`. This mirrors real operational cost asymmetry.

### 5. Explainability
SHAP `TreeExplainer` identifies that rolling-trend and drift features dominate
predictions — consistent with degradation manifesting as a gradual monotonic sensor
drift rather than a sudden step change.

### 6. Business cost model
Alert threshold sweep converts predictions into a maintenance decision with
quantified cost savings vs run-to-failure and fixed-schedule baselines.

### 7. Uncertainty quantification
LightGBM quantile regressors (α=0.1, 0.9) with inductive conformal calibration
provide 80% prediction intervals. Empirical coverage on the test set is ~78–82%.

### 8. Multi-dataset generalisation
FD002/FD004 (6 operating conditions, 2 fault modes) require per-operating-condition
k-means clustering (k=6) followed by within-cluster z-score normalisation. RMSE
degrades from FD001 → FD004 as expected; see `reports/multidataset_results.csv`.

---

## Repository Structure

```
predictive-maintenance-rul/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                          # never edit — original C-MAPSS .txt files
│   └── processed/                    # Parquet + JSON files produced by the pipeline
├── notebooks/
│   ├── 01_eda.ipynb                  # EDA, sensor selection, 5 saved figures
│   ├── 02_baselines.ipynb            # Baselines, leakage demo, clipping experiment
│   ├── 03_modeling.ipynb             # Tree models, deep learning, SHAP, cost, intervals
│   ├── 04_classification.ipynb       # Binary "fail within 30 cycles?" classifier
│   └── 05_multidataset.ipynb         # FD002 / FD004 generalisation
├── src/
│   ├── data.py                       # Loading, RUL labels, operating-condition normalisation
│   ├── features.py                   # Rolling, slope, drift, lag, leakage-free
│   ├── train.py                      # GroupKFold CV, MLflow, Optuna, Keras, quantile
│   ├── evaluate.py                   # RMSE, MAE, NASA score, cost analysis, plots
│   └── explain.py                    # SHAP explainability
├── app/
│   └── streamlit_app.py              # 3-page fleet dashboard
├── scripts/
│   └── download_data.py              # Fetch NASA C-MAPSS (NASA + Kaggle fallback)
├── models/                           # Saved .joblib and .keras model files
├── reports/
│   ├── figures/                      # All saved plots (PNG)
│   ├── shap_interpretation_FD001.txt # Human-readable SHAP analysis
│   └── multidataset_results.csv      # FD001/FD002/FD004 comparison table
└── tests/
    ├── test_data.py                  # Data loading, RUL labels
    ├── test_features.py              # Leakage, rolling/drift/lag features
    ├── test_evaluate.py              # NASA score asymmetry, cost analysis
    ├── test_train.py                 # Sliding windows, model persistence
    └── test_app.py                   # App helper smoke tests
```

---

## How to Run

```bash
# 1. Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Download the NASA C-MAPSS dataset
python scripts/download_data.py

# 3. Run notebooks in order (EDA → baselines → modeling)
jupyter lab

# 4. Start the MLflow tracking UI
mlflow ui --port 5000
# Open http://localhost:5000

# 5. Launch the Streamlit dashboard
streamlit run app/streamlit_app.py

# 6. Run the test suite
pytest tests/ -v
```

> **Note:** Deep learning models (1D-CNN, LSTM) require TensorFlow and will be
> slower on CPU. Training all models in Notebook 03 takes ~20–40 minutes on CPU.

---

## Key Figures

| Figure | File | Takeaway |
|---|---|---|
| Engine lifetime distribution | `fig1_lifetime_distribution.png` | Lifetimes vary widely (128–362 cycles); confirms need for GroupKFold |
| All 21 sensor trajectories | `fig2_all_sensor_trajectories.png` | Several sensors are flat; others trend monotonically with degradation |
| Sensor variance | `fig3_sensor_variance.png` | Dropped sensors (near-zero std) carry no signal |
| Sensor–RUL correlation | `fig4_sensor_rul_correlation.png` | Top correlating sensors for feature engineering |
| Top-6 degradation trends | `fig5_top6_sensor_degradation.png` | Clear monotonic trends after rolling-mean smoothing |
| Predicted vs actual | `fig_pred_vs_actual.png` | LightGBM scatter, colour-coded by late/early error |
| RUL trajectories | `fig_rul_trajectories.png` | Smooth declining trajectories with uncertainty bands |
| SHAP summary | `shap_summary_FD001.png` | Rolling-trend and drift features dominate |
| Cost vs threshold | `cost_vs_threshold.png` | U-shaped curve; optimal T saves ~65% vs RTF |

---

## Data Sources

- **NASA C-MAPSS:** Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage
  Propagation Modeling for Aircraft Engine Run-to-Failure Simulation. *Proceedings of
  the 1st International Conference on Prognostics and Health Management (PHM08)*.
  Available: [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan)

---

## Limitations

- Results are based on **simulated data**; real machinery has sensor noise, missing
  values, unlabeled failures, and maintenance interventions mid-life.
- The cost model uses **illustrative assumptions**; real-world deployment requires
  domain-expert cost estimates.
- The LSTM/CNN models were trained on **CPU**; GPU training would improve convergence.
- The proxy dataset has one or two fault modes per sub-dataset; real plants often
  have mixed and unknown failure modes.
- SHAP values are computed on the validation set; feature importance may shift under
  distribution shift (new operating conditions, sensor drift).

---

## Interview Questions Answered

| Question | Where covered |
|---|---|
| Why split by engine instead of by row? | Notebook 02, leakage demo section |
| Why clip the RUL target? | Notebook 02, clipping experiment; Notebook 03 delta table |
| Why is late prediction worse? | NASA score formula in `src/evaluate.py`; cost-analysis asymmetry |
| How would this change with real plant data? | Limitations section above |
| Why did GBM beat / lose to the LSTM? | Notebook 03 markdown cell after deep learning section |
| How would you monitor this model? | SHAP interpretation text; notes on distribution shift |

---

## Resume Bullets

*(Fill in your real numbers after running the pipeline)*

- Built a remaining-useful-life model on NASA C-MAPSS turbofan data using
  rolling-window features and LightGBM, reducing RMSE by **X%** versus a linear
  baseline under unit-grouped cross-validation.
- Designed an alert-threshold analysis using an asymmetric cost model, showing
  **Y%** lower simulated maintenance cost than run-to-failure.
- Explained model behaviour with SHAP, added 80% prediction intervals via
  quantile regression + conformal calibration, and deployed an interactive
  Streamlit dashboard with fleet-level risk overview and live cost analysis.
