# Predictive Maintenance & Remaining Useful Life (RUL)

> A tire plant runs hundreds of machines — mixers, extruders, curing presses. An unplanned
> breakdown stops a line, wastes material, and delays orders. Can we predict how many cycles
> a machine has left before failure, so maintenance can be scheduled **just in time** instead
> of too early (waste) or too late (breakdown)?

## Proxy Data Disclaimer

This project uses the **NASA C-MAPSS turbofan engine degradation dataset** as a proxy for
industrial equipment. The problem structure — many units, multi-sensor degradation,
run-to-failure — is identical to real plant machinery, which is why C-MAPSS is the standard
benchmark in the predictive-maintenance literature. Results should not be extrapolated to
production without real sensor data, labeled failures, and domain expert validation.

---

## Results Summary

| Model | RMSE | MAE | NASA Score |
|---|---|---|---|
| Constant baseline | — | — | — |
| Ridge regression | — | — | — |
| Random Forest | — | — | — |
| LightGBM (tuned) | — | — | — |
| XGBoost | — | — | — |
| 1D-CNN | — | — | — |
| LSTM | — | — | — |

*Numbers will be filled in after training completes.*

**Business impact:** Optimal alert threshold T=? cycles saves ?% vs. run-to-failure.

---

## Approach

1. **Feature engineering** — rolling windows (mean, std, min, max), sensor slopes, drift from
   unit baseline, all computed backward-only to prevent leakage.
2. **Leakage-safe validation** — `GroupKFold` splits by engine unit so no rows from the same
   engine appear in both train and validation.
3. **RUL clipping** — cap the target at 125 cycles (degradation is invisible in early life);
   experiments run with and without clipping to document the difference.
4. **Asymmetric evaluation** — NASA scoring function penalizes late predictions (under-
   estimating remaining life) more than early ones, mirroring real-world cost asymmetry.
5. **Explainability** — SHAP TreeExplainer identifies which sensors and feature types (trend
   vs. level) drive predictions.
6. **Business cost model** — alert threshold sweep converts predictions into a maintenance
   decision with quantified cost savings vs. run-to-failure and fixed-schedule baselines.
7. **Uncertainty quantification** — quantile regression + conformal prediction add prediction
   intervals so the alert system can account for model uncertainty.
8. **Multi-dataset generalization** — FD001/FD002/FD004 comparison with per-operating-
   condition normalization.

---

## Repository Structure

```
predictive-maintenance-rul/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              # never edit — original C-MAPSS .txt files
│   └── processed/        # Parquet files produced by the pipeline
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_baselines.ipynb        # Baselines + leakage-safe validation
│   ├── 03_modeling.ipynb         # Tree models + deep learning
│   ├── 04_classification.ipynb   # Classification stretch goal
│   └── 05_multidataset.ipynb     # FD002/FD004 comparison
├── src/
│   ├── data.py           # Loading, RUL labels, operating-condition normalisation
│   ├── features.py       # Rolling features, slopes, drift
│   ├── train.py          # Training, GroupKFold CV, MLflow logging, deep learning
│   ├── evaluate.py       # Metrics, NASA score, cost analysis
│   └── explain.py        # SHAP explainability
├── app/
│   └── streamlit_app.py  # Fleet dashboard (3 pages)
├── scripts/
│   └── download_data.py  # Fetch NASA C-MAPSS dataset
├── models/               # Saved models (.joblib / .keras)
├── reports/
│   ├── figures/          # All saved plots
│   ├── shap_interpretation.txt
│   └── multidataset_results.csv
└── tests/
    ├── test_data.py
    ├── test_features.py
    ├── test_evaluate.py
    ├── test_train.py
    └── test_app.py
```

---

## How to Run

```bash
# 1. Install dependencies
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download the NASA C-MAPSS dataset
python scripts/download_data.py

# 3. Run the Streamlit dashboard
streamlit run app/streamlit_app.py
```

To run notebooks in order (EDA → baselines → modeling):
```bash
jupyter lab
```

To start the MLflow tracking UI:
```bash
mlflow ui --port 5000
```

---

## Data Sources

- **NASA C-MAPSS:** Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage
  Propagation Modeling for Aircraft Engine Run-to-Failure Simulation. *Proceedings of the 1st
  International Conference on Prognostics and Health Management (PHM08)*, Denver CO, Oct 2008.
  Available at the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

---

## Limitations

- Results are based on simulated data; real machinery will have sensor noise, missing values,
  unlabeled failures, and maintenance interventions mid-life.
- The cost model uses illustrative assumptions; real-world deployment requires domain-expert
  cost estimates.
- The LSTM/CNN models were trained on CPU for reproducibility; GPU training may improve results.
- The proxy dataset has one fault mode per sub-dataset; real plants often have mixed failure modes.

---

## Resume Bullets

- Built a remaining-useful-life model on NASA C-MAPSS turbofan data using rolling-window
  features and LightGBM, reducing RMSE by **X%** versus a linear baseline under unit-grouped
  cross-validation.
- Designed an alert-threshold analysis using an asymmetric cost model, showing **Y%** lower
  simulated maintenance cost than run-to-failure.
- Explained model behavior with SHAP and deployed an interactive Streamlit dashboard with
  fleet-level risk overview, per-engine RUL trajectories with uncertainty bands, and live
  cost-vs-threshold analysis.
