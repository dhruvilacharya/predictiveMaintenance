# Project 1: Predictive Maintenance and Remaining Useful Life (RUL)

**Skill focus:** tabular and time-series ML, feature engineering, leakage-free evaluation, explainability, business reasoning
**Estimated time:** about 14 working days at 2-3 hours/day
**Difficulty:** Beginner to intermediate

---

## 1. Business framing

> *"A tire plant runs hundreds of machines (mixers, extruders, curing presses). An unplanned breakdown stops a line, wastes material and delays orders. Can we predict how many cycles a machine has left before failure, so maintenance can be scheduled just in time instead of too early (waste) or too late (breakdown)?"*

Write this framing at the top of your README. Interviewers read the first 10 lines of a repo.

**Be transparent about the proxy.** You will use NASA turbofan engine data, not tire-plant data. Say so explicitly: the problem structure (many units, multi-sensor degradation, run-to-failure) is identical to industrial equipment, which is why it is a standard benchmark.

## 2. What you will learn and demonstrate

| Skill | Where it shows up |
|---|---|
| Time-series feature engineering | Rolling windows, trends, deltas |
| Correct validation | Splitting by machine unit, not random rows |
| Model comparison | Baseline vs. tree models vs. optional deep learning |
| Asymmetric error costs | NASA scoring function and cost analysis |
| Explainability | SHAP |
| Communication | Business-cost conclusion and a demo app |

## 3. Datasets

**Primary: NASA C-MAPSS (Turbofan Engine Degradation Simulation)**
- Four sub-datasets: FD001 to FD004, increasing in difficulty (more operating conditions and fault modes).
- Each row is one engine at one cycle: `unit`, `cycle`, 3 operational settings, 21 sensors.
- Training engines run to failure. Test engines are cut off before failure and a separate file gives the true RUL.
- **Start with FD001** (one operating condition, one fault mode). Move to FD002 or FD004 only as a stretch goal.

**Secondary (optional): AI4I 2020 Predictive Maintenance** (10,000 rows, binary failure label plus failure modes). Good for a quick classification add-on, but it is synthetic and not time-series, so keep it secondary.

Find both by searching for the dataset names on the NASA Prognostics Data Repository, Kaggle, or the UCI repository. Check the license and cite the source in your README.

## 4. Repository structure

```
predictive-maintenance-rul/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              # never edit these files
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data.py           # loading, RUL labels
│   ├── features.py       # rolling features
│   ├── train.py          # training and CV
│   ├── evaluate.py       # metrics, scoring function
│   └── explain.py        # SHAP
├── app/
│   └── streamlit_app.py
├── models/
├── reports/figures/
└── tests/
    └── test_features.py  # at least a couple of tests
```

## 5. Step-by-step plan

### Phase 0: Setup (Day 1)

- Create a GitHub repo, a virtual environment, and a `requirements.txt` (pandas, numpy, scikit-learn, lightgbm, xgboost, shap, matplotlib, seaborn, streamlit, mlflow, optionally torch).
- Commit early and often with meaningful messages. A commit history that shows steady progress looks very different from one giant upload.
- Write a draft README with the business problem. You will refine it at the end.

### Phase 1: Understand the data (Day 2)

Load FD001:

```python
import pandas as pd

cols = ["unit", "cycle"] + [f"op{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
train = pd.read_csv("data/raw/train_FD001.txt", sep=r"\s+", header=None, names=cols)
test = pd.read_csv("data/raw/test_FD001.txt", sep=r"\s+", header=None, names=cols)
rul_test = pd.read_csv("data/raw/RUL_FD001.txt", header=None, names=["rul"])
```

Create the training label. For each unit, RUL = (max cycle of that unit) - (current cycle):

```python
max_cycle = train.groupby("unit")["cycle"].transform("max")
train["rul"] = max_cycle - train["cycle"]
```

**Checkpoint questions (write answers in the notebook):**
- How many engines? What is the shortest and longest life?
- Is every engine's life the same length? (No. This matters for how you split.)
- Why does the test set contain truncated engines?

### Phase 2: Exploratory data analysis (Day 3)

- Plot every sensor against cycle for 5-10 engines. Some sensors trend with degradation; others are flat.
- Identify **constant or near-constant sensors** (in FD001 several carry no information) and drop them, with a documented reason.
- Plot the distribution of engine lifetimes.
- Check correlations between sensors and RUL. Remember that correlation is only a hint here since relationships are non-linear.

**Deliverable:** 4-5 clear figures saved in `reports/figures/`, each with a one-line takeaway.

### Phase 3: Baselines and the right validation (Day 4)

Never build a fancy model before a baseline. Baselines to implement:

1. **Constant baseline:** predict the mean training RUL for everything.
2. **Linear regression** on the raw sensors.

**Validation rule (the most common beginner mistake):** split by **engine unit**, not by random rows. Rows from the same engine are highly correlated, so a random split leaks information and gives falsely great scores.

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for tr_idx, val_idx in gkf.split(X, y, groups=train["unit"]):
    ...
```

**RUL clipping:** early in an engine's life, degradation is not visible, so predicting exact RUL (say 300 vs. 280) is meaningless. A common practice is to clip the target at about 125 cycles. Try with and without clipping and report the difference. This kind of experiment impresses interviewers.

### Phase 4: Feature engineering (Days 5-6)

Raw sensor values at one instant are noisy. Engineer features **per unit, using only past data**:

```python
def add_rolling_features(df, sensors, windows=(5, 10, 20)):
    df = df.sort_values(["unit", "cycle"]).copy()
    g = df.groupby("unit")
    for s in sensors:
        for w in windows:
            df[f"{s}_mean_{w}"] = g[s].transform(lambda x: x.rolling(w, min_periods=1).mean())
            df[f"{s}_std_{w}"] = g[s].transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
        df[f"{s}_diff_1"] = g[s].diff().fillna(0)
    return df
```

Ideas to try:
- Rolling mean, std, min, max
- Slope of each sensor over the last N cycles
- Difference from the unit's own baseline (first 10-20 cycles), which captures *relative* drift
- Cycle number itself (engine age)

**Leakage check:** write a small test confirming that no feature at cycle *t* uses data from cycle *t+1* or later. Rolling windows with `.rolling()` look backward by default, but verify rather than assume.

### Phase 5: Modeling (Days 7-8)

Compare, in order:

| Model | Why |
|---|---|
| Linear / Ridge | Simple reference |
| Random Forest | Non-linear baseline |
| LightGBM / XGBoost | Usually the best tabular performer |
| (Optional) 1D-CNN or LSTM on sliding windows | Shows deep learning exposure |

For the deep-learning option, build sliding windows of the last 30 cycles per engine as input sequences. Do not skip the tree models; interviewers care that you know a gradient-boosted model is often the right answer and a neural net is not automatically better.

Track every run (parameters, metrics) in **MLflow** or even a simple CSV log. Tune lightly (Optuna with 30-50 trials is plenty). Do not spend days on tuning.

### Phase 6: Evaluation and error analysis (Day 9)

Metrics:
- **RMSE** and **MAE** on RUL
- **NASA scoring function**, which penalizes *late* predictions (predicting more life than actually remains) more than early ones. This mirrors reality: a late prediction means an unexpected breakdown.

```python
import numpy as np

def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    return np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))
```

Evaluate on the official test set using the **last cycle of each test engine** against the provided RUL file.

Error analysis (this is what separates good from average):
- Plot predicted vs. actual RUL. Where is the model worst? Typically early in life (hard) and near the end (critical).
- Plot predicted RUL over time for 3-4 individual engines. Does it track smoothly down?
- Which engines are badly predicted, and do they share anything?

### Phase 7: Explainability (Day 10)

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val)
```

Write 3-4 sentences interpreting the result: *which sensors and which kinds of features (trend vs. level) drive predictions, and does that match physical intuition?*

### Phase 8: Business and cost analysis (Day 11)

Convert the model into a decision. Define an alert rule: "raise a maintenance alert when predicted RUL falls below threshold T cycles."

- Assign illustrative costs (state clearly that these are assumptions): e.g., an unplanned failure costs 10x a planned maintenance; an early replacement wastes some remaining life.
- Sweep T and plot **total cost vs. threshold**.
- Report the best T and the saving relative to "run to failure" and to "replace on a fixed schedule".

This single analysis turns a student project into something that looks like a real data science deliverable.

### Phase 9: Package and demo (Days 12-13)

- Streamlit app: pick an engine from the test set, show its sensor trends, predicted RUL over time, and an alert status.
- Save the trained model and feature code so the app can run from a clean clone.
- Add 3-5 unit tests (feature functions, scoring function, label creation).
- Make sure `pip install -r requirements.txt` followed by `streamlit run app/streamlit_app.py` works on a fresh machine.

### Phase 10: Write-up (Day 14)

README sections: Problem, Data (with proxy disclaimer), Approach, Results table, Key figures, Business impact, Limitations, How to run.

## 6. Common pitfalls

- Random train/test split (leakage), which is the number one thing interviewers look for.
- Using future data in rolling features.
- Reporting only RMSE and ignoring the asymmetric cost of late predictions.
- Tuning for days on a model whose features are weak. Features usually beat tuning.
- Overclaiming ("this would work at Michelin"). Say what would be needed: real sensor data, labeled failures, domain experts.

## 7. Stretch goals (pick one or two)

- Move to FD002/FD004 (multiple operating conditions) and normalize sensors per operating condition.
- Add prediction intervals (quantile regression or conformal prediction) so the alert uses uncertainty.
- Add a classification variant: "fail within 30 cycles?" and compare precision/recall.
- Add drift monitoring: simulate sensor drift and show how performance degrades.

## 8. Definition of done

- [ ] Clean repo, runs from scratch
- [ ] 2+ baselines and 2+ real models compared fairly
- [ ] Unit-grouped validation, documented
- [ ] SHAP plot with interpretation
- [ ] Cost-vs-threshold analysis
- [ ] Streamlit demo
- [ ] README with results table and limitations

## 9. Resume bullets (adapt with your real numbers)

- Built a remaining-useful-life model on NASA C-MAPSS turbofan data using rolling-window features and LightGBM, reducing RMSE by **X%** versus a linear baseline under unit-grouped cross-validation.
- Designed an alert-threshold analysis using an asymmetric cost model, showing **Y%** lower simulated maintenance cost than run-to-failure.
- Explained model behavior with SHAP and deployed an interactive Streamlit dashboard.

## 10. Interview questions to prepare

1. Why did you split by engine instead of by row?
2. Why clip the RUL target? What happens if you do not?
3. Why is predicting too-late worse than too-early here, and how does your metric reflect that?
4. How would this change with real plant data (missing sensors, unlabeled failures, maintenance interventions mid-life)?
5. Why did gradient boosting beat or lose to the LSTM in your experiments?
6. How would you monitor this model after deployment?
