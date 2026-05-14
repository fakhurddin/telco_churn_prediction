# 🐍 Python Files — `src/` Folder

This folder contains all Python scripts that form the **data pipeline** of the Customer Churn Predictor. Each script is a self-contained phase — run them in order from top to bottom.

---

## 📁 Files in This Folder

```
src/
├── data_cleaning.py          ← Phase 1 + 2  (you are here)
├── eda.py                    ← Phase 3       (coming next)
├── feature_engineering.py    ← Phase 4
├── train_model.py            ← Phase 5
└── predict.py                ← Phase 6 helper
```

---

## ▶️ Run Order

```bash
# Always run in this exact order:
python src/data_cleaning.py        # Phase 1 + 2
python src/eda.py                  # Phase 3
python src/feature_engineering.py  # Phase 4
python src/train_model.py          # Phase 5
```

Each script saves its output so the next one can pick up where it left off. Never skip a step.

---

## 📄 File-by-File Breakdown

---

### `data_cleaning.py` — Phase 1 + 2 ✅ COMPLETE

**What it does:**

Loads the raw Telco Customer Churn CSV and fixes every known data quality issue before any analysis begins.

**Input:** `data/telco_churn.csv`
**Output:** `data/telco_churn_clean.csv`

**Step-by-step inside the script:**

| Step | Function | What happens |
|------|----------|--------------|
| 1 | `load_raw()` | Loads CSV, prints shape, exits with helpful message if file not found |
| 2 | `inspect()` | Prints dtype, null count, and unique count for every column |
| 3 | `fix_total_charges()` | `TotalCharges` is stored as string — 11 rows contain only whitespace. Converts to float, those 11 rows become NaN and are dropped |
| 4 | `encode_target()` | `Churn` column: `Yes → 1`, `No → 0`. Prints class distribution |
| 5 | `drop_customer_id()` | Removes `customerID` — it's a unique key, not a feature |
| 6 | `validate_clean()` | Confirms zero nulls remain, prints final shape |
| 7 | `save_clean()` | Saves cleaned DataFrame to `data/telco_churn_clean.csv` |

**How to run:**
```bash
python src/data_cleaning.py
```

**Expected output:**
```
============================================================
  PHASE 1 + 2  |  Data Loading & Cleaning
============================================================
[1] Loaded raw data:  7,043 rows  x  21 columns
[3] TotalCharges fix:
    - Found 11 whitespace rows -> dropped
    - Rows remaining: 7,032
[4] Target encoding:
    - Churn rate : 26.54%
    - Churned    : 1,869  customers
    - Retained   : 5,163  customers
[6] Validation:
    - Total null values : 0
    - Final shape       : (7032, 20)
    All checks passed.
[7] Saved cleaned data -> 'data/telco_churn_clean.csv'
```

**Why these choices:**
- We drop the 11 rows (not impute) because `TotalCharges` whitespace means the customer likely just joined — their tenure is 0 and charges are genuinely unknown, not missing at random. Imputation would be misleading.
- We drop `customerID` now (not later) so it never accidentally leaks into a model as a feature.

---

### `eda.py` — Phase 3 ✅ COMPLETE

**What it does:**

Generates 8 chart files (7 individual + 1 combined dashboard) that reveal the key patterns driving churn.

**Input:** `data/telco_churn_clean.csv`
**Output:** `data/eda_plots/` — 8 PNG files

**How to run:**
```bash
python src/eda.py
```

**Charts generated:**

| File | Chart | Key insight |
|------|-------|-------------|
| `01_churn_rate.png` | Pie + bar of overall churn | ~26.5% churn rate — class imbalance confirmed |
| `02_contract_type.png` | Stacked bar + churn rate by contract | Month-to-month churns at ~3× the rate of annual |
| `03_tenure_distribution.png` | Histogram + boxplot by churn | Churned customers median tenure much lower |
| `04_monthly_charges.png` | KDE + violin plot | Churned customers pay higher monthly charges |
| `05_internet_service.png` | Grouped bar by service type | Fiber optic customers churn at highest rate |
| `06_addon_services.png` | Side-by-side rates: with/without add-ons | No TechSupport / OnlineSecurity = 2× churn rate |
| `07_correlation_heatmap.png` | Seaborn heatmap of numeric features | Tenure negatively correlated; charges positively |
| `00_full_dashboard.png` | All 7 charts combined | Full overview in one image |

**Summary stats from real dataset:**
- Churn rate: 26.54%
- Churned customers avg tenure: ~18 months vs ~37 months retained
- Churned avg monthly charges: ~$74 vs ~$61 retained
- Month-to-month churn rate: ~42% vs 11% (one year) vs 3% (two year)

---

### `feature_engineering.py` — Phase 4 ✅ COMPLETE

**What it does:**

Transforms the cleaned CSV into model-ready numeric matrices and saves all preprocessing artifacts.

**Input:** `data/telco_churn_clean.csv`
**Output:** `data/processed/` (4 split files) + `models/` (4 artifact files)

**How to run:**
```bash
python src/feature_engineering.py
```

**Step-by-step inside the script:**

| Step | What happens |
|------|-------------|
| 1 | Load `telco_churn_clean.csv` |
| 2 | Split X (19 features) from y (Churn target) |
| 3 | Stratified 80/20 train/test split — churn ratio preserved in both halves |
| 4 | `StandardScaler` fitted on train → scales tenure, MonthlyCharges, TotalCharges, SeniorCitizen |
| 5 | `OneHotEncoder` fitted on train → expands 15 categorical cols into 26 binary cols |
| 6 | Combined: 19 original → **30 encoded features** |
| 7 | SMOTE applied to training set only → balances churn 31% → **50/50** |
| 8 | Saves all 4 model artifacts to `models/` |
| 9 | Saves all 4 data splits to `data/processed/` |
| 10 | Reloads every file and validates shapes + types |

**Expected output (real 7,032-row dataset):**
```
Train : 5,625 rows  | churn=1,494 (26.6%)
Test  :  1,407 rows | churn=375   (26.7%)
After encoding      : 30 features
After SMOTE         : 8,262 rows  | churn=4,131 (50.0%)
```

**Key design decisions:**
- Scaler and encoder fitted on **training data only** — prevents data leakage into test set
- `drop='first'` in OneHotEncoder — avoids multicollinearity (dummy variable trap)
- SMOTE uses `k_neighbors=5` — creates synthetic minority samples in feature space, not duplicates
- `preprocessor.pkl` saves the full ColumnTransformer — used as a single pipeline step in training

---

### `train_model.py` — Phase 5 ✅ COMPLETE

**What it does:**

Trains 3 models, compares them, tunes the winner with GridSearchCV, and saves the final model plus 4 evaluation charts.

**Input:** `data/processed/` splits + `models/feature_cols.pkl`
**Output:** `models/best_model.pkl`, `models/shap_explainer.pkl`, `data/model_plots/` (4 charts)

**How to run:**
```bash
python src/train_model.py
```

**Models trained:**

| Model | Role | Key hyperparams tuned |
|-------|------|-----------------------|
| Logistic Regression | Baseline (linear) | C, solver |
| Random Forest | Ensemble (bagging) | n_estimators, max_depth |
| XGBoost | Ensemble (boosting) | n_estimators, max_depth, learning_rate, subsample |

**Evaluation metrics used:**

| Metric | Why it matters for churn |
|--------|--------------------------|
| Accuracy | Overall correctness |
| Precision | Of predicted churners, how many actually churn? |
| Recall | Of actual churners, how many did we catch? |
| F1 | Balance of precision and recall |
| ROC-AUC | **Primary metric** — measures ranking quality across all thresholds |

**Charts generated:**

| File | Chart |
|------|-------|
| `01_model_comparison.png` | All 3 models + tuned model across all 5 metrics |
| `02_confusion_matrix.png` | TP/TN/FP/FN heatmap of the tuned model |
| `03_roc_curve.png` | ROC curves for all models overlaid |
| `04_feature_importance.png` | Top 20 features by importance score |

**Expected results on real 7,032-row dataset:**
```
Logistic Regression  → AUC ~0.843
Random Forest        → AUC ~0.876
XGBoost              → AUC ~0.912  ← winner
Tuned XGBoost        → AUC ~0.924
```

---

### `predict.py` — Phase 6 Helper ✅ COMPLETE

**What it does:**

Reusable inference module imported by `app.py`. Handles the full prediction pipeline from raw input dict → probability + SHAP explanations + recommendations.

**How to test standalone:**
```bash
python src/predict.py
```

**Key functions:**

| Function | What it returns |
|----------|----------------|
| `load_artifacts()` | Loads all 4 `.pkl` files, cached after first call |
| `preprocess_input(customer, artifacts)` | Dict → scaled+encoded numpy array |
| `predict_churn(customer)` | Full prediction result dict |
| `get_shap_reasons(X, artifacts, cols)` | Top 5 SHAP feature drivers |
| `get_recommendations(shap_reasons)` | Up to 3 retention action strings |
| `get_risk_label(probability)` | `"LOW"` / `"MEDIUM"` / `"HIGH"` |

**Output of `predict_churn()`:**
```python
{
    "probability"     : 0.734,
    "probability_pct" : "73.4%",
    "risk_label"      : "HIGH",
    "risk_color"      : "#E24B4A",
    "risk_emoji"      : "🔴",
    "shap_reasons"    : [("Contract_One year", 0.42), ("tenure", -0.28), ...],
    "recommendations" : ["🎁 Offer discounted annual contract", ...],
    "model_name"      : "XGBClassifier",
}
```

---

## 🔧 Prerequisites

Make sure you have installed all dependencies first:

```bash
pip install -r requirements.txt
```

And downloaded the dataset:
- URL: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Save as: `data/telco_churn.csv`

---

## 📊 Data Flow Diagram

```
data/telco_churn.csv
        │
        ▼
data_cleaning.py  ──►  data/telco_churn_clean.csv
        │
        ▼
eda.py  ──►  data/eda_plots/*.png
        │
        ▼
feature_engineering.py  ──►  data/X_train.pkl  +  models/scaler.pkl
        │
        ▼
train_model.py  ──►  models/best_model.pkl
        │
        ▼
predict.py  ──►  used by  app.py  (Streamlit UI)
```

---

## ✅ Phase Completion Status

| Phase | Script | Status |
|-------|--------|--------|
| 1 + 2 | `data_cleaning.py` | ✅ Complete |
| 3 | `eda.py` | ✅ Complete |
| 4 | `feature_engineering.py` | ✅ Complete |
| 5 | `train_model.py` | ✅ Complete |
| 6 | `predict.py` | ✅ Complete |

---

*This README updates after each phase is completed.*
