# 🤖 Model Artifacts — `models/` Folder

This folder stores all **trained model files** saved by the ML pipeline. These `.pkl` files are what the Streamlit app loads at runtime to make predictions — no retraining needed.

---

## 📁 Files in This Folder

```
models/
├── best_model.pkl        ← Trained XGBoost classifier       (Phase 5)
├── scaler.pkl            ← StandardScaler fitted on training data  (Phase 4)
├── encoder.pkl           ← OneHotEncoder fitted on training data   (Phase 4)
└── feature_cols.pkl      ← Ordered list of feature column names    (Phase 4)
```

> ⚠️ This folder starts **empty**. Files are generated when you run the pipeline scripts in `src/`.

---

## 📄 File-by-File Breakdown

---

### `scaler.pkl` — StandardScaler ✅ Generated in Phase 4

**What it is:**
A fitted `sklearn.preprocessing.StandardScaler` object.

**What it does:**
Normalises the three continuous numeric columns so they all live on a similar scale:

| Column | Raw range | After scaling |
|--------|-----------|---------------|
| `tenure` | 0 – 72 months | mean≈0, std≈1 |
| `MonthlyCharges` | $18 – $119 | mean≈0, std≈1 |
| `TotalCharges` | $18 – $8,685 | mean≈0, std≈1 |

**Why we need this:**
XGBoost doesn't strictly need scaling, but Logistic Regression does. We scale before training all models so the comparison is fair. The scaler is fitted **only on training data** to prevent data leakage.

**How to load it:**
```python
import joblib
scaler = joblib.load("models/scaler.pkl")
X_scaled = scaler.transform(X_new)
```

---

### `encoder.pkl` — OneHotEncoder ✅ Generated in Phase 4

**What it is:**
A fitted `sklearn.preprocessing.OneHotEncoder` object.

**What it does:**
Converts categorical string columns into binary numeric columns that models can understand:

| Original column | Example value | Encoded as |
|-----------------|---------------|------------|
| `Contract` | `Month-to-month` | `Contract_Month-to-month = 1`, others = 0 |
| `InternetService` | `Fiber optic` | `InternetService_Fiber optic = 1`, others = 0 |
| `PaymentMethod` | `Electronic check` | `PaymentMethod_Electronic check = 1`, others = 0 |
| ... | ... | ... |

**Categorical columns encoded (15 total):**
`gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`

**Why we need this:**
ML models only accept numbers. The encoder is fitted on training data only, then used to transform both train and test sets consistently.

**How to load it:**
```python
import joblib
encoder = joblib.load("models/encoder.pkl")
X_encoded = encoder.transform(X_categorical)
```

---

### `feature_cols.pkl` — Feature Column Order ✅ Generated in Phase 4

**What it is:**
A Python list of strings — the exact column names and order used during training.

**What it does:**
When the Streamlit app takes user input and assembles a DataFrame, it must pass columns to the model in the **exact same order** they were in during training. This file stores that order.

**Actual contents (30 features after encoding):**
```python
['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen',
 'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
 'MultipleLines_No phone service', 'MultipleLines_Yes',
 'InternetService_Fiber optic', 'InternetService_No',
 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
 'TechSupport_No internet service', 'TechSupport_Yes',
 'StreamingTV_No internet service', 'StreamingTV_Yes',
 'StreamingMovies_No internet service', 'StreamingMovies_Yes',
 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
 'PaymentMethod_Credit card (automatic)',
 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']
```

**How to load it:**
```python
import joblib
feature_cols = joblib.load("models/feature_cols.pkl")
X_input = X_input[feature_cols]   # reorder/align columns
```

---

### `best_model.pkl` — Trained Model ✅ Generated in Phase 5

**What it is:**
A fitted tree-based classifier (XGBoost on the real dataset, Random Forest on small datasets) — the final production model.

**Expected performance on real 7,032-row dataset:**

| Metric | Value |
|--------|-------|
| Accuracy | ~87% |
| F1 Score | ~0.71 |
| ROC-AUC | ~0.92 |
| Precision | ~0.73 |
| Recall | ~0.69 |

**Why XGBoost wins on the real dataset:**
After comparing all 3 models on the test set, XGBoost consistently achieves the best ROC-AUC. It handles mixed feature types, class imbalance, and feature interactions automatically.

**Hyperparameters tuned (GridSearchCV 5-fold):**
```python
param_grid = {
    "n_estimators" : [100, 200, 300],
    "max_depth"    : [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample"    : [0.8, 1.0],
}
```

---

### `shap_explainer.pkl` — SHAP TreeExplainer ✅ Generated in Phase 5

**What it is:**
A fitted `shap.TreeExplainer` object for the best model.

**What it does:**
For any new customer, computes a SHAP value for every one of the 30 features — showing exactly how much each feature pushed the prediction toward or away from churn. The Streamlit app uses this to show the top 5 drivers.

---

## 🔄 How These Files Connect

```
Phase 4 — feature_engineering.py
    │
    ├──► scaler.pkl       (fitted StandardScaler)
    ├──► encoder.pkl      (fitted OneHotEncoder)
    └──► feature_cols.pkl (column order list)
                │
                ▼
Phase 5 — train_model.py
    │
    └──► best_model.pkl   (trained + tuned XGBoost)
                │
                ▼
Phase 7 — app.py  (Streamlit UI)
    Loads ALL 4 files at startup → makes predictions on new customers
```

---

## ⚠️ Important Notes

**Never retrain on the full dataset.**
The scaler and encoder are fitted on the **training split only**. If you refit them on all data and then evaluate, your metrics will be artificially inflated (data leakage).

**Never edit these files manually.**
They are binary pickle files. Any manual edit corrupts them.

**Version these files with your code.**
If you change the feature set or retrain the model, regenerate all 4 files together. A mismatch between `encoder.pkl` and `best_model.pkl` will crash the app.

**Python version matters.**
Pickle files are tied to the Python version they were created with. If you share this project, make sure collaborators use the same Python version listed in `requirements.txt`.

---

## ✅ Generation Status

| File | Generated by | Status |
|------|-------------|--------|
| `scaler.pkl` | `src/feature_engineering.py` | ✅ Phase 4 complete |
| `encoder.pkl` | `src/feature_engineering.py` | ✅ Phase 4 complete |
| `feature_cols.pkl` | `src/feature_engineering.py` | ✅ Phase 4 complete |
| `best_model.pkl` | `src/train_model.py` | ✅ Phase 5 complete |
| `shap_explainer.pkl` | `src/train_model.py` | ✅ Phase 5 complete |
| `model_results.pkl` | `src/train_model.py` | ✅ Phase 5 complete |

---

*This README updates after each phase is completed.*
