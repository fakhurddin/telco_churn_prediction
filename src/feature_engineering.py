"""
feature_engineering.py
=======================
Phase 4: Transform the cleaned dataset into model-ready format.

What this file does (in order):
  1. Loads  data/telco_churn_clean.csv
  2. Splits features (X) from target (y)
  3. Separates numeric and categorical columns
  4. Applies StandardScaler  to numeric columns
  5. Applies OneHotEncoder   to categorical columns
  6. Combines into a single feature matrix
  7. Saves the feature column order  -> models/feature_cols.pkl
  8. Saves the fitted scaler         -> models/scaler.pkl
  9. Saves the fitted encoder        -> models/encoder.pkl
 10. Performs stratified 80/20 train/test split
 11. Applies SMOTE on the TRAINING set only (prevents data leakage)
 12. Saves X_train, X_test, y_train, y_test -> data/processed/
 13. Prints a full report so you can verify every transformation

Run:
    python src/feature_engineering.py
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import StandardScaler, OneHotEncoder
from sklearn.compose          import ColumnTransformer
from sklearn.pipeline         import Pipeline
from imblearn.over_sampling   import SMOTE

# ── paths ────────────────────────────────────────────────────────────────────
CLEAN_PATH   = os.path.join("data", "telco_churn_clean.csv")
PROCESSED    = os.path.join("data", "processed")
MODELS_DIR   = "models"

X_TRAIN_PATH    = os.path.join(PROCESSED, "X_train.pkl")
X_TEST_PATH     = os.path.join(PROCESSED, "X_test.pkl")
Y_TRAIN_PATH    = os.path.join(PROCESSED, "y_train.pkl")
Y_TEST_PATH     = os.path.join(PROCESSED, "y_test.pkl")
SCALER_PATH     = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_PATH    = os.path.join(MODELS_DIR, "encoder.pkl")
FEATURE_PATH    = os.path.join(MODELS_DIR, "feature_cols.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")

RANDOM_STATE = 42
TEST_SIZE    = 0.20


# ── column definitions ───────────────────────────────────────────────────────
NUMERIC_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
]

CATEGORICAL_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

TARGET_COL = "Churn"


# ── step 1: load ─────────────────────────────────────────────────────────────
def load_clean(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"\n  ERROR: '{path}' not found.")
        print("  Run  python src/data_cleaning.py  first.\n")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"[1] Loaded clean data: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


# ── step 2: split X / y ──────────────────────────────────────────────────────
def split_xy(df: pd.DataFrame):
    # Keep only the columns we defined above (guard against extra cols)
    all_cols = NUMERIC_COLS + CATEGORICAL_COLS
    missing  = [c for c in all_cols if c not in df.columns]
    if missing:
        print(f"  WARNING: These columns not found and will be skipped: {missing}")

    available_num = [c for c in NUMERIC_COLS     if c in df.columns]
    available_cat = [c for c in CATEGORICAL_COLS if c in df.columns]

    X = df[available_num + available_cat].copy()
    y = df[TARGET_COL].copy()

    print(f"\n[2] Features / target split:")
    print(f"    X shape        : {X.shape}")
    print(f"    y shape        : {y.shape}")
    print(f"    Numeric cols   : {available_num}")
    print(f"    Categorical cols ({len(available_cat)}): {available_cat}")
    return X, y, available_num, available_cat


# ── step 3: build preprocessor ───────────────────────────────────────────────
def build_preprocessor(numeric_cols, categorical_cols):
    """
    ColumnTransformer that:
      - StandardScaler  on numeric columns
      - OneHotEncoder   on categorical columns (drop='first' avoids dummy trap)
    """
    numeric_transformer     = Pipeline([("scaler",  StandardScaler())])
    categorical_transformer = Pipeline([("encoder", OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore",
    ))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,     numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


# ── step 4: train/test split ──────────────────────────────────────────────────
def split_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y,       # preserve churn ratio in both splits
    )
    print(f"\n[3] Train / test split (stratified 80/20):")
    print(f"    Train : {X_train.shape[0]:,} rows  "
          f"| churn={y_train.sum():,} ({y_train.mean()*100:.1f}%)")
    print(f"    Test  : {X_test.shape[0]:,} rows  "
          f"| churn={y_test.sum():,}  ({y_test.mean()*100:.1f}%)")
    return X_train, X_test, y_train, y_test


# ── step 5: fit + transform ───────────────────────────────────────────────────
def fit_transform(preprocessor, X_train, X_test, numeric_cols, categorical_cols):
    # Fit ONLY on training data
    preprocessor.fit(X_train)

    X_train_proc = preprocessor.transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    # Get feature names after encoding
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        # Fallback for older sklearn
        num_names = numeric_cols
        cat_enc   = preprocessor.named_transformers_["cat"]["encoder"]
        cat_names = cat_enc.get_feature_names_out(categorical_cols).tolist()
        feature_names = num_names + cat_names

    feature_names = list(feature_names)

    print(f"\n[4] Preprocessing (fit on train only):")
    print(f"    Original feature count   : {X_train.shape[1]}")
    print(f"    After encoding           : {len(feature_names)} features")
    print(f"    X_train_proc shape       : {X_train_proc.shape}")
    print(f"    X_test_proc  shape       : {X_test_proc.shape}")
    print(f"\n    Feature names ({len(feature_names)} total):")
    for i, name in enumerate(feature_names):
        print(f"      [{i:02d}] {name}")

    return X_train_proc, X_test_proc, feature_names


# ── step 6: SMOTE ─────────────────────────────────────────────────────────────
def apply_smote(X_train_proc, y_train):
    """
    SMOTE: Synthetic Minority Oversampling Technique.
    Creates synthetic churn=1 samples so the training set is balanced.
    Applied ONLY to training data — never touch the test set.
    """
    before_total   = len(y_train)
    before_churn   = y_train.sum()
    before_noChurn = before_total - before_churn

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, before_churn - 1))
    X_resampled, y_resampled = smote.fit_resample(X_train_proc, y_train)

    after_total  = len(y_resampled)
    after_churn  = y_resampled.sum()
    after_retain = after_total - after_churn

    print(f"\n[5] SMOTE applied to training set:")
    print(f"    Before — total: {before_total:,}  "
          f"| churn: {before_churn:,} ({before_churn/before_total*100:.1f}%)  "
          f"| retained: {before_noChurn:,} ({before_noChurn/before_total*100:.1f}%)")
    print(f"    After  — total: {after_total:,}  "
          f"| churn: {after_churn:,} ({after_churn/after_total*100:.1f}%)  "
          f"| retained: {after_retain:,} ({after_retain/after_total*100:.1f}%)")
    print(f"    Synthetic samples created: {after_total - before_total:,}")

    return X_resampled, y_resampled


# ── step 7: save artifacts ────────────────────────────────────────────────────
def save_artifacts(preprocessor, feature_names,
                   X_train, X_test, y_train, y_test):
    os.makedirs(PROCESSED,  exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Extract individual scaler and encoder from the ColumnTransformer
    scaler  = preprocessor.named_transformers_["num"]["scaler"]
    encoder = preprocessor.named_transformers_["cat"]["encoder"]

    # Save individual components (used by predict.py and app.py)
    joblib.dump(scaler,        SCALER_PATH)
    joblib.dump(encoder,       ENCODER_PATH)
    joblib.dump(feature_names, FEATURE_PATH)
    joblib.dump(preprocessor,  PREPROCESSOR_PATH)

    # Save processed splits
    joblib.dump(X_train, X_TRAIN_PATH)
    joblib.dump(X_test,  X_TEST_PATH)
    joblib.dump(y_train, Y_TRAIN_PATH)
    joblib.dump(y_test,  Y_TEST_PATH)

    print(f"\n[6] Saved artifacts:")
    print(f"    models/scaler.pkl          -> StandardScaler")
    print(f"    models/encoder.pkl         -> OneHotEncoder")
    print(f"    models/feature_cols.pkl    -> {len(feature_names)} feature names")
    print(f"    models/preprocessor.pkl    -> full ColumnTransformer pipeline")
    print(f"    data/processed/X_train.pkl -> shape {X_train.shape}")
    print(f"    data/processed/X_test.pkl  -> shape {X_test.shape}")
    print(f"    data/processed/y_train.pkl -> shape {y_train.shape}")
    print(f"    data/processed/y_test.pkl  -> shape {y_test.shape}")


# ── step 8: validation ────────────────────────────────────────────────────────
def validate_artifacts(feature_names):
    """Reload every artifact and confirm shapes + contents match."""
    print(f"\n[7] Validation — reloading saved artifacts:")

    scaler_loaded  = joblib.load(SCALER_PATH)
    encoder_loaded = joblib.load(ENCODER_PATH)
    feat_loaded    = joblib.load(FEATURE_PATH)
    X_tr           = joblib.load(X_TRAIN_PATH)
    X_te           = joblib.load(X_TEST_PATH)
    y_tr           = joblib.load(Y_TRAIN_PATH)
    y_te           = joblib.load(Y_TEST_PATH)

    print(f"    scaler.pkl     : {type(scaler_loaded).__name__}  ✓")
    print(f"    encoder.pkl    : {type(encoder_loaded).__name__}  ✓")
    print(f"    feature_cols   : {len(feat_loaded)} features match={feat_loaded == feature_names}  ✓")
    print(f"    X_train shape  : {X_tr.shape}  ✓")
    print(f"    X_test  shape  : {X_te.shape}  ✓")
    print(f"    y_train dist   : churn={int(y_tr.sum())} / retained={int((y_tr==0).sum())}")
    print(f"    y_test  dist   : churn={int(y_te.sum())} / retained={int((y_te==0).sum())}")

    assert X_tr.shape[1] == len(feature_names), "Feature count mismatch!"
    print(f"\n    All artifacts validated successfully.")


# ── main ──────────────────────────────────────────────────────────────────────
def run_pipeline():
    print("=" * 60)
    print("  PHASE 4  |  Feature Engineering")
    print("=" * 60)

    df = load_clean(CLEAN_PATH)

    X, y, num_cols, cat_cols = split_xy(df)

    X_train_raw, X_test_raw, y_train, y_test = split_train_test(X, y)

    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_train_proc, X_test_proc, feature_names = fit_transform(
        preprocessor, X_train_raw, X_test_raw, num_cols, cat_cols
    )

    X_train_smote, y_train_smote = apply_smote(X_train_proc, y_train)

    save_artifacts(
        preprocessor, feature_names,
        X_train_smote, X_test_proc,
        y_train_smote, y_test,
    )

    validate_artifacts(feature_names)

    print("\n" + "=" * 60)
    print("  Phase 4 complete.  Run next:  python src/train_model.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_pipeline()
