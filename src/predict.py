"""
predict.py
==========
Phase 5 helper — reusable inference module.

NOT run directly. Imported by app.py (Streamlit UI).

What this module provides:
  - load_artifacts()       : loads model + scaler + encoder + feature_cols
  - preprocess_input()     : converts raw customer dict → model-ready array
  - predict_churn()        : returns probability + risk label + SHAP reasons
  - get_recommendations()  : maps SHAP drivers → retention action items

Quick test (standalone):
    python src/predict.py
"""

import numpy as np
import pandas as pd
import joblib
import os

# ── paths ────────────────────────────────────────────────────────────────────
MODELS_DIR   = "models"
MODEL_PATH   = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH  = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "encoder.pkl")
FEATURE_PATH = os.path.join(MODELS_DIR, "feature_cols.pkl")
SHAP_PATH    = os.path.join(MODELS_DIR, "shap_explainer.pkl")

# ── numeric and categorical columns (must match feature_engineering.py) ──────
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

# ── risk thresholds ───────────────────────────────────────────────────────────
RISK_THRESHOLDS = {
    "LOW"    : (0.00, 0.30),
    "MEDIUM" : (0.30, 0.60),
    "HIGH"   : (0.60, 1.01),
}

RISK_COLORS = {
    "LOW"   : "#1D9E75",   # green
    "MEDIUM": "#E8A838",   # amber
    "HIGH"  : "#E24B4A",   # red
}

RISK_EMOJI = {
    "LOW"   : "🟢",
    "MEDIUM": "🟡",
    "HIGH"  : "🔴",
}

# ── retention recommendations ─────────────────────────────────────────────────
RECOMMENDATIONS = {
    "Contract_One year"          : "🎁 Offer a discounted upgrade to an annual contract.",
    "Contract_Two year"          : "🎁 Offer a discounted upgrade to a two-year contract.",
    "tenure"                     : "👋 Trigger the new-customer success program — early engagement reduces churn.",
    "MonthlyCharges"             : "💰 Offer a personalised loyalty discount or price-match guarantee.",
    "TotalCharges"               : "💰 Offer a personalised loyalty discount based on lifetime value.",
    "TechSupport_Yes"            : "🛠️  Offer a free 3-month tech support add-on trial.",
    "TechSupport_No internet service": "🛠️  Offer a bundled tech support and internet package.",
    "OnlineSecurity_Yes"         : "🔒 Offer a discounted online security bundle.",
    "OnlineSecurity_No internet service": "🔒 Offer a bundled security and internet package.",
    "InternetService_Fiber optic": "📶 Fibre optic customers churn more — offer a loyalty bonus or speed upgrade.",
    "InternetService_No"         : "📶 Offer an introductory internet service bundle.",
    "PaymentMethod_Electronic check": "💳 Encourage switching to auto-pay — it reduces friction and churn.",
    "PaperlessBilling_Yes"       : "📧 Paperless users are higher risk — send personalised retention emails.",
    "SeniorCitizen"              : "👴 Offer a senior citizen loyalty plan with dedicated support.",
    "MultipleLines_Yes"          : "📱 Offer a multi-line family plan discount.",
    "StreamingTV_Yes"            : "📺 Bundle streaming services at a discounted rate.",
    "StreamingMovies_Yes"        : "🎬 Bundle streaming services at a discounted rate.",
}

DEFAULT_RECOMMENDATION = "📞 Schedule a proactive outreach call with a retention specialist."


# ── artifact loader ───────────────────────────────────────────────────────────
_cache = {}   # module-level cache so Streamlit doesn't reload on every interaction

def load_artifacts():
    """Load all model artifacts. Cached after first call."""
    global _cache
    if _cache:
        return _cache

    required = {
        "model"       : MODEL_PATH,
        "scaler"      : SCALER_PATH,
        "encoder"     : ENCODER_PATH,
        "feature_cols": FEATURE_PATH,
    }

    for key, path in required.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required artifact not found: '{path}'\n"
                f"Run the full pipeline first:\n"
                f"  python src/data_cleaning.py\n"
                f"  python src/feature_engineering.py\n"
                f"  python src/train_model.py"
            )
        _cache[key] = joblib.load(path)

    # SHAP is optional
    if os.path.exists(SHAP_PATH):
        _cache["shap_explainer"] = joblib.load(SHAP_PATH)
    else:
        _cache["shap_explainer"] = None

    return _cache


# ── preprocessing ─────────────────────────────────────────────────────────────
def preprocess_input(customer: dict, artifacts: dict) -> np.ndarray:
    """
    Convert a raw customer input dict into a model-ready numpy array.

    Parameters
    ----------
    customer : dict
        Keys must match the 19 feature columns exactly.
        Example:
        {
            "tenure"          : 12,
            "MonthlyCharges"  : 79.50,
            "TotalCharges"    : 954.0,
            "SeniorCitizen"   : 0,
            "gender"          : "Male",
            "Partner"         : "Yes",
            "Dependents"      : "No",
            "PhoneService"    : "Yes",
            "MultipleLines"   : "No",
            "InternetService" : "Fiber optic",
            "OnlineSecurity"  : "No",
            "OnlineBackup"    : "No",
            "DeviceProtection": "No",
            "TechSupport"     : "No",
            "StreamingTV"     : "Yes",
            "StreamingMovies" : "Yes",
            "Contract"        : "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod"   : "Electronic check",
        }

    Returns
    -------
    np.ndarray of shape (1, n_features) — scaled and encoded, ready for model
    """
    scaler       = artifacts["scaler"]
    encoder      = artifacts["encoder"]
    feature_cols = artifacts["feature_cols"]

    # Build a single-row DataFrame
    df = pd.DataFrame([customer])

    # Scale numeric columns
    num_scaled = scaler.transform(df[NUMERIC_COLS])

    # Encode categorical columns
    cat_encoded = encoder.transform(df[CATEGORICAL_COLS])

    # Combine
    X = np.hstack([num_scaled, cat_encoded])

    # Safety check: feature count must match
    if X.shape[1] != len(feature_cols):
        raise ValueError(
            f"Feature count mismatch: got {X.shape[1]}, expected {len(feature_cols)}.\n"
            f"Re-run feature_engineering.py and train_model.py to regenerate artifacts."
        )

    return X


# ── SHAP explanations ─────────────────────────────────────────────────────────
def get_shap_reasons(X_processed: np.ndarray, artifacts: dict,
                     feature_cols: list, top_n: int = 5) -> list:
    """
    Returns a list of (feature_name, shap_value) tuples — top churn drivers.

    Parameters
    ----------
    X_processed : np.ndarray of shape (1, n_features)
    artifacts   : dict from load_artifacts()
    feature_cols: list of feature names
    top_n       : how many top drivers to return

    Returns
    -------
    list of (feature_name, shap_value) sorted by |shap_value| descending
    """
    explainer = artifacts.get("shap_explainer")
    if explainer is None:
        return []

    try:
        shap_vals = explainer.shap_values(X_processed)

        # Tree models return list [class_0, class_1] or array of shape (1, features, 2)
        if isinstance(shap_vals, list):
            sv = np.array(shap_vals[1])[0]          # class 1 (churn) for first row
        elif shap_vals.ndim == 3:
            sv = shap_vals[0, :, 1]                  # shape (features, 2) → class 1
        else:
            sv = shap_vals[0]

        pairs = sorted(zip(feature_cols, sv),
                       key=lambda x: abs(x[1]),
                       reverse=True)
        return pairs[:top_n]

    except Exception:
        return []


# ── risk label ────────────────────────────────────────────────────────────────
def get_risk_label(probability: float) -> str:
    for label, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= probability < hi:
            return label
    return "HIGH"


# ── recommendations ───────────────────────────────────────────────────────────
def get_recommendations(shap_reasons: list) -> list:
    """
    Map top SHAP features to actionable retention recommendations.
    Returns a list of recommendation strings (up to 3).
    """
    seen  = set()
    recs  = []
    for feat, val in shap_reasons:
        if val <= 0:
            continue       # only features pushing TOWARD churn
        rec = RECOMMENDATIONS.get(feat, DEFAULT_RECOMMENDATION)
        if rec not in seen:
            seen.add(rec)
            recs.append(rec)
        if len(recs) == 3:
            break

    if not recs:
        recs.append(DEFAULT_RECOMMENDATION)

    return recs


# ── main prediction function ──────────────────────────────────────────────────
def predict_churn(customer: dict) -> dict:
    """
    End-to-end prediction for a single customer.

    Parameters
    ----------
    customer : dict — raw customer input (19 feature values)

    Returns
    -------
    dict with keys:
        probability     : float  (0.0 – 1.0)
        probability_pct : str    e.g. "73.4%"
        risk_label      : str    "LOW" | "MEDIUM" | "HIGH"
        risk_color      : str    hex colour string
        risk_emoji      : str    emoji
        shap_reasons    : list of (feature, value) tuples
        recommendations : list of str
        model_name      : str
    """
    artifacts    = load_artifacts()
    model        = artifacts["model"]
    feature_cols = artifacts["feature_cols"]

    X_proc = preprocess_input(customer, artifacts)

    prob        = float(model.predict_proba(X_proc)[0][1])
    risk_label  = get_risk_label(prob)
    shap_reasons= get_shap_reasons(X_proc, artifacts, feature_cols)
    recs        = get_recommendations(shap_reasons)

    return {
        "probability"     : prob,
        "probability_pct" : f"{prob * 100:.1f}%",
        "risk_label"      : risk_label,
        "risk_color"      : RISK_COLORS[risk_label],
        "risk_emoji"      : RISK_EMOJI[risk_label],
        "shap_reasons"    : shap_reasons,
        "recommendations" : recs,
        "model_name"      : type(model).__name__,
    }


# ── standalone test ───────────────────────────────────────────────────────────
def run_test():
    print("=" * 60)
    print("  predict.py — Standalone Test")
    print("=" * 60)

    # Two test customers: one high-risk, one low-risk
    test_customers = [
        {
            "label"           : "HIGH-RISK customer",
            "tenure"          : 2,
            "MonthlyCharges"  : 95.50,
            "TotalCharges"    : 191.0,
            "SeniorCitizen"   : 0,
            "gender"          : "Male",
            "Partner"         : "No",
            "Dependents"      : "No",
            "PhoneService"    : "Yes",
            "MultipleLines"   : "No",
            "InternetService" : "Fiber optic",
            "OnlineSecurity"  : "No",
            "OnlineBackup"    : "No",
            "DeviceProtection": "No",
            "TechSupport"     : "No",
            "StreamingTV"     : "Yes",
            "StreamingMovies" : "Yes",
            "Contract"        : "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod"   : "Electronic check",
        },
        {
            "label"           : "LOW-RISK customer",
            "tenure"          : 60,
            "MonthlyCharges"  : 45.00,
            "TotalCharges"    : 2700.0,
            "SeniorCitizen"   : 0,
            "gender"          : "Female",
            "Partner"         : "Yes",
            "Dependents"      : "Yes",
            "PhoneService"    : "Yes",
            "MultipleLines"   : "Yes",
            "InternetService" : "DSL",
            "OnlineSecurity"  : "Yes",
            "OnlineBackup"    : "Yes",
            "DeviceProtection": "Yes",
            "TechSupport"     : "Yes",
            "StreamingTV"     : "No",
            "StreamingMovies" : "No",
            "Contract"        : "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod"   : "Bank transfer (automatic)",
        },
    ]

    for tc in test_customers:
        label = tc.pop("label")
        print(f"\n  Test: {label}")
        print(f"  {'─'*50}")
        try:
            result = predict_churn(tc)
            print(f"  Churn probability : {result['probability_pct']}")
            print(f"  Risk              : {result['risk_emoji']} {result['risk_label']}")
            print(f"  Model used        : {result['model_name']}")
            if result["shap_reasons"]:
                print(f"  Top SHAP drivers  :")
                for feat, val in result["shap_reasons"]:
                    direction = "▲ toward churn" if val > 0 else "▼ away from churn"
                    print(f"    {feat:<45} {val:+.4f}  {direction}")
            print(f"  Recommendations   :")
            for rec in result["recommendations"]:
                print(f"    {rec}")
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("  predict.py test complete.")
    print("  Ready for:  streamlit run app.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_test()
