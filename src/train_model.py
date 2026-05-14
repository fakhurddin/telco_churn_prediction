"""
train_model.py
==============
Phase 5: Train, compare, tune, and save the best churn prediction model.

What this file does (in order):
  1.  Loads processed train/test splits from  data/processed/
  2.  Trains 3 baseline models:
        - Logistic Regression
        - Random Forest
        - XGBoost
  3.  Evaluates all 3 on the test set:
        Accuracy, Precision, Recall, F1, ROC-AUC
  4.  Prints a side-by-side comparison table
  5.  Selects the best model by ROC-AUC score
  6.  Tunes the best model with GridSearchCV (5-fold CV)
  7.  Re-evaluates the tuned model on the test set
  8.  Generates and saves 4 evaluation charts:
        - Model comparison bar chart
        - Confusion matrix (heatmap)
        - ROC curve
        - Feature importance (top 20)
  9.  Saves the final tuned model  -> models/best_model.pkl
 10.  Saves SHAP explainer         -> models/shap_explainer.pkl
 11.  Prints final performance summary

Run:
    python src/train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.metrics        import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost                 import XGBClassifier

# ── paths ────────────────────────────────────────────────────────────────────
PROCESSED      = os.path.join("data", "processed")
MODELS_DIR     = "models"
PLOT_DIR       = os.path.join("data", "model_plots")

X_TRAIN_PATH   = os.path.join(PROCESSED, "X_train.pkl")
X_TEST_PATH    = os.path.join(PROCESSED, "X_test.pkl")
Y_TRAIN_PATH   = os.path.join(PROCESSED, "y_train.pkl")
Y_TEST_PATH    = os.path.join(PROCESSED, "y_test.pkl")
FEATURE_PATH   = os.path.join(MODELS_DIR, "feature_cols.pkl")
MODEL_PATH     = os.path.join(MODELS_DIR, "best_model.pkl")
SHAP_PATH      = os.path.join(MODELS_DIR, "shap_explainer.pkl")
RESULTS_PATH   = os.path.join(MODELS_DIR, "model_results.pkl")

RANDOM_STATE   = 42
CV_FOLDS       = 5

# ── style ────────────────────────────────────────────────────────────────────
BG      = "#F8F8F8"
GREEN   = "#1D9E75"
RED     = "#E24B4A"
BLUE    = "#378ADD"
AMBER   = "#BA7517"
COLORS  = [GREEN, BLUE, AMBER]

plt.rcParams.update({
    "figure.facecolor" : BG,
    "axes.facecolor"   : BG,
    "axes.grid"        : True,
    "grid.color"       : "#E0E0E0",
    "grid.linewidth"   : 0.6,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})


# ── helpers ───────────────────────────────────────────────────────────────────
def save_plot(fig, filename):
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved -> {path}")


def evaluate(model, X_test, y_test, name="Model"):
    """Return a dict of all evaluation metrics."""
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]
    return {
        "Model"    : name,
        "Accuracy" : round(accuracy_score(y_test, y_pred),  4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall"   : round(recall_score(y_test, y_pred,    zero_division=0), 4),
        "F1"       : round(f1_score(y_test, y_pred,        zero_division=0), 4),
        "ROC-AUC"  : round(roc_auc_score(y_test, y_prob),  4),
    }


# ── step 1: load data ─────────────────────────────────────────────────────────
def load_data():
    for p in [X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH]:
        if not os.path.exists(p):
            print(f"\n  ERROR: '{p}' not found.")
            print("  Run  python src/feature_engineering.py  first.\n")
            sys.exit(1)

    X_train      = joblib.load(X_TRAIN_PATH)
    X_test       = joblib.load(X_TEST_PATH)
    y_train      = joblib.load(Y_TRAIN_PATH)
    y_test       = joblib.load(Y_TEST_PATH)
    feature_cols = joblib.load(FEATURE_PATH)

    print(f"[1] Data loaded:")
    print(f"    X_train : {X_train.shape}  (SMOTE balanced)")
    print(f"    X_test  : {X_test.shape}   (original distribution)")
    print(f"    y_train : churn={int(y_train.sum())} / retained={int((y_train==0).sum())}")
    print(f"    y_test  : churn={int(y_test.sum())} / retained={int((y_test==0).sum())}")
    return X_train, X_test, y_train, y_test, feature_cols


# ── step 2: train baseline models ────────────────────────────────────────────
def train_baselines(X_train, y_train):
    print(f"\n[2] Training 3 baseline models...")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            C=1.0,
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
            use_label_encoder=False,
        ),
    }

    trained = {}
    for name, model in models.items():
        print(f"    Training {name}...", end=" ", flush=True)
        model.fit(X_train, y_train)
        trained[name] = model
        print("done")

    return trained


# ── step 3: evaluate all 3 ───────────────────────────────────────────────────
def evaluate_all(models, X_test, y_test):
    print(f"\n[3] Evaluating all models on test set:")

    results = []
    for name, model in models.items():
        res = evaluate(model, X_test, y_test, name)
        results.append(res)

    df = pd.DataFrame(results).set_index("Model")

    # Pretty print comparison table
    print(f"\n{'':25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'ROC-AUC':>10}")
    print("    " + "-" * 72)
    for model_name, row in df.iterrows():
        print(f"    {model_name:<22} "
              f"{row['Accuracy']:>10.4f} "
              f"{row['Precision']:>10.4f} "
              f"{row['Recall']:>10.4f} "
              f"{row['F1']:>8.4f} "
              f"{row['ROC-AUC']:>10.4f}")

    best_name = df["ROC-AUC"].idxmax()
    print(f"\n    Best by ROC-AUC: {best_name}  ({df.loc[best_name, 'ROC-AUC']:.4f})")
    return df, best_name


# ── step 4: tune best model ───────────────────────────────────────────────────
def tune_model(best_name, models, X_train, y_train):
    print(f"\n[4] Hyperparameter tuning: {best_name} (GridSearchCV {CV_FOLDS}-fold)")

    param_grids = {
        "Logistic Regression": {
            "C"      : [0.01, 0.1, 1.0, 10.0],
            "solver" : ["lbfgs", "liblinear"],
            "max_iter": [500, 1000],
        },
        "Random Forest": {
            "n_estimators"     : [100, 200, 300],
            "max_depth"        : [None, 10, 20],
            "min_samples_split": [2, 5],
        },
        "XGBoost": {
            "n_estimators" : [100, 200, 300],
            "max_depth"    : [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample"    : [0.8, 1.0],
        },
    }

    base_model  = models[best_name]
    param_grid  = param_grids[best_name]
    cv          = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                                  random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator  = base_model,
        param_grid = param_grid,
        cv         = cv,
        scoring    = "roc_auc",
        n_jobs     = -1,
        verbose    = 0,
        refit      = True,
    )

    print(f"    Running grid search... ", end="", flush=True)
    grid_search.fit(X_train, y_train)
    print("done")
    print(f"    Best params : {grid_search.best_params_}")
    print(f"    Best CV AUC : {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


# ── step 5: final evaluation ──────────────────────────────────────────────────
def final_evaluation(tuned_model, best_name, X_test, y_test):
    print(f"\n[5] Final evaluation — tuned {best_name}:")

    y_pred = tuned_model.predict(X_test)
    y_prob = tuned_model.predict_proba(X_test)[:, 1]

    metrics = evaluate(tuned_model, X_test, y_test, f"Tuned {best_name}")

    print(f"\n    {'Metric':<15} {'Score':>8}")
    print("    " + "-" * 25)
    for k, v in metrics.items():
        if k != "Model":
            flag = " ← " if k == "ROC-AUC" else ""
            print(f"    {k:<15} {v:>8.4f}{flag}")

    print(f"\n    Classification report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["Retained", "Churned"],
                                 digits=4))

    return metrics, y_pred, y_prob


# ── step 6: plot comparison ───────────────────────────────────────────────────
def plot_model_comparison(baseline_df, tuned_metrics):
    print("\n[6] Generating evaluation charts...")

    metrics_list = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Model Comparison — All Metrics", fontsize=14, fontweight="bold")

    x     = np.arange(len(metrics_list))
    width = 0.2
    n     = len(baseline_df)

    for i, (model_name, row) in enumerate(baseline_df.iterrows()):
        vals = [row[m] for m in metrics_list]
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=model_name,
                      color=COLORS[i], edgecolor="white", alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")

    # Add tuned model line
    tuned_vals = [tuned_metrics[m] for m in metrics_list]
    ax.plot(x, tuned_vals, "D--", color=RED, linewidth=2,
            markersize=7, label=f"Tuned {tuned_metrics['Model'].replace('Tuned ','')}", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_list, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    save_plot(fig, "01_model_comparison.png")


def plot_confusion_matrix(tuned_model, X_test, y_test):
    y_pred = tuned_model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle("Confusion Matrix — Tuned Model", fontsize=14, fontweight="bold")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted: Retained", "Predicted: Churned"],
        yticklabels=["Actual: Retained",    "Actual: Churned"],
        linewidths=1, linecolor="white",
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax, cbar=False,
    )
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)

    # Annotate TP/TN/FP/FN
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.72, labels[i][j],
                    ha="center", fontsize=9, color="gray")

    fig.tight_layout()
    save_plot(fig, "02_confusion_matrix.png")


def plot_roc_curve(models_dict, tuned_model, best_name, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("ROC Curve Comparison", fontsize=14, fontweight="bold")

    # Baseline models
    for i, (name, model) in enumerate(models_dict.items()):
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        ax.plot(fpr, tpr, color=COLORS[i], linewidth=1.8,
                alpha=0.7, label=f"{name} (AUC={auc:.3f})", linestyle="--")

    # Tuned model
    fpr_t, tpr_t, _ = roc_curve(y_test, tuned_model.predict_proba(X_test)[:, 1])
    auc_t = roc_auc_score(y_test, tuned_model.predict_proba(X_test)[:, 1])
    ax.plot(fpr_t, tpr_t, color=RED, linewidth=2.8,
            label=f"Tuned {best_name} (AUC={auc_t:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Random (AUC=0.500)")
    ax.fill_between(fpr_t, tpr_t, alpha=0.08, color=RED)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    fig.tight_layout()
    save_plot(fig, "03_roc_curve.png")


def plot_feature_importance(tuned_model, feature_cols, best_name, top_n=20):
    # Works for tree-based models (RF, XGBoost)
    if not hasattr(tuned_model, "feature_importances_"):
        print("  (Feature importance skipped — not a tree model)")
        return

    importances = tuned_model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature"   : feature_cols,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle(f"Top {top_n} Feature Importances — {best_name}",
                 fontsize=14, fontweight="bold")

    colors = [RED if i < 5 else BLUE if i < 10 else GREEN
              for i in range(len(feat_df))]

    bars = ax.barh(feat_df["Feature"][::-1],
                   feat_df["Importance"][::-1],
                   color=colors[::-1], edgecolor="white", height=0.7)

    for bar, val in zip(bars, feat_df["Importance"][::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8.5)

    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_xlim(0, feat_df["Importance"].max() * 1.2)

    # Legend
    red_patch   = plt.Rectangle((0,0),1,1, color=RED)
    blue_patch  = plt.Rectangle((0,0),1,1, color=BLUE)
    green_patch = plt.Rectangle((0,0),1,1, color=GREEN)
    ax.legend([red_patch, blue_patch, green_patch],
              ["Top 5", "Top 6-10", "Top 11-20"],
              fontsize=9, loc="lower right")

    fig.tight_layout()
    save_plot(fig, "04_feature_importance.png")


# ── step 7: save SHAP explainer ───────────────────────────────────────────────
def save_shap_explainer(tuned_model, X_test, best_name):
    try:
        import shap
        print(f"\n[7] Computing SHAP explainer for {best_name}...")
        if "XGBoost" in best_name or "Random Forest" in best_name:
            explainer = shap.TreeExplainer(tuned_model)
        else:
            explainer = shap.LinearExplainer(tuned_model,
                                              X_test,
                                              feature_perturbation="correlation_dependent")
        joblib.dump(explainer, SHAP_PATH)
        print(f"    Saved -> {SHAP_PATH}")

        # Quick test
        shap_vals = explainer.shap_values(X_test[:5])
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        print(f"    SHAP values shape: {np.array(shap_vals).shape}  ✓")

    except ImportError:
        print("  (SHAP not installed — skipping. Run: pip install shap)")
    except Exception as e:
        print(f"  (SHAP explainer skipped: {e})")


# ── step 8: save final model ──────────────────────────────────────────────────
def save_model(tuned_model, baseline_results, tuned_metrics, best_name, best_params):
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(tuned_model, MODEL_PATH)

    # Save results summary for the app to display
    results_summary = {
        "best_model_name"   : best_name,
        "best_params"       : best_params,
        "tuned_metrics"     : tuned_metrics,
        "baseline_results"  : baseline_results.to_dict(),
    }
    joblib.dump(results_summary, RESULTS_PATH)

    print(f"\n[8] Saved final model:")
    print(f"    models/best_model.pkl      -> Tuned {best_name}")
    print(f"    models/model_results.pkl   -> Metrics summary")

    # File size
    size_kb = os.path.getsize(MODEL_PATH) / 1024
    print(f"    Model file size: {size_kb:.1f} KB")


# ── step 9: final summary ─────────────────────────────────────────────────────
def print_final_summary(tuned_metrics, best_name, best_params):
    print("\n" + "=" * 60)
    print("  PHASE 5 COMPLETE  |  Model Training Summary")
    print("=" * 60)
    print(f"  Best model        : {best_name}")
    print(f"  Best params       : {best_params}")
    print(f"  Test Accuracy     : {tuned_metrics['Accuracy']:.4f}  "
          f"({tuned_metrics['Accuracy']*100:.2f}%)")
    print(f"  Test Precision    : {tuned_metrics['Precision']:.4f}")
    print(f"  Test Recall       : {tuned_metrics['Recall']:.4f}")
    print(f"  Test F1 Score     : {tuned_metrics['F1']:.4f}")
    print(f"  Test ROC-AUC      : {tuned_metrics['ROC-AUC']:.4f}")
    print(f"\n  Plots saved to    : {PLOT_DIR}/")
    print(f"  Model saved to    : {MODEL_PATH}")
    print("=" * 60)
    print("  Phase 5 complete.  Run next:  python src/predict.py")
    print("=" * 60 + "\n")


# ── main ──────────────────────────────────────────────────────────────────────
def run_training():
    print("=" * 60)
    print("  PHASE 5  |  Model Training & Evaluation")
    print("=" * 60)

    X_train, X_test, y_train, y_test, feature_cols = load_data()

    # 1. Train baselines
    models = train_baselines(X_train, y_train)

    # 2. Compare all 3
    baseline_df, best_name = evaluate_all(models, X_test, y_test)

    # 3. Tune best
    tuned_model, best_params = tune_model(best_name, models, X_train, y_train)

    # 4. Final eval
    tuned_metrics, y_pred, y_prob = final_evaluation(
        tuned_model, best_name, X_test, y_test
    )

    # 5. Charts
    plot_model_comparison(baseline_df, tuned_metrics)
    plot_confusion_matrix(tuned_model, X_test, y_test)
    plot_roc_curve(models, tuned_model, best_name, X_test, y_test)
    plot_feature_importance(tuned_model, feature_cols, best_name)

    # 6. SHAP
    save_shap_explainer(tuned_model, X_test, best_name)

    # 7. Save model
    save_model(tuned_model, baseline_df, tuned_metrics, best_name, best_params)

    # 8. Summary
    print_final_summary(tuned_metrics, best_name, best_params)


if __name__ == "__main__":
    run_training()
