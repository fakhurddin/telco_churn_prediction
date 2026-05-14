"""
eda.py
======
Phase 3: Exploratory Data Analysis (EDA)
Reads the cleaned dataset and generates 7 charts that reveal
the key patterns driving customer churn.

Charts produced:
  1. Overall churn rate (pie + bar)
  2. Churn by contract type
  3. Tenure distribution by churn status
  4. Monthly charges distribution by churn status
  5. Churn rate by internet service type
  6. Churn by key add-on services (TechSupport, OnlineSecurity, etc.)
  7. Correlation heatmap of numeric features

Run:
    python src/eda.py

Output:
    data/eda_plots/01_churn_rate.png
    data/eda_plots/02_contract_type.png
    data/eda_plots/03_tenure_distribution.png
    data/eda_plots/04_monthly_charges.png
    data/eda_plots/05_internet_service.png
    data/eda_plots/06_addon_services.png
    data/eda_plots/07_correlation_heatmap.png
    data/eda_plots/00_full_dashboard.png   (all 7 in one image)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import sys

# ── paths ───────────────────────────────────────────────────────────────────
CLEAN_PATH = os.path.join("data", "telco_churn_clean.csv")
PLOT_DIR   = os.path.join("data", "eda_plots")

# ── style ────────────────────────────────────────────────────────────────────
CHURN_COLORS   = {"Churned": "#E24B4A", "Retained": "#1D9E75"}
PALETTE        = ["#1D9E75", "#E24B4A"]          # green=retained, red=churned
BG_COLOR       = "#F8F8F8"
GRID_COLOR     = "#E0E0E0"
TITLE_SIZE     = 14
LABEL_SIZE     = 11
TICK_SIZE      = 10


def setup():
    """Create output directory and set global matplotlib style."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor"  : BG_COLOR,
        "axes.facecolor"    : BG_COLOR,
        "axes.grid"         : True,
        "grid.color"        : GRID_COLOR,
        "grid.linewidth"    : 0.6,
        "font.family"       : "DejaVu Sans",
        "axes.spines.top"   : False,
        "axes.spines.right" : False,
    })


def load(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"\n  ERROR: '{path}' not found.")
        print("  Run  python src/data_cleaning.py  first.\n")
        sys.exit(1)
    df = pd.read_csv(path)
    # Add a readable label column for plotting
    df["ChurnLabel"] = df["Churn"].map({1: "Churned", 0: "Retained"})
    print(f"[load] {df.shape[0]:,} rows loaded from '{path}'")
    return df


def save(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved -> {path}")


# ── Chart 1: Overall churn rate ─────────────────────────────────────────────
def chart_01_churn_rate(df: pd.DataFrame) -> None:
    print("\n[Chart 1] Overall churn rate")

    total     = len(df)
    churned   = df["Churn"].sum()
    retained  = total - churned
    churn_pct = churned / total * 100

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Overall Customer Churn Rate", fontsize=TITLE_SIZE + 1, fontweight="bold", y=1.01)

    # Left: pie chart
    ax = axes[0]
    wedges, texts, autotexts = ax.pie(
        [retained, churned],
        labels=["Retained", "Churned"],
        colors=[CHURN_COLORS["Retained"], CHURN_COLORS["Churned"]],
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": LABEL_SIZE},
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
        at.set_color("white")
    ax.set_title("Proportion", fontsize=LABEL_SIZE, pad=10)

    # Right: count bar
    ax2 = axes[1]
    bars = ax2.bar(
        ["Retained", "Churned"],
        [retained, churned],
        color=[CHURN_COLORS["Retained"], CHURN_COLORS["Churned"]],
        width=0.45,
        edgecolor="white",
        linewidth=1.2,
    )
    for bar, val in zip(bars, [retained, churned]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 40,
            f"{val:,}\n({val/total*100:.1f}%)",
            ha="center", va="bottom", fontsize=LABEL_SIZE, fontweight="bold",
        )
    ax2.set_ylabel("Number of Customers", fontsize=LABEL_SIZE)
    ax2.set_title("Counts", fontsize=LABEL_SIZE, pad=10)
    ax2.set_ylim(0, retained * 1.18)
    ax2.tick_params(labelsize=TICK_SIZE)

    # Key insight annotation
    fig.text(
        0.5, -0.04,
        f"Key insight: {churn_pct:.1f}% of customers churned — "
        f"significant class imbalance that SMOTE will address in Phase 4.",
        ha="center", fontsize=10, color="#555", style="italic",
    )

    fig.tight_layout()
    save(fig, "01_churn_rate.png")


# ── Chart 2: Churn by contract type ─────────────────────────────────────────
def chart_02_contract_type(df: pd.DataFrame) -> None:
    print("[Chart 2] Churn by contract type")

    contract_churn = (
        df.groupby("Contract")["Churn"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "Churned", "count": "Total"})
    )
    contract_churn["ChurnRate"] = contract_churn["Churned"] / contract_churn["Total"] * 100
    contract_churn = contract_churn.sort_values("ChurnRate", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Churn by Contract Type", fontsize=TITLE_SIZE + 1, fontweight="bold")

    # Left: stacked bar (counts)
    ax = axes[0]
    x     = range(len(contract_churn))
    retained_counts = contract_churn["Total"] - contract_churn["Churned"]
    b1 = ax.bar(x, retained_counts,         label="Retained", color=CHURN_COLORS["Retained"], edgecolor="white")
    b2 = ax.bar(x, contract_churn["Churned"], bottom=retained_counts, label="Churned", color=CHURN_COLORS["Churned"], edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(contract_churn.index, fontsize=TICK_SIZE)
    ax.set_ylabel("Number of Customers", fontsize=LABEL_SIZE)
    ax.set_title("Customer counts (stacked)", fontsize=LABEL_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    for bar in b2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + h/2,
                    f"{int(h):,}", ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")

    # Right: churn rate bar
    ax2 = axes[1]
    colors = [CHURN_COLORS["Churned"] if r > 30 else "#E8A838"
              if r > 10 else CHURN_COLORS["Retained"]
              for r in contract_churn["ChurnRate"]]
    bars = ax2.barh(contract_churn.index, contract_churn["ChurnRate"],
                    color=colors, edgecolor="white", height=0.45)
    for bar, val in zip(bars, contract_churn["ChurnRate"]):
        ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=LABEL_SIZE, fontweight="bold")
    ax2.set_xlabel("Churn Rate (%)", fontsize=LABEL_SIZE)
    ax2.set_title("Churn rate per contract", fontsize=LABEL_SIZE)
    ax2.set_xlim(0, contract_churn["ChurnRate"].max() * 1.2)
    ax2.tick_params(labelsize=TICK_SIZE)

    fig.text(0.5, -0.04,
             "Key insight: Month-to-month customers churn at ~3× the rate of annual subscribers.",
             ha="center", fontsize=10, color="#555", style="italic")

    fig.tight_layout()
    save(fig, "02_contract_type.png")


# ── Chart 3: Tenure distribution ─────────────────────────────────────────────
def chart_03_tenure(df: pd.DataFrame) -> None:
    print("[Chart 3] Tenure distribution by churn")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Tenure Distribution by Churn Status", fontsize=TITLE_SIZE + 1, fontweight="bold")

    # Left: overlapping histograms
    ax = axes[0]
    for label, color in CHURN_COLORS.items():
        subset = df[df["ChurnLabel"] == label]["tenure"]
        ax.hist(subset, bins=30, alpha=0.65, color=color, label=label,
                edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Tenure (months)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Number of Customers", fontsize=LABEL_SIZE)
    ax.set_title("Histogram", fontsize=LABEL_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)

    # Right: box plot
    ax2 = axes[1]
    churned_tenure  = df[df["Churn"] == 1]["tenure"]
    retained_tenure = df[df["Churn"] == 0]["tenure"]
    bp = ax2.boxplot(
        [retained_tenure, churned_tenure],
        labels=["Retained", "Churned"],
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 2.5},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
    )
    for patch, color in zip(bp["boxes"], [CHURN_COLORS["Retained"], CHURN_COLORS["Churned"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax2.set_ylabel("Tenure (months)", fontsize=LABEL_SIZE)
    ax2.set_title("Box plot (median & spread)", fontsize=LABEL_SIZE)
    ax2.tick_params(labelsize=TICK_SIZE)

    med_retained = retained_tenure.median()
    med_churned  = churned_tenure.median()
    fig.text(0.5, -0.04,
             f"Key insight: Churned customers median tenure = {med_churned:.0f} months vs "
             f"{med_retained:.0f} months for retained. New customers churn far more.",
             ha="center", fontsize=10, color="#555", style="italic")

    fig.tight_layout()
    save(fig, "03_tenure_distribution.png")


# ── Chart 4: Monthly charges ─────────────────────────────────────────────────
def chart_04_monthly_charges(df: pd.DataFrame) -> None:
    print("[Chart 4] Monthly charges distribution")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Monthly Charges by Churn Status", fontsize=TITLE_SIZE + 1, fontweight="bold")

    # Left: KDE plot
    ax = axes[0]
    for label, color in CHURN_COLORS.items():
        subset = df[df["ChurnLabel"] == label]["MonthlyCharges"]
        subset.plot.kde(ax=ax, color=color, label=label, linewidth=2.2)
        ax.axvline(subset.median(), color=color, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("Monthly Charges ($)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Density", fontsize=LABEL_SIZE)
    ax.set_title("Density (KDE) — dashed = median", fontsize=LABEL_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)

    # Right: violin plot
    ax2 = axes[1]
    parts = ax2.violinplot(
        [df[df["Churn"] == 0]["MonthlyCharges"],
         df[df["Churn"] == 1]["MonthlyCharges"]],
        positions=[1, 2],
        showmedians=True,
        showextrema=True,
    )
    colors = [CHURN_COLORS["Retained"], CHURN_COLORS["Churned"]]
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.75)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["Retained", "Churned"], fontsize=TICK_SIZE)
    ax2.set_ylabel("Monthly Charges ($)", fontsize=LABEL_SIZE)
    ax2.set_title("Violin plot", fontsize=LABEL_SIZE)
    ax2.tick_params(labelsize=TICK_SIZE)

    med_c = df[df["Churn"] == 1]["MonthlyCharges"].median()
    med_r = df[df["Churn"] == 0]["MonthlyCharges"].median()
    fig.text(0.5, -0.04,
             f"Key insight: Churned customers pay ${med_c:.0f}/mo median vs ${med_r:.0f}/mo for retained. "
             "Higher bills strongly predict churn.",
             ha="center", fontsize=10, color="#555", style="italic")

    fig.tight_layout()
    save(fig, "04_monthly_charges.png")


# ── Chart 5: Internet service type ──────────────────────────────────────────
def chart_05_internet_service(df: pd.DataFrame) -> None:
    print("[Chart 5] Churn by internet service")

    grp = (df.groupby(["InternetService", "ChurnLabel"])
             .size().reset_index(name="Count"))
    pivot = grp.pivot(index="InternetService", columns="ChurnLabel", values="Count").fillna(0)
    pivot["ChurnRate"] = pivot["Churned"] / (pivot["Churned"] + pivot["Retained"]) * 100
    pivot = pivot.sort_values("ChurnRate", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Churn Rate by Internet Service Type", fontsize=TITLE_SIZE + 1, fontweight="bold")

    x      = np.arange(len(pivot))
    width  = 0.35
    b1 = ax.bar(x - width/2, pivot["Retained"], width, label="Retained",
                color=CHURN_COLORS["Retained"], edgecolor="white")
    b2 = ax.bar(x + width/2, pivot["Churned"],  width, label="Churned",
                color=CHURN_COLORS["Churned"],  edgecolor="white")

    for bar in [*b1, *b2]:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 15,
                f"{int(h):,}", ha="center", va="bottom", fontsize=9)

    # Churn rate labels above bars
    for i, (idx, row) in enumerate(pivot.iterrows()):
        ax.text(i, max(row["Retained"], row["Churned"]) + 120,
                f"Churn rate:\n{row['ChurnRate']:.1f}%",
                ha="center", fontsize=9, color="#333", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3CD", edgecolor="#E8A838", linewidth=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, fontsize=TICK_SIZE)
    ax.set_ylabel("Number of Customers", fontsize=LABEL_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)

    fig.text(0.5, -0.04,
             "Key insight: Fiber optic customers churn at the highest rate despite paying the most.",
             ha="center", fontsize=10, color="#555", style="italic")

    fig.tight_layout()
    save(fig, "05_internet_service.png")


# ── Chart 6: Add-on services ─────────────────────────────────────────────────
def chart_06_addon_services(df: pd.DataFrame) -> None:
    print("[Chart 6] Churn by add-on services")

    services = ["TechSupport", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "StreamingTV", "StreamingMovies"]

    # Compute churn rate for Yes vs No (exclude "No internet service")
    rows = []
    for svc in services:
        for val in ["Yes", "No"]:
            subset = df[df[svc] == val]
            if len(subset) > 0:
                rows.append({
                    "Service"   : svc.replace("Streaming", "Streaming\n"),
                    "HasService": val,
                    "ChurnRate" : subset["Churn"].mean() * 100,
                    "Count"     : len(subset),
                })
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle("Churn Rate: With vs Without Add-on Services",
                 fontsize=TITLE_SIZE + 1, fontweight="bold")

    svc_labels = plot_df["Service"].unique()
    x     = np.arange(len(svc_labels))
    width = 0.35

    yes_rates = plot_df[plot_df["HasService"] == "Yes"].set_index("Service")["ChurnRate"]
    no_rates  = plot_df[plot_df["HasService"] == "No"].set_index("Service")["ChurnRate"]

    b1 = ax.bar(x - width/2, [yes_rates.get(s, 0) for s in svc_labels],
                width, label="Has service",    color="#1D9E75", edgecolor="white")
    b2 = ax.bar(x + width/2, [no_rates.get(s, 0) for s in svc_labels],
                width, label="No service",     color="#E24B4A", edgecolor="white")

    for bar in [*b1, *b2]:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.4,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(svc_labels, fontsize=TICK_SIZE)
    ax.set_ylabel("Churn Rate (%)", fontsize=LABEL_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)

    fig.text(0.5, -0.04,
             "Key insight: Customers WITHOUT TechSupport or OnlineSecurity "
             "churn at significantly higher rates.",
             ha="center", fontsize=10, color="#555", style="italic")

    fig.tight_layout()
    save(fig, "06_addon_services.png")


# ── Chart 7: Correlation heatmap ─────────────────────────────────────────────
def chart_07_correlation(df: pd.DataFrame) -> None:
    print("[Chart 7] Correlation heatmap")

    # Binary-encode Yes/No columns for correlation
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include="object").columns:
        if set(df_enc[col].dropna().unique()).issubset({"Yes", "No"}):
            df_enc[col] = df_enc[col].map({"Yes": 1, "No": 0})
        else:
            df_enc.drop(columns=[col], inplace=True)

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges",
                    "SeniorCitizen", "Churn"]
    available = [c for c in numeric_cols if c in df_enc.columns]
    corr = df_enc[available].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Correlation Heatmap — Numeric Features vs Churn",
                 fontsize=TITLE_SIZE + 1, fontweight="bold")

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 11, "weight": "bold"},
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.tick_params(labelsize=TICK_SIZE, rotation=30)

    # Highlight the Churn row/column
    ax.add_patch(plt.Rectangle(
        (0, corr.columns.get_loc("Churn")), len(corr.columns), 1,
        fill=False, edgecolor="#E24B4A", lw=2.5, clip_on=False
    ))

    fig.text(0.5, -0.04,
             "Key insight: Tenure has a negative correlation with churn. "
             "MonthlyCharges has a positive correlation.",
             ha="center", fontsize=10, color="#555", style="italic")

    fig.tight_layout()
    save(fig, "07_correlation_heatmap.png")


# ── Dashboard: all 7 in one image ───────────────────────────────────────────
def chart_00_dashboard(df: pd.DataFrame) -> None:
    print("\n[Dashboard] Generating combined dashboard...")

    fig = plt.figure(figsize=(20, 24))
    fig.suptitle("Customer Churn — Full EDA Dashboard",
                 fontsize=18, fontweight="bold", y=0.99)

    # Load the 7 saved PNGs and place them
    filenames = [
        "01_churn_rate.png", "02_contract_type.png",
        "03_tenure_distribution.png", "04_monthly_charges.png",
        "05_internet_service.png", "06_addon_services.png",
        "07_correlation_heatmap.png",
    ]

    from matplotlib.image import imread
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.15)
    axes_specs = [
        gs[0, 0], gs[0, 1],
        gs[1, 0], gs[1, 1],
        gs[2, 0], gs[2, 1],
        gs[3, :],            # last chart full width
    ]

    for spec, fname in zip(axes_specs, filenames):
        fpath = os.path.join(PLOT_DIR, fname)
        if os.path.exists(fpath):
            ax = fig.add_subplot(spec)
            img = imread(fpath)
            ax.imshow(img)
            ax.axis("off")

    save(fig, "00_full_dashboard.png")


# ── Print summary stats ──────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("  EDA SUMMARY STATISTICS")
    print("=" * 60)
    print(f"  Total customers      : {len(df):,}")
    print(f"  Churned              : {df['Churn'].sum():,} ({df['Churn'].mean()*100:.2f}%)")
    print(f"  Retained             : {(df['Churn']==0).sum():,}")
    print(f"\n  Tenure  — mean       : {df['tenure'].mean():.1f} months")
    print(f"  Tenure  — churned    : {df[df['Churn']==1]['tenure'].mean():.1f} months")
    print(f"  Tenure  — retained   : {df[df['Churn']==0]['tenure'].mean():.1f} months")
    print(f"\n  Monthly charges mean : ${df['MonthlyCharges'].mean():.2f}")
    print(f"  Charges — churned    : ${df[df['Churn']==1]['MonthlyCharges'].mean():.2f}")
    print(f"  Charges — retained   : ${df[df['Churn']==0]['MonthlyCharges'].mean():.2f}")
    print(f"\n  Contract breakdown:")
    for ctype, grp in df.groupby("Contract"):
        print(f"    {ctype:<22} churn rate: {grp['Churn'].mean()*100:.1f}%")
    print("=" * 60)


# ── Main ─────────────────────────────────────────────────────────────────────
def run_eda() -> None:
    print("=" * 60)
    print("  PHASE 3  |  Exploratory Data Analysis")
    print("=" * 60)

    setup()
    df = load(CLEAN_PATH)
    print_summary(df)

    print("\nGenerating charts...")
    chart_01_churn_rate(df)
    chart_02_contract_type(df)
    chart_03_tenure(df)
    chart_04_monthly_charges(df)
    chart_05_internet_service(df)
    chart_06_addon_services(df)
    chart_07_correlation(df)
    chart_00_dashboard(df)

    print("\n" + "=" * 60)
    print(f"  All 8 files saved to  '{PLOT_DIR}/'")
    print("  Phase 3 complete.  Run next:  python src/feature_engineering.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_eda()
