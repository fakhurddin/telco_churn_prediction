"""
data_cleaning.py
================
Phase 1 + Phase 2: Load the Telco Customer Churn dataset, fix known
data-quality issues, and save a clean version ready for EDA and modelling.

What this file does:
  1. Loads raw CSV from  data/telco_churn.csv
  2. Fixes TotalCharges (stored as string with whitespace for 11 rows)
  3. Drops those 11 blank rows (~0.15% of data — safe to remove)
  4. Encodes the target column: Churn  Yes -> 1 / No -> 0
  5. Resets the index after dropping
  6. Saves the cleaned file to  data/telco_churn_clean.csv
  7. Prints a full summary report so you can verify every step

Run:
    python src/data_cleaning.py
"""

import pandas as pd
import numpy as np
import os
import sys


# ── paths ──────────────────────────────────────────────────────────────────────
RAW_PATH   = os.path.join("data", "telco_churn.csv")
CLEAN_PATH = os.path.join("data", "telco_churn_clean.csv")


def load_raw(path: str) -> pd.DataFrame:
    """Load raw CSV and perform initial sanity checks."""
    if not os.path.exists(path):
        print(f"\n  ERROR: Dataset not found at '{path}'")
        print("  Please download it from:")
        print("  https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        print("  and place the CSV inside the  data/  folder.\n")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"[1] Loaded raw data:  {df.shape[0]:,} rows  x  {df.shape[1]} columns")
    return df


def inspect(df: pd.DataFrame) -> None:
    """Print a readable summary of dtypes, nulls, and unique counts."""
    print("\n[2] Column overview:")
    print(f"    {'Column':<25} {'Dtype':<12} {'Nulls':>6}  {'Unique':>7}")
    print("    " + "-" * 55)
    for col in df.columns:
        nulls  = df[col].isnull().sum()
        unique = df[col].nunique()
        print(f"    {col:<25} {str(df[col].dtype):<12} {nulls:>6}  {unique:>7}")


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    TotalCharges is read as object because 11 rows contain only whitespace.
    Convert to float; rows that cannot convert become NaN and are dropped.
    """
    original_count = len(df)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    bad_rows = df["TotalCharges"].isnull().sum()

    df.dropna(subset=["TotalCharges"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"\n[3] TotalCharges fix:")
    print(f"    - Converted from object -> float64")
    print(f"    - Found {bad_rows} whitespace rows -> dropped")
    print(f"    - Rows remaining: {len(df):,}  (was {original_count:,})")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Churn: Yes -> 1, No -> 0."""
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    churn_rate  = df["Churn"].mean() * 100
    print(f"\n[4] Target encoding:")
    print(f"    - 'Yes' -> 1  |  'No' -> 0")
    print(f"    - Churn rate : {churn_rate:.2f}%")
    print(f"    - Churned    : {df['Churn'].sum():,}  customers")
    print(f"    - Retained   : {(df['Churn'] == 0).sum():,}  customers")
    return df


def drop_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    """customerID is a unique key — not a feature. Drop it."""
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)
        print(f"\n[5] Dropped 'customerID' column (not a feature)")
    return df


def validate_clean(df: pd.DataFrame) -> None:
    """Final validation — confirm zero nulls and correct dtypes."""
    null_total = df.isnull().sum().sum()
    print(f"\n[6] Validation:")
    print(f"    - Total null values : {null_total}")
    print(f"    - Final shape       : {df.shape}")
    print(f"    - Numeric columns   : {list(df.select_dtypes(include='number').columns)}")
    print(f"    - Object  columns   : {list(df.select_dtypes(include='object').columns)}")

    if null_total > 0:
        print("    WARNING: Null values still present — review pipeline.")
    else:
        print("    All checks passed.")


def save_clean(df: pd.DataFrame, path: str) -> None:
    """Save cleaned DataFrame to CSV."""
    df.to_csv(path, index=False)
    print(f"\n[7] Saved cleaned data -> '{path}'")


def run_pipeline() -> pd.DataFrame:
    """Execute the full cleaning pipeline end-to-end."""
    print("=" * 60)
    print("  PHASE 1 + 2  |  Data Loading & Cleaning")
    print("=" * 60)

    df = load_raw(RAW_PATH)
    inspect(df)
    df = fix_total_charges(df)
    df = encode_target(df)
    df = drop_customer_id(df)
    validate_clean(df)
    save_clean(df, CLEAN_PATH)

    print("\n" + "=" * 60)
    print("  Phase 1 + 2 complete.  Run next:  python src/eda.py")
    print("=" * 60 + "\n")
    return df


if __name__ == "__main__":
    run_pipeline()
