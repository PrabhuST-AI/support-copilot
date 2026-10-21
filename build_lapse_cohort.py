"""build_lapse_cohort.py -- Define the customer lapse prediction problem and feature set.

Cutoff Date: 2025-09-01 00:00:00
Target Window: 2025-09-01 through 2025-09-30

- Cohort: All customers with >= 1 completed order before the cutoff (1,156 customers).
- Target (lapsed): 1 if customer has NO completed orders in the target window, 0 otherwise (443 lapsed, 38.32%).
- Features engineered strictly before cutoff:
    * recency: whole days from cutoff to last completed order
    * frequency: count of completed orders before cutoff
    * monetary: sum of completed order_amount before cutoff
    * avg_discount_pct: mean discount percentage across completed orders
    * cancellation_rate: cancelled orders / all orders with final status before cutoff
    * avg_delivery_minutes: mean delivery minutes of completed orders
    * most_frequent_category: mode product category among completed orders
    * city, age, gender: demographic data from clean customers table
    * tenure_days: whole days from signup_date to cutoff
- Anti-leakage guarantees:
    * Zero post-cutoff order information used in feature generation.
    * No RFM clustering/segmentation labels included.
- Split: train_test_split(test_size=0.2, stratify=y, random_state=42) -> 924 train / 232 test rows.
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

DB_PATH = "data/instamart.db"
OUTPUT_PATH = "outputs/lapse_cohort.csv"
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")

def build_lapse_dataset():
    conn = sqlite3.connect(DB_PATH)
    orders = pd.read_sql_query("SELECT * FROM orders", conn)
    customers = pd.read_sql_query("SELECT * FROM customers", conn)
    products = pd.read_sql_query("SELECT product_id, category FROM products", conn)
    conn.close()

    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])
    orders = orders.merge(products, on="product_id", how="left")

    # 1. Temporal Partitioning
    orders_pre = orders[orders["order_ts"] < CUTOFF].copy()
    orders_post = orders[orders["order_ts"] >= CUTOFF].copy()

    # 2. Cohort Definition: >= 1 completed order before cutoff
    completed_pre = orders_pre[orders_pre["status"] == "completed"].copy()
    cohort_cids = sorted(completed_pre["customer_id"].unique())

    # 3. Target Label (lapsed): 1 if no completed order in Sep 2025, else 0
    active_post_cids = set(orders_post[orders_post["status"] == "completed"]["customer_id"].unique())
    target_map = {cid: (0 if cid in active_post_cids else 1) for cid in cohort_cids}

    # 4. Feature Engineering (Strictly before cutoff)
    # Aggregated metrics on completed orders
    comp_agg = completed_pre.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean")
    ).reset_index()

    comp_agg["recency"] = (CUTOFF - comp_agg["last_order_ts"]).dt.days

    # Most frequent category among completed orders
    mode_cat = completed_pre.groupby("customer_id")["category"].agg(
        lambda s: s.mode().iloc[0]
    ).reset_index()
    mode_cat.rename(columns={"category": "most_frequent_category"}, inplace=True)

    # Cancellation rate before cutoff (cancelled / all orders with final status)
    all_pre_counts = orders_pre.groupby("customer_id")["order_id"].count()
    canc_pre_counts = orders_pre[orders_pre["status"] == "cancelled"].groupby("customer_id")["order_id"].count()
    canc_rate = (canc_pre_counts / all_pre_counts).fillna(0.0).reset_index()
    canc_rate.rename(columns={"order_id": "cancellation_rate"}, inplace=True)

    # Assemble feature matrix
    df = pd.DataFrame({"customer_id": cohort_cids})
    df = df.merge(comp_agg[["customer_id", "recency", "frequency", "monetary", "avg_discount_pct", "avg_delivery_minutes"]], on="customer_id", how="left")
    df = df.merge(canc_rate, on="customer_id", how="left").fillna({"cancellation_rate": 0.0})
    df = df.merge(mode_cat, on="customer_id", how="left")
    df = df.merge(customers[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id", how="left")
    df["tenure_days"] = (CUTOFF - df["signup_date"]).dt.days
    df["lapsed"] = df["customer_id"].map(target_map)

    # Round continuous metrics for cleanliness
    df["monetary"] = df["monetary"].round(2)
    df["avg_discount_pct"] = df["avg_discount_pct"].round(2)
    df["avg_delivery_minutes"] = df["avg_delivery_minutes"].round(2)
    df["cancellation_rate"] = df["cancellation_rate"].round(4)

    # Save dataset to outputs/
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Cohort dataset saved to {OUTPUT_PATH} ({len(df)} rows)")

    return df

def evaluate_split_and_baseline(df):
    feature_cols = [
        "recency", "frequency", "monetary", "avg_discount_pct",
        "cancellation_rate", "avg_delivery_minutes",
        "most_frequent_category", "city", "age", "gender", "tenure_days"
    ]
    X = df[feature_cols]
    y = df["lapsed"]

    # Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\n" + "=" * 70)
    print("  LAPSE PROBLEM DEFINITION & DATASET SUMMARY")
    print("=" * 70)
    cohort_size = len(df)
    lapsed_count = int(y.sum())
    lapsed_pct = (lapsed_count / cohort_size) * 100
    train_size = len(X_train)
    test_size = len(X_test)

    print(f"Cohort Size:          {cohort_size:,} customers (Reference: 1,156)")
    print(f"Lapsed Count:         {lapsed_count:,} ({lapsed_pct:.2f}%) (Reference: 443 / 38.32%)")
    print(f"Non-Lapsed Count:     {cohort_size - lapsed_count:,} ({(100 - lapsed_pct):.2f}%)")
    print(f"Training Set Size:    {train_size:,} rows (Reference: 924)")
    print(f"Test Set Size:        {test_size:,} rows (Reference: 232)")

    # Baseline: Always predict 0 ("not lapsed")
    y_pred_majority = np.zeros_like(y_test)
    acc = accuracy_score(y_test, y_pred_majority)
    f1_lapsed = f1_score(y_test, y_pred_majority, zero_division=0)

    print("\n" + "=" * 70)
    print("  DUMMY / MAJORITY BASELINE (PREDICTING ALWAYS 'NOT LAPSED')")
    print("=" * 70)
    print(f"Test Accuracy:        {acc * 100:.1f}% (Reference: 61.6%)")
    print(f"Test Lapsed-Class F1: {f1_lapsed:.1f} (Reference: 0)")

    # Assertions for verification
    assert cohort_size == 1156, f"Expected 1156, got {cohort_size}"
    assert lapsed_count == 443, f"Expected 443, got {lapsed_count}"
    assert train_size == 924, f"Expected 924, got {train_size}"
    assert test_size == 232, f"Expected 232, got {test_size}"
    assert abs(acc - 0.61637) < 0.005, f"Expected ~61.6%, got {acc*100:.2f}%"
    assert f1_lapsed == 0.0, f"Expected 0.0, got {f1_lapsed}"

    print("-" * 70)
    print("ALL REFERENCE METRICS VERIFIED SUCCESSFULLY! ✓\n")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    cohort_df = build_lapse_dataset()
    evaluate_split_and_baseline(cohort_df)
