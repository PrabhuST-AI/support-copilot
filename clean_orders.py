"""clean_orders.py -- Clean orders dataset according to specification:
  1. Convert order_ts to a datetime.
  2. Remove exact duplicate rows.
  3. Keep only the record with the latest order_ts for each order_id.
  4. Convert order_amount to numeric, turning unparseable entries into missing values,
     then fill each missing amount from the product price:
     order_amount = round(price_inr * quantity * (1 - discount_pct / 100), 2).
  5. Set negative delivery_minutes to missing. For completed orders, fill every missing
     delivery_minutes with the median delivery time of that order's product category
     among completed orders (using groupby and transform). For cancelled orders, leave
     delivery_minutes missing.
Saves cleaned dataset to data/clean/orders.csv and verifies against reference metrics.
"""

import os
import numpy as np
import pandas as pd

RAW_ORDERS_PATH = "data/raw/orders.csv"
RAW_PRODUCTS_PATH = "data/raw/products.csv"
CLEAN_ORDERS_PATH = "data/clean/orders.csv"

def clean_orders():
    # Load raw data
    orders = pd.read_csv(RAW_ORDERS_PATH, keep_default_na=False)
    products = pd.read_csv(RAW_PRODUCTS_PATH, keep_default_na=False)
    print(f"Loaded raw orders: {len(orders)} rows")

    # Rule 1: Convert order_ts to a datetime
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])

    # Rule 2: Remove exact duplicate rows
    initial_len = len(orders)
    orders = orders.drop_duplicates()
    print(f"Removed {initial_len - len(orders)} exact duplicate rows -> {len(orders)} rows remain")

    # Rule 3: For each order_id keep only the record with the latest order_ts
    # (Resolves earlier pending records vs later final completed/cancelled records)
    orders = orders.sort_values(by=["order_id", "order_ts"]).drop_duplicates(
        subset=["order_id"], keep="last"
    ).reset_index(drop=True)
    print(f"Kept latest order_ts per order_id -> {len(orders)} distinct orders remain")

    # Rule 4: Convert order_amount to numeric, fill missing from product formula:
    # order_amount = round(price_inr * quantity * (1 - discount_pct / 100), 2)
    orders = orders.merge(
        products[["product_id", "price_inr", "category"]],
        on="product_id",
        how="left"
    )

    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    orders["quantity"] = pd.to_numeric(orders["quantity"])
    orders["discount_pct"] = pd.to_numeric(orders["discount_pct"])
    orders["price_inr"] = pd.to_numeric(orders["price_inr"])

    calculated_amount = (
        orders["price_inr"] * orders["quantity"] * (1.0 - orders["discount_pct"] / 100.0)
    ).round(2)
    orders["order_amount"] = orders["order_amount"].fillna(calculated_amount)

    # Rule 5: Set negative delivery_minutes to missing.
    # For completed orders: fill missing delivery_minutes with median delivery time of product category
    # among completed orders using groupby and transform.
    # For cancelled orders: leave delivery_minutes missing.
    orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan

    completed_mask = orders["status"] == "completed"
    category_medians = orders.loc[completed_mask].groupby("category")["delivery_minutes"].transform("median")
    orders.loc[completed_mask, "delivery_minutes"] = orders.loc[completed_mask, "delivery_minutes"].fillna(category_medians)

    # Retain the canonical 9 columns in original schema order
    cols = [
        "order_id", "customer_id", "product_id", "quantity",
        "order_ts", "status", "discount_pct", "order_amount", "delivery_minutes"
    ]
    clean_df = orders[cols].copy()

    # Save cleaned output
    os.makedirs(os.path.dirname(CLEAN_ORDERS_PATH), exist_ok=True)
    clean_df.to_csv(CLEAN_ORDERS_PATH, index=False)
    print(f"Saved cleaned orders to {CLEAN_ORDERS_PATH}")

    return clean_df

def verify_cleaning(df):
    print("\n" + "=" * 70)
    print("  VERIFICATION AGAINST REFERENCE METRICS")
    print("=" * 70)

    total_rows = len(df)
    unique_orders = df["order_id"].nunique()
    status_counts = df["status"].value_counts().to_dict()
    completed_rows = status_counts.get("completed", 0)
    cancelled_rows = status_counts.get("cancelled", 0)
    pending_rows = status_counts.get("pending", 0)

    missing_amount = int(df["order_amount"].isna().sum())

    completed_df = df[df["status"] == "completed"]
    cancelled_df = df[df["status"] == "cancelled"]

    missing_deliv_completed = int(completed_df["delivery_minutes"].isna().sum())
    missing_deliv_cancelled = int(cancelled_df["delivery_minutes"].isna().sum())

    sum_completed_amount = float(completed_df["order_amount"].sum())

    rows = [
        ("Total rows", total_rows, 12309),
        ("Distinct order_id (one per order)", unique_orders, 12309),
        ("Completed orders count", completed_rows, 11110),
        ("Cancelled orders count", cancelled_rows, 1199),
        ("Pending orders count", pending_rows, 0),
        ("Missing order_amount count", missing_amount, 0),
        ("Missing delivery_minutes (completed)", missing_deliv_completed, 0),
        ("Missing delivery_minutes (cancelled)", missing_deliv_cancelled, 1199),
        ("Sum order_amount completed (INR)", f"{sum_completed_amount:,.2f}", "2,998,668.80"),
    ]

    print(f"{'Check / Metric':<42} {'Got':>13}  {'Expected':>13}  {'OK?':>5}")
    print("-" * 77)
    all_passed = True
    for label, got, exp in rows:
        ok = str(got) == str(exp)
        if not ok:
            all_passed = False
        mark = "✓" if ok else "✗"
        print(f"{label:<42} {str(got):>13}  {str(exp):>13}  {mark:>5}")

    print("-" * 77)
    if all_passed:
        print("ALL REFERENCE METRICS PASSED SUCCESSFULLY! ✓\n")
    else:
        print("SOME CHECKS FAILED! ✗\n")

if __name__ == "__main__":
    cleaned = clean_orders()
    verify_cleaning(cleaned)
