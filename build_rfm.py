"""build_rfm.py -- Compute RFM (Recency, Frequency, Monetary) metrics
for every customer with at least one completed order.

Snapshot moment: 2025-10-01 00:00:00
- recency: whole days between snapshot and most recent completed order ((snapshot - last_order_ts).dt.days)
- frequency: number of distinct completed orders
- monetary: sum of completed order_amount
"""

import os
import sqlite3
import pandas as pd

DB_PATH = "data/instamart.db"
CLEAN_RFM_PATH = "data/clean/rfm.csv"
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")

def build_rfm():
    # Load completed orders
    conn = sqlite3.connect(DB_PATH)
    orders = pd.read_sql_query(
        "SELECT order_id, customer_id, order_ts, order_amount FROM orders WHERE status = 'completed'",
        conn
    )

    orders["order_ts"] = pd.to_datetime(orders["order_ts"])

    # Group by customer_id
    rfm = orders.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()

    # Recency: whole days between snapshot and most recent order
    rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days

    # Round monetary to 2 decimal places
    rfm["monetary"] = rfm["monetary"].round(2)

    # Order columns
    rfm_table = rfm[["customer_id", "recency", "frequency", "monetary"]].sort_values("customer_id").reset_index(drop=True)

    # Save to CSV
    os.makedirs(os.path.dirname(CLEAN_RFM_PATH), exist_ok=True)
    rfm_table.to_csv(CLEAN_RFM_PATH, index=False)

    # Save to SQLite table 'rfm'
    rfm_table.to_sql("rfm", conn, if_exists="replace", index=False)
    conn.close()

    print(f"RFM table saved to {CLEAN_RFM_PATH} and table 'rfm' in {DB_PATH}")
    return rfm_table

def verify_rfm(df):
    print("\n" + "=" * 65)
    print("  VERIFICATION AGAINST REFERENCE METRICS")
    print("=" * 65)

    n_cust = len(df)
    rec_med = float(df["recency"].median())
    rec_min = int(df["recency"].min())
    rec_max = int(df["recency"].max())

    freq_mean = float(df["frequency"].mean())
    freq_max = int(df["frequency"].max())

    mon_mean = float(df["monetary"].mean())
    mon_max = float(df["monetary"].max())

    rows = [
        ("Customer count", n_cust, 1175),
        ("Recency median", rec_med, 19.0),
        ("Recency minimum", rec_min, 0),
        ("Recency maximum", rec_max, 180),
        ("Frequency mean", f"{freq_mean:.2f}", "9.46"),
        ("Frequency maximum", freq_max, 33),
        ("Monetary mean (INR)", f"{mon_mean:,.2f}", "2,552.06"),
        ("Monetary maximum (INR)", f"{mon_max:,.2f}", "14,766.55"),
    ]

    print(f"{'Metric / Check':<30} {'Got':>15}  {'Expected':>15}  {'OK?':>5}")
    print("-" * 70)
    all_ok = True
    for label, got, exp in rows:
        ok = str(got) == str(exp)
        if not ok:
            all_ok = False
        mark = "✓" if ok else "✗"
        print(f"{label:<30} {str(got):>15}  {str(exp):>15}  {mark:>5}")

    print("-" * 70)
    if all_ok:
        print("ALL REFERENCE METRICS PASSED! ✓\n")
    else:
        print("SOME CHECKS FAILED! ✗\n")

if __name__ == "__main__":
    rfm_df = build_rfm()
    verify_rfm(rfm_df)
    print("Sample (First 10 rows):")
    print(rfm_df.head(10).to_string(index=False))
