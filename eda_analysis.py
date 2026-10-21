"""eda_analysis.py -- pandas cross-check and exploratory data analysis.
(a) Rebuild Q2 with pd.pivot_table(..., margins=True) and verify grand total;
    Groupby and named aggregation per city (sum, mean, count).
(b) Distribution stats for completed order_amount (mean, median, skewness, quartiles,
    1.5xIQR fences, outlier counts), and correlation matrix.
"""

import sqlite3
import pandas as pd

DB_PATH = "data/instamart.db"

def run_eda():
    conn = sqlite3.connect(DB_PATH)
    orders = pd.read_sql_query("SELECT * FROM orders WHERE status = 'completed'", conn)
    customers = pd.read_sql_query("SELECT customer_id, city FROM customers", conn)
    products = pd.read_sql_query("SELECT product_id, category FROM products", conn)
    conn.close()

    df = orders.merge(customers, on="customer_id").merge(products, on="product_id")

    print("=" * 75)
    print("  PART (A): PANDAS CROSS-CHECK (Q2 REBUILD & CITY NAMED AGGREGATION)")
    print("=" * 75)

    # Rebuild Q2 with pivot_table(..., margins=True)
    pt = pd.pivot_table(
        df,
        values="order_amount",
        index="city",
        columns="category",
        aggfunc="sum",
        margins=True
    )
    grand_total = pt.loc["All", "All"]
    print("Revenue Pivot Table by City and Category (with Margins):\n")
    print(pt.map(lambda v: f"{v:,.2f}").to_string())
    print(f"\nPivot Table Grand Total: INR {grand_total:,.2f} (Expected: INR 2,998,668.80)")
    assert abs(grand_total - 2998668.80) < 0.01, "Grand total mismatch"

    # Groupby with named aggregation per city
    city_agg = df.groupby("city").agg(
        sum_order_amount=("order_amount", "sum"),
        mean_order_amount=("order_amount", "mean"),
        count_orders=("order_amount", "count")
    ).reset_index()

    print("\nCompleted Order Amount Aggregations by City:")
    display_city = city_agg.copy()
    display_city["sum_order_amount"] = display_city["sum_order_amount"].apply(lambda v: f"{v:,.2f}")
    display_city["mean_order_amount"] = display_city["mean_order_amount"].apply(lambda v: f"{v:,.2f}")
    display_city["count_orders"] = display_city["count_orders"].apply(lambda v: f"{v:,}")
    print(display_city.to_string(index=False))

    # Verify reference values
    bengaluru = city_agg[city_agg["city"] == "Bengaluru"].iloc[0]
    mumbai = city_agg[city_agg["city"] == "Mumbai"].iloc[0]
    assert abs(bengaluru["sum_order_amount"] - 523643.40) < 0.01
    assert abs(bengaluru["mean_order_amount"] - 276.47) < 0.01
    assert bengaluru["count_orders"] == 1894
    assert abs(mumbai["sum_order_amount"] - 479058.50) < 0.01
    assert abs(mumbai["mean_order_amount"] - 297.00) < 0.01
    assert mumbai["count_orders"] == 1613

    print("\n" + "=" * 75)
    print("  PART (B): EXPLORATORY ANALYSIS (DISTRIBUTIONS, OUTLIERS & CORRELATIONS)")
    print("=" * 75)

    amt = df["order_amount"]
    mean_val = amt.mean()
    median_val = amt.median()
    skew_val = amt.skew()
    q1 = amt.quantile(0.25)
    q3 = amt.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers_above = int((amt > upper_fence).sum())
    outliers_below = int((amt < lower_fence).sum())

    print(f"Mean:         INR {mean_val:.2f}")
    print(f"Median:       INR {median_val:.2f} (Right-skewed: Mean > Median)")
    print(f"Skewness:     {skew_val:.2f}")
    print(f"Q1 (25%):     INR {q1:.2f}")
    print(f"Q3 (75%):     INR {q3:.2f}")
    print(f"IQR:          INR {iqr:.2f}")
    print(f"Lower Fence:  INR {lower_fence:.2f} -> Outliers below: {outliers_below}")
    print(f"Upper Fence:  INR {upper_fence:.2f} -> Outliers above: {outliers_above}")

    # Correlation matrix
    corr_cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
    corr = df[corr_cols].corr()
    print("\nCorrelation Matrix (Completed Orders):")
    print(corr.round(4).to_string())

    print("\n" + "=" * 75)
    print("ALL CHECKS AND EXPLORATORY ANALYSIS COMPLETED SUCCESSFULLY! ✓")
    print("=" * 75)

if __name__ == "__main__":
    run_eda()
