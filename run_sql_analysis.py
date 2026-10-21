"""run_sql_analysis.py -- Execute all 7 SQL queries from sql/queries.sql
against data/instamart.db using pd.read_sql_query and report results.
"""

import sqlite3
import pandas as pd

DB_PATH = "data/instamart.db"

def run_queries():
    conn = sqlite3.connect(DB_PATH)

    print("=" * 75)
    print("  INSTAMART SQL ANALYSIS (7 QUERIES)")
    print("=" * 75)

    # --------------------------------------------------------------------------
    # Q1: WHERE then HAVING
    # --------------------------------------------------------------------------
    print("\n--- Q1: WHERE then HAVING ---")
    q1 = """
    SELECT 
        customer_id, 
        ROUND(SUM(order_amount), 2) AS total_spend
    FROM orders
    WHERE status = 'completed' AND order_amount > 500
    GROUP BY customer_id
    HAVING SUM(order_amount) > 8000
    ORDER BY total_spend DESC;
    """
    df1 = pd.read_sql_query(q1, conn)
    print(f"Result count: {len(df1)} customers (Reference: 30)")
    print(f"All 30 customers matching criteria:")
    print(df1.to_string(index=False))
    assert len(df1) == 30, f"Q1 expected 30, got {len(df1)}"

    # --------------------------------------------------------------------------
    # Q2: Three-Table Join
    # --------------------------------------------------------------------------
    print("\n--- Q2: Three-Table Join ---")
    q2 = """
    SELECT 
        c.city, 
        p.category, 
        ROUND(SUM(o.order_amount), 2) AS revenue
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN products p ON o.product_id = p.product_id
    WHERE o.status = 'completed'
    GROUP BY c.city, p.category
    ORDER BY revenue DESC;
    """
    df2 = pd.read_sql_query(q2, conn)
    total_rev = df2["revenue"].sum()
    top_cell = df2.iloc[0]
    print(f"Result count: {len(df2)} rows (Reference: 30)")
    print(f"Total revenue across all 30 cells: INR {total_rev:,.2f} (Reference: 2,998,668.80)")
    print(f"Largest cell: {top_cell['city']} / {top_cell['category']} at INR {top_cell['revenue']:,.2f} (Reference: Bengaluru / Electronics at 233,909.45)")
    print("\nPivot Table (City x Category Revenue in INR):")
    pivot2 = df2.pivot(index="city", columns="category", values="revenue").fillna(0)
    print(pivot2.map(lambda v: f"{v:,.2f}").to_string())
    assert len(df2) == 30
    assert abs(total_rev - 2998668.80) < 0.01
    assert top_cell["city"] == "Bengaluru" and top_cell["category"] == "Electronics" and abs(top_cell["revenue"] - 233909.45) < 0.01

    # --------------------------------------------------------------------------
    # Q3: Scalar Subquery
    # --------------------------------------------------------------------------
    print("\n--- Q3: Scalar Subquery ---")
    q3 = """
    SELECT 
        product_id, 
        product_name, 
        category, 
        price_inr
    FROM products
    WHERE price_inr > (SELECT AVG(price_inr) FROM products)
    ORDER BY price_inr DESC;
    """
    df3 = pd.read_sql_query(q3, conn)
    avg_price = pd.read_sql_query("SELECT ROUND(AVG(price_inr), 2) FROM products", conn).iloc[0, 0]
    print(f"Average catalogue price: INR {avg_price:.2f}")
    print(f"Products priced above average: {len(df3)} products (Reference: 11)")
    print(df3.to_string(index=False))
    assert len(df3) == 11, f"Q3 expected 11, got {len(df3)}"

    # --------------------------------------------------------------------------
    # Q4: LEFT JOIN
    # --------------------------------------------------------------------------
    print("\n--- Q4: LEFT JOIN ---")
    q4 = """
    SELECT 
        c.customer_id, 
        c.city, 
        c.signup_date
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
    WHERE o.order_id IS NULL
    ORDER BY c.customer_id ASC;
    """
    df4 = pd.read_sql_query(q4, conn)
    print(f"Customers with no completed orders: {len(df4)} customers (Reference: 25)")
    print(df4.to_string(index=False))
    assert len(df4) == 25, f"Q4 expected 25, got {len(df4)}"

    # --------------------------------------------------------------------------
    # Q5: Common Table Expression (CTE)
    # --------------------------------------------------------------------------
    print("\n--- Q5: Common Table Expression (CTE) ---")
    q5 = """
    WITH customer_spend AS (
        SELECT 
            customer_id, 
            SUM(order_amount) AS total_spend
        FROM orders
        WHERE status = 'completed'
        GROUP BY customer_id
    ),
    avg_spend AS (
        SELECT 
            AVG(total_spend) AS avg_total_spend
        FROM customer_spend
    )
    SELECT 
        cs.customer_id, 
        ROUND(cs.total_spend, 2) AS total_spend,
        ROUND(a.avg_total_spend, 2) AS benchmark_avg
    FROM customer_spend cs, avg_spend a
    WHERE cs.total_spend > a.avg_total_spend
    ORDER BY cs.total_spend DESC;
    """
    df5 = pd.read_sql_query(q5, conn)
    avg_val = df5["benchmark_avg"].iloc[0]
    print(f"Benchmark average completed customer spend: INR {avg_val:,.2f} (Reference: 2,552.06)")
    print(f"Customers above average: {len(df5)} customers (Reference: 550)")
    print(f"Top 5 spenders above average:\n{df5.head(5).to_string(index=False)}")
    assert len(df5) == 550, f"Q5 expected 550, got {len(df5)}"
    assert abs(avg_val - 2552.06) < 0.01

    # --------------------------------------------------------------------------
    # Q6: Ranking Window Functions
    # --------------------------------------------------------------------------
    print("\n--- Q6: Ranking Window Functions ---")
    q6 = """
    WITH cust_spend AS (
        SELECT 
            c.customer_id, 
            c.city, 
            ROUND(SUM(o.order_amount), 2) AS total_spend
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        WHERE o.status = 'completed'
        GROUP BY c.customer_id, c.city
    )
    SELECT
        customer_id,
        city,
        total_spend,
        RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
        DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
        ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
    FROM cust_spend
    ORDER BY city ASC, total_spend DESC;
    """
    df6 = pd.read_sql_query(q6, conn)
    top_per_city = df6[df6["row_num"] == 1].sort_values("city")
    print("Top customer per city:")
    for _, row in top_per_city.iterrows():
        print(f"  {row['city']} {row['customer_id']} ({row['total_spend']:,.2f})")

    diff_rows = df6[df6["rnk"] != df6["row_num"]]
    print(f"\nRows where RANK() != ROW_NUMBER(): {len(diff_rows)} rows (Reference: 8)")
    print(diff_rows[["city", "customer_id", "total_spend", "rnk", "dense_rnk", "row_num"]].to_string(index=False))

    expected_tops = {
        "Bengaluru": ("C0082", 12856.20),
        "Chennai": ("C0930", 10151.40),
        "Delhi": ("C0291", 13742.60),
        "Hyderabad": ("C0743", 12730.05),
        "Mumbai": ("C0693", 14766.55),
        "Pune": ("C0426", 10377.10),
    }
    for _, row in top_per_city.iterrows():
        exp_cid, exp_amt = expected_tops[row["city"]]
        assert row["customer_id"] == exp_cid and abs(row["total_spend"] - exp_amt) < 0.01
    assert len(diff_rows) == 8, f"Q6 expected 8 diff rows, got {len(diff_rows)}"

    # --------------------------------------------------------------------------
    # Q7: Running Total
    # --------------------------------------------------------------------------
    print("\n--- Q7: Running Total ---")
    q7 = """
    WITH daily_revenue AS (
        SELECT
            SUBSTR(order_ts, 1, 10) AS day,
            ROUND(SUM(order_amount), 2) AS daily_revenue
        FROM orders
        WHERE status = 'completed'
        GROUP BY SUBSTR(order_ts, 1, 10)
    )
    SELECT
        day,
        daily_revenue,
        ROUND(SUM(daily_revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
    FROM daily_revenue
    ORDER BY day ASC;
    """
    df7 = pd.read_sql_query(q7, conn)
    last_val = df7["cumulative_revenue"].iloc[-1]
    print(f"Total daily rows: {len(df7)} rows (Reference: 183)")
    print(f"First day: {df7.iloc[0]['day']} | Daily: INR {df7.iloc[0]['daily_revenue']:,.2f} | Cum: INR {df7.iloc[0]['cumulative_revenue']:,.2f}")
    print(f"Last day:  {df7.iloc[-1]['day']} | Daily: INR {df7.iloc[-1]['daily_revenue']:,.2f} | Cum: INR {last_val:,.2f} (Reference: 2,998,668.80)")
    assert len(df7) == 183, f"Q7 expected 183 rows, got {len(df7)}"
    assert abs(last_val - 2998668.80) < 0.01

    conn.close()
    print("\n" + "=" * 75)
    print("ALL 7 SQL QUERIES EXECUTED AND VALIDATED SUCCESSFULLY! ✓")
    print("=" * 75)

if __name__ == "__main__":
    run_queries()
