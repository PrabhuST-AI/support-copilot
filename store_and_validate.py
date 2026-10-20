"""store_and_validate.py -- Store clean tables to CSV and SQLite, then validate checkpoints.
Tasks:
  1. Write the three clean tables to data/clean/ as CSV files:
     - customers.csv
     - products.csv
     - orders.csv
  2. Write to SQLite database data/instamart.db as tables named:
     - customers
     - products
     - orders (order_ts stored as text in the form YYYY-MM-DD HH:MM:SS)
  3. Validate reference checkpoints from tasks 3 & 4.
  4. Print exactly: ALL CHECKPOINTS PASSED
"""

import os
import sqlite3
import sys
import pandas as pd

DB_PATH = "data/instamart.db"
CLEAN_DIR = "data/clean"

def store_clean_tables():
    """Ensure data/clean/ CSVs and data/instamart.db tables are populated."""
    os.makedirs(CLEAN_DIR, exist_ok=True)

    # 1. Customers
    cust_path = os.path.join(CLEAN_DIR, "customers.csv")
    if not os.path.exists(cust_path) or os.path.getsize(cust_path) == 0:
        from clean_customers import clean_customers
        customers = clean_customers()
    else:
        customers = pd.read_csv(cust_path)
    customers["signup_date"] = pd.to_datetime(customers["signup_date"]).dt.strftime("%Y-%m-%d")
    customers.to_csv(cust_path, index=False)

    # 2. Products
    prod_path = os.path.join(CLEAN_DIR, "products.csv")
    if not os.path.exists(prod_path) or os.path.getsize(prod_path) == 0:
        products = pd.read_csv("data/raw/products.csv")
        products.to_csv(prod_path, index=False)
    else:
        products = pd.read_csv(prod_path)

    # 3. Orders
    ord_path = os.path.join(CLEAN_DIR, "orders.csv")
    if not os.path.exists(ord_path) or os.path.getsize(ord_path) == 0:
        from clean_orders import clean_orders
        orders = clean_orders()
    else:
        orders = pd.read_csv(ord_path)

    # Format order_ts as text YYYY-MM-DD HH:MM:SS
    orders["order_ts"] = pd.to_datetime(orders["order_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    orders.to_csv(ord_path, index=False)

    # Store to SQLite instamart.db
    conn = sqlite3.connect(DB_PATH)
    customers.to_sql("customers", conn, if_exists="replace", index=False)
    products.to_sql("products", conn, if_exists="replace", index=False)
    orders.to_sql("orders", conn, if_exists="replace", index=False)
    conn.close()

def run_validations(verbose=False):
    """Assert checkpoints from tasks 3 & 4 against instamart.db and clean CSVs."""
    # Ensure database file exists
    assert os.path.exists(DB_PATH), f"Database file {DB_PATH} not found"

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Checkpoint 1 (Task 3a): Exactly 6 cities with exact distributions
    cur.execute("SELECT city, COUNT(*) FROM customers GROUP BY city")
    city_counts = dict(cur.fetchall())
    expected_cities = {
        "Pune": 227, "Delhi": 210, "Bengaluru": 197,
        "Chennai": 191, "Hyderabad": 191, "Mumbai": 184
    }
    assert city_counts == expected_cities, f"City counts mismatch: {city_counts}"
    if verbose:
        print("[Checkpoint 1] 6 cities match reference.")

    # Checkpoint 2 (Task 3b): Genders female 610, male 560, unknown 30
    cur.execute("SELECT gender, COUNT(*) FROM customers GROUP BY gender")
    gender_counts = dict(cur.fetchall())
    expected_genders = {"female": 610, "male": 560, "unknown": 30}
    assert gender_counts == expected_genders, f"Gender counts mismatch: {gender_counts}"
    if verbose:
        print("[Checkpoint 2] Genders match reference.")

    # Checkpoint 3 (Task 3c): 0 missing ages
    cur.execute("SELECT COUNT(*) FROM customers WHERE age IS NULL")
    missing_ages = cur.fetchone()[0]
    assert missing_ages == 0, f"Expected 0 missing ages, got {missing_ages}"
    if verbose:
        print("[Checkpoint 3] 0 missing ages.")

    # Checkpoint 4 (Task 3c): All customer ages between 18 and 60
    cur.execute("SELECT MIN(age), MAX(age) FROM customers")
    min_age, max_age = cur.fetchone()
    cur.execute("SELECT COUNT(*) FROM customers WHERE age < 18 OR age > 60")
    invalid_ages = cur.fetchone()[0]
    assert invalid_ages == 0 and min_age >= 18 and max_age <= 60, (
        f"Age bounds failed: min={min_age}, max={max_age}, invalid={invalid_ages}"
    )
    if verbose:
        print("[Checkpoint 4] Customer ages between 18 and 60.")

    # Checkpoint 5 (Task 4): Exactly 12,309 rows (one per order_id)
    cur.execute("SELECT COUNT(*), COUNT(DISTINCT order_id) FROM orders")
    total_rows, distinct_orders = cur.fetchone()
    assert total_rows == 12309 and distinct_orders == 12309, (
        f"Orders count mismatch: total={total_rows}, distinct={distinct_orders}"
    )
    if verbose:
        print("[Checkpoint 5] Exactly 12,309 distinct orders.")

    # Checkpoint 6 (Task 4): 11,110 completed and 1,199 cancelled
    cur.execute("SELECT status, COUNT(*) FROM orders GROUP BY status")
    status_counts = dict(cur.fetchall())
    assert status_counts == {"completed": 11110, "cancelled": 1199}, (
        f"Status counts mismatch: {status_counts}"
    )
    if verbose:
        print("[Checkpoint 6] 11,110 completed and 1,199 cancelled orders.")

    # Checkpoint 7 (Task 4): 0 pending rows
    cur.execute("SELECT COUNT(*) FROM orders WHERE status = 'pending'")
    pending_count = cur.fetchone()[0]
    assert pending_count == 0, f"Found pending orders: {pending_count}"
    if verbose:
        print("[Checkpoint 7] 0 pending orders.")

    # Checkpoint 8 (Task 4): 0 missing order_amount
    cur.execute("SELECT COUNT(*) FROM orders WHERE order_amount IS NULL")
    missing_amount = cur.fetchone()[0]
    assert missing_amount == 0, f"Found missing order_amount: {missing_amount}"
    if verbose:
        print("[Checkpoint 8] 0 missing order_amount.")

    # Checkpoint 9 (Task 4): 0 missing delivery_minutes among completed orders
    cur.execute("SELECT COUNT(*) FROM orders WHERE status = 'completed' AND delivery_minutes IS NULL")
    missing_deliv_comp = cur.fetchone()[0]
    assert missing_deliv_comp == 0, f"Found missing delivery_minutes in completed: {missing_deliv_comp}"
    if verbose:
        print("[Checkpoint 9] 0 missing delivery_minutes in completed orders.")

    # Checkpoint 10 (Task 4): 1,199 missing delivery_minutes among cancelled orders
    cur.execute("SELECT COUNT(*) FROM orders WHERE status = 'cancelled' AND delivery_minutes IS NULL")
    missing_deliv_canc = cur.fetchone()[0]
    assert missing_deliv_canc == 1199, (
        f"Expected 1,199 missing delivery_minutes in cancelled, got {missing_deliv_canc}"
    )
    if verbose:
        print("[Checkpoint 10] 1,199 missing delivery_minutes in cancelled orders.")

    # Checkpoint 11 (Task 4): Sum of order_amount over completed orders is INR 2,998,668.80
    cur.execute("SELECT ROUND(SUM(order_amount), 2) FROM orders WHERE status = 'completed'")
    sum_completed = cur.fetchone()[0]
    assert abs(sum_completed - 2998668.80) < 0.01, (
        f"Expected sum 2998668.80, got {sum_completed}"
    )
    if verbose:
        print(f"[Checkpoint 11] Sum order_amount over completed: {sum_completed:,.2f}.")

    # Checkpoint 12 (Task 5 requirement): order_ts is text formatted YYYY-MM-DD HH:MM:SS
    cur.execute("SELECT typeof(order_ts) FROM orders LIMIT 1")
    assert cur.fetchone()[0] == "text", "order_ts is not stored as text in SQLite"
    cur.execute(
        "SELECT COUNT(*) FROM orders WHERE length(order_ts) != 19 "
        "OR substr(order_ts, 5, 1) != '-' OR substr(order_ts, 8, 1) != '-' "
        "OR substr(order_ts, 11, 1) != ' '"
    )
    invalid_ts = cur.fetchone()[0]
    assert invalid_ts == 0, f"Found {invalid_ts} non-conforming order_ts values"
    if verbose:
        print("[Checkpoint 12] order_ts stored as text in YYYY-MM-DD HH:MM:SS format.")

    conn.close()

if __name__ == "__main__":
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    store_clean_tables()
    run_validations(verbose=verbose)
    print("ALL CHECKPOINTS PASSED")
