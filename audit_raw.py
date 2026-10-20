"""audit_raw.py -- Audit the three raw tables and report shape, dtypes,
missing-value summaries, and the specific figures required by the brief."""

import pandas as pd

RAW = "data/raw"

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def is_blank(series):
    """True where value is NaN OR empty/whitespace string."""
    return series.isna() | series.fillna("").str.strip().eq("")

def missing_report(df):
    total = len(df)
    rows = []
    for col in df.columns:
        n = int(is_blank(df[col]).sum())
        rows.append({"column": col, "missing_count": n, "missing_pct": f"{100*n/total:.1f}%"})
    print(pd.DataFrame(rows).to_string(index=False))

def to_float(v):
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False

def to_int(v):
    try:
        return int(v)
    except (ValueError, TypeError):
        return None

# ──────────────────────────────────────────────
# Load — keep_default_na=False so "" stays ""
# ──────────────────────────────────────────────
orders    = pd.read_csv(f"{RAW}/orders.csv",    dtype=str, keep_default_na=False)
customers = pd.read_csv(f"{RAW}/customers.csv", dtype=str, keep_default_na=False)
tickets   = pd.read_csv(f"{RAW}/tickets.csv",   dtype=str, keep_default_na=False)

# ──────────────────────────────────────────────
# ORDERS
# ──────────────────────────────────────────────
section("ORDERS — shape, dtypes, missing values")
print(f"Shape : {orders.shape}")
print(f"\nData types:\n{orders.dtypes.to_string()}")
print("\nMissing / blank per column:")
missing_report(orders)

section("ORDERS — specific audit figures")
total_rows      = len(orders)
exact_dupes     = int(orders.duplicated().sum())
pending_rows    = int((orders["status"] == "pending").sum())
distinct_oid    = int(orders["order_id"].nunique())

# non-numeric: blank counts too (blank is not a number)
non_numeric_amt = int((~orders["order_amount"].apply(to_float)).sum())

# negative delivery_minutes (NaN/blank not counted)
neg_deliv = int(orders["delivery_minutes"].apply(
    lambda x: (v := to_int(x)) is not None and v < 0).sum())

# blank delivery_minutes
blank_deliv = int(is_blank(orders["delivery_minutes"]).sum())

rows = [
    ("Total rows",                             total_rows,      13_084),
    ("Exact duplicate rows",                   exact_dupes,        119),
    ("Rows with status=pending",               pending_rows,       656),
    ("Distinct order_id values",               distinct_oid,    12_309),
    ("Non-numeric order_amount entries",       non_numeric_amt,    120),
    ("Negative delivery_minutes",              neg_deliv,          106),
    ("Blank delivery_minutes",                 blank_deliv,      2_541),
]
print(f"\n{'Metric':<40} {'Got':>8}  {'Expected':>10}  {'OK?':>5}")
print("-" * 68)
for label, got, exp in rows:
    ok = "✓" if got == exp else "✗"
    print(f"{label:<40} {got:>8}  {exp:>10}  {ok:>5}")

# ──────────────────────────────────────────────
# CUSTOMERS
# ──────────────────────────────────────────────
section("CUSTOMERS — shape, dtypes, missing values")
print(f"Shape : {customers.shape}")
print(f"\nData types:\n{customers.dtypes.to_string()}")
print("\nMissing / blank per column:")
missing_report(customers)

section("CUSTOMERS — specific audit figures")
blank_age        = int(is_blank(customers["age"]).sum())
age_outliers     = int(customers["age"].apply(
    lambda v: (a := to_int(v)) is not None and (a < 10 or a > 100)).sum())
raw_city_spell   = int(customers["city"].nunique())
non_blank_gender = customers["gender"][~is_blank(customers["gender"])]
raw_gender_spell = int(non_blank_gender.nunique())
blank_gender     = int(is_blank(customers["gender"]).sum())

cust_rows = [
    ("Blank age",                               blank_age,        61),
    ("Age values outside 10–100",              age_outliers,      6),
    ("Distinct raw city spellings",             raw_city_spell,   24),
    ("Distinct non-blank raw gender spellings", raw_gender_spell, 10),
    ("Blank gender",                            blank_gender,     30),
]
print(f"\n{'Metric':<45} {'Got':>6}  {'Expected':>10}  {'OK?':>5}")
print("-" * 70)
for label, got, exp in cust_rows:
    ok = "✓" if got == exp else "✗"
    print(f"{label:<45} {got:>6}  {exp:>10}  {ok:>5}")

# ──────────────────────────────────────────────
# TICKETS
# ──────────────────────────────────────────────
section("TICKETS — shape, dtypes, missing values")
print(f"Shape : {tickets.shape}")
print(f"\nData types:\n{tickets.dtypes.to_string()}")
print("\nMissing / blank per column:")
missing_report(tickets)
print(f"\nNote: loaded with keep_default_na=False — empty text stays as '' not NaN")
print(f"  Category distribution:\n{tickets['true_category'].value_counts().to_string()}")
