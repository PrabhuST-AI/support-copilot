"""clean_customers.py -- Clean customers dataset according to specification:
  (a) Standardise city (strip spaces, title case -> 6 cities)
  (b) Map gender to male, female, unknown (blank -> unknown)
  (c) Treat ages outside 10-100 as missing, fill missing with city median via groupby + transform
  (d) Convert signup_date to datetime
Saves cleaned dataset to data/clean/customers.csv and verifies against reference metrics.
"""

import os
import pandas as pd

RAW_PATH = "data/raw/customers.csv"
CLEAN_PATH = "data/clean/customers.csv"

def clean_customers():
    # Load raw customers (keep_default_na=False to preserve empty strings)
    df = pd.read_csv(RAW_PATH, keep_default_na=False)
    print(f"Loaded raw customers: {len(df)} rows")

    # (a) Standardise city by removing leading/trailing spaces and applying title case
    df["city"] = df["city"].astype(str).str.strip().str.title()

    # (b) Map every gender spelling to male, female, or unknown (blank -> unknown)
    gender_norm = df["gender"].astype(str).str.strip().str.lower()
    gender_map = {
        "m": "male",
        "male": "male",
        "f": "female",
        "female": "female",
    }
    df["gender"] = gender_norm.map(gender_map).fillna("unknown")

    # (c) Treat ages outside 10–100 as missing, then fill all missing ages with the
    # median age of the customer's city, computed with groupby and transform.
    age_num = pd.to_numeric(df["age"], errors="coerce")
    df["age"] = age_num.where((age_num >= 10) & (age_num <= 100))
    city_median_age = df.groupby("city")["age"].transform("median")
    df["age"] = df["age"].fillna(city_median_age)

    # (d) Convert signup_date to a datetime
    df["signup_date"] = pd.to_datetime(df["signup_date"])

    # Ensure output directory exists and save cleaned data
    os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Saved cleaned customers to {CLEAN_PATH}")

    return df

def verify_cleaning(df):
    print("\n" + "=" * 65)
    print("  VERIFICATION AGAINST REFERENCE METRICS")
    print("=" * 65)

    # Reference metrics
    ref_cities = {
        "Pune": 227,
        "Delhi": 210,
        "Bengaluru": 197,
        "Chennai": 191,
        "Hyderabad": 191,
        "Mumbai": 184,
    }

    ref_genders = {
        "female": 610,
        "male": 560,
        "unknown": 30,
    }

    city_counts = df["city"].value_counts().to_dict()
    gender_counts = df["gender"].value_counts().to_dict()
    missing_age = int(df["age"].isna().sum())
    min_age = float(df["age"].min())
    max_age = float(df["age"].max())
    all_ages_in_range = bool(((df["age"] >= 18) & (df["age"] <= 60)).all())
    signup_is_datetime = bool(pd.api.types.is_datetime64_any_dtype(df["signup_date"]))

    rows = []
    # City checks
    rows.append(("Number of distinct cities", len(city_counts), 6))
    for city, exp in ref_cities.items():
        rows.append((f"City: {city}", city_counts.get(city, 0), exp))

    # Gender checks
    for g, exp in ref_genders.items():
        rows.append((f"Gender: {g}", gender_counts.get(g, 0), exp))

    # Age checks
    rows.append(("Missing age count", missing_age, 0))
    rows.append(("Ages >= 18 and <= 60 (all valid)", all_ages_in_range, True))
    rows.append(("Minimum age", min_age, 18.0))
    rows.append(("Maximum age", max_age, 60.0))

    # Datetime check
    rows.append(("signup_date is datetime", signup_is_datetime, True))

    print(f"{'Check / Metric':<36} {'Got':>10}  {'Expected':>10}  {'OK?':>5}")
    print("-" * 67)
    all_passed = True
    for label, got, exp in rows:
        ok = got == exp
        if not ok:
            all_passed = False
        mark = "✓" if ok else "✗"
        print(f"{label:<36} {str(got):>10}  {str(exp):>10}  {mark:>5}")

    print("-" * 67)
    if all_passed:
        print("ALL REFERENCE METRICS PASSED SUCCESSFULLY! ✓\n")
    else:
        print("SOME CHECKS FAILED! ✗\n")

if __name__ == "__main__":
    cleaned_df = clean_customers()
    verify_cleaning(cleaned_df)
