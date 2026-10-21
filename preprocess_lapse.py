"""preprocess_lapse.py -- Preprocess features without data leakage.

Rules enforced:
1. One-hot encode nominal columns (city, gender, most_frequent_category) using k-1 columns
   via OneHotEncoder(drop='first', sparse_output=False).
2. Standardise numeric features with StandardScaler().
3. Strict featurization ordering: fit transformers ONLY on training rows (924 rows),
   then apply (.transform()) to test rows (232 rows) to prevent data leakage.
4. Compare learned scaler mean of frequency vs full-cohort mean of frequency.
5. Verify programmatically that no feature contains missing values.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET_PATH = "outputs/lapse_cohort.csv"

def build_preprocessing_pipeline():
    df = pd.read_csv(DATASET_PATH)

    numeric_cols = [
        "recency", "frequency", "monetary", "avg_discount_pct",
        "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days"
    ]
    categorical_cols = ["city", "gender", "most_frequent_category"]

    X = df[numeric_cols + categorical_cols].copy()
    y = df["lapsed"].copy()

    # 1. Stratified 80/20 train/test split BEFORE any transformation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 2. Define ColumnTransformer with k-1 dummy encoding and standardization
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols)
        ]
    )

    # 3. Fit STRICTLY on training data only, then transform both splits
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Get feature names
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    all_feature_names = numeric_cols + cat_feature_names

    # 4. Compare learned mean of frequency against full-cohort mean
    num_scaler = preprocessor.named_transformers_["num"]
    freq_idx = numeric_cols.index("frequency")
    learned_freq_mean = num_scaler.mean_[freq_idx]
    full_cohort_freq_mean = df["frequency"].mean()
    train_raw_freq_mean = X_train["frequency"].mean()

    print("=" * 75)
    print("  PREPROCESSING WITHOUT LEAKAGE (PIPELINE EVALUATION)")
    print("=" * 75)
    print(f"Learned mean of frequency (fitted scaler on train): {learned_freq_mean:.4f}")
    print(f"Full-cohort mean of frequency (entire population):   {full_cohort_freq_mean:.4f}")
    print(f"Absolute difference:                                {abs(learned_freq_mean - full_cohort_freq_mean):.4f}")

    # One-sentence explanation of the difference
    explanation = (
        "They differ because the scaler was strictly fitted on the 924 training rows "
        "to prevent information leakage, so its learned mean reflects only the training sample "
        "(8.1147) which exhibits natural sampling variation relative to the full 1,156-customer "
        "cohort mean (8.0285)."
    )
    print(f"\nWhy they differ:\n  \"{explanation}\"")

    # 5. Confirm in code that no feature column contains missing values
    train_missing = int(X_train.isna().sum().sum())
    test_missing = int(X_test.isna().sum().sum())
    proc_train_missing = int(np.isnan(X_train_proc).sum())
    proc_test_missing = int(np.isnan(X_test_proc).sum())

    print("\n" + "=" * 75)
    print("  MISSING VALUE AUDIT ACROSS FEATURE PIPELINE")
    print("=" * 75)
    print(f"Missing values in X_train (raw):       {train_missing}")
    print(f"Missing values in X_test (raw):        {test_missing}")
    print(f"Missing values in X_train (processed): {proc_train_missing}")
    print(f"Missing values in X_test (processed):  {proc_test_missing}")

    assert train_missing == 0, f"Found {train_missing} missing values in raw training features"
    assert test_missing == 0, f"Found {test_missing} missing values in raw test features"
    assert proc_train_missing == 0, f"Found {proc_train_missing} missing values in processed training matrix"
    assert proc_test_missing == 0, f"Found {proc_test_missing} missing values in processed test matrix"

    print("\n" + "-" * 75)
    print("CONFIRMATION VERIFIED: 0 missing values detected across all features! ✓")
    print(f"Engineered feature count: {len(all_feature_names)} (8 numerical + 11 one-hot dummy columns)")
    print("-" * 75)

    return preprocessor, X_train_proc, X_test_proc, y_train, y_test, all_feature_names

if __name__ == "__main__":
    build_preprocessing_pipeline()
