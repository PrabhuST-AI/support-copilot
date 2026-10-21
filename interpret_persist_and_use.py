"""interpret_persist_and_use.py -- Model interpretation, persistence, and retention list generation.

Tasks (3 marks):
(a) Report the five most important features of your best Random Forest (feature_importances_)
    and the sign of the coefficient of three features in your best Logistic Regression,
    with a one-line business reading each.
(b) Save your selected model with joblib.dump to outputs/lapse_model.joblib, reload it with
    joblib.load, and show that the reloaded model reproduces the original test predictions exactly.
(c) Among test-set customers who are in the Premium segment, rank by the model's predicted
    lapse probability and write the top 10 (or all, if fewer than 10) to outputs/retention_list.csv
    with columns: customer_id, lapse_probability, actually_lapsed; report how many of them truly lapsed.
    State that the retention list is illustrative and not a leakage-free evaluation because the
    Premium flag is a post-hoc whole-window label.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COHORT_PATH = "outputs/lapse_cohort.csv"
SEGMENTS_PATH = "outputs/customer_segments.csv"
MODEL_SAVE_PATH = "outputs/lapse_model.joblib"
RETENTION_LIST_PATH = "outputs/retention_list.csv"


def main():
    # 1. Load data
    df_cohort = pd.read_csv(COHORT_PATH)
    df_segments = pd.read_csv(SEGMENTS_PATH)

    # Merge post-hoc Premium segment flag from Task 3
    df = df_cohort.merge(
        df_segments[["customer_id", "segment_name", "is_premium"]],
        on="customer_id",
        how="left"
    )

    numeric_cols = [
        "recency", "frequency", "monetary", "avg_discount_pct",
        "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days"
    ]
    categorical_cols = ["city", "gender", "most_frequent_category"]

    X = df[numeric_cols + categorical_cols]
    y = df["lapsed"]

    # Stratified 80/20 train/test split (exact same split as Task 7, 8, 9, 10)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols)
        ]
    )

    # =========================================================================
    # PART (A): FIT MODELS & REPORT FEATURE IMPORTANCES / COEFFICIENTS
    # =========================================================================
    print("=" * 80)
    print("  PART (A): MODEL INTERPRETATION")
    print("=" * 80)

    # Best Random Forest from 5-fold CV tuning: max_depth=8, min_samples_leaf=10
    pipe_rf = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=42))
    ])
    pipe_rf.fit(X_train, y_train)

    # Best Logistic Regression from 5-fold CV tuning: C=100 (or C=1.0)
    pipe_lr = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(C=100, max_iter=1000, random_state=42))
    ])
    pipe_lr.fit(X_train, y_train)

    # Feature names from ColumnTransformer
    feature_names = pipe_rf.named_steps["prep"].get_feature_names_out()
    clean_feature_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]

    # Random Forest Feature Importances
    rf_importances = pipe_rf.named_steps["clf"].feature_importances_
    df_rf_imp = pd.DataFrame({
        "raw_feature": feature_names,
        "feature": clean_feature_names,
        "importance": rf_importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    print("\n--- Top 5 Most Important Features in Best Random Forest ---")
    for idx in range(5):
        row = df_rf_imp.iloc[idx]
        print(f"  {idx+1}. {row['feature']:<25}: {row['importance']:.4f} ({row['importance']*100:.2f}%)")

    # Logistic Regression Coefficients
    lr_coefs = pipe_lr.named_steps["clf"].coef_[0]
    df_lr_coef = pd.DataFrame({
        "raw_feature": feature_names,
        "feature": clean_feature_names,
        "coef": lr_coefs
    }).sort_values(by="coef", key=abs, ascending=False).reset_index(drop=True)

    print("\n--- Key Coefficients in Best Logistic Regression ---")
    key_lr_features = ["frequency", "recency", "avg_delivery_minutes", "most_frequent_category_Electronics"]
    for feat in key_lr_features:
        row = df_lr_coef[df_lr_coef["feature"] == feat].iloc[0]
        sign_str = "Negative (-)" if row["coef"] < 0 else "Positive (+)"
        print(f"  - {row['feature']:<35}: coef = {row['coef']:+.4f} | Sign: {sign_str}")

    # =========================================================================
    # PART (B): SAVE SELECTED MODEL, RELOAD, AND VERIFY EXACT PREDICTIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PART (B): MODEL PERSISTENCE & EXACT REPRODUCTION")
    print("=" * 80)

    # Selected model: Tuned Random Forest (achieving highest test F1 = 0.6554)
    selected_model = pipe_rf
    original_test_preds = selected_model.predict(X_test)
    original_test_probs = selected_model.predict_proba(X_test)

    # Persist with joblib.dump
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(selected_model, MODEL_SAVE_PATH)
    print(f"Successfully saved selected model to: {MODEL_SAVE_PATH}")
    file_size_kb = os.path.getsize(MODEL_SAVE_PATH) / 1024
    print(f"Saved model file size: {file_size_kb:.1f} KB")

    # Reload with joblib.load
    reloaded_model = joblib.load(MODEL_SAVE_PATH)
    print(f"Successfully reloaded model from: {MODEL_SAVE_PATH}")

    # Verify reproduction
    reloaded_test_preds = reloaded_model.predict(X_test)
    reloaded_test_probs = reloaded_model.predict_proba(X_test)

    preds_match = np.array_equal(original_test_preds, reloaded_test_preds)
    probs_match = np.allclose(original_test_probs, reloaded_test_probs, atol=1e-12)

    print(f"Predictions exactly identical: {preds_match}")
    print(f"Probabilities strictly match:  {probs_match}")
    assert preds_match, "Reloaded model must reproduce original test predictions exactly!"
    assert probs_match, "Reloaded model must reproduce original test probabilities exactly!"
    print("--> CHECKPOINT PASSED: Reloaded model perfectly reproduces original test predictions.")

    # =========================================================================
    # PART (C): RANK PREMIUM TEST CUSTOMERS & EXPORT RETENTION LIST
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PART (C): RETENTION LIST FOR PREMIUM SEGMENT")
    print("=" * 80)

    test_df = df.loc[idx_test].copy()
    test_df["lapse_probability"] = reloaded_model.predict_proba(X_test)[:, 1]

    # Filter test-set customers in the Premium segment
    test_premium = test_df[test_df["is_premium"] == True].copy()
    print(f"Total customers in test set: {len(test_df)}")
    print(f"Test customers belonging to Premium segment: {len(test_premium)}")

    # Rank by predicted lapse probability descending
    test_premium_ranked = test_premium.sort_values(by="lapse_probability", ascending=False)

    # Select top 10 (or all if fewer than 10)
    retention_top10 = test_premium_ranked.head(10)[["customer_id", "lapse_probability", "lapsed"]].copy()
    retention_top10.rename(columns={"lapsed": "actually_lapsed"}, inplace=True)

    # Write to outputs/retention_list.csv
    retention_top10.to_csv(RETENTION_LIST_PATH, index=False)
    print(f"Successfully wrote top 10 Premium retention list to: {RETENTION_LIST_PATH}")

    num_truly_lapsed = int(retention_top10["actually_lapsed"].sum())
    print(f"\nTop 10 Retention List Table:")
    print(retention_top10.to_string(index=False))
    print(f"\nNumber of customers in the top 10 who truly lapsed: {num_truly_lapsed} / 10 ({num_truly_lapsed * 10}%)")

    # Also calculate for Logistic Regression for comprehensive comparison
    test_df["lapse_prob_lr"] = pipe_lr.predict_proba(X_test)[:, 1]
    test_prem_lr = test_df[test_df["is_premium"] == True].sort_values(by="lapse_prob_lr", ascending=False).head(10)
    num_lr_lapsed = int(test_prem_lr["lapsed"].sum())
    print(f"(Comparative Note: With tuned Logistic Regression, {num_lr_lapsed} / 10 truly lapsed.)")

    statement = (
        "The retention list is illustrative and not a leakage-free evaluation because the Premium "
        "customer segment flag is a post-hoc whole-window label derived from RFM clustering across "
        "the entire observation window (including the September window that defines the lapse target label)."
    )
    print("\n" + "=" * 80)
    print("  LEAKAGE STATEMENT")
    print("=" * 80)
    print(statement)
    print("=" * 80)


if __name__ == "__main__":
    main()
