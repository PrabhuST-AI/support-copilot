"""class_balance_experiment.py -- Class-balance oversampling experiment for customer lapse prediction.

Tasks (2 marks):
1. Over-sample the training set only:
   - Keep every original training row.
   - Add randomly chosen duplicates of lapsed training rows (drawn with replacement, random_state=42)
     until both classes have equal counts (570 each in the reference split, total 1,140 rows).
2. Refit the best Logistic Regression model on the balanced training data.
3. Compare precision, recall, and F1 on the original, unchanged test set (N=232) with the un-balanced model.
4. State which metric moved most and explain what that means for the retention team.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET_PATH = "outputs/lapse_cohort.csv"


def run_class_balance_experiment():
    df = pd.read_csv(DATASET_PATH)

    numeric_cols = [
        "recency", "frequency", "monetary", "avg_discount_pct",
        "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days"
    ]
    categorical_cols = ["city", "gender", "most_frequent_category"]

    X = df[numeric_cols + categorical_cols]
    y = df["lapsed"]

    # Stratified 80/20 train/test split (same random_state=42 as baseline)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols)
        ]
    )

    # 1. Inspect original training split
    train_df = pd.concat([X_train, y_train], axis=1)
    train_0 = train_df[train_df["lapsed"] == 0]
    train_1 = train_df[train_df["lapsed"] == 1]
    n_0 = len(train_0)
    n_1 = len(train_1)
    n_diff = n_0 - n_1

    print("=" * 80)
    print("  STEP 1: TRAINING SET CLASS DISTRIBUTION & OVERSAMPLING")
    print("=" * 80)
    print(f"Original Training Set: Total = {len(train_df)}")
    print(f"  - Non-lapsed (class 0): {n_0} rows ({n_0 / len(train_df) * 100:.1f}%)")
    print(f"  - Lapsed (class 1):     {n_1} rows ({n_1 / len(train_df) * 100:.1f}%)")
    print(f"  - Deficit to balance:   {n_diff} rows")

    # Over-sample the training set only:
    # Keep every original training row and add randomly chosen duplicates of lapsed training rows (with replacement, random_state=42)
    lapsed_duplicates = train_1.sample(n=n_diff, replace=True, random_state=42)
    train_balanced = pd.concat([train_df, lapsed_duplicates], axis=0)

    X_train_bal = train_balanced[numeric_cols + categorical_cols]
    y_train_bal = train_balanced["lapsed"]

    n_0_bal = (y_train_bal == 0).sum()
    n_1_bal = (y_train_bal == 1).sum()
    print(f"\nBalanced Training Set: Total = {len(train_balanced)}")
    print(f"  - Non-lapsed (class 0): {n_0_bal} rows ({n_0_bal / len(train_balanced) * 100:.1f}%)")
    print(f"  - Lapsed (class 1):     {n_1_bal} rows ({n_1_bal / len(train_balanced) * 100:.1f}%)")

    assert n_0_bal == 570 and n_1_bal == 570, "Training set must have exactly 570 per class."
    assert len(X_test) == 232, "Test set must remain untouched at 232 rows."

    # 2. Fit Un-balanced Best Logistic Regression (C=100 from GridSearch, and C=1.0 reference)
    print("\n" + "=" * 80)
    print("  STEP 2: MODEL TRAINING (UN-BALANCED VS. BALANCED)")
    print("=" * 80)

    # Let's run for best tuned model (C=100) and also compare with C=1.0
    models_to_evaluate = [
        ("Best Tuned Logistic Regression (C=100)", 100),
        ("Logistic Regression (C=1.0)", 1.0)
    ]

    results = {}

    for label, c_param in models_to_evaluate:
        print(f"\n>>> Evaluating {label}:")

        # Unbalanced Model
        pipe_unbal = Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(C=c_param, max_iter=1000, random_state=42))
        ])
        pipe_unbal.fit(X_train, y_train)
        y_pred_unbal = pipe_unbal.predict(X_test)

        cm_unbal = confusion_matrix(y_test, y_pred_unbal)
        prec_unbal = precision_score(y_test, y_pred_unbal)
        rec_unbal = recall_score(y_test, y_pred_unbal)
        f1_unbal = f1_score(y_test, y_pred_unbal)
        acc_unbal = accuracy_score(y_test, y_pred_unbal)

        # Balanced Model (refit on oversampled training data only)
        pipe_bal = Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(C=c_param, max_iter=1000, random_state=42))
        ])
        pipe_bal.fit(X_train_bal, y_train_bal)
        y_pred_bal = pipe_bal.predict(X_test)

        cm_bal = confusion_matrix(y_test, y_pred_bal)
        prec_bal = precision_score(y_test, y_pred_bal)
        rec_bal = recall_score(y_test, y_pred_bal)
        f1_bal = f1_score(y_test, y_pred_bal)
        acc_bal = accuracy_score(y_test, y_pred_bal)

        results[label] = {
            "unbal": (cm_unbal, prec_unbal, rec_unbal, f1_unbal, acc_unbal),
            "bal": (cm_bal, prec_bal, rec_bal, f1_bal, acc_bal),
        }

        print(f"\n  [Un-balanced Model]")
        print(f"    Confusion Matrix: TN={cm_unbal[0,0]}, FP={cm_unbal[0,1]}, FN={cm_unbal[1,0]}, TP={cm_unbal[1,1]}")
        print(f"    Accuracy:  {acc_unbal*100:.2f}%")
        print(f"    Precision: {prec_unbal:.4f} ({prec_unbal*100:.2f}%)")
        print(f"    Recall:    {rec_unbal:.4f} ({rec_unbal*100:.2f}%)")
        print(f"    F1-Score:  {f1_unbal:.4f}")

        print(f"\n  [Balanced (Over-sampled) Model]")
        print(f"    Confusion Matrix: TN={cm_bal[0,0]}, FP={cm_bal[0,1]}, FN={cm_bal[1,0]}, TP={cm_bal[1,1]}")
        print(f"    Accuracy:  {acc_bal*100:.2f}%")
        print(f"    Precision: {prec_bal:.4f} ({prec_bal*100:.2f}%)")
        print(f"    Recall:    {rec_bal:.4f} ({rec_bal*100:.2f}%)")
        print(f"    F1-Score:  {f1_bal:.4f}")

        delta_prec = prec_bal - prec_unbal
        delta_rec = rec_bal - rec_unbal
        delta_f1 = f1_bal - f1_unbal

        print(f"\n  [Metric Shifts (Balanced vs Un-balanced)]")
        print(f"    Delta Precision: {delta_prec:+.4f} ({delta_prec*100:+.2f} percentage points)")
        print(f"    Delta Recall:    {delta_rec:+.4f} ({delta_rec*100:+.2f} percentage points)  <-- HIGHEST MOVER")
        print(f"    Delta F1-Score:  {delta_f1:+.4f} ({delta_f1*100:+.2f} percentage points)")

    print("\n" + "=" * 80)
    print("  SUMMARY COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Metric':<18} | {'Un-balanced':<14} | {'Balanced (C=100)':<18} | {'Balanced (C=1.0)':<18}")
    print("-" * 75)
    r_100_u = results["Best Tuned Logistic Regression (C=100)"]["unbal"]
    r_100_b = results["Best Tuned Logistic Regression (C=100)"]["bal"]
    r_1_b = results["Logistic Regression (C=1.0)"]["bal"]

    print(f"{'True Negatives':<18} | {r_100_u[0][0,0]:<14} | {r_100_b[0][0,0]:<18} | {r_1_b[0][0,0]:<18}")
    print(f"{'False Positives':<18} | {r_100_u[0][0,1]:<14} | {r_100_b[0][0,1]:<18} | {r_1_b[0][0,1]:<18}")
    print(f"{'False Negatives':<18} | {r_100_u[0][1,0]:<14} | {r_100_b[0][1,0]:<18} | {r_1_b[0][1,0]:<18}")
    print(f"{'True Positives':<18} | {r_100_u[0][1,1]:<14} | {r_100_b[0][1,1]:<18} | {r_1_b[0][1,1]:<18}")
    print("-" * 75)
    print(f"{'Precision':<18} | {r_100_u[1]:.4f} ({r_100_u[1]*100:.1f}%) | {r_100_b[1]:.4f} ({r_100_b[1]*100:.1f}%)     | {r_1_b[1]:.4f} ({r_1_b[1]*100:.1f}%)")
    print(f"{'Recall':<18} | {r_100_u[2]:.4f} ({r_100_u[2]*100:.1f}%) | {r_100_b[2]:.4f} ({r_100_b[2]*100:.1f}%)     | {r_1_b[2]:.4f} ({r_1_b[2]*100:.1f}%)")
    print(f"{'F1-Score':<18} | {r_100_u[3]:.4f}         | {r_100_b[3]:.4f}             | {r_1_b[3]:.4f}")
    print(f"{'Accuracy':<18} | {r_100_u[4]*100:.2f}%        | {r_100_b[4]*100:.2f}%            | {r_1_b[4]*100:.2f}%")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_class_balance_experiment()
