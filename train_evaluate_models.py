"""train_evaluate_models.py -- Train, tune, and evaluate classification models for customer lapse.

Tasks:
(a) Report majority baseline (accuracy: 61.6%, lapsed F1: 0.0).
(b) Tune LogisticRegression(max_iter=1000) over C in {0.01, 0.1, 1, 10, 100}, cv=5, scoring="f1".
(c) Tune RandomForestClassifier(n_estimators=100, random_state=42) over
    max_depth in {3, 5, 8} and min_samples_leaf in {5, 10, 20}, cv=5, scoring="f1" (45 fits).
(d) Evaluate both tuned models on the untouched test set:
    - Confusion matrix
    - Precision, recall, and F1 of the lapsed class
    - Justify model selection considering asymmetric business costs.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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

def train_and_evaluate():
    df = pd.read_csv(DATASET_PATH)

    numeric_cols = [
        "recency", "frequency", "monetary", "avg_discount_pct",
        "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days"
    ]
    categorical_cols = ["city", "gender", "most_frequent_category"]

    X = df[numeric_cols + categorical_cols]
    y = df["lapsed"]

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols)
        ]
    )

    print("=" * 80)
    print("  (A) MAJORITY CLASS BASELINE")
    print("=" * 80)
    y_pred_dummy = np.zeros_like(y_test)
    base_acc = accuracy_score(y_test, y_pred_dummy)
    base_f1 = f1_score(y_test, y_pred_dummy, zero_division=0)
    base_prec = precision_score(y_test, y_pred_dummy, zero_division=0)
    base_rec = recall_score(y_test, y_pred_dummy, zero_division=0)
    print(f"Strategy: Always predict 'not lapsed' (class 0)")
    print(f"Test Accuracy: {base_acc*100:.1f}%")
    print(f"Lapsed Class Precision: {base_prec:.4f}")
    print(f"Lapsed Class Recall:    {base_rec:.4f}")
    print(f"Lapsed Class F1-Score:  {base_f1:.4f}")

    # (B) Tune Logistic Regression
    print("\n" + "=" * 80)
    print("  (B) LOGISTIC REGRESSION TUNING (GridSearchCV, 5-Fold CV)")
    print("=" * 80)
    pipe_lr = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    param_grid_lr = {"clf__C": [0.01, 0.1, 1, 10, 100]}
    grid_lr = GridSearchCV(pipe_lr, param_grid=param_grid_lr, cv=5, scoring="f1")
    grid_lr.fit(X_train, y_train)

    lr_cv_results = pd.DataFrame(grid_lr.cv_results_)
    for _, r in lr_cv_results.iterrows():
        print(f"C={r['param_clf__C']:<6}: Mean CV F1 = {r['mean_test_score']:.4f} (std: {r['std_test_score']:.4f})")
    print(f"--> Best Parameters: {grid_lr.best_params_}")
    print(f"--> Best 5-Fold CV F1: {grid_lr.best_score_:.4f}")

    # (C) Tune Random Forest Classifier
    print("\n" + "=" * 80)
    print("  (C) RANDOM FOREST TUNING (GridSearchCV, 5-Fold CV, 45 fits)")
    print("=" * 80)
    pipe_rf = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    param_grid_rf = {
        "clf__max_depth": [3, 5, 8],
        "clf__min_samples_leaf": [5, 10, 20]
    }
    grid_rf = GridSearchCV(pipe_rf, param_grid=param_grid_rf, cv=5, scoring="f1")
    grid_rf.fit(X_train, y_train)

    rf_cv_results = pd.DataFrame(grid_rf.cv_results_)
    for _, r in rf_cv_results.iterrows():
        print(f"depth={r['param_clf__max_depth']}, leaf={r['param_clf__min_samples_leaf']:<2}: Mean CV F1 = {r['mean_test_score']:.4f} (std: {r['std_test_score']:.4f})")
    print(f"--> Best Parameters: {grid_rf.best_params_}")
    print(f"--> Best 5-Fold CV F1: {grid_rf.best_score_:.4f}")

    # (D) Evaluation on untouched Test Set
    print("\n" + "=" * 80)
    print("  (D) UNTOUCHED TEST SET EVALUATION")
    print("=" * 80)

    # 1. Logistic Regression Test Evaluation
    best_lr = grid_lr.best_estimator_
    y_pred_lr = best_lr.predict(X_test)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    prec_lr = precision_score(y_test, y_pred_lr)
    rec_lr = recall_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    # 2. Random Forest Test Evaluation
    best_rf = grid_rf.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    prec_rf = precision_score(y_test, y_pred_rf)
    rec_rf = recall_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    print("--- Logistic Regression (Best Model) ---")
    print(f"Confusion Matrix [TN={cm_lr[0,0]}, FP={cm_lr[0,1]}, FN={cm_lr[1,0]}, TP={cm_lr[1,1]}]:")
    print(cm_lr)
    print(f"Accuracy:  {acc_lr*100:.1f}%")
    print(f"Precision: {prec_lr:.4f} ({prec_lr*100:.2f}%)")
    print(f"Recall:    {rec_lr:.4f} ({rec_lr*100:.2f}%)")
    print(f"F1-Score:  {f1_lr:.4f}")

    print("\n--- Random Forest (Best Model) ---")
    print(f"Confusion Matrix [TN={cm_rf[0,0]}, FP={cm_rf[0,1]}, FN={cm_rf[1,0]}, TP={cm_rf[1,1]}]:")
    print(cm_rf)
    print(f"Accuracy:  {acc_rf*100:.1f}%")
    print(f"Precision: {prec_rf:.4f} ({prec_rf*100:.2f}%)")
    print(f"Recall:    {rec_rf:.4f} ({rec_rf*100:.2f}%)")
    print(f"F1-Score:  {f1_rf:.4f}")

    assert f1_rf >= 0.55, f"Selected model must reach F1 >= 0.55, got {f1_rf}"
    assert f1_lr >= 0.55, f"Selected model must reach F1 >= 0.55, got {f1_lr}"

    return {
        "grid_lr": grid_lr,
        "grid_rf": grid_rf,
        "lr_metrics": (cm_lr, prec_lr, rec_lr, f1_lr, acc_lr),
        "rf_metrics": (cm_rf, prec_rf, rec_rf, f1_rf, acc_rf),
    }

if __name__ == "__main__":
    train_and_evaluate()
