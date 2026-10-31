"""
Week 4: Model Evaluation and Validation
Telecom Customer Churn Prediction
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

MODEL_PATH = BASE_DIR / "Week-3" / "outputs" / "logistic_regression_model.joblib"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def load_data():
    """Load and validate the original dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()

    if data.empty:
        raise ValueError("Dataset is empty.")

    if "Churn" not in data.columns:
        raise ValueError("Churn target column is missing.")

    return data


def prepare_data(data):
    """Prepare features and binary target labels."""

    data = data.copy()

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )

    y = data["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Invalid or missing churn labels found.")

    X = data.drop(columns=["Churn", "customerID"])

    return X, y


def main():

    print("Loading dataset...")
    data = load_data()

    X, y = prepare_data(data)

    print("Dataset shape:", data.shape)

    # Reproduce the Week-3 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    # Load the Week-3 trained pipeline
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    # Test-set predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, y_pred, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }

    print("\nTest Set Evaluation:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # Save evaluation metrics
    with open(OUTPUT_DIR / "evaluation_metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    # Classification report
    report = classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"],
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    pd.DataFrame(
        cm,
        index=["Actual_No_Churn", "Actual_Churn"],
        columns=["Predicted_No_Churn", "Predicted_Churn"]
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    # Stratified 5-Fold Cross-Validation
    print("\nRunning 5-Fold Stratified Cross-Validation...")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ]

    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    cv_summary = {}

    for metric in scoring:
        scores = cv_results[f"test_{metric}"]

        cv_summary[metric] = {
            "fold_scores": scores.tolist(),
            "mean": float(np.mean(scores)),
            "standard_deviation": float(np.std(scores))
        }

        print(
            f"{metric}: Mean = {np.mean(scores):.4f}, "
            f"Std = {np.std(scores):.4f}"
        )

    with open(OUTPUT_DIR / "cross_validation_results.json", "w") as file:
        json.dump(cv_summary, file, indent=4)

    # Training versus validation performance
    comparison = {
        "metric": scoring,
        "mean_training_score": [
            float(np.mean(cv_results[f"train_{m}"]))
            for m in scoring
        ],
        "mean_validation_score": [
            float(np.mean(cv_results[f"test_{m}"]))
            for m in scoring
        ]
    }

    pd.DataFrame(comparison).to_csv(
        OUTPUT_DIR / "training_validation_comparison.csv",
        index=False
    )

    # Error analysis: False Positives and False Negatives
    error_analysis = X_test.copy()
    error_analysis["Actual_Churn"] = y_test
    error_analysis["Predicted_Churn"] = y_pred

    error_analysis["Error_Type"] = np.select(
        [
            (y_test == 0) & (y_pred == 1),
            (y_test == 1) & (y_pred == 0)
        ],
        [
            "False Positive",
            "False Negative"
        ],
        default="Correct Prediction"
    )

    error_analysis.to_csv(
        OUTPUT_DIR / "error_analysis.csv",
        index=False
    )

    print("\nWeek-4 evaluation completed successfully.")
    print("Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()