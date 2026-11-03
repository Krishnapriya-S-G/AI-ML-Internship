from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

BASELINE_MODEL_PATH = (
    PROJECT_DIR
    / "Week-3"
    / "outputs"
    / "logistic_regression_model.joblib"
)

OPTIMIZED_MODEL_PATH = (
    PROJECT_DIR
    / "Week-5"
    / "outputs"
    / "best_optimized_model.joblib"
)

OUTPUT_DIR = PROJECT_DIR / "Week-6" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    # 1. Load dataset
    print("Loading dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)

    # 2. Prepare target variable
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )

    data["Churn"] = data["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    data = data.dropna(subset=["Churn"])

    X = data.drop(columns=["customerID", "Churn"])
    y = data["Churn"].astype(int)

    # 3. Recreate the same train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    # 4. Load saved models
    print("\nLoading trained models...")

    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Baseline model not found: {BASELINE_MODEL_PATH}"
        )

    if not OPTIMIZED_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Optimized model not found: {OPTIMIZED_MODEL_PATH}"
        )

    models = {
        "Baseline Model": joblib.load(BASELINE_MODEL_PATH),
        "Optimized Model": joblib.load(OPTIMIZED_MODEL_PATH)
    }

    results = []

    # 5. Evaluate both models
    for model_name, model in models.items():

        print(f"\nEvaluating {model_name}...")

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(y_test, probabilities)
        }

        results.append(metrics)

        # Save confusion matrix
        matrix = confusion_matrix(y_test, predictions)

        matrix_df = pd.DataFrame(
            matrix,
            index=["Actual No Churn", "Actual Churn"],
            columns=["Predicted No Churn", "Predicted Churn"]
        )

        filename = (
            model_name.lower().replace(" ", "_")
            + "_confusion_matrix.csv"
        )

        matrix_df.to_csv(OUTPUT_DIR / filename)

        print("Accuracy:", round(metrics["Accuracy"], 4))
        print("Precision:", round(metrics["Precision"], 4))
        print("Recall:", round(metrics["Recall"], 4))
        print("F1 Score:", round(metrics["F1 Score"], 4))
        print("ROC-AUC:", round(metrics["ROC-AUC"], 4))

    # 6. Save comparison results
    comparison = pd.DataFrame(results)

    comparison.to_csv(
        OUTPUT_DIR / "final_model_comparison.csv",
        index=False
    )

    print("\nFinal Model Comparison:")
    print(comparison.to_string(index=False))

    print("\nAnalysis completed successfully!")
    print("Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()