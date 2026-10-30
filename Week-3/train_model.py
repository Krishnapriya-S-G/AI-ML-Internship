"""
Telecom Customer Churn Prediction
Week 3: Model Implementation and Code Documentation

Purpose:
    Train a baseline Logistic Regression classifier to predict whether a
    telecommunications customer is likely to churn.

Dataset:
    IBM Telco Customer Churn sample CSV:
    WA_Fn-UseC_-Telco-Customer-Churn.csv

Expected project structure:
    project-root/
        data/
            WA_Fn-UseC_-Telco-Customer-Churn.csv
        train_model.py

Install dependencies:
    python -m pip install pandas numpy scikit-learn

Run from the project root:
    python train_model.py

Important:
    This script prints metrics calculated from your local dataset. Results
    are not hard-coded and may vary if the dataset version changes.
"""

from pathlib import Path
import sys

import pandas as pd
import json
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Locate the CSV relative to the folder from which the script is run.
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Fixed seed makes the baseline train/test split reproducible.
RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    """Load the dataset and validate its basic structure."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}\n"
            "Download the IBM Telco Customer Churn CSV and place it in the "
            "'data' folder, or update DATA_PATH at the top of this script."
        )

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError("The CSV was loaded, but it contains no rows.")

    if "Churn" not in data.columns:
        raise ValueError(
            "The required target column 'Churn' was not found. "
            "Check that you downloaded the expected dataset."
        )

    # Remove whitespace around column names to avoid accidental mismatches.
    data.columns = data.columns.astype(str).str.strip()

    if "Churn" not in data.columns:
        raise ValueError("Target column 'Churn' is missing after column cleanup.")

    return data


def prepare_data(data: pd.DataFrame):
    """Prepare features and binary target labels."""
    data = data.copy()

    # Convert TotalCharges to numeric where that column exists.
    # Empty strings or invalid values become NaN and will be imputed later.
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"], errors="coerce"
        )

    # Map the expected target labels to 0 and 1.
    target = data["Churn"].astype(str).str.strip().map(
        {"No": 0, "Yes": 1}
    )

    if target.isna().any():
        unexpected = data.loc[target.isna(), "Churn"].dropna().unique().tolist()
        raise ValueError(
            "The Churn column contains missing or unexpected labels. "
            f"Unexpected values: {unexpected}"
        )

    # Remove the target and identifier from model inputs.
    # An identifier is not a meaningful generalizable customer feature.
    features = data.drop(
        columns=["Churn", "customerID"], errors="ignore"
    )

    if features.shape[1] == 0:
        raise ValueError("No feature columns remain after removing target/ID.")

    return features, target.astype(int)


def build_model(X_train: pd.DataFrame) -> Pipeline:
    """Build a preprocessing pipeline and Logistic Regression classifier."""
    # Determine column types using the training data only.
    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Median imputation and standardization for numeric features.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Most-frequent imputation and one-hot encoding for categories.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Apply each transformation only to its corresponding column group.
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    # Balanced class weights are a baseline choice for potentially imbalanced
    # churn labels. Evaluate whether they help; they are not always optimal.
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    # The pipeline prevents preprocessing from being fitted on test data.
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )

    return model

def save_model(model: Pipeline, output_dir: Path) -> None:
    """Save the trained model to a file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "logistic_regression_model.joblib"
    joblib.dump(model, model_path)

    print(f"\nModel saved to: {model_path.resolve()}")

def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    """Print standard classification metrics for the held-out test set."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    with open(output_dir / "evaluation_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["No Churn", "Churn"],
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report).transpose().to_csv(
        output_dir / "classification_report.csv"
    )

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    pd.DataFrame(
        matrix,
        index=["Actual No Churn", "Actual Churn"],
        columns=["Predicted No Churn", "Predicted Churn"],
    ).to_csv(output_dir / "confusion_matrix.csv")

    print(f"Evaluation files saved in: {output_dir.resolve()}")
    print("\n" + "=" * 60)
    print("MODEL EVALUATION — HELD-OUT TEST SET")
    print("=" * 60)
    print(f"Test records: {len(y_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

    print("\nConfusion matrix")
    print("Rows = actual class; columns = predicted class")
    print("Class order: 0 = No Churn, 1 = Churn")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\nClassification report")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No Churn", "Churn"],
            zero_division=0,
        )
    )

    # ROC-AUC uses probability scores rather than predicted class labels.
    if y_test.nunique() == 2:
        print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    else:
        print("ROC-AUC was not calculated because the test set has one class.")

    # Basic checks that help detect output-shape or probability problems.
    assert len(predictions) == len(y_test), "Prediction count does not match test labels."
    assert len(probabilities) == len(y_test), "Probability count does not match test labels."
    assert ((probabilities >= 0) & (probabilities <= 1)).all(), (
        "Predicted probabilities must be between 0 and 1."
    )


def main() -> int:
    """Run the end-to-end training and evaluation workflow."""
    try:
        print(f"Loading dataset from: {DATA_PATH.resolve()}")
        data = load_data(DATA_PATH)

        print(f"Dataset shape: {data.shape}")
        print("\nTarget distribution:")
        print(data["Churn"].value_counts(dropna=False))

        X, y = prepare_data(data)

        # Stratification preserves the approximate churn/non-churn ratio.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        print(f"\nTraining records: {len(X_train)}")
        print(f"Testing records: {len(X_test)}")

        model = build_model(X_train)
        print("\nTraining Logistic Regression baseline...")
        model.fit(X_train, y_train)
        save_model(model, Path(__file__).resolve().parent / "outputs")
        evaluate_model(model, X_test, y_test)

        print("\nCompleted successfully.")
        print(
            "Note: these are baseline results. Use cross-validation on the "
            "training set for model selection; do not tune repeatedly against "
            "the held-out test set."
        )
        return 0

    except FileNotFoundError as error:
        print(f"\nFILE ERROR:\n{error}", file=sys.stderr)
        return 1
    except (ValueError, pd.errors.ParserError) as error:
        print(f"\nDATA ERROR:\n{error}", file=sys.stderr)
        return 1
    except PermissionError as error:
        print(f"\nPERMISSION ERROR:\n{error}", file=sys.stderr)
        return 1
    except Exception as error:
        # Keep a final safety net while retaining the actual error message.
        print(
            f"\nUNEXPECTED ERROR: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    # This guard allows the file to be imported without starting training.
    raise SystemExit(main())
