import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

warnings.filterwarnings("ignore")

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    # Preprocessing
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Hyperparameter search space
    param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__class_weight": [None, "balanced"],
        "classifier__solver": ["liblinear", "lbfgs"]
    }

    scoring = "roc_auc"

    # Grid Search
    print("\nRunning Grid Search...")

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("Grid Search Best Parameters:")
    print(grid_search.best_params_)

    # Randomized Search
    print("\nRunning Randomized Search...")

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=15,
        scoring=scoring,
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    random_search.fit(X_train, y_train)

    print("Random Search Best Parameters:")
    print(random_search.best_params_)

    # Evaluate both optimized models
    experiments = {
        "Grid Search": grid_search.best_estimator_,
        "Randomized Search": random_search.best_estimator_
    }

    results = []
    best_model = None
    best_score = -1
    best_name = None

    for name, model in experiments.items():

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
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

        print(f"\n{name} Test Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

        if metrics["ROC-AUC"] > best_score:
            best_score = metrics["ROC-AUC"]
            best_model = model
            best_name = name

    # Save experiment results
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        OUTPUT_DIR / "optimization_comparison.csv",
        index=False
    )

    # Save best model
    joblib.dump(
        best_model,
        OUTPUT_DIR / "best_optimized_model.joblib"
    )

    # Save best parameters
    best_parameters = {
        "best_search_method": best_name,
        "best_roc_auc": float(best_score),
        "grid_search_parameters": grid_search.best_params_,
        "random_search_parameters": random_search.best_params_
    }

    with open(
        OUTPUT_DIR / "best_parameters.json",
        "w"
    ) as file:
        json.dump(best_parameters, file, indent=4)

    print("\nBest Model:", best_name)
    print("Best ROC-AUC:", round(best_score, 4))
    print("\nWeek-5 optimization completed successfully!")
    print("Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()