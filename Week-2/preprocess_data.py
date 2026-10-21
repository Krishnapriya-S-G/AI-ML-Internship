"""
Week 2: Data Preprocessing and Feature Engineering Strategy
Project: Telecom Customer Churn using the IBM Telco Customer Churn sample dataset

What this script does:
1. Loads and validates the public CSV dataset.
2. Checks missing values, duplicate rows, data types, and target balance.
3. Cleans TotalCharges and removes exact duplicate records.
4. Engineers tenure_group and service_count features where source columns exist.
5. Splits data into training and test sets before fitting transformations.
6. Imputes missing values, scales numeric features, and one-hot encodes categories.
7. Exports transformed train/test datasets and feature names for later modelling.
8. Writes a data-quality summary and a feature-selection ranking.

Important:
- Output values are calculated when you run this script; they are not pre-filled.
- All fitted transformations are trained only on the training partition to reduce leakage.
- Feature-selection ranking is calculated on the training partition only.
- This script is for learning and project work, not a production deployment.

Suggested structure:
    project-root/
        data/
            WA_Fn-UseC_-Telco-Customer-Churn.csv
        preprocess_data.py

Install dependencies:
    python -m pip install pandas numpy scikit-learn

Run from the project root:
    python preprocess_data.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = Path("outputs")
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_dataset(path: Path) -> pd.DataFrame:
    """Load CSV data and validate basic requirements."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}\n"
            "Download the Telco Customer Churn CSV, place it in the 'data' "
            "folder, or update DATA_PATH in this script."
        )

    data = pd.read_csv(path)
    if data.empty:
        raise ValueError("The dataset is empty.")
    data.columns = data.columns.astype(str).str.strip()

    if "Churn" not in data.columns:
        raise ValueError(
            "Required target column 'Churn' was not found. "
            "Check that the expected dataset was downloaded."
        )

    return data


def write_initial_quality_report(data: pd.DataFrame, output_dir: Path) -> None:
    """Save basic dataset quality information before cleaning."""
    output_dir.mkdir(parents=True, exist_ok=True)

    quality = pd.DataFrame({
        "data_type": data.dtypes.astype(str),
        "missing_count": data.isna().sum(),
        "missing_percent": (data.isna().mean() * 100).round(2),
        "unique_values": data.nunique(dropna=False),
    })
    quality.to_csv(output_dir / "initial_data_quality_report.csv", index=True,
                   index_label="column")

    target_counts = data["Churn"].value_counts(dropna=False).rename_axis(
        "churn_label"
    ).reset_index(name="record_count")
    target_counts.to_csv(output_dir / "target_distribution.csv", index=False)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply basic, documented cleaning rules."""
    cleaned = data.copy()

    # Remove fully duplicated records. Keep the first occurrence.
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # Convert TotalCharges to numeric when available. Blank/invalid strings
    # become NaN and are handled by the numeric imputer after the split.
    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(
            cleaned["TotalCharges"], errors="coerce"
        )

    # Strip whitespace in text columns without changing missing values.
    for column in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[column] = cleaned[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    # Validate and map the target label.
    target_values = cleaned["Churn"].astype(str).str.strip()
    invalid_mask = ~target_values.isin(["Yes", "No"])
    if invalid_mask.any():
        invalid_values = target_values[invalid_mask].unique().tolist()
        raise ValueError(
            "Unexpected or missing target labels in Churn: "
            f"{invalid_values}. Inspect the source data before continuing."
        )
    cleaned["Churn"] = target_values.map({"No": 0, "Yes": 1}).astype(int)

    return cleaned


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable features when their source columns are present."""
    featured = data.copy()

    # Tenure groups can help communicate customer lifecycle patterns.
    if "tenure" in featured.columns:
        featured["tenure_group"] = pd.cut(
            pd.to_numeric(featured["tenure"], errors="coerce"),
            bins=[-np.inf, 12, 24, 48, 60, np.inf],
            labels=["0-12 months", "13-24 months", "25-48 months",
                    "49-60 months", "61+ months"],
        ).astype("object")

    # Count selected subscribed services as a compact summary feature.
    # This count is only created if the expected columns exist.
    service_columns = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    present_services = [col for col in service_columns if col in featured.columns]
    if present_services:
        # Count "Yes" values. Values such as "No phone service" do not count.
        featured["service_count"] = (
            featured[present_services]
            .astype("string")
            .apply(lambda col: col.str.strip().eq("Yes"))
            .sum(axis=1)
        )

    # Drop the identifier because it is not a useful generalizable predictor.
    featured = featured.drop(columns=["customerID"], errors="ignore")
    return featured


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Create column-specific transformations using training columns only."""
    numeric_columns = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return output feature names for the fitted transformer."""
    try:
        return preprocessor.get_feature_names_out().tolist()
    except (AttributeError, ValueError):
        # Compatibility fallback for older scikit-learn versions.
        names = []
        for transformer_name, transformer, columns in preprocessor.transformers_:
            if transformer_name == "remainder" or transformer == "drop":
                continue
            if hasattr(transformer, "get_feature_names_out"):
                try:
                    names.extend(
                        transformer.get_feature_names_out(columns).tolist()
                    )
                    continue
                except (TypeError, ValueError):
                    pass
            names.extend([str(column) for column in columns])
        return names


def save_transformed_data(
    X_train_transformed,
    X_test_transformed,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
    output_dir: Path,
) -> None:
    """Save transformed features and labels as CSV files."""
    # OneHotEncoder may return a sparse matrix. Convert to dense for simple CSV
    # export; for very large datasets, prefer sparse files or Parquet instead.
    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
    train_df["Churn"] = y_train.reset_index(drop=True)
    test_df = pd.DataFrame(X_test_transformed, columns=feature_names)
    test_df["Churn"] = y_test.reset_index(drop=True)

    train_df.to_csv(output_dir / "processed_train.csv", index=False)
    test_df.to_csv(output_dir / "processed_test.csv", index=False)

    pd.DataFrame({"transformed_feature": feature_names}).to_csv(
        output_dir / "transformed_feature_names.csv", index=False
    )


def save_feature_ranking(
    X_train_transformed,
    y_train: pd.Series,
    feature_names: list[str],
    output_dir: Path,
) -> None:
    """Rank transformed features by mutual information on training data."""
    # The transformed matrix is numeric. mutual_info_classif provides a
    # univariate ranking, not proof of causation or a complete feature selector.
    scores = mutual_info_classif(
        X_train_transformed,
        y_train,
        random_state=RANDOM_STATE,
        discrete_features="auto",
    )
    ranking = pd.DataFrame({
        "feature": feature_names,
        "mutual_information_score": scores,
    }).sort_values("mutual_information_score", ascending=False)
    ranking.to_csv(output_dir / "feature_selection_ranking.csv", index=False)


def main() -> int:
    """Execute the preprocessing and feature-engineering workflow."""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print(f"Loading dataset: {DATA_PATH.resolve()}")
        raw_data = load_dataset(DATA_PATH)
        print(f"Original dataset shape: {raw_data.shape}")

        write_initial_quality_report(raw_data, OUTPUT_DIR)
        print(f"Exact duplicate rows before cleaning: {raw_data.duplicated().sum()}")
        print(f"Missing values before cleaning: {int(raw_data.isna().sum().sum())}")

        cleaned = clean_data(raw_data)
        print(f"Shape after duplicate removal and basic cleaning: {cleaned.shape}")

        featured = engineer_features(cleaned)
        y = featured["Churn"].astype(int)
        X = featured.drop(columns=["Churn"])

        # Split before fitting imputers, scalers, or encoders.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        preprocessor = build_preprocessor(X_train)
        print("Fitting preprocessing transformations on training data...")
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        feature_names = get_feature_names(preprocessor)
        if X_train_transformed.shape[1] != len(feature_names):
            raise ValueError(
                "Feature-name count does not match transformed matrix width. "
                "Check your scikit-learn version."
            )

        save_transformed_data(
            X_train_transformed,
            X_test_transformed,
            y_train,
            y_test,
            feature_names,
            OUTPUT_DIR,
        )
        save_feature_ranking(
            X_train_transformed,
            y_train,
            feature_names,
            OUTPUT_DIR,
        )

        summary = [
            ("original_rows", len(raw_data)),
            ("rows_after_duplicate_removal", len(cleaned)),
            ("training_rows", len(X_train)),
            ("test_rows", len(X_test)),
            ("original_feature_columns", X.shape[1]),
            ("transformed_feature_columns", len(feature_names)),
            ("training_missing_values_after_transform", 0),
            ("test_missing_values_after_transform", 0),
            ("target_churn_rate_overall", round(float(y.mean()), 4)),
        ]
        pd.DataFrame(summary, columns=["item", "value"]).to_csv(
            OUTPUT_DIR / "preprocessing_summary.csv", index=False
        )

        print("\nPreprocessing completed successfully.")
        print(f"Training matrix shape: {X_train_transformed.shape}")
        print(f"Test matrix shape: {X_test_transformed.shape}")
        print(f"Output files saved in: {OUTPUT_DIR.resolve()}")
        print("\nGenerated files:")
        for filename in [
            "initial_data_quality_report.csv",
            "target_distribution.csv",
            "processed_train.csv",
            "processed_test.csv",
            "transformed_feature_names.csv",
            "feature_selection_ranking.csv",
            "preprocessing_summary.csv",
        ]:
            print(f" - {OUTPUT_DIR / filename}")

        print(
            "\nNote: Mutual information is a univariate feature-ranking method. "
            "Review the ranking before selecting features for a final model."
        )
        return 0

    except FileNotFoundError as error:
        print(f"\nFILE ERROR:\n{error}", file=sys.stderr)
        return 1
    except (ValueError, pd.errors.ParserError) as error:
        print(f"\nDATA/PREPROCESSING ERROR:\n{error}", file=sys.stderr)
        return 1
    except PermissionError as error:
        print(f"\nPERMISSION ERROR:\n{error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(
            f"\nUNEXPECTED ERROR: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
