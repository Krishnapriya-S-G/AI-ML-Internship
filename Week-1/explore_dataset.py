from pathlib import Path

import pandas as pd

# Locate the dataset in the main project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Load the dataset
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("WEEK 1: DATASET EXPLORATION")
print("=" * 60)

# 1. Dataset dimensions
print("\n1. DATASET SHAPE")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# 2. Column names and data types
print("\n2. COLUMN NAMES AND DATA TYPES")
print(df.dtypes.to_string())

# 3. First five records
print("\n3. FIRST FIVE RECORDS")
print(df.head().to_string())

# 4. Missing values
print("\n4. MISSING VALUES")
print(df.isnull().sum().to_string())

# 5. Duplicate records
print("\n5. DUPLICATE RECORDS")
print(f"Duplicate rows: {df.duplicated().sum()}")

# 6. Churn distribution
print("\n6. CHURN DISTRIBUTION")
print(df["Churn"].value_counts().to_string())
print("\nChurn percentage:")
print((df["Churn"].value_counts(normalize=True) * 100).round(2).to_string())

# 7. Numerical summary
print("\n7. NUMERICAL SUMMARY")
print(df.describe(include="all").transpose().to_string())

# 8. Save a data quality report
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

quality_report = pd.DataFrame({
    "data_type": df.dtypes.astype(str),
    "missing_values": df.isnull().sum(),
    "unique_values": df.nunique(),
})

quality_report.to_csv(output_dir / "data_quality_report.csv")

# Save churn distribution
churn_report = df["Churn"].value_counts().rename_axis("Churn").reset_index(name="Count")
churn_report["Percentage"] = (
    churn_report["Count"] / len(df) * 100
).round(2)

churn_report.to_csv(output_dir / "churn_distribution.csv", index=False)

print("\n8. REPORTS SAVED")
print(f"Output folder: {output_dir.resolve()}")
print("Created: data_quality_report.csv")
print("Created: churn_distribution.csv")

print("\nDataset exploration completed successfully.")
