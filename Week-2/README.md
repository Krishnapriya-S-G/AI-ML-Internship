# Telco Customer Churn Analysis

## Week 2: Data Preprocessing and Feature Engineering

### Project Overview

This project focuses on preparing the IBM Telco Customer Churn dataset for machine learning. The objective is to inspect data quality, clean the dataset, encode categorical variables, scale numerical variables, and prepare training and testing datasets.

### Dataset

* **Dataset:** IBM Telco Customer Churn
* **Original records:** 7,043
* **Original columns:** 21
* **Target variable:** `Churn`
* **Source:** [IBM Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset is stored in the shared `data` folder at the project root.

Dataset filename: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

### Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

### Project Structure

```text
Telco_Customer_Churn_Project/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── Week-2/
│   ├── outputs/
│   ├── models/
│   ├── preprocess_data.py
│   └── README.md
├── Week-3/
├── Week-4/
├── Week-5/
└── Week-6/
```

### Preprocessing Steps

1. Loaded and inspected the dataset.
2. Checked for missing values and duplicate records.
3. Cleaned text fields and converted `TotalCharges` to numeric format.
4. Encoded the target variable.
5. Prepared numerical and categorical features.
6. Split the dataset into training and testing subsets.
7. Applied numerical scaling and categorical encoding.
8. Generated a mutual-information feature ranking.

### How to Run

Install the required libraries:

```bash
python -m pip install pandas numpy scikit-learn matplotlib seaborn
```

From the project root, navigate to the Week-2 directory:

```powershell
cd Week-2
python preprocess_data.py
```

### Actual Execution Results

* **Original dataset:** 7,043 rows × 21 columns
* **Exact duplicate rows detected:** 0
* **Missing values detected before cleaning:** 0
* **Training dataset:** 5,634 rows
* **Testing dataset:** 1,409 rows
* **Transformed feature count:** 51
* **Execution status:** Completed successfully

### Generated Outputs

The `Week-2/outputs/` folder contains:

* `initial_data_quality_report.csv`
* `target_distribution.csv`
* `processed_train.csv`
* `processed_test.csv`
* `transformed_feature_names.csv`
* `feature_selection_ranking.csv`
* `preprocessing_summary.csv`

### Notes

The preprocessing transformations were fitted using the training data to reduce data leakage. Mutual-information ranking indicates statistical association between individual features and the target; it does not establish causation.

### Author

Krishnapriya S G
