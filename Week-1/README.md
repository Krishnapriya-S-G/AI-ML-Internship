# Telco Customer Churn Analysis

## Week 1: Dataset Exploration and Understanding

### Project Overview

This project analyzes the IBM Telco Customer Churn dataset to understand customer characteristics and explore patterns related to customer churn. The overall project covers data exploration, preprocessing, machine learning, model evaluation, and final documentation over six weeks.

### Objectives

* Understand the customer churn problem.
* Inspect dataset dimensions, columns, and data types.
* Check for missing values and duplicate records.
* Examine the distribution of the target variable.
* Generate initial data quality and churn distribution reports.

### Dataset Information

* **Dataset:** IBM Telco Customer Churn
* **Records:** 7,043
* **Columns:** 21
* **Target variable:** `Churn`
* **Dataset file:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset is stored in the shared `data/` folder at the project root.

### Technologies Used

* Python
* Pandas

### Analysis Results

| Metric                                              |         Result |
| --------------------------------------------------- | -------------: |
| Total records                                       |          7,043 |
| Total columns                                       |             21 |
| Duplicate rows detected                             |              0 |
| Missing values detected by the initial Pandas check |              0 |
| Customers who churned                               | 1,869 (26.54%) |
| Customers who did not churn                         | 5,174 (73.46%) |
| Average tenure                                      |   32.37 months |
| Average monthly charges                             |          64.76 |

### Key Observations

* Approximately 26.54% of customers in the dataset have churned.
* The dataset contains customer demographics, subscribed services, contract details, billing information, and the churn target.
* `TotalCharges` was initially loaded as a text column and requires numeric conversion during preprocessing.
* Although the initial missing-value check reported zero missing values, blank or whitespace-only values in text columns may require additional checks.

### How to Run

Run the following command from the project root:

```bash
python Week-1/explore_dataset.py
```

### Generated Outputs

The `Week-1/outputs/` folder contains:

* `data_quality_report.csv` — column data types, missing-value counts, and unique-value counts.
* `churn_distribution.csv` — customer churn counts and percentages.

### Project Structure

```text
Telco_Customer_Churn_Project/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── Week-1/
│   ├── outputs/
│   ├── explore_dataset.py
│   └── README.md
├── Week-2/
├── Week-3/
├── Week-4/
├── Week-5/
└── Week-6/
```

### Author

Krishnapriya S G
