# Week 3: Telecom Customer Churn Prediction

## Project Overview

This week focuses on developing and evaluating a baseline Machine Learning model to predict customer churn using the IBM Telco Customer Churn dataset.

A Logistic Regression classifier is implemented using a Scikit-learn pipeline. The pipeline handles data preprocessing and model training, ensuring that preprocessing transformations are fitted only on the training data.

## Objectives

* Load and validate the customer churn dataset.
* Prepare input features and the target variable.
* Split the dataset into training and testing sets.
* Build a preprocessing pipeline for numerical and categorical features.
* Train a Logistic Regression classifier.
* Evaluate the model using classification metrics.
* Save the trained model and evaluation results.

## Dataset

* **Dataset:** IBM Telco Customer Churn
* **Source file:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
* **Total records:** 7,043
* **Total columns:** 21
* **Target variable:** `Churn`
* **Target classes:** `No` and `Yes`

The `customerID` column is excluded because it is an identifier and does not represent a generalizable customer characteristic.

The script loads the original dataset from the project's root `data/` directory and performs preprocessing within its own pipeline.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* JSON

## Methodology

### 1. Data Loading and Preparation

* Loaded the dataset using Pandas.
* Checked whether the dataset file and required target column were available.
* Converted `TotalCharges` into numeric values using `pd.to_numeric()` with invalid values converted to missing values.
* Converted target labels into binary values: `No = 0` and `Yes = 1`.
* Removed the `Churn` target and `customerID` identifier from the input features.

### 2. Data Splitting

The dataset was divided into training and testing sets using an 80:20 split.

Stratified sampling was applied to preserve the approximate distribution of churn and non-churn customers.

* **Training records:** 5,634
* **Testing records:** 1,409
* **Test size:** 20%
* **Random state:** 42

### 3. Data Preprocessing

A Scikit-learn preprocessing pipeline was created using a `ColumnTransformer`.

**Numerical features:**

* Missing values were handled using median imputation.
* Features were standardized using `StandardScaler`.

**Categorical features:**

* Missing values were handled using most-frequent imputation.
* Categorical values were converted into numerical features using `OneHotEncoder`.
* Unknown categories were handled using `handle_unknown="ignore"`.

The preprocessing steps were fitted through the model pipeline using the training data, helping prevent data leakage from the test set.

### 4. Model Training

A Logistic Regression classifier was selected as the baseline model.

Configuration:

* **Algorithm:** Logistic Regression
* **Maximum iterations:** 1,000
* **Class weighting:** Balanced
* **Random state:** 42

Balanced class weighting was used to account for the difference in the number of churn and non-churn customers.

### 5. Model Evaluation

The trained model was evaluated using the held-out test dataset.

The evaluation includes:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* ROC-AUC score

The script also checks prediction lengths and verifies that predicted probabilities are within the valid range.

## Results

Evaluation metrics are calculated when the script is executed and saved in the `outputs/` directory.

The following results should be taken from the generated output files after running the latest version of the script:

* Test accuracy: Refer to `evaluation_metrics.json`.
* ROC-AUC: Refer to `evaluation_metrics.json`.
* Precision, recall and F1-score: Refer to `classification_report.csv`.
* Confusion matrix: Refer to `confusion_matrix.csv`.

### Interpretation

The evaluation results help assess how effectively the baseline model identifies customers who are likely to leave the service.

Recall is particularly important because identifying actual churn customers is a key objective of this project. Accuracy should not be considered in isolation; precision, recall, F1-score and ROC-AUC also provide useful information about model performance.

## Generated Outputs

The following files are saved in the `Week-3/outputs/` directory:

* `logistic_regression_model.joblib` — saved preprocessing and classification pipeline.
* `evaluation_metrics.json` — accuracy and ROC-AUC results.
* `classification_report.csv` — precision, recall, F1-score and support.
* `confusion_matrix.csv` — actual versus predicted class counts.

## How to Run

Run the following commands from the project root directory:

```bash
python -m pip install pandas numpy scikit-learn joblib
python Week-3/train_model.py
```

Alternatively, from inside the `Week-3` directory, run:

```bash
python -m pip install pandas numpy scikit-learn joblib
python train_model.py
```

Ensure that the dataset is available at:

```text
project-root/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── Week-3/
    └── train_model.py
```

## Limitations and Next Steps

* Logistic Regression is used as a baseline model.
* Further model comparison and validation are required.
* Cross-validation can be used on the training data when comparing models.
* The held-out test set should not be repeatedly used for model tuning.
* Alternative classification algorithms and evaluation strategies can be explored in subsequent project weeks.

## Conclusion

A baseline Logistic Regression classification pipeline was developed for telecom customer churn prediction using the IBM Telco Customer Churn dataset.

The implementation includes data preparation, numerical and categorical preprocessing, model training, evaluation and model saving. The generated model and evaluation files provide a foundation for further experimentation and model comparison in the upcoming project weeks.