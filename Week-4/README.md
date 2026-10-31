# Week 4: Model Evaluation and Validation Techniques

## Project Overview

This week focuses on evaluating and validating the Logistic Regression model developed during Week 3 for Telecom Customer Churn Prediction using the IBM Telco Customer Churn dataset.

The objective is to assess model performance using classification metrics, cross-validation techniques, and error analysis. The evaluation helps understand the model's reliability, consistency, and ability to identify customers who are likely to churn.

## Objectives

* Evaluate the trained machine learning model.
* Calculate accuracy, precision, recall, F1-score, and ROC-AUC.
* Apply Stratified K-Fold Cross-Validation.
* Compare training and validation performance.
* Generate a confusion matrix and classification report.
* Identify false positive and false negative predictions.
* Save evaluation results for further analysis.

## Dataset

* **Dataset:** IBM Telco Customer Churn
* **Source file:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
* **Total records:** 7,043
* **Total columns:** 21
* **Target variable:** `Churn`

The original dataset is loaded, and the same 80:20 stratified train-test split configuration used in Week 3 is reproduced using random state 42.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* JSON

## Methodology

### 1. Model Loading

The trained Logistic Regression preprocessing and classification pipeline from Week 3 is loaded using Joblib.

### 2. Test Set Evaluation

The model predicts churn labels and probabilities for the held-out test dataset.

The following metrics are calculated:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

### 3. Evaluation Results

| Metric    | Test Result |
| --------- | ----------: |
| Accuracy  |      73.81% |
| Precision |      50.43% |
| Recall    |      78.34% |
| F1-score  |      61.36% |
| ROC-AUC   |      84.13% |

### 4. Cross-Validation

Stratified 5-Fold Cross-Validation is performed on the training dataset.

The training data is divided into five folds while maintaining approximately the same class distribution in each fold. Each fold is used for validation once, while the remaining folds are used for training.

**Cross-Validation Results**

| Metric    | Mean Score | Standard Deviation |
| --------- | ---------: | -----------------: |
| Accuracy  |     74.81% |             0.0153 |
| Precision |     51.65% |             0.0189 |
| Recall    |     80.13% |             0.0379 |
| F1-score  |     62.79% |             0.0232 |
| ROC-AUC   |     84.59% |             0.0124 |

The relatively low standard deviations indicate that the model achieved reasonably consistent results across the five validation folds.

### 5. Training and Validation Comparison

The model's training and validation scores are compared across the five folds.

This comparison helps identify possible overfitting or underfitting. A substantial difference between training and validation performance may indicate overfitting, while consistently low scores on both may suggest underfitting.

### 6. Error Analysis

The evaluation script identifies two important types of classification errors:

* **False Positive:** A customer who does not churn but is predicted as a churn customer.
* **False Negative:** A customer who actually churns but is predicted as a non-churn customer.

These errors are saved in `error_analysis.csv` for further investigation.

## Interpretation

The test-set recall of 78.34% indicates that the model identifies a substantial proportion of actual churn customers. Precision of 50.43% indicates that approximately half of the customers predicted as churners are actual churners.

The cross-validation ROC-AUC mean of 84.59% is close to the test-set ROC-AUC of 84.13%. The results suggest reasonably consistent performance across the validation folds and held-out test set.

However, these results alone do not prove that the model is free from overfitting or underfitting. Training and validation score differences should also be examined.

## Generated Outputs

The following files are saved in the `outputs/` directory:

* `evaluation_metrics.json` — test-set evaluation metrics.
* `classification_report.csv` — precision, recall, F1-score, and support.
* `confusion_matrix.csv` — actual versus predicted class counts.
* `cross_validation_results.json` — five-fold validation scores and summary statistics.
* `training_validation_comparison.csv` — mean training and validation scores.
* `error_analysis.csv` — individual predictions and their error categories.

## How to Run

Install the required packages:

```bash
python -m pip install pandas numpy scikit-learn joblib
```

Run the evaluation script from the project root directory:

```bash
python Week-4/evaluate_model.py
```

Ensure that the original dataset and Week-3 trained model are available in their respective project directories.

## Limitations and Future Improvements

* Further investigation is required to determine the causes of false positive and false negative predictions.
* Additional classifiers can be evaluated and compared.
* More detailed feature-level error analysis can help identify patterns in incorrect predictions.
* Different classification thresholds can be explored to balance precision and recall.
* Additional validation and model tuning can be performed using the training data.

## Conclusion

The Week-3 Logistic Regression model was evaluated using classification metrics, Stratified 5-Fold Cross-Validation, training-validation comparison, and error analysis.

The results provide a better understanding of the model's predictive performance and consistency. The generated evaluation files establish a foundation for further model improvement and experimentation in the upcoming project weeks.