# Week 6 – Final Analysis and Project Report

## Project Title

Telecom Customer Churn Prediction Using Machine Learning

## Overview

Week 6 focuses on the final analysis and documentation of the Telecom Customer Churn Prediction project. The main objective is to compare the performance of the Baseline Logistic Regression Model developed in Week 3 with the Optimized Logistic Regression Model developed in Week 5.

A final analysis script was developed to evaluate both saved models using the same training and testing split. The evaluation results were saved as CSV files for further analysis and documentation.

## Objectives

* Perform a final comparison of the baseline and optimized models.
* Evaluate model performance using standard classification metrics.
* Generate confusion matrices for both models.
* Save the final evaluation results in CSV format.
* Document the complete machine learning project lifecycle.
* Identify limitations and possible future improvements.

## Dataset

* Dataset: IBM Telco Customer Churn
* Total records: 7,043
* Total columns: 21
* Training records: 5,634
* Testing records: 1,409
* Target variable: Churn

The dataset was divided using an 80:20 stratified train-test split with `random_state=42`.

## Implementation

### Python Script

`final_analysis.py`

The script performs the following operations:

1. Loads the original Telco Customer Churn dataset.
2. Converts the `TotalCharges` column into numeric format.
3. Encodes the target variable `Churn` into binary values.
4. Recreates the training and testing split.
5. Loads the saved baseline and optimized models.
6. Calculates Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
7. Generates confusion matrices.
8. Saves the model comparison results as CSV files.

## Final Model Comparison

| Evaluation Metric | Baseline Model | Optimized Model |
| ----------------- | -------------: | --------------: |
| Accuracy          |         73.81% |          80.55% |
| Precision         |         50.43% |          65.72% |
| Recall            |         78.34% |          55.88% |
| F1 Score          |         61.36% |          60.40% |
| ROC-AUC           |         84.13% |          84.12% |

## Key Findings

* The optimized model achieved higher accuracy and precision.
* The baseline model achieved higher recall and F1 Score.
* Both models produced nearly identical ROC-AUC scores.
* The optimized model missed a larger proportion of actual churn customers because of its lower recall.
* Model selection should depend on business objectives, especially the importance of identifying customers who may leave.

## Output Files

The `outputs/` directory contains:

* `final_model_comparison.csv` – Performance comparison of both models.
* `baseline_model_confusion_matrix.csv` – Confusion matrix of the baseline model.
* `optimized_model_confusion_matrix.csv` – Confusion matrix of the optimized model.

## Project Structure

```text
Week-6/
│
├── final_analysis.py
├── README.md
├── Week-6_Final_Telecom_Customer_Churn_ML_Report.docx
│
└── outputs/
    ├── final_model_comparison.csv
    ├── baseline_model_confusion_matrix.csv
    └── optimized_model_confusion_matrix.csv
```

## How to Run

Run the following command from the main project directory:

```bash
python Week-6/final_analysis.py
```

Required Python libraries include:

* pandas
* joblib
* scikit-learn

The script also requires the original dataset and the saved model files from Week-3 and Week-5.

## Future Improvements

* Perform classification threshold tuning.
* Explore alternative machine learning algorithms.
* Improve recall while maintaining acceptable precision.
* Apply additional feature selection techniques.
* Evaluate models using business-focused cost metrics.
* Test the selected model on new customer data.

## Conclusion

Week 6 completes the final analysis and documentation of the Telecom Customer Churn Prediction project. The comparison demonstrates the trade-off between accuracy, precision, and recall when optimizing a classification model. The project provides practical experience in data exploration, preprocessing, model development, evaluation, optimization, and final reporting.

---

**Project Status:** Completed – Week 6