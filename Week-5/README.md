# Week-5: Model Optimization and Experimentation

## Project Overview

This week focuses on optimizing the Logistic Regression model developed for the Telecom Customer Churn Prediction project. Grid Search and Randomized Search were used to identify suitable hyperparameter configurations and compare their performance.

## Objectives

* Understand hyperparameter tuning.
* Apply Grid Search and Randomized Search.
* Compare different model configurations.
* Evaluate optimized models using classification metrics.
* Examine the effect of optimization on churn prediction.

## Dataset

IBM Telco Customer Churn Dataset

* Total records: 7,043
* Training records: 5,634
* Testing records: 1,409

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

## Optimization Methods

### 1. Grid Search

GridSearchCV tested 20 parameter combinations using 5-fold stratified cross-validation.

### 2. Randomized Search

RandomizedSearchCV tested 15 sampled parameter combinations using 5-fold stratified cross-validation.

## Best Parameters

Grid Search:

* C: 10
* Class Weight: None
* Solver: liblinear

Randomized Search:

* C: 10
* Class Weight: None
* Solver: lbfgs

## Experimental Results

| Metric    | Grid Search | Randomized Search |
| --------- | ----------: | ----------------: |
| Accuracy  |      80.48% |            80.55% |
| Precision |      65.52% |            65.72% |
| Recall    |      55.88% |            55.88% |
| F1-score  |      60.32% |            60.40% |
| ROC-AUC   |      84.11% |            84.12% |

Randomized Search achieved marginally better test accuracy, precision, F1-score, and ROC-AUC than Grid Search.

## Key Observations

* Accuracy and precision improved compared with the Week-4 baseline.
* Recall decreased, indicating that more actual churn customers were missed.
* ROC-AUC remained almost unchanged.
* Further experiments involving class weighting and decision-threshold tuning may improve churn detection.

## Output Files

* `optimization_comparison.csv`
* `best_parameters.json`
* `best_optimized_model.joblib`

## How to Run

From the project root directory:

```bash
python Week-5/optimize_model.py
```

## Conclusion

Week-5 provided practical experience in hyperparameter optimization and model experimentation. Grid Search and Randomized Search helped identify configurations with improved accuracy and precision. However, the reduction in recall highlights the need to balance performance metrics according to the business objective. Further optimization can focus on improving churn detection while maintaining reliable generalization.