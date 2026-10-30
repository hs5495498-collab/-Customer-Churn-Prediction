# Customer Churn Prediction using Logistic Regression

A machine learning mini project for **CSE3141 — Predictive Analytics**, predicting whether a telecom customer is likely to churn (cancel their subscription) using Logistic Regression.

## Project Overview

Customer churn is when a customer stops using a company's service. For subscription-based businesses like telecom providers, retaining an existing customer is far cheaper than acquiring a new one, so predicting churn in advance lets a company intervene (offer discounts, improve service, etc.) before the customer leaves.

This project builds a Logistic Regression model that estimates the probability that a given customer will churn, based on their account and service details, and compares it against a Linear Discriminant Analysis (LDA) baseline.

## Problem Statement

Given a customer's demographic information, account details, and subscribed services, predict whether they will churn (`Yes`/`No`).

## Business Objective

Identify customers at high risk of churning so the business can proactively offer retention incentives, reducing revenue loss.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset from Kaggle — 7,043 customer records, 21 columns, including demographics, account information, subscribed services, and the target column `Churn`.

The raw CSV is **not included in this repository** (see Dataset Setup below).

## Technologies Used

- Python 3.12
- pandas, numpy — data loading and manipulation
- matplotlib, seaborn — visualization
- scikit-learn — preprocessing, Logistic Regression, LDA, model evaluation
- joblib — saving/loading the trained model
- Jupyter Notebook — exploratory analysis and experimentation
- Streamlit — interactive demo web app
- pytest — automated tests

## Project Structure

```
Customer-Churn-Prediction/
│
├── data/                   # Raw dataset (not committed, see below)
├── notebooks/
│   └── churn_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train.py
│   └── validation.py
├── models/                 # Saved model artifacts (not committed, see below)
├── app/
│   └── app.py
├── reports/
│   └── figures/            # Saved chart PNGs
├── tests/
│   ├── test_preprocess.py
│   ├── test_train.py
│   └── test_validation.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Setup

1. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from the [Kaggle Telco Customer Churn page](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
2. Place it at: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Running the Notebook

```powershell
jupyter notebook notebooks/churn_analysis.ipynb
```

## Training the Model

```powershell
python -m src.train
```

This fits the preprocessing pipeline and baseline Logistic Regression model (`class_weight="balanced"`, default `C=1.0`) on the 80% training split and saves the fitted pipeline to `models/logistic_regression_pipeline.joblib`. The full cross-validation, hyperparameter tuning, model comparison, and final model selection described below are run from the notebook (Section 11), which saves the selected final pipeline to `models/final_churn_model.joblib`.

## Running the Streamlit App

```powershell
streamlit run app/app.py
```

## Model Validation Methodology

**5-fold stratified cross-validation:** rather than trusting a single 80/20 split, the baseline Logistic Regression was re-evaluated with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the training data only, giving 5 independent accuracy/precision/recall/F1/ROC-AUC estimates and their standard deviation.

**Hyperparameter tuning (GridSearchCV):** the regularization strength `C` was tuned over `[0.001, 0.01, 0.1, 1, 10, 100]` using the same 5-fold stratified CV and `scoring="roc_auc"`, with the full preprocessing-plus-classifier pipeline passed as the estimator so preprocessing is refit inside every fold (never on the full training set before splitting). **Best C: 10**, cross-validated ROC-AUC: **0.8463**.

**Logistic Regression vs. LDA:** a Linear Discriminant Analysis model (reusing the same preprocessing pipeline, verified compatible with dense output for this dataset) was compared against both the baseline and tuned Logistic Regression models under identical 5-fold cross-validation:

| Model (5-fold CV mean) | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline, C=1.0) | 0.7490 | 0.5176 | 0.8020 | 0.6289 | 0.8462 |
| Logistic Regression (tuned, C=10) | 0.7483 | 0.5166 | 0.8007 | 0.6279 | 0.8463 |
| LDA | 0.7984 | 0.6396 | 0.5525 | 0.5923 | 0.8391 |

LDA achieved higher accuracy and precision, but noticeably lower recall and ROC-AUC than Logistic Regression. Since catching actual churners (recall) and threshold-independent ranking quality (ROC-AUC) matter more than raw accuracy for a retention use case, **tuned Logistic Regression (C=10) was selected as the final model** — LDA did not outperform it on the primary selection metric (ROC-AUC).

**Threshold selection:** classification thresholds from 0.30 to 0.60 were evaluated using leakage-safe out-of-fold cross-validated probabilities on the training set (never the test set). The threshold maximizing F1-score within that range was **0.60**, which trades some recall for higher precision compared to the default 0.50 — this trade-off is reported transparently rather than hidden (see the notebook, Section 11.5, for the full table and discussion).

## Model Evaluation Results

**Stage 4 baseline** (single 80/20 split, default C=1.0, threshold=0.50):

| Metric | Value |
|---|---|
| Accuracy | 0.7374 |
| Precision | 0.5034 |
| Recall | 0.7834 |
| F1-score | 0.6130 |
| ROC-AUC | 0.8418 |

**Stage 5 final model** (tuned Logistic Regression, C=10) evaluated once on the same held-out test set, reported at both the default threshold and the selected F1-optimal threshold:

| Metric | @ threshold 0.50 | @ selected threshold 0.60 |
|---|---|---|
| Accuracy | 0.7388 | 0.7622 |
| Precision | 0.5052 | 0.5399 |
| Recall | 0.7834 | 0.7059 |
| F1-score | 0.6143 | 0.6118 |
| ROC-AUC | 0.8409 | 0.8409 |

Confusion matrix @ selected threshold 0.60 (test set, n=1,409): TN=810, FP=225, FN=110, TP=264.

Hyperparameter tuning alone (C=1.0 → C=10, threshold held at 0.50) had a small effect on this dataset — ROC-AUC barely moved (0.8418 → 0.8409). Raising the threshold to 0.60 clearly traded recall for precision, exactly as the threshold analysis predicted.

## Logistic Regression Explanation

Logistic Regression models the probability that a customer churns by passing a weighted sum of the input features through the sigmoid function, `σ(z) = 1 / (1 + e^-z)`, which maps any real number to a probability between 0 and 1. Each feature gets a learned coefficient; its sign shows whether that feature increases or decreases predicted churn risk, and `exp(coefficient)` gives an odds ratio — how many times more (or less) likely churn becomes per unit (or per category). `class_weight="balanced"` compensates for the dataset's ~26.5% churn rate by weighting the minority class more heavily during training. See `notebooks/churn_analysis.ipynb` Section 10.8 for the full coefficient table and Section 11.3 for how its regularization strength (`C`) was tuned.

## LDA Comparison

See the Model Validation Methodology section above and `notebooks/churn_analysis.ipynb` Section 11.4 for the full comparison, including why LDA's higher accuracy did not make it the better choice for this churn-prediction use case.

## Key Findings

- Churn is heavily concentrated in **Month-to-month contracts** (42.71% churn rate) vs. One-year (11.27%) and Two-year (2.83%).
- Longer-tenured customers churn far less: churned customers average ~18.0 months tenure vs. ~37.6 months for retained customers.
- The top predictive features by model coefficient magnitude are `tenure`, `Contract_Two year`, `InternetService_Fiber optic`, `MonthlyCharges`, and `Contract_Month-to-month` (see notebook Section 10.9).
- Hyperparameter tuning gave only a marginal ROC-AUC improvement over the untuned baseline, suggesting the model/feature set is not highly sensitive to this particular regularization setting.
- LDA achieved higher accuracy than Logistic Regression but substantially lower recall, illustrating why accuracy alone is a misleading metric on an imbalanced dataset.

## Limitations

- The dataset is a static snapshot and does not reflect real-time customer behavior changes.
- Model performance is tied to this specific dataset and may not generalize to other telecom providers without retraining.

## Ethical Considerations

Gender is retained in the dataset for descriptive analysis only and is **excluded from the predictive model** to avoid using a protected attribute in retention decisions.

## Conclusion

_To be added in Stage 7 after all results are finalized._
