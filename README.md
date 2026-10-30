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
│   └── train.py
├── models/                 # Saved model + scaler (not committed)
├── app/
│   └── app.py
├── reports/
│   └── figures/            # Saved chart PNGs
├── tests/
│   ├── test_preprocess.py
│   └── test_train.py
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
python src/train.py
```

This fits the preprocessing pipeline and Logistic Regression model on the training split and saves the trained model and scaler to `models/`.

## Running the Streamlit App

```powershell
streamlit run app/app.py
```

## Model Evaluation Results

_To be filled in with real metrics once Stage 5 (model validation and tuning) is complete._

## Logistic Regression Explanation

_To be added in Stage 4._

## LDA Comparison

_To be added in Stage 5._

## Key Findings

_To be added once EDA and modeling are complete._

## Limitations

- The dataset is a static snapshot and does not reflect real-time customer behavior changes.
- Model performance is tied to this specific dataset and may not generalize to other telecom providers without retraining.

## Ethical Considerations

Gender is retained in the dataset for descriptive analysis only and is **excluded from the predictive model** to avoid using a protected attribute in retention decisions.

## Conclusion

_To be added in Stage 7 after all results are finalized._
