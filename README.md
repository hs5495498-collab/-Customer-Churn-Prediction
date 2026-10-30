# Customer Churn Prediction Using Logistic Regression

A machine learning mini project for **CSE3141 — Predictive Analytics**, predicting whether a telecom customer is likely to churn (cancel their subscription), using Logistic Regression, validated with cross-validation and hyperparameter tuning, and deployed as an interactive Streamlit application.

## Project Overview

Customer churn is when a customer stops using a company's service. For subscription-based businesses like telecom providers, retaining an existing customer is far cheaper than acquiring a new one, so predicting churn in advance lets a company intervene (loyalty offers, service improvements, proactive support) before the customer actually leaves.

This project builds a complete, reproducible pipeline — from raw data to a usable application:
1. Clean and explore the real Telco Customer Churn dataset.
2. Train a Logistic Regression classifier to estimate each customer's churn probability.
3. Validate it rigorously with 5-fold cross-validation, tune its regularization strength, and compare it against a second algorithm (LDA).
4. Select a final model and classification threshold using only training data, then evaluate once on a held-out test set.
5. Serve the trained model through a Streamlit web app where a user can enter a hypothetical customer's details and see a churn risk assessment.

## Problem Statement

Given a customer's account details and subscribed services (never their gender or customer ID — see Ethical Considerations), predict whether they will churn (`Yes`/`No`), and estimate how likely that is as a probability.

## Project Objectives

- **Business objective:** identify customers at high risk of churning so a business can prioritize retention efforts toward those customers specifically.
- **Machine learning objective:** build and rigorously validate a binary classifier that outputs both a churn probability and a risk label, using a methodology that avoids data leakage at every stage (preprocessing, cross-validation, hyperparameter tuning, and threshold selection).
- **Educational objective:** demonstrate a complete, explainable predictive-analytics workflow — problem framing, cleaning, EDA, modeling, validation, and deployment — suitable for a CSE3141 viva.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset from Kaggle (originally published by IBM).

- **Rows:** 7,043 customers
- **Original columns:** 21 (20 descriptive/input columns + 1 target column)
- **Target variable:** `Churn` (`Yes`/`No`) — 1,869 churned (26.54%), 5,174 retained (73.46%)

**Important input features** (17 used for modeling, after dropping `customerID` and excluding `gender` — see Ethical Considerations below): `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`.

**Obtaining the dataset:** download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from the [Kaggle Telco Customer Churn page](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (a free Kaggle account is required). The raw CSV is **not included in this repository** (see Dataset Setup below) and is not redistributed here due to its licensing on Kaggle.

**Expected local path:** `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Technologies Used

- **Python 3.12**
- **pandas**, **numpy** — data loading and manipulation
- **matplotlib**, **seaborn** — visualization
- **scikit-learn** — preprocessing (`ColumnTransformer`, `OneHotEncoder`, `StandardScaler`), Logistic Regression, LDA, `GridSearchCV`, cross-validation, evaluation metrics
- **joblib** — saving/loading the trained model pipeline
- **Jupyter Notebook** — exploratory analysis and the full documented modeling workflow
- **Streamlit** — the interactive demo web application
- **pytest** — automated tests (69 tests across the project)

## Project Directory Structure

```
Customer-Churn-Prediction/
│
├── data/                              # Raw dataset (not committed — see Dataset Setup)
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── churn_analysis.ipynb           # Full documented workflow, Stages 1-5
├── src/
│   ├── __init__.py
│   ├── preprocess.py                  # Stage 2: cleaning, ColumnTransformer
│   ├── train.py                       # Stage 4: split, baseline LR pipeline, coefficients
│   └── validation.py                  # Stage 5: cross-validation, tuning, LDA, threshold, final model
├── models/                            # Saved model artifacts (gitignored — see Model Artifact Setup)
├── app/
│   ├── app.py                         # Streamlit application
│   └── utils.py                       # App logic: validation, prediction, recommendations
├── reports/
│   ├── figures/                       # Saved chart PNGs (EDA, evaluation)
│   ├── screenshots/                   # Application screenshots (manual capture — see its README)
│   ├── final_project_report.md        # Full academic project report
│   ├── demo_guide.md                  # Live demonstration script
│   ├── viva_questions.md              # 50+ viva questions and answers
│   └── setup_and_execution_guide.md   # Fresh-machine setup walkthrough
├── tests/
│   ├── test_preprocess.py
│   ├── test_train.py
│   ├── test_validation.py
│   └── test_app.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Data Cleaning and Preprocessing

Implemented in `src/preprocess.py`, Stage 2:

- **Missing-value treatment:** `TotalCharges` is stored as text in the raw CSV with 11 blank-string rows. These are converted to proper missing values with `pd.to_numeric(errors="coerce")`, then verified to all belong to customers with `tenure == 0` (no completed billing cycle yet) before being filled with `0.0` — never filled blindly.
- **TotalCharges conversion:** from text to a proper numeric (`float64`) type.
- **Duplicate checking:** `df.drop_duplicates()` is run (0 duplicate rows found in this dataset).
- **customerID exclusion:** dropped — a unique identifier has no predictive value.
- **Gender exclusion from predictive features:** see Ethical Considerations.
- **Outlier investigation:** the IQR method was applied to `tenure`, `MonthlyCharges`, and `TotalCharges` — 0 outliers were found in all three (reported honestly, not assumed).
- **One-Hot Encoding:** `OneHotEncoder(handle_unknown="ignore")` converts each categorical column (e.g. `Contract`) into binary columns; `handle_unknown="ignore"` means a category never seen during training is encoded as all zeros instead of crashing.
- **StandardScaler:** rescales numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`) to mean 0, standard deviation 1.
- **ColumnTransformer and Pipeline:** `build_preprocessing_pipeline()` bundles the encoder and scaler into one `ColumnTransformer`, which is itself one step in a full scikit-learn `Pipeline` alongside the classifier — so `.fit()` on the pipeline fits preprocessing and the model together, only ever on training data (see Model Validation Methodology for how this prevents data leakage).

## Exploratory Data Analysis

Implemented in `notebooks/churn_analysis.ipynb`, Section 9, with charts saved to `reports/figures/`:

1. **Overall churn distribution (pie chart)** — 26.54% churned vs. 73.46% retained, confirming class imbalance.
2. **Churn rate by contract type (bar chart)** — Month-to-month 42.71%, One year 11.27%, Two year 2.83%: contract length is one of the strongest behavioral signals of churn.
3. **Tenure distribution (histogram)** — a bimodal shape, with large groups of customers near 0 months (recently joined) and near 72 months (long-term loyal), fewer in between.
4. **Correlation heatmap** — `tenure` is the numeric feature most correlated with churn (−0.35), followed by `MonthlyCharges` (+0.19); `tenure` and `TotalCharges` correlate strongly with each other (+0.83), a mechanical relationship (TotalCharges accumulates over tenure), not a causal one.

Additional real findings: churned customers average ~18.0 months tenure vs. ~37.6 months for retained customers, and churned customers pay a *higher* average monthly charge (~$74.44) than retained customers (~$61.27).

## Model Building

Implemented in `src/train.py`, Stage 4:

- **Logistic Regression** models the probability of churn by passing a weighted sum of input features through the **sigmoid function**, `σ(z) = 1 / (1 + e^-z)`, which maps any real number to a value between 0 and 1 — a valid probability.
- **80/20 stratified train-test split** (`stratify=y`, `random_state=42`) keeps the ~26.5% churn ratio consistent between training and test sets.
- **`class_weight="balanced"`** automatically weights the minority (churn) class more heavily during training, compensating for the imbalance without altering the dataset itself.
- Each feature gets a learned **coefficient**; `exp(coefficient)` gives an **odds ratio** showing how much more (or less) likely churn becomes per unit (or category) change. The five most influential features in the Stage 4 baseline model, by coefficient magnitude, were `tenure`, `Contract_Two year`, `InternetService_Fiber optic`, `MonthlyCharges`, and `Contract_Month-to-month`.

## Model Validation and Hyperparameter Tuning

Implemented in `src/validation.py` and `notebooks/churn_analysis.ipynb` Section 11, Stage 5:

- **Stratified K-Fold / 5-fold cross-validation:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` splits the training data into 5 folds that each preserve the ~26.5% churn ratio, giving 5 independent performance estimates instead of trusting one lucky/unlucky split.
- **GridSearchCV:** tunes the regularization strength `C` over `[0.001, 0.01, 0.1, 1, 10, 100]` using `scoring="roc_auc"` and the same 5-fold CV, with the **full preprocessing-plus-classifier pipeline** passed as the estimator so preprocessing is refit inside every fold — never on the complete training set before splitting.
- **C parameter:** the inverse of regularization strength — small `C` means strong regularization (risk of underfitting), large `C` means weak regularization (risk of overfitting).
- **LDA comparison:** a Linear Discriminant Analysis model was compared under identical cross-validation (compatibility with the existing dense-output preprocessing pipeline was verified first, not assumed).
- **Classification threshold selection:** thresholds from 0.30 to 0.60 were evaluated using leakage-safe **out-of-fold** cross-validated probabilities on the training set only (never the test set), then the F1-maximizing threshold was selected.

### Cross-Validation Results (training data only)

| Model (5-fold CV mean) | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline, C=1.0) | 0.7490 | 0.5176 | 0.8020 | 0.6289 | 0.8462 |
| Logistic Regression (tuned, C=10) | 0.7483 | 0.5166 | 0.8007 | 0.6279 | **0.8463** |
| LDA | 0.7984 | 0.6396 | 0.5525 | 0.5923 | 0.8391 |

**Best C: 10**, selected via `GridSearchCV` with cross-validated ROC-AUC **0.8463**. LDA achieved higher accuracy and precision but noticeably lower recall and ROC-AUC — since catching actual churners (recall) and threshold-independent ranking quality (ROC-AUC) matter more than raw accuracy for a retention use case, **tuned Logistic Regression (C=10) was selected as the final model**. This comparison is reported honestly: LDA did not outperform Logistic Regression on the primary selection metric.

**Selected classification threshold: 0.60** (maximizes F1-score on out-of-fold training predictions within the tested range).

## Final Model Performance

These results are kept clearly distinct from the cross-validation results above: **cross-validation** metrics were used to *select* the model and threshold (training data only); the numbers below come from a **single evaluation on the held-out test set**, performed only once, after selection was already finalized.

**Held-out test results at the selected threshold (0.60):**

| Metric | Value |
|---|---|
| Accuracy | 0.7622 |
| Precision | 0.5399 |
| Recall | 0.7059 |
| F1-score | 0.6118 |
| ROC-AUC | 0.8409 |

**Confusion matrix (test set, n=1,409):**

|  | Predicted No Churn | Predicted Churn |
|---|---|---|
| **Actual No Churn** | TN = 810 | FP = 225 |
| **Actual Churn** | FN = 110 | TP = 264 |

For comparison, the same tuned model at the conventional default threshold (0.50) scored Accuracy 0.7388, Precision 0.5052, Recall 0.7834, F1 0.6143, ROC-AUC 0.8409 — illustrating the precision/recall trade-off from raising the threshold. The original Stage 4 baseline (C=1.0, threshold 0.50) scored Accuracy 0.7374, Precision 0.5034, Recall 0.7834, F1 0.6130, ROC-AUC 0.8418. Full details and discussion: `notebooks/churn_analysis.ipynb` Section 11.8.

## Streamlit Application

`app/app.py` ("Customer Churn Prediction System") loads the Stage 5 final model artifact and uses it, unmodified, to score a single customer entered through a form:

- **Input form:** tenure, monthly/total charges, contract type, billing/payment method, internet service and 6 service add-ons, and demographic details (senior citizen, partner, dependents, phone service, multiple lines) — **never gender or customer ID**. Dropdown options are read directly from the fitted `OneHotEncoder`'s categories, so they can never drift out of sync with what the model was trained on.
- **Input validation:** numeric ranges checked against real observed dataset bounds; categorical values checked against the exact fitted categories; a tenure=0 customer is required to have TotalCharges=0, consistent with Stage 2's cleaning logic.
- **Prediction workflow:** the validated input is built into a single-row DataFrame with the exact feature columns and order the saved pipeline expects, then `pipeline.predict_proba()` returns a churn probability, which is compared against the **saved threshold (0.60)** — not the default 0.50 — to produce a "Likely to Churn"/"Likely to Stay" label and a High Risk/Low Risk badge.
- **Retention recommendations:** simple, explicitly-labeled rule-based suggestions (e.g. "month-to-month contract → consider a longer-term plan") derived from the entered details — clearly distinguished from the model's own output.
- **Model information section:** shows the real Stage 5 cross-validation and held-out test metrics, algorithm, hyperparameter, and threshold.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Setup

1. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from the [Kaggle Telco Customer Churn page](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
2. Place it at: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Model Artifact Setup

**Important:** `models/final_churn_model.joblib` (the trained pipeline the Streamlit app loads) is **excluded from Git** via `.gitignore` (`models/*.joblib`) — it is a large, regenerable binary artifact, not source code, so a fresh clone of this repository will **not** include it.

To generate it after cloning, with the real dataset in place (see Dataset Setup above), run **one** of the following — both execute the exact same leakage-safe Stage 5 methodology (cross-validation and hyperparameter tuning on training data only, a single held-out test evaluation, no shortcuts, no different model):

**Option A — standalone script (faster, no Jupyter needed):**
```powershell
python -m src.validation
```
This loads and cleans the data, tunes Logistic Regression via `GridSearchCV`, compares it against LDA, selects the final model and threshold using cross-validation only, evaluates once on the held-out test set, and saves the artifact to `models/final_churn_model.joblib`. It prints every intermediate result as it runs.

**Option B — run the notebook:**
```powershell
jupyter notebook notebooks/churn_analysis.ipynb
```
Run all cells; Section 11 performs the same steps and saves the same artifact.

Both options are deterministic (every `random_state` in the pipeline is fixed to `42`), so either one reproduces the documented results above: best C = 10, selected threshold = 0.60, held-out test ROC-AUC ≈ 0.8409.

## Running the Notebook

```powershell
jupyter notebook notebooks/churn_analysis.ipynb
```

## Running the Application

```powershell
streamlit run app/app.py
```

Run from the **project root** (not from inside `app/`). The app will show a clear error and refuse to proceed if `models/final_churn_model.joblib` is missing — see Model Artifact Setup above.

## Running Tests

```powershell
pytest tests/ -v
```

69 tests across `tests/test_preprocess.py`, `tests/test_train.py`, `tests/test_validation.py`, and `tests/test_app.py` — covering data cleaning, the modeling pipeline, cross-validation/tuning, and the Streamlit app's logic (including two headless end-to-end app tests).

## Limitations

- **Predictions are statistical estimates, not guarantees.** The model outputs a probability based on patterns learned from historical data — it cannot account for information outside the dataset (e.g. a customer's recent support interactions, competitor offers, or personal circumstances).
- **The model learns correlational patterns from historical data,** not proven causal drivers of churn — see the next point.
- **Correlation does not establish causation.** A feature can be a strong, genuine statistical predictor (e.g. Fiber optic internet service) without being the actual cause of a customer leaving; there could be confounding factors behind the association.
- **Dataset limitations:** a single static snapshot from one telecom provider; does not capture behavior changes over time, and may not generalize to a different company's customer base without retraining.
- **Threshold trade-offs:** the selected threshold (0.60) was chosen by maximizing F1-score within a specific tested range (0.30–0.60); F1 was still rising at the top of that range, so a wider search might find a different optimum, and a real business might deliberately choose a different threshold based on the actual cost of a missed churner vs. a false alarm (see `notebooks/churn_analysis.ipynb` Section 11.5).
- **Single held-out test evaluation:** even after cross-validation-based selection, the final numbers come from one test split; results could shift somewhat on a different random split.

## Future Improvements

- Expand the hyperparameter search (finer/randomized grid, additional regularization types such as L1 for built-in feature selection).
- Try additional algorithms (e.g. Random Forest, Gradient Boosting) for comparison, using the same leakage-safe cross-validation methodology already in place.
- Incorporate a business-specific cost matrix (cost of a false negative vs. a false positive) to select the threshold analytically rather than by F1-maximization alone.
- Add a persistent prediction log (with explicit user consent) to let the Streamlit app show genuine historical prediction trends, rather than only single-customer predictions.
- Explore feature engineering (e.g. interaction terms between contract type and tenure) to see whether it improves ranking performance.

## Ethical Considerations

`gender` is retained in the dataset and notebook for descriptive analysis only (e.g. EDA) and is **excluded from the predictive feature set** in `src/preprocess.py`, `src/train.py`, and the Streamlit app — it is never used by the trained model or to influence any retention decision. `customerID` is similarly excluded as a non-predictive identifier.

## Screenshots

Application screenshots require manual capture in this environment (no automated browser screenshot tool was available) — see `reports/screenshots/README.md` for exact capture instructions. No screenshot images are embedded here until those files are actually added, to avoid referencing non-existent images.

## Additional Documentation

- [`reports/final_project_report.md`](reports/final_project_report.md) — full academic-style project report
- [`reports/demo_guide.md`](reports/demo_guide.md) — live demonstration script
- [`reports/viva_questions.md`](reports/viva_questions.md) — 50+ viva questions and answers
- [`reports/setup_and_execution_guide.md`](reports/setup_and_execution_guide.md) — fresh-machine setup walkthrough

## Conclusion

This project demonstrates a complete, honestly-evaluated predictive analytics workflow: cleaning and exploring a real dataset, building an interpretable Logistic Regression model, validating it rigorously with cross-validation and hyperparameter tuning, comparing it against a second algorithm, and deploying the selected model through a working interactive application — with every reported number generated from actual code execution against the real Telco Customer Churn dataset, and every limitation reported alongside the results rather than hidden.

## Author and Academic Project Information

- **Course:** CSE3141 — Predictive Analytics
- **Project:** Customer Churn Prediction Using Logistic Regression
- **Student:** Harsh Kumar Singh
