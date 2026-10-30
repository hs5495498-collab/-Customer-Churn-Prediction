# Project Demonstration Guide

An 8–10 minute walkthrough script for demonstrating this project live (to a professor, examiner, or peer). Each step lists what to show on screen, what to say, and which concept to explain if asked.

**Before you start:** have the following ready in separate terminal/browser tabs:
- VS Code (or terminal) open at the project root with the `.venv` activated
- `notebooks/churn_analysis.ipynb` open in Jupyter (already executed, so outputs are visible without re-running)
- A terminal ready to run `streamlit run app/app.py`

---

## 1. Introduction (≈45 seconds)

**Show:** the GitHub repository page or the project folder in VS Code.

**Say:** "This is my CSE3141 Predictive Analytics mini project — a customer churn prediction system for a telecom company. Churn means a customer cancels their subscription. The goal is to predict, from a customer's account details, how likely they are to churn, so the business can act before losing them."

**Concept to mention if asked:** churn prediction is a binary classification problem — the output is one of two classes, `Churn` or `No Churn`.

## 2. Business Problem (≈30 seconds)

**Show:** `README.md`, Project Overview section.

**Say:** "Acquiring a new customer costs more than keeping an existing one, so even a small improvement in identifying at-risk customers early has real business value. That's the motivation for this whole project."

## 3. Dataset Explanation (≈1 minute)

**Show:** the notebook, Section 4 (Dataset Description), and the actual `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` file if available.

**Say:** "I used the real Telco Customer Churn dataset from Kaggle — 7,043 customers, 21 original columns including demographics, account details, and which services they're subscribed to. The target column is `Churn`, and in this data, about 26.5% of customers churned."

**Concept to mention:** class imbalance — explain that roughly 1 in 4 customers churned, which shaped several later decisions (class weighting, stratified splitting, threshold analysis).

## 4. Data Preprocessing (≈1 minute)

**Show:** `src/preprocess.py`, and the notebook Section 8 (Stage 2) output cells showing the 11 blank `TotalCharges` rows and their `tenure == 0` values.

**Say:** "The raw data needed cleaning: `TotalCharges` was stored as text with 11 blank values. I checked those rows specifically and found every one of them had zero tenure — meaning they're brand-new customers who hadn't been billed yet — so filling them with 0 was the logically correct choice, not a guess. I also dropped `customerID` since it has no predictive value, and I excluded `gender` from the model entirely as an ethical decision — it's never used to make a retention decision."

**Concept to mention if asked:** One-Hot Encoding (turning categories like `Contract` into separate 0/1 columns) and StandardScaler (rescaling numeric columns to comparable ranges), both bundled into a scikit-learn `ColumnTransformer`.

## 5. EDA Demonstration (≈1.5 minutes)

**Show:** the notebook, Section 9, scrolling through the four charts: churn pie chart, churn rate by contract bar chart, tenure histogram, correlation heatmap. Alternatively, open the saved PNGs directly from `reports/figures/`.

**Say:** "The clearest pattern is contract type: month-to-month customers churn at 42.7%, versus just 2.8% for two-year contracts. Churned customers also have much shorter average tenure — about 18 months versus 37.6 months for retained customers. The correlation heatmap confirms tenure has the strongest numeric relationship with churn."

**Concept to mention if asked:** correlation does not imply causation — point to how `tenure` and `TotalCharges` correlate strongly with each other for a purely mechanical reason (billing accumulates over time), not a causal one.

## 6. Logistic Regression Explanation (≈1 minute)

**Show:** notebook Section 10, the coefficient/odds-ratio table.

**Say:** "I used Logistic Regression because it's interpretable and well-suited to a yes/no prediction. It passes a weighted sum of the features through the sigmoid function, which outputs a value between 0 and 1 — a probability. Each feature gets a coefficient; exponentiating it gives an odds ratio. For example, tenure has a negative coefficient — longer-tenured customers are less likely to churn — while Fiber optic internet has a positive coefficient, roughly doubling the odds of churn compared to the baseline category."

**Concept to mention if asked:** `class_weight="balanced"` — compensates for the imbalance by weighting the minority (churn) class more heavily during training.

## 7. Model Validation and Evaluation (≈2 minutes)

**Show:** notebook Section 11 — the cross-validation output, the GridSearchCV ranking table, the LDA comparison table, and the threshold analysis chart.

**Say:** "Rather than trusting one train-test split, I used 5-fold cross-validation, which gave a mean ROC-AUC of 0.846 with a small standard deviation — showing the result is stable. I then tuned the regularization parameter C using GridSearchCV; C=10 came out best, though only marginally better than the default. I also compared against Linear Discriminant Analysis — LDA actually had higher accuracy, but noticeably lower recall and ROC-AUC, so I didn't let accuracy alone decide; I picked Logistic Regression because it's better at actually catching churners. Finally, I selected a classification threshold of 0.60 using cross-validated training predictions only — never touching the test set — and only then evaluated once on the real held-out test set, getting an accuracy of 76.2% and an ROC-AUC of 0.84."

**Concept to mention if asked:** data leakage — explain that the full pipeline (preprocessing + model) is refit inside every single cross-validation fold, so no fold's validation data ever influences its own training.

## 8. Streamlit Application Demonstration (≈1.5 minutes)

**Show:** run `streamlit run app/app.py` live, let it open in the browser.

**Say:** "This app loads the exact trained model from Stage 5 — it doesn't retrain anything. I'll fill in a sample customer." (Fill the form with a month-to-month, short-tenure, Fiber optic customer with no security add-ons.) "Here's the predicted churn probability, and whether it crosses our selected 0.60 threshold to be flagged High Risk. Below that are some simple rule-based suggestions — these are separate from the model, clearly labeled as heuristics, not ML output."

## 9. Example Customer Prediction (≈1 minute)

**Show:** submit the form from Step 8, point to the probability number, the risk badge, and the recommendations list.

**Say:** "For this customer — 2 months tenure, month-to-month contract, high monthly charges, no security add-ons — the model estimates a churn probability around 52%, which is below our 60% threshold, so it's labeled Low Risk in this particular run. I'll also show a clearly low-risk customer for contrast" (switch to a long-tenure, two-year-contract profile) "— here the probability drops to around 20%."

**Concept to mention if asked:** the app uses `predict_proba()` and the *saved* threshold (0.60), not the default 0.50, so the labels shown here exactly match how the model was evaluated in Stage 5.

## 10. Conclusion (≈45 seconds)

**Show:** `README.md`, Conclusion section, or the sidebar's model info panel in the running app.

**Say:** "To summarize: I built a complete pipeline from raw data to a deployed app, validated the model honestly with cross-validation instead of just one split, compared it against a second algorithm instead of assuming Logistic Regression was automatically best, and reported the real test results — accuracy 76.2%, ROC-AUC 0.84 — along with the limitations, like the fact that this model finds correlations, not proven causes of churn. That's the project."

---

## Quick reference: key numbers to have ready

- Dataset: 7,043 customers, 26.54% churn rate
- Baseline test metrics (Stage 4): Accuracy 0.7374, ROC-AUC 0.8418
- Best hyperparameter: C = 10, cross-validated ROC-AUC 0.8463
- LDA comparison: lower ROC-AUC (0.8391) and recall (0.5525) than Logistic Regression
- Selected threshold: 0.60
- Final test metrics: Accuracy 0.7622, Precision 0.5399, Recall 0.7059, F1 0.6118, ROC-AUC 0.8409
- 69 automated tests, all passing
