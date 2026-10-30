# Viva Question Bank

50+ likely viva questions and answers, organized by topic, with real project-specific values where relevant. Answers are written to be understandable and technically accurate for a second-year B.Tech CSE student.

---

## A. Project Overview

**1. What is this project about?**
A predictive analytics system that estimates the probability a telecom customer will churn (cancel their subscription), built using Logistic Regression, validated with cross-validation and hyperparameter tuning, and deployed as a Streamlit web application.

**2. What were the main stages of this project?**
Problem definition, data cleaning and preprocessing, exploratory data analysis, Logistic Regression model building, model validation and hyperparameter tuning, a Streamlit application, and final documentation — each developed and merged through its own Git branch and pull request.

**3. Why is churn prediction important for a business?**
Acquiring a new customer costs significantly more than retaining an existing one. Predicting churn in advance lets a business proactively intervene (discounts, better service, outreach) before losing a customer, rather than reacting after the fact.

**4. What was your overall development workflow?**
Each stage was implemented on its own feature branch, tested (automated tests plus notebook execution), reviewed via `git diff`, committed, pushed, opened as a pull request, and merged into `main` — then the next stage pulled the updated `main` before starting.

**5. What is the final output of this project?**
A trained, validated Logistic Regression pipeline (`models/final_churn_model.joblib`) and a Streamlit web app that lets a user enter a hypothetical customer's details and see a churn probability, a risk label, and rule-based retention suggestions.

---

## B. Dataset and Business Problem

**6. Which dataset did you use, and how large is it?**
The Telco Customer Churn dataset from Kaggle — 7,043 customer records, 21 original columns.

**7. What is the target variable, and what is its distribution?**
`Churn` (`Yes`/`No`). Real distribution: 1,869 churned (26.54%), 5,174 retained (73.46%) — a moderately imbalanced dataset.

**8. Why does the churn rate matter for modeling choices?**
Because of the imbalance, accuracy alone can be misleading (always predicting "No churn" would already score ~73.5%). This motivated `class_weight="balanced"`, stratified splitting, and ROC-AUC/recall-focused model selection.

**9. Which columns did you exclude from the model, and why?**
`customerID` (a unique identifier with no predictive value) and `gender` (excluded as an ethical safeguard — see Section C).

**10. What were the input features actually used?**
17 features: `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`.

---

## C. Data Cleaning and Preprocessing

**11. What problem did you find with the `TotalCharges` column, and how did you fix it?**
It was stored as text, with 11 rows containing a blank string instead of a number. `pd.to_numeric(..., errors="coerce")` converted these to `NaN`, then I verified all 11 affected rows had `tenure == 0` (no completed billing cycle) before filling them with 0.0 — a justified value, not a guess.

**12. Why did you check `tenure` before filling the missing `TotalCharges` values?**
To confirm the fill was logically correct rather than arbitrary. If any missing row had non-zero tenure, filling with 0 would misrepresent a customer who had actually been billed — my code explicitly raises an error in that case instead of silently filling.

**13. Did you find any duplicate rows?**
No — `df.drop_duplicates()` found 0 duplicate rows in this dataset.

**14. How did you detect outliers, and what did you find?**
Using the IQR (Interquartile Range) method: `Q1`/`Q3` (25th/75th percentiles), `IQR = Q3 - Q1`, bounds at `Q1 - 1.5×IQR` and `Q3 + 1.5×IQR`. All three numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`) had **0 outliers** — a genuine result, not an oversight, so no rows were removed.

**15. What is One-Hot Encoding, and why did you need it?**
It converts a categorical column (e.g. `Contract` with values `Month-to-month`/`One year`/`Two year`) into multiple binary (0/1) columns, since Logistic Regression requires numeric input and has no natural way to interpret category text.

**16. What does `handle_unknown="ignore"` do in `OneHotEncoder`, and why is it useful here?**
If a category appears at prediction time that wasn't seen during training, it's encoded as all zeros instead of raising an error — important for the Streamlit app, which could in principle receive an unexpected value.

**17. What is StandardScaler, and why use it?**
It rescales a numeric column to mean 0 and standard deviation 1, so columns on very different natural scales (months vs. currency) contribute more comparably to the model instead of the larger-magnitude column dominating purely due to its units.

**18. What is a ColumnTransformer?**
A scikit-learn tool that applies different transformations to different columns in one object — here, `OneHotEncoder` to categorical columns and `StandardScaler` to numeric columns — bundled into a single reusable preprocessing step.

**19. Why was the preprocessing pipeline left unfitted in `src/preprocess.py`?**
To prevent data leakage — it must only be fit on training data (or a cross-validation fold's training portion), never on the complete dataset before splitting, which the actual fitting code in `src/train.py`/`src/validation.py` enforces.

---

## D. Exploratory Data Analysis

**20. What are the four main charts you created, and what did each show?**
(1) Pie chart — 26.54% churn overall. (2) Bar chart — churn rate by contract type: Month-to-month 42.71%, One year 11.27%, Two year 2.83%. (3) Histogram — tenure distribution is bimodal, with many customers near 0 and near 72 months. (4) Correlation heatmap — tenure is the numeric feature most correlated with churn (−0.35).

**21. Why did you compute churn *rate* by contract type instead of raw counts?**
Because contract groups have very different sizes (3,875 Month-to-month vs. 1,473 One-year vs. 1,695 Two-year customers); a rate normalizes for that and allows a fair comparison.

**22. What does a negative correlation between tenure and churn mean?**
As tenure increases, churn (coded as 1=Yes, 0=No) tends to decrease — longer-tenured customers are less likely to churn, on average, in this dataset.

**23. Why does `tenure` correlate so strongly with `TotalCharges` (+0.83)?**
Because `TotalCharges` accumulates from monthly billing over the customer's tenure — a mechanical/mathematical relationship, not evidence that tenure causes higher total billing in a meaningful business sense.

**24. Why is correlation not the same as causation?**
Two variables can move together without one causing the other — there could be a third factor driving both, or the relationship could be a structural artifact of how the data was generated (as with tenure and TotalCharges above).

**25. What did you find about monthly charges and churn?**
Counterintuitively, churned customers had a *higher* average monthly charge (~$74.44) than retained customers (~$61.27) — suggesting price sensitivity plays some role, though tenure/contract length showed a stronger relationship with churn overall.

---

## E. Logistic Regression

**26. Why did you choose Logistic Regression for this problem?**
It directly models the probability of a binary outcome via the sigmoid function, is interpretable (coefficients show direction and relative strength of association), is a standard, fast baseline for tabular binary classification, and supports `class_weight="balanced"` for imbalanced data.

**27. Explain the sigmoid function and why it's used.**
`σ(z) = 1 / (1 + e^-z)`. It maps any real number `z` to a value strictly between 0 and 1, which is exactly what's needed to represent a probability — unlike plain linear regression, which could output values outside [0, 1].

**28. What is `z` in the Logistic Regression formula?**
A linear combination of the input features: `z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ`, where `β₀` is the intercept and each `βᵢ` is a learned coefficient.

**29. What does a coefficient represent in Logistic Regression?**
Its sign shows whether that feature increases (`positive`) or decreases (`negative`) the predicted log-odds of churn; its magnitude shows the strength of that association, holding other features constant.

**30. What is an odds ratio, and how do you compute it?**
`exp(coefficient)`. An odds ratio above 1 means the feature is associated with increased churn odds; below 1 means decreased odds. Example: `InternetService_Fiber optic` had coefficient +0.701, giving an odds ratio of 2.016 — roughly double the churn odds compared to the reference category.

**31. What were the five most influential features in your baseline model?**
By coefficient magnitude: `tenure` (−1.153), `Contract_Two year` (−0.778), `InternetService_Fiber optic` (+0.701), `MonthlyCharges` (−0.675), `Contract_Month-to-month` (+0.651).

**32. Why can't you say a feature "causes" churn just because it has a large coefficient?**
Logistic Regression finds statistical associations in observational historical data; it cannot rule out confounding variables or reverse causation. Proving causation would require a controlled experiment.

**33. What does `class_weight="balanced"` do?**
It automatically re-weights the loss function so that misclassifying the minority class (churners) is penalized more, proportional to class frequency — compensating for the ~26.5%/73.5% imbalance without changing the data itself.

---

## F. Train-Test Split and Data Leakage

**34. What split ratio did you use, and why?**
80% training (5,634 customers), 20% testing (1,409 customers) — a standard split that leaves enough data to train a reliable model while keeping a meaningfully sized held-out test set.

**35. What does `stratify=y` do, and why is it important here?**
It forces both the training and test sets to preserve the same class ratio as the full dataset (~26.5% churn). Without it, a random split could, by chance, put too few churners in the test set, making metrics like recall unreliable.

**36. How did you verify stratification worked?**
By printing the churn rate in both sets — both came out to exactly 0.2654, matching the full dataset's rate.

**37. What is data leakage, and how did you prevent it?**
Data leakage is when information from data the model shouldn't yet "see" (like the test set, or a cross-validation fold's validation portion) influences training or tuning, producing misleadingly optimistic results. I prevented it by always bundling preprocessing and the classifier into one scikit-learn `Pipeline` and fitting that single pipeline only on training data (or a fold's training portion) — the `OneHotEncoder`/`StandardScaler` statistics are never learned from data outside that fit call.

**38. How do you know preprocessing wasn't accidentally fit on the full dataset?**
I wrote an explicit test comparing a `StandardScaler` fit on training data only against one fit on the combined train+test data — their learned means differ, proving the training-only fit actually changes what's learned (i.e., the test rows weren't silently included).

**39. Why is `random_state=42` used everywhere?**
To make every split, shuffle, and model fit reproducible — running the code again produces the identical result, which matters for being able to explain and reproduce specific reported numbers.

---

## G. Cross-Validation and GridSearchCV

**40. What is k-fold cross-validation, and why did you use 5 folds?**
The training data is split into k parts; the model trains k times, each time validating on a different held-out part, producing k independent performance estimates. 5 is a standard choice — enough for a reliable estimate without the computational cost of very high fold counts.

**41. Why `StratifiedKFold` instead of plain `KFold`?**
It keeps each fold's churn ratio consistent with the full dataset, avoiding folds with too few churners to evaluate meaningfully, given the dataset's imbalance.

**42. What were your cross-validated baseline results?**
Mean Accuracy 0.7490 (±0.0150), Precision 0.5176 (±0.0186), Recall 0.8020 (±0.0376), F1 0.6289 (±0.0228), ROC-AUC 0.8462 (±0.0125) — the small standard deviations show the result is stable across folds.

**43. What is `GridSearchCV`, and what did you tune with it?**
An exhaustive search tool that tries every value in a parameter grid, scores each with cross-validation, and returns the best. I tuned Logistic Regression's regularization parameter `C` over `[0.001, 0.01, 0.1, 1, 10, 100]`.

**44. What is regularization, and what does `C` control?**
Regularization penalizes large coefficients to reduce overfitting. `C` is the *inverse* of regularization strength in scikit-learn: small `C` = strong regularization (risk of underfitting); large `C` = weak regularization (risk of overfitting).

**45. What was your best `C`, and what was its score?**
C = 10, with cross-validated ROC-AUC of 0.8463 — only marginally better than the default `C=1` (0.8462), suggesting this hyperparameter has limited impact for this dataset/feature set.

**46. Why did you use `scoring="roc_auc"` for `GridSearchCV` instead of accuracy?**
ROC-AUC evaluates ranking quality across all thresholds, which is more informative than accuracy at one fixed threshold on an imbalanced dataset.

**47. How does cross-validation inside `GridSearchCV` avoid data leakage?**
Because the full pipeline (preprocessing + classifier) is passed as the estimator, scikit-learn refits the `OneHotEncoder`/`StandardScaler` from scratch inside every single fold, using only that fold's training portion — never the validation fold or the test set.

---

## H. LDA and Model Comparison

**48. What is Linear Discriminant Analysis (LDA)?**
A linear classification algorithm that models each class's features as Gaussian distributions with a shared covariance matrix, then classifies a new point via Bayes' theorem, based on which class's distribution it's statistically closer to.

**49. How does LDA differ from Logistic Regression?**
Logistic Regression directly models `P(churn | features)` via the sigmoid function without assuming how features are distributed. LDA assumes features within each class are approximately normally distributed with equal covariance — an assumption one-hot encoded binary columns don't strictly satisfy.

**50. Why did you need to check LDA's compatibility with your preprocessing pipeline?**
Because LDA, unlike Logistic Regression, cannot fit on a sparse matrix. I verified the existing `ColumnTransformer` outputs a dense array for this specific dataset (its one-hot encoded output is dense enough — ~42% non-zero — to cross the `ColumnTransformer`'s default density threshold), so reusing the same pipeline was safe.

**51. How did LDA compare to Logistic Regression?**
LDA: Accuracy 0.7984, Precision 0.6396, Recall 0.5525, ROC-AUC 0.8391. Logistic Regression (tuned): Accuracy 0.7483, Precision 0.5166, Recall 0.8007, ROC-AUC 0.8463. LDA had higher accuracy and precision but lower recall and ROC-AUC.

**52. Why didn't you pick LDA, given its higher accuracy?**
Because accuracy alone is misleading on an imbalanced dataset — a model predicting the majority class more often naturally scores higher accuracy while missing more actual churners. I prioritized ROC-AUC and recall, where Logistic Regression won, since catching at-risk customers is the actual business goal.

**53. Why is comparing multiple algorithms useful?**
No single algorithm is guaranteed best for every dataset; comparing a second, differently-structured algorithm under identical cross-validation is a sanity check on whether the first model's assumptions are actually a good fit, rather than just assuming so.

---

## I. Evaluation Metrics

**54. Define precision, recall, and F1-score, with formulas.**
Precision = TP/(TP+FP) — of predicted churners, how many actually churned. Recall = TP/(TP+FN) — of actual churners, how many were caught. F1 = 2·(Precision·Recall)/(Precision+Recall) — their harmonic mean.

**55. What does ROC-AUC measure?**
The probability that the model ranks a randomly chosen churner's predicted probability higher than a randomly chosen non-churner's. 0.5 = random guessing, 1.0 = perfect ranking; this project's final model scored 0.8409.

**56. Why is accuracy alone insufficient here?**
With ~73.5% of customers not churning, a model that always predicts "No churn" would score ~73.5% accuracy while being useless for identifying at-risk customers — accuracy doesn't distinguish this failure mode from genuine skill.

**57. Explain your final confusion matrix.**
Of 1,409 test customers: TN=810 (correctly predicted to stay), FP=225 (predicted to churn but stayed), FN=110 (predicted to stay but actually churned), TP=264 (correctly predicted to churn).

**58. Why might a business care more about recall than precision here?**
Missing an at-risk customer (a false negative) typically costs more — lost revenue with no chance to intervene — than a false alarm (contacting a customer who would have stayed anyway), which is usually cheap by comparison.

---

## J. Threshold Selection

**59. What is a classification threshold?**
The probability cutoff above which a prediction is labeled "churn." The scikit-learn default is 0.50, but this is a convention, not a law.

**60. How does moving the threshold affect precision and recall?**
Lowering it catches more actual churners (higher recall) but raises more false alarms (lower precision). Raising it does the opposite — fewer false alarms, but more missed churners.

**61. What threshold did you select, and how?**
0.60, selected by maximizing F1-score on leakage-safe out-of-fold cross-validated training predictions (never the test set), searched over the range 0.30–0.60.

**62. Why didn't you select the threshold by looking at test-set results?**
That would bias the "held-out" evaluation — if the threshold is tuned to the test set, the final reported metrics are no longer an honest estimate of performance on genuinely unseen data.

**63. What's a limitation of your threshold selection?**
F1 was still rising at the top of the tested range (0.60), meaning a wider search might find an even higher-F1 threshold beyond that range — reported explicitly rather than hidden.

**64. How did raising the threshold from 0.50 to 0.60 change the final test results?**
Recall dropped from 0.7834 to 0.7059, while precision rose from 0.5052 to 0.5399 — exactly the expected precision/recall trade-off, with ROC-AUC staying essentially flat (~0.8409) since it's threshold-independent.

---

## K. Streamlit and Model Integration

**65. How does the Streamlit app get its predictions?**
It loads the saved pipeline (`models/final_churn_model.joblib`) via `joblib.load()` and calls `.predict_proba()` on a user-constructed single-row DataFrame — no retraining, no separately reimplemented preprocessing.

**66. Why does the app use `predict_proba()` instead of `predict()`?**
`.predict()` uses the model's built-in default 0.5 cutoff; `.predict_proba()` returns the raw probability, letting the app apply the actual Stage 5-selected threshold (0.60) itself.

**67. How does the app ensure its form dropdowns match what the model expects?**
`get_categorical_options()` reads the allowed categories directly from the fitted `OneHotEncoder`'s `categories_` attribute inside the saved pipeline, rather than a hardcoded list that could drift out of sync.

**68. What happens if the model artifact file is missing?**
The app catches the resulting `FileNotFoundError`, displays a clear error message, and stops — it does not silently train a replacement model.

**69. What is `st.cache_resource`, and why use it?**
A Streamlit decorator that loads the model once and reuses the same in-memory object across reruns (Streamlit reruns the whole script on every user interaction), avoiding reloading the file from disk repeatedly.

**70. Are the retention recommendations generated by the ML model?**
No — they are simple, explicitly-labeled rule-based heuristics derived directly from the entered form values (e.g. "month-to-month contract → suggest a longer-term plan"), kept clearly separate from the model's probability output.

---

## L. Python Libraries and GitHub

**71. What is the role of pandas in this project?**
Loading, cleaning, and manipulating the tabular dataset (DataFrames) throughout preprocessing, EDA, and feature construction.

**72. What is the role of scikit-learn?**
Preprocessing (`ColumnTransformer`, `OneHotEncoder`, `StandardScaler`), the Logistic Regression and LDA classifiers, train-test splitting, cross-validation, `GridSearchCV`, and all evaluation metrics.

**73. What is joblib used for, and why not plain `pickle`?**
Saving and loading the fitted pipeline object to/from disk; `joblib` is optimized for objects containing large NumPy arrays (common in scikit-learn models), making it more efficient than plain `pickle` for this use case.

**74. What is a Git feature branch, and why did you use one per stage?**
An isolated line of development off `main`, so each stage's work-in-progress doesn't affect `main` until it's reviewed and merged — keeping `main` always in a working state and giving a clean history of each stage's changes.

**75. What is a pull request, and why open one for every stage?**
A request to merge a feature branch into `main`, which documents what changed and why, and provides a natural checkpoint for review before the change becomes permanent.

**76. How did you verify no data-leakage-causing files (raw CSV, model artifacts) were committed to Git?**
Via `.gitignore` entries (`data/*.csv`, `models/*.joblib`) and by explicitly checking `git status`/`git diff --staged` before every commit throughout the project to confirm only intended files were staged.
