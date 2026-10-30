"""
Logistic Regression training functions for the Telco Customer Churn dataset.

This module builds on src/preprocess.py — it does NOT duplicate cleaning or
encoding logic. It only adds what Stage 2 deliberately left out: splitting
into train/test, wrapping the unfitted ColumnTransformer in a full
scikit-learn Pipeline with a classifier, fitting that pipeline (which fits
the preprocessing ONLY on the training split, never on the full dataset),
and extracting predictions, coefficients, and odds ratios.

DATA LEAKAGE: split_data() must always be called BEFORE
build_model_pipeline()'s pipeline is fit. The pipeline's .fit() call fits
the ColumnTransformer (OneHotEncoder + StandardScaler) using X_train only;
.predict()/.predict_proba() on X_test only ever calls .transform(), reusing
statistics learned from training data. This is enforced structurally by
scikit-learn's Pipeline: calling .fit() once fits every step in sequence
using only the data passed to that single call.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import clean_dataset, identify_feature_types, build_preprocessing_pipeline

RANDOM_STATE = 42


def load_and_clean_data(csv_path):
    """Load and clean the raw CSV using the Stage 2 cleaning pipeline."""
    return clean_dataset(csv_path)


def prepare_features_and_target(df, target_col="Churn"):
    """
    Split the cleaned dataframe into a feature matrix X and a binary
    target vector y.

    y is encoded as 1 = churned ("Yes"), 0 = retained ("No") so that
    downstream metrics (ROC-AUC, odds ratios) have an unambiguous
    "positive class" to reference.

    X excludes the target column and 'gender' (ethical requirement: gender
    must never be a predictive input). customerID was already dropped by
    clean_dataset() in Stage 2.
    """
    y = (df[target_col] == "Yes").astype(int)
    X = df.drop(columns=[target_col, "gender"], errors="ignore")
    return X, y


def split_data(X, y, test_size=0.2, random_state=RANDOM_STATE):
    """
    Perform an 80/20 stratified train-test split.

    stratify=y ensures both the training and testing sets preserve the
    same churn/non-churn ratio as the full dataset. Without it, a random
    split could by chance put too few churners in the test set, making
    evaluation metrics (especially recall) unreliable on an already
    imbalanced dataset (~26.5% churn). random_state is fixed so the split
    is reproducible across runs.
    """
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def build_model_pipeline(categorical_cols, numerical_cols):
    """
    Build (unfitted) a full scikit-learn Pipeline:
        1. preprocessing (OneHotEncoder + StandardScaler), reused from
           src/preprocess.py
        2. LogisticRegression classifier

    class_weight="balanced" automatically re-weights the two classes
    based on their frequency in the training data, compensating for the
    ~73.5% / 26.5% imbalance without altering the dataset itself.
    """
    preprocessor = build_preprocessing_pipeline(categorical_cols, numerical_cols)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", model),
    ])
    return pipeline


def train_model(pipeline, X_train, y_train):
    """
    Fit the full pipeline on the training data only.

    Because `pipeline` bundles preprocessing and the classifier together,
    this single .fit() call fits the OneHotEncoder/StandardScaler AND the
    LogisticRegression classifier using only X_train/y_train -- the test
    set is never touched here, which is what prevents data leakage.
    """
    pipeline.fit(X_train, y_train)
    return pipeline


def get_predictions(pipeline, X):
    """
    Return (predicted_labels, predicted_probabilities) for the given
    feature matrix X, using an already-fitted pipeline.

    predicted_probabilities is the probability of the positive class
    (churn = 1), taken from predict_proba()[:, 1].
    """
    predicted_labels = pipeline.predict(X)
    predicted_probabilities = pipeline.predict_proba(X)[:, 1]
    return predicted_labels, predicted_probabilities


def get_feature_names(pipeline):
    """
    Return the list of feature names AFTER preprocessing (i.e. after
    one-hot encoding expands each category into its own column), in the
    same order the LogisticRegression classifier sees them.
    """
    preprocessor = pipeline.named_steps["preprocessing"]
    raw_names = preprocessor.get_feature_names_out()
    # Strip the ColumnTransformer's "categorical__"/"numerical__" prefixes
    # to keep the names readable in the coefficient table.
    cleaned_names = [name.split("__", 1)[-1] for name in raw_names]
    return cleaned_names


def get_coefficients_and_odds_ratios(pipeline):
    """
    Extract the trained LogisticRegression coefficients, paired with
    their (post-preprocessing) feature names, and compute each feature's
    odds ratio as exp(coefficient).

    Returns a DataFrame sorted by absolute coefficient size (descending),
    with columns: feature, coefficient, odds_ratio, direction.

    Interpretation:
      - A positive coefficient means an increase in that feature's
        (standardized, or one-hot 0/1) value is associated with higher
        predicted log-odds of churn.
      - A negative coefficient means the opposite: higher values are
        associated with lower predicted log-odds of churn.
      - odds_ratio > 1 means the odds of churn increase for a one-unit
        increase in the feature; odds_ratio < 1 means the odds decrease.
      - These are associations learned from this dataset, not proof of
        causation -- a feature can be a strong statistical predictor
        without being the underlying cause of churn.
    """
    feature_names = get_feature_names(pipeline)
    coefficients = pipeline.named_steps["classifier"].coef_[0]

    odds_ratios = np.exp(coefficients)
    direction = np.where(coefficients > 0, "increases churn odds", "decreases churn odds")

    table = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "odds_ratio": odds_ratios,
        "direction": direction,
    })
    table["abs_coefficient"] = table["coefficient"].abs()
    table = table.sort_values("abs_coefficient", ascending=False).drop(columns="abs_coefficient")
    table = table.reset_index(drop=True)
    return table


def save_model(pipeline, model_path):
    """Save the complete fitted pipeline (preprocessing + classifier) with joblib."""
    import joblib
    import os
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    return model_path


if __name__ == "__main__":
    DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    MODEL_PATH = "models/logistic_regression_pipeline.joblib"

    cleaned_df = load_and_clean_data(DATA_PATH)
    X, y = prepare_features_and_target(cleaned_df)

    X_train, X_test, y_train, y_test = split_data(X, y)
    print("Training set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)
    print("Training churn rate:", y_train.mean().round(4))
    print("Testing churn rate:", y_test.mean().round(4))

    categorical_cols, numerical_cols = identify_feature_types(cleaned_df)
    pipeline = build_model_pipeline(categorical_cols, numerical_cols)
    pipeline = train_model(pipeline, X_train, y_train)

    y_pred, y_proba = get_predictions(pipeline, X_test)
    print("Sample predictions:", y_pred[:10])
    print("Sample probabilities:", y_proba[:10].round(3))

    coef_table = get_coefficients_and_odds_ratios(pipeline)
    print("\nTop 5 most influential features:")
    print(coef_table.head(5).to_string(index=False))

    saved_path = save_model(pipeline, MODEL_PATH)
    print("\nModel saved to:", saved_path)
