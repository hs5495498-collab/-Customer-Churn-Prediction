"""
Unit tests for src/train.py.

These tests run against the real Telco Customer Churn dataset (small and
fast enough to load/split/fit in a test suite) -- never synthetic data --
since Stage 4 is specifically about validating the real training pipeline,
not just isolated function logic.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import identify_feature_types
from src.train import (
    load_and_clean_data,
    prepare_features_and_target,
    split_data,
    build_model_pipeline,
    train_model,
    get_predictions,
    get_feature_names,
    get_coefficients_and_odds_ratios,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")


@pytest.fixture(scope="module")
def cleaned_df():
    return load_and_clean_data(DATA_PATH)


@pytest.fixture(scope="module")
def features_and_target(cleaned_df):
    return prepare_features_and_target(cleaned_df)


@pytest.fixture(scope="module")
def split(features_and_target):
    X, y = features_and_target
    return split_data(X, y)


@pytest.fixture(scope="module")
def fitted_pipeline(cleaned_df, split):
    X_train, X_test, y_train, y_test = split
    categorical_cols, numerical_cols = identify_feature_types(cleaned_df)
    pipeline = build_model_pipeline(categorical_cols, numerical_cols)
    pipeline = train_model(pipeline, X_train, y_train)
    return pipeline


# ---------------------------------------------------------------------
# Target and feature separation
# ---------------------------------------------------------------------

def test_prepare_features_and_target_excludes_target_and_gender(features_and_target):
    X, y = features_and_target
    assert "Churn" not in X.columns
    assert "gender" not in X.columns


def test_prepare_features_and_target_excludes_customer_id(features_and_target):
    X, _ = features_and_target
    assert "customerID" not in X.columns


def test_target_is_binary_encoded(features_and_target):
    _, y = features_and_target
    assert set(y.unique()) == {0, 1}


# ---------------------------------------------------------------------
# Train-test split dimensions, stratification, reproducibility
# ---------------------------------------------------------------------

def test_split_is_80_20(features_and_target):
    X, y = features_and_target
    X_train, X_test, y_train, y_test = split_data(X, y)

    total = len(X)
    assert len(X_train) == pytest.approx(total * 0.8, abs=2)
    assert len(X_test) == pytest.approx(total * 0.2, abs=2)
    assert len(X_train) + len(X_test) == total


def test_split_is_stratified(features_and_target):
    X, y = features_and_target
    X_train, X_test, y_train, y_test = split_data(X, y)

    full_rate = y.mean()
    train_rate = y_train.mean()
    test_rate = y_test.mean()

    # Stratification should keep the churn rate very close across all three sets
    assert abs(train_rate - full_rate) < 0.01
    assert abs(test_rate - full_rate) < 0.01


def test_split_is_reproducible(features_and_target):
    X, y = features_and_target
    X_train_1, X_test_1, y_train_1, y_test_1 = split_data(X, y, random_state=42)
    X_train_2, X_test_2, y_train_2, y_test_2 = split_data(X, y, random_state=42)

    assert X_train_1.index.equals(X_train_2.index)
    assert X_test_1.index.equals(X_test_2.index)


# ---------------------------------------------------------------------
# Pipeline fitting and predictions
# ---------------------------------------------------------------------

def test_pipeline_fits_without_error(fitted_pipeline):
    assert fitted_pipeline is not None
    # A fitted LogisticRegression has learned coefficients
    assert fitted_pipeline.named_steps["classifier"].coef_ is not None


def test_predictions_are_binary(fitted_pipeline, split):
    _, X_test, _, _ = split
    y_pred, _ = get_predictions(fitted_pipeline, X_test)
    assert set(np.unique(y_pred)).issubset({0, 1})
    assert len(y_pred) == len(X_test)


def test_probabilities_are_between_0_and_1(fitted_pipeline, split):
    _, X_test, _, _ = split
    _, y_proba = get_predictions(fitted_pipeline, X_test)
    assert (y_proba >= 0).all()
    assert (y_proba <= 1).all()


# ---------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------

def test_metrics_are_calculated_in_valid_ranges(fitted_pipeline, split):
    _, X_test, _, y_test = split
    y_pred, y_proba = get_predictions(fitted_pipeline, X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    for metric in (accuracy, precision, recall, f1, roc_auc):
        assert 0.0 <= metric <= 1.0

    # A reasonably trained model on this dataset should clearly beat
    # random guessing (ROC-AUC of 0.5), without hardcoding an exact value.
    assert roc_auc > 0.7


# ---------------------------------------------------------------------
# Coefficient and odds ratio extraction
# ---------------------------------------------------------------------

def test_feature_names_exclude_gender_and_customer_id(fitted_pipeline):
    names = get_feature_names(fitted_pipeline)
    assert not any("gender" in name for name in names)
    assert not any("customerID" in name for name in names)


def test_coefficient_table_has_expected_columns(fitted_pipeline):
    table = get_coefficients_and_odds_ratios(fitted_pipeline)
    assert list(table.columns) == ["feature", "coefficient", "odds_ratio", "direction"]
    assert len(table) == len(get_feature_names(fitted_pipeline))


def test_odds_ratio_equals_exp_of_coefficient(fitted_pipeline):
    table = get_coefficients_and_odds_ratios(fitted_pipeline)
    expected_odds_ratios = np.exp(table["coefficient"].values)
    np.testing.assert_allclose(table["odds_ratio"].values, expected_odds_ratios, rtol=1e-6)


def test_coefficient_table_sorted_by_absolute_magnitude(fitted_pipeline):
    table = get_coefficients_and_odds_ratios(fitted_pipeline)
    abs_coefs = table["coefficient"].abs().values
    assert all(abs_coefs[i] >= abs_coefs[i + 1] for i in range(len(abs_coefs) - 1))
