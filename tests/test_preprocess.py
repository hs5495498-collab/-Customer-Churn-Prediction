"""
Unit tests for src/preprocess.py.

Small, hand-built fixtures (a handful of rows) are used here ONLY to test
that each function behaves correctly in isolation and in edge cases. These
fixtures never replace the real Kaggle dataset in the project's actual
analysis -- the notebook and src/train.py always load the real CSV at
data/WA_Fn-UseC_-Telco-Customer-Churn.csv.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import (
    load_data,
    inspect_data,
    remove_duplicates,
    convert_total_charges,
    handle_missing_total_charges,
    drop_customer_id,
    detect_outliers_iqr,
    identify_feature_types,
    build_preprocessing_pipeline,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")


def make_fixture_df():
    """A tiny hand-built DataFrame mimicking the real dataset's shape."""
    return pd.DataFrame({
        "customerID": ["1001-AAA", "1002-BBB", "1003-CCC", "1004-DDD"],
        "gender": ["Female", "Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0, 0],
        "Partner": ["Yes", "No", "Yes", "No"],
        "Dependents": ["No", "No", "Yes", "No"],
        "tenure": [0, 34, 2, 45],
        "PhoneService": ["Yes", "Yes", "Yes", "No"],
        "MultipleLines": ["No", "Yes", "No", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "DSL", "No"],
        "OnlineSecurity": ["Yes", "No", "No", "No internet service"],
        "OnlineBackup": ["No", "Yes", "No", "No internet service"],
        "DeviceProtection": ["No", "No", "Yes", "No internet service"],
        "TechSupport": ["No", "No", "No", "No internet service"],
        "StreamingTV": ["No", "Yes", "No", "No internet service"],
        "StreamingMovies": ["No", "Yes", "No", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "Yes", "No", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Electronic check", "Bank transfer (automatic)"],
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30],
        "TotalCharges": [" ", "1889.5", "108.15", "1840.75"],
        "Churn": ["No", "No", "Yes", "No"],
    })


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def test_load_data_reads_real_csv():
    df = load_data(DATA_PATH)
    assert df.shape[0] == 7043
    assert df.shape[1] == 21
    assert "Churn" in df.columns


def test_inspect_data_returns_expected_keys():
    df = make_fixture_df()
    summary = inspect_data(df)
    assert summary["n_rows"] == 4
    assert summary["n_columns"] == 21
    assert "churn_counts" in summary


# ---------------------------------------------------------------------
# Duplicate handling
# ---------------------------------------------------------------------

def test_remove_duplicates_drops_exact_duplicate_rows():
    df = make_fixture_df()
    df_with_dupe = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert df_with_dupe.shape[0] == 5

    deduped = remove_duplicates(df_with_dupe)
    assert deduped.shape[0] == 4


# ---------------------------------------------------------------------
# TotalCharges conversion and missing-value handling
# ---------------------------------------------------------------------

def test_convert_total_charges_turns_blank_into_nan():
    df = make_fixture_df()
    converted = convert_total_charges(df)
    assert converted["TotalCharges"].dtype == np.float64
    assert pd.isna(converted.loc[0, "TotalCharges"])
    assert converted.loc[1, "TotalCharges"] == 1889.5


def test_handle_missing_total_charges_fills_zero_tenure_rows_with_zero():
    df = make_fixture_df()
    converted = convert_total_charges(df)
    filled = handle_missing_total_charges(converted)

    assert filled["TotalCharges"].isnull().sum() == 0
    # Row 0 had tenure == 0 and a blank TotalCharges -> should become 0.0
    assert filled.loc[0, "TotalCharges"] == 0.0


def test_handle_missing_total_charges_raises_if_nonzero_tenure_missing():
    df = make_fixture_df()
    df.loc[0, "tenure"] = 5  # now the blank TotalCharges row has tenure != 0
    converted = convert_total_charges(df)

    with pytest.raises(ValueError):
        handle_missing_total_charges(converted)


# ---------------------------------------------------------------------
# customerID removal
# ---------------------------------------------------------------------

def test_drop_customer_id_removes_column():
    df = make_fixture_df()
    dropped = drop_customer_id(df)
    assert "customerID" not in dropped.columns
    assert dropped.shape[1] == df.shape[1] - 1


# ---------------------------------------------------------------------
# Gender exclusion (ethical requirement)
# ---------------------------------------------------------------------

def test_identify_feature_types_excludes_gender_by_default():
    df = make_fixture_df()
    cat_cols, num_cols = identify_feature_types(df)
    assert "gender" not in cat_cols
    assert "gender" not in num_cols


def test_identify_feature_types_excludes_target_column():
    df = make_fixture_df()
    cat_cols, num_cols = identify_feature_types(df)
    assert "Churn" not in cat_cols
    assert "Churn" not in num_cols


# ---------------------------------------------------------------------
# Numerical / categorical feature identification
# ---------------------------------------------------------------------

def test_identify_feature_types_splits_correctly():
    df = make_fixture_df()
    cat_cols, num_cols = identify_feature_types(df)

    assert "Contract" in cat_cols
    assert "PaymentMethod" in cat_cols
    assert "tenure" in num_cols
    assert "MonthlyCharges" in num_cols


# ---------------------------------------------------------------------
# Outlier detection (IQR)
# ---------------------------------------------------------------------

def test_detect_outliers_iqr_flags_extreme_value():
    df = pd.DataFrame({"tenure": [1, 2, 3, 4, 5, 100]})
    report = detect_outliers_iqr(df, ["tenure"])
    assert report["tenure"]["outlier_count"] >= 1
    assert 5 in report["tenure"]["outlier_indices"]


def test_detect_outliers_iqr_no_outliers_in_uniform_data():
    df = pd.DataFrame({"tenure": [10, 11, 12, 13, 14]})
    report = detect_outliers_iqr(df, ["tenure"])
    assert report["tenure"]["outlier_count"] == 0


# ---------------------------------------------------------------------
# Preprocessing pipeline output and unseen categories
# ---------------------------------------------------------------------

def test_build_preprocessing_pipeline_transforms_known_categories():
    df = make_fixture_df()
    converted = convert_total_charges(df)
    filled = handle_missing_total_charges(converted)
    cleaned = drop_customer_id(filled)

    cat_cols, num_cols = identify_feature_types(cleaned)
    pipeline = build_preprocessing_pipeline(cat_cols, num_cols)

    X = cleaned.drop(columns=["Churn", "gender"])
    transformed = pipeline.fit_transform(X)

    # Output should be a 2D numeric array with one row per input row
    assert transformed.shape[0] == X.shape[0]
    assert np.isfinite(transformed.toarray() if hasattr(transformed, "toarray") else transformed).all()


def test_pipeline_handles_unseen_category_without_crashing():
    df = make_fixture_df()
    converted = convert_total_charges(df)
    filled = handle_missing_total_charges(converted)
    cleaned = drop_customer_id(filled)

    cat_cols, num_cols = identify_feature_types(cleaned)
    pipeline = build_preprocessing_pipeline(cat_cols, num_cols)

    X = cleaned.drop(columns=["Churn", "gender"])
    pipeline.fit(X)

    # A brand-new PaymentMethod value never seen during fit
    X_new = X.iloc[[0]].copy()
    X_new["PaymentMethod"] = "Cryptocurrency"

    # handle_unknown='ignore' should encode this as all-zeros, not raise
    transformed_new = pipeline.transform(X_new)
    assert transformed_new.shape[0] == 1
