"""
Tests for the Streamlit application's logic (app/utils.py) and, where the
model artifact is available locally, an integration test against the real
Stage 5 model artifact.

Pure logic (validation, dataframe construction, threshold classification,
recommendations) is tested directly without needing a running Streamlit
session. A separate test using streamlit.testing.v1.AppTest exercises the
actual app.py script headlessly (import + render + a full form submission)
to catch integration issues a unit test on utils.py alone would miss --
this is exactly how the "pyarrow requires NumPy 2.0" bug in st.table() was
caught and fixed during development.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import (
    load_model_artifact,
    get_categorical_options,
    validate_customer_input,
    build_input_dataframe,
    predict_churn,
    generate_retention_recommendations,
    REQUIRED_ARTIFACT_KEYS,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "final_churn_model.joblib")
APP_PATH = os.path.join(os.path.dirname(__file__), "..", "app", "app.py")

VALID_INPUT = {
    "SeniorCitizen": "No", "Partner": "No", "Dependents": "No", "tenure": 12,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 80.0, "TotalCharges": 960.0,
}

FEATURE_COLUMNS = [
    "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

CATEGORICAL_OPTIONS = {
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
}


# ---------------------------------------------------------------------
# 1. Model artifact loading
# ---------------------------------------------------------------------

def test_load_model_artifact_raises_if_missing(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.joblib")
    with pytest.raises(FileNotFoundError):
        load_model_artifact(missing_path)


def test_load_model_artifact_raises_on_malformed_artifact(tmp_path):
    import joblib
    bad_path = str(tmp_path / "bad.joblib")
    joblib.dump({"pipeline": "not a real pipeline"}, bad_path)  # missing keys
    with pytest.raises(ValueError):
        load_model_artifact(bad_path)


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Stage 5 model artifact not present locally")
def test_load_real_model_artifact_has_expected_keys():
    artifact = load_model_artifact(MODEL_PATH)
    for key in REQUIRED_ARTIFACT_KEYS:
        assert key in artifact


# ---------------------------------------------------------------------
# 2. Model pipeline structure
# ---------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Stage 5 model artifact not present locally")
def test_real_pipeline_has_preprocessing_and_classifier_steps():
    artifact = load_model_artifact(MODEL_PATH)
    pipeline = artifact["pipeline"]
    assert "preprocessing" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Stage 5 model artifact not present locally")
def test_get_categorical_options_matches_fitted_encoder():
    artifact = load_model_artifact(MODEL_PATH)
    options = get_categorical_options(artifact["pipeline"])
    assert options["Contract"] == ["Month-to-month", "One year", "Two year"]
    assert "gender" not in options
    assert "customerID" not in options


# ---------------------------------------------------------------------
# 3-4. Valid input and feature names
# ---------------------------------------------------------------------

def test_valid_customer_input_passes_validation():
    is_valid, errors = validate_customer_input(VALID_INPUT, CATEGORICAL_OPTIONS)
    assert is_valid
    assert errors == []


def test_build_input_dataframe_has_exact_feature_columns():
    df = build_input_dataframe(VALID_INPUT, FEATURE_COLUMNS)
    assert list(df.columns) == FEATURE_COLUMNS
    assert len(df) == 1
    assert "gender" not in df.columns
    assert "customerID" not in df.columns


def test_build_input_dataframe_converts_senior_citizen_to_binary():
    df_yes = build_input_dataframe({**VALID_INPUT, "SeniorCitizen": "Yes"}, FEATURE_COLUMNS)
    df_no = build_input_dataframe({**VALID_INPUT, "SeniorCitizen": "No"}, FEATURE_COLUMNS)
    assert df_yes.loc[0, "SeniorCitizen"] == 1
    assert df_no.loc[0, "SeniorCitizen"] == 0


# ---------------------------------------------------------------------
# 5. Missing input handling
# ---------------------------------------------------------------------

def test_missing_required_field_fails_validation():
    incomplete_input = dict(VALID_INPUT)
    del incomplete_input["tenure"]
    is_valid, errors = validate_customer_input(incomplete_input, CATEGORICAL_OPTIONS)
    assert not is_valid
    assert any("Tenure" in e for e in errors)


# ---------------------------------------------------------------------
# 6-7. Prediction probability range and threshold-based classification
# ---------------------------------------------------------------------

class _FakePipeline:
    """A minimal stand-in for a fitted sklearn pipeline, for testing
    predict_churn()'s threshold logic in isolation from the real model."""
    def __init__(self, probability):
        self._probability = probability

    def predict_proba(self, X):
        return np.array([[1 - self._probability, self._probability]])


def test_predict_churn_probability_is_between_0_and_1():
    df = build_input_dataframe(VALID_INPUT, FEATURE_COLUMNS)
    result = predict_churn(_FakePipeline(0.73), df, threshold=0.5)
    assert 0.0 <= result["probability"] <= 1.0


def test_predict_churn_above_threshold_is_high_risk():
    df = build_input_dataframe(VALID_INPUT, FEATURE_COLUMNS)
    result = predict_churn(_FakePipeline(0.65), df, threshold=0.6)
    assert result["risk_category"] == "High Risk"
    assert result["prediction_label"] == "Likely to Churn"


def test_predict_churn_below_threshold_is_low_risk():
    df = build_input_dataframe(VALID_INPUT, FEATURE_COLUMNS)
    result = predict_churn(_FakePipeline(0.55), df, threshold=0.6)
    assert result["risk_category"] == "Low Risk"
    assert result["prediction_label"] == "Likely to Stay"


def test_predict_churn_exactly_at_threshold_is_high_risk():
    # probability >= threshold -> churn, per the Stage 5 methodology
    df = build_input_dataframe(VALID_INPUT, FEATURE_COLUMNS)
    result = predict_churn(_FakePipeline(0.6), df, threshold=0.6)
    assert result["risk_category"] == "High Risk"


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Stage 5 model artifact not present locally")
def test_real_model_prediction_uses_saved_threshold_not_default():
    artifact = load_model_artifact(MODEL_PATH)
    assert artifact["threshold"] == pytest.approx(0.6)

    df = build_input_dataframe(VALID_INPUT, artifact["feature_columns"])
    result = predict_churn(artifact["pipeline"], df, artifact["threshold"])
    assert 0.0 <= result["probability"] <= 1.0
    assert result["threshold"] == artifact["threshold"]


# ---------------------------------------------------------------------
# 9. Retention recommendation generation
# ---------------------------------------------------------------------

def test_recommendations_flag_month_to_month_contract():
    recs = generate_retention_recommendations(VALID_INPUT)
    assert any("month-to-month" in r.lower() for r in recs)


def test_recommendations_flag_short_tenure():
    recs = generate_retention_recommendations({**VALID_INPUT, "tenure": 3})
    assert any("tenure" in r.lower() for r in recs)


def test_recommendations_for_low_risk_profile_still_returns_a_list():
    low_risk_input = {
        **VALID_INPUT,
        "Contract": "Two year", "MonthlyCharges": 40.0, "OnlineSecurity": "Yes",
        "TechSupport": "Yes", "tenure": 60, "InternetService": "DSL",
    }
    recs = generate_retention_recommendations(low_risk_input)
    assert isinstance(recs, list)
    assert len(recs) >= 1


# ---------------------------------------------------------------------
# 10-11. Invalid numeric values and categorical values
# ---------------------------------------------------------------------

def test_negative_monthly_charges_fails_validation():
    bad_input = {**VALID_INPUT, "MonthlyCharges": -10.0}
    is_valid, errors = validate_customer_input(bad_input, CATEGORICAL_OPTIONS)
    assert not is_valid
    assert any("Monthly Charges" in e for e in errors)


def test_tenure_out_of_range_fails_validation():
    bad_input = {**VALID_INPUT, "tenure": 200}
    is_valid, errors = validate_customer_input(bad_input, CATEGORICAL_OPTIONS)
    assert not is_valid
    assert any("Tenure" in e for e in errors)


def test_invalid_category_fails_validation():
    bad_input = {**VALID_INPUT, "Contract": "Three year"}
    is_valid, errors = validate_customer_input(bad_input, CATEGORICAL_OPTIONS)
    assert not is_valid
    assert any("Contract" in e for e in errors)


def test_tenure_zero_with_nonzero_total_charges_fails_validation():
    bad_input = {**VALID_INPUT, "tenure": 0, "TotalCharges": 500.0}
    is_valid, errors = validate_customer_input(bad_input, CATEGORICAL_OPTIONS)
    assert not is_valid


# ---------------------------------------------------------------------
# 12. Prediction output consistency
# ---------------------------------------------------------------------

def test_same_input_produces_same_prediction():
    df1 = build_input_dataframe(VALID_INPUT, FEATURE_COLUMNS)
    df2 = build_input_dataframe(VALID_INPUT, FEATURE_COLUMNS)
    fake_pipeline = _FakePipeline(0.42)
    result1 = predict_churn(fake_pipeline, df1, threshold=0.5)
    result2 = predict_churn(fake_pipeline, df2, threshold=0.5)
    assert result1 == result2


# ---------------------------------------------------------------------
# Integration: run the actual Streamlit app headlessly end-to-end
# ---------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Stage 5 model artifact not present locally")
def test_streamlit_app_runs_without_exceptions():
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    at = streamlit_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert len(at.exception) == 0
    assert len(at.selectbox) == 15
    assert len(at.slider) == 1


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Stage 5 model artifact not present locally")
def test_streamlit_app_full_form_submission_produces_prediction():
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    at = streamlit_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    at.slider[0].set_value(5)
    at.number_input[0].set_value(85.0)
    at.number_input[1].set_value(425.0)

    submit_button = [b for b in at.button if b.label == "Predict Churn Risk"][0]
    submit_button.click()
    at.run(timeout=30)

    assert len(at.exception) == 0
    assert any("risk-badge" in el.value for el in at.markdown)
    assert any(m.label == "Predicted Churn Probability" for m in at.metric)
