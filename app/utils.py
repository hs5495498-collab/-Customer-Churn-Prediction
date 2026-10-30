"""
Reusable, Streamlit-independent logic for the churn prediction app.

Kept separate from app.py so these functions can be unit tested directly
(without needing to run inside a Streamlit session), and so app.py stays
focused on layout/UI.

IMPORTANT: this module does NOT reimplement any preprocessing. It only
builds a plain pandas DataFrame with the exact raw column names/values the
saved pipeline's ColumnTransformer expects -- the pipeline itself (loaded
from models/final_churn_model.joblib) performs all one-hot encoding and
scaling, exactly as it did during Stage 4/5 training.
"""

import os

import joblib
import pandas as pd

REQUIRED_ARTIFACT_KEYS = ["pipeline", "model_name", "threshold", "metrics", "feature_columns"]

# Real min/max values observed in the actual Telco Customer Churn dataset
# (data/WA_Fn-UseC_-Telco-Customer-Churn.csv), used only to bound the input
# sliders -- not fabricated.
TENURE_MIN, TENURE_MAX = 0, 72
MONTHLY_CHARGES_MIN, MONTHLY_CHARGES_MAX = 18.25, 118.75
TOTAL_CHARGES_MIN, TOTAL_CHARGES_MAX = 0.0, 8684.80


def load_model_artifact(model_path):
    """
    Load the Stage 5 final model artifact and verify it has the expected
    structure before returning it.

    Raises FileNotFoundError if the artifact is missing (the app must not
    silently train or invent a replacement model), and ValueError if the
    file loads but is missing expected keys (so a malformed/unexpected
    artifact is reported explicitly, not guessed at).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at '{model_path}'. "
            "This file is produced by running the notebook's Stage 5 "
            "section (or src/validation.py's save_final_model()) and is "
            "intentionally excluded from Git. Run the notebook first, "
            "then restart the app."
        )

    artifact = joblib.load(model_path)

    if not isinstance(artifact, dict):
        raise ValueError(
            f"Expected the model artifact to be a dict with keys "
            f"{REQUIRED_ARTIFACT_KEYS}, but got {type(artifact)}."
        )

    missing_keys = [key for key in REQUIRED_ARTIFACT_KEYS if key not in artifact]
    if missing_keys:
        raise ValueError(
            f"Model artifact at '{model_path}' is missing expected keys: "
            f"{missing_keys}. Found keys: {list(artifact.keys())}."
        )

    if artifact["threshold"] is None:
        raise ValueError(
            "Model artifact has no classification threshold saved. "
            "Refusing to guess a default -- re-save the artifact with a "
            "valid threshold."
        )

    return artifact


def get_categorical_options(pipeline):
    """
    Extract the exact category values the fitted OneHotEncoder inside the
    pipeline expects for each categorical column, read directly from the
    fitted encoder (`categories_`) rather than hardcoded -- so the UI
    dropdowns can never drift out of sync with what the model was
    actually trained on.

    Returns a dict: {column_name: [allowed_value, ...]}.
    """
    preprocessing = pipeline.named_steps["preprocessing"]
    cat_transformer = preprocessing.named_transformers_["categorical"]
    cat_cols = preprocessing.transformers_[0][2]

    return {
        col: list(categories)
        for col, categories in zip(cat_cols, cat_transformer.categories_)
    }


def validate_customer_input(input_dict, categorical_options):
    """
    Validate a raw customer input dict before it is turned into a model
    input. Returns (is_valid: bool, errors: list[str]).

    Numeric fields are checked against real observed ranges from the
    training dataset; categorical fields are checked against the exact
    categories the fitted encoder accepts (from get_categorical_options).
    """
    errors = []

    tenure = input_dict.get("tenure")
    if tenure is None or not (TENURE_MIN <= tenure <= TENURE_MAX):
        errors.append(f"Tenure must be between {TENURE_MIN} and {TENURE_MAX} months.")

    monthly_charges = input_dict.get("MonthlyCharges")
    if monthly_charges is None or monthly_charges < 0:
        errors.append("Monthly Charges must be a non-negative number.")

    total_charges = input_dict.get("TotalCharges")
    if total_charges is None or total_charges < 0:
        errors.append("Total Charges must be a non-negative number.")

    if tenure == 0 and total_charges is not None and total_charges != 0:
        errors.append(
            "A customer with 0 months tenure has not completed a billing "
            "cycle yet -- Total Charges should be 0 for tenure = 0, "
            "consistent with how the training data was cleaned (Stage 2)."
        )

    for col, allowed_values in categorical_options.items():
        value = input_dict.get(col)
        if value not in allowed_values:
            errors.append(
                f"'{col}' must be one of {allowed_values}, got '{value}'."
            )

    return (len(errors) == 0), errors


def build_input_dataframe(input_dict, feature_columns):
    """
    Build a single-row pandas DataFrame with exactly `feature_columns` as
    its columns, in that order, ready to be passed straight into the
    saved pipeline's .predict_proba().

    SeniorCitizen is accepted here as "Yes"/"No" (friendlier for a form
    UI) and converted to the 1/0 the trained pipeline expects, matching
    the raw dataset's original encoding (Stage 2 did not alter this
    column).
    """
    row = dict(input_dict)
    row["SeniorCitizen"] = 1 if row.get("SeniorCitizen") == "Yes" else 0
    row["tenure"] = int(row["tenure"])
    row["MonthlyCharges"] = float(row["MonthlyCharges"])
    row["TotalCharges"] = float(row["TotalCharges"])

    return pd.DataFrame([row], columns=feature_columns)


def predict_churn(pipeline, input_df, threshold):
    """
    Run the saved pipeline's predict_proba() on a single-row input
    DataFrame and classify it using `threshold` (NOT the pipeline's
    built-in 0.5 default), matching the Stage 5 methodology exactly.

    Returns a dict: probability, prediction_label, risk_category.
    """
    probability = float(pipeline.predict_proba(input_df)[0, 1])

    if probability >= threshold:
        prediction_label = "Likely to Churn"
        risk_category = "High Risk"
    else:
        prediction_label = "Likely to Stay"
        risk_category = "Low Risk"

    return {
        "probability": probability,
        "prediction_label": prediction_label,
        "risk_category": risk_category,
        "threshold": threshold,
    }


def generate_retention_recommendations(input_dict):
    """
    Return a list of simple, RULE-BASED retention suggestions derived
    directly from the entered customer details. These are heuristics for
    a human to consider, not output from the trained model, and not a
    guarantee that any action prevents churn -- the app must label them
    as such when displaying them.
    """
    recommendations = []

    if input_dict.get("Contract") == "Month-to-month":
        recommendations.append(
            "Customer is on a month-to-month contract — consider offering "
            "a discounted one-year or two-year plan, since longer "
            "contracts are strongly associated with lower churn in this "
            "dataset."
        )

    monthly_charges = input_dict.get("MonthlyCharges", 0)
    if monthly_charges is not None and monthly_charges > 70:
        recommendations.append(
            "Monthly charges are relatively high — consider reviewing "
            "pricing or offering a loyalty discount."
        )

    if input_dict.get("OnlineSecurity") == "No":
        recommendations.append(
            "Customer has no Online Security add-on — consider explaining "
            "this available service."
        )

    if input_dict.get("TechSupport") == "No":
        recommendations.append(
            "Customer has no Tech Support add-on — consider offering a "
            "support package."
        )

    tenure = input_dict.get("tenure", 0)
    if tenure is not None and tenure < 12:
        recommendations.append(
            "Customer has under a year of tenure — consider an onboarding "
            "check-in or an early-loyalty incentive."
        )

    if input_dict.get("InternetService") == "Fiber optic":
        recommendations.append(
            "Customer has Fiber optic internet — this group showed a "
            "higher churn association in this dataset's analysis; "
            "consider a service satisfaction check-in."
        )

    if not recommendations:
        recommendations.append(
            "No specific risk factors flagged by these simple rules based "
            "on the entered details."
        )

    return recommendations
