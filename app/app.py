"""
Customer Churn Prediction System — Streamlit demo application.

Loads the Stage 5 final trained pipeline (models/final_churn_model.joblib)
and uses it, unmodified, to score a single customer entered through this
form. No model is retrained here, and no preprocessing step is
reimplemented -- the saved pipeline performs all encoding/scaling exactly
as it did during training, so the only new code in this file is the UI
and simple rule-based retention suggestions.

Run from the project root with:
    streamlit run app/app.py
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    load_model_artifact,
    get_categorical_options,
    validate_customer_input,
    build_input_dataframe,
    predict_churn,
    generate_retention_recommendations,
    TENURE_MIN, TENURE_MAX,
    MONTHLY_CHARGES_MIN, MONTHLY_CHARGES_MAX,
    TOTAL_CHARGES_MIN, TOTAL_CHARGES_MAX,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "final_churn_model.joblib")

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📉",
    layout="wide",
)

CUSTOM_CSS = """
<style>
.main-title { font-size: 2.1rem; font-weight: 700; margin-bottom: 0; }
.main-subtitle { font-size: 1.05rem; color: #6b7280; margin-top: 0.2rem; margin-bottom: 1.5rem; }
.section-card {
    background-color: var(--background-color, #ffffff);
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.risk-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1.05rem;
}
.risk-high { background-color: #fee2e2; color: #b91c1c; }
.risk-low { background-color: #dcfce7; color: #15803d; }
.disclaimer {
    font-size: 0.85rem;
    color: #6b7280;
    border-top: 1px solid #e5e7eb;
    padding-top: 0.75rem;
    margin-top: 1rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_artifact():
    return load_model_artifact(MODEL_PATH)


# ---------------------------------------------------------------------
# Load the real Stage 5 model artifact. If it's missing, stop here with
# a clear message instead of silently training or inventing a model.
# ---------------------------------------------------------------------
try:
    artifact = get_artifact()
except (FileNotFoundError, ValueError) as e:
    st.error(str(e))
    st.stop()

pipeline = artifact["pipeline"]
model_name = artifact["model_name"]
threshold = artifact["threshold"]
saved_metrics = artifact["metrics"]
feature_columns = artifact["feature_columns"]
categorical_options = get_categorical_options(pipeline)

# ---------------------------------------------------------------------
# Sidebar: project info, model info, how it works, disclaimer
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("About This Project")
    st.write(
        "CSE3141 — Predictive Analytics mini project. Predicts the "
        "likelihood that a telecom customer will churn, using a "
        "Logistic Regression model trained on the Kaggle Telco Customer "
        "Churn dataset."
    )

    st.header("About the Model")
    st.markdown(f"""
- **Algorithm:** Logistic Regression
- **Selected model:** `{model_name}`
- **Regularization (C):** 10
- **Classification threshold:** {threshold:.2f}
- **Training methodology:** 5-fold stratified cross-validation + GridSearchCV
- **Dataset:** Telco Customer Churn (7,043 customers)
""")

    st.caption("Cross-validation ROC-AUC: 0.8463 (from Stage 5)")

    st.subheader("Held-Out Test Metrics (Stage 5)")
    # Rendered as markdown rather than st.table()/st.dataframe(): those
    # widgets serialize pandas data via pyarrow, and this environment's
    # installed pyarrow requires NumPy 2.0 while the project pins NumPy
    # 1.26.4 for scikit-learn/pandas compatibility -- markdown avoids
    # that dependency entirely for this simple, static table.
    metrics_rows = [
        ("Accuracy", saved_metrics["accuracy"]),
        ("Precision", saved_metrics["precision"]),
        ("Recall", saved_metrics["recall"]),
        ("F1-score", saved_metrics["f1"]),
        ("ROC-AUC", saved_metrics["roc_auc"]),
    ]
    metrics_markdown = "| Metric | Value |\n|---|---|\n" + "\n".join(
        f"| {name} | {value:.4f} |" for name, value in metrics_rows
    )
    st.markdown(metrics_markdown)
    st.caption(
        "These are real held-out test-set results from Stage 5, not "
        "estimates — cross-validation metrics (used for model selection) "
        "and test metrics (used for final evaluation) are kept distinct."
    )

    with st.expander("How does this prediction work?"):
        st.write(
            "The entered customer details are passed through the exact "
            "same preprocessing (one-hot encoding + scaling) and "
            "Logistic Regression model that were fitted and validated in "
            "Stages 2–5. The model outputs a churn probability between 0 "
            "and 1; if that probability is at or above the selected "
            f"threshold ({threshold:.2f}), the customer is classified as "
            "'Likely to Churn' / High Risk."
        )

    st.markdown(
        '<div class="disclaimer">⚠️ This is a statistical prediction from '
        'a model trained on historical data — not a guarantee of any '
        'individual customer\'s future behavior. Predictions should '
        'support, not replace, human judgment.</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------
st.markdown('<p class="main-title">Customer Churn Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">AI-powered customer retention and churn risk analysis.</p>', unsafe_allow_html=True)

if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0

reset_col, _ = st.columns([1, 5])
with reset_col:
    if st.button("🔄 Reset Form"):
        st.session_state.reset_counter += 1
        st.rerun()

form_key = f"customer_form_{st.session_state.reset_counter}"

with st.form(key=form_key):
    st.markdown("#### Customer Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        tenure = st.slider("Tenure (months)", min_value=TENURE_MIN, max_value=TENURE_MAX, value=12)
    with c2:
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=MONTHLY_CHARGES_MIN, max_value=MONTHLY_CHARGES_MAX,
            value=65.0, step=0.5,
        )
    with c3:
        total_charges = st.number_input(
            "Total Charges ($)", min_value=TOTAL_CHARGES_MIN, max_value=TOTAL_CHARGES_MAX,
            value=float(round(monthly_charges * max(tenure, 1), 2)), step=1.0,
        )

    st.markdown("#### Contract Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        contract = st.selectbox("Contract Type", categorical_options["Contract"])
    with c2:
        paperless_billing = st.selectbox("Paperless Billing", categorical_options["PaperlessBilling"])
    with c3:
        payment_method = st.selectbox("Payment Method", categorical_options["PaymentMethod"])

    st.markdown("#### Internet and Services")
    c1, c2, c3 = st.columns(3)
    with c1:
        internet_service = st.selectbox("Internet Service", categorical_options["InternetService"])
        online_security = st.selectbox("Online Security", categorical_options["OnlineSecurity"])
    with c2:
        online_backup = st.selectbox("Online Backup", categorical_options["OnlineBackup"])
        device_protection = st.selectbox("Device Protection", categorical_options["DeviceProtection"])
    with c3:
        tech_support = st.selectbox("Tech Support", categorical_options["TechSupport"])
        streaming_tv = st.selectbox("Streaming TV", categorical_options["StreamingTV"])
    streaming_movies = st.selectbox("Streaming Movies", categorical_options["StreamingMovies"])

    st.markdown("#### Other Customer Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", categorical_options["Partner"])
    with c2:
        dependents = st.selectbox("Dependents", categorical_options["Dependents"])
        phone_service = st.selectbox("Phone Service", categorical_options["PhoneService"])
    with c3:
        multiple_lines = st.selectbox("Multiple Lines", categorical_options["MultipleLines"])

    submitted = st.form_submit_button("Predict Churn Risk", type="primary")

if submitted:
    input_dict = {
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    is_valid, errors = validate_customer_input(input_dict, categorical_options)

    if not is_valid:
        st.error("Please fix the following before predicting:")
        for err in errors:
            st.write(f"- {err}")
    else:
        input_df = build_input_dataframe(input_dict, feature_columns)
        result = predict_churn(pipeline, input_df, threshold)

        st.markdown("---")
        st.markdown("### Prediction Result")

        result_col, chart_col = st.columns([2, 1])

        with result_col:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)

            badge_class = "risk-high" if result["risk_category"] == "High Risk" else "risk-low"
            st.markdown(
                f'<span class="risk-badge {badge_class}">{result["risk_category"]}</span>'
                f'&nbsp;&nbsp;<strong>{result["prediction_label"]}</strong>',
                unsafe_allow_html=True,
            )

            st.metric("Predicted Churn Probability", f"{result['probability'] * 100:.1f}%")
            st.progress(min(max(result["probability"], 0.0), 1.0))
            st.caption(
                f"Classification threshold used: {result['threshold']:.2f} "
                f"(the Stage 5-selected threshold, not the default 0.50). "
                f"Probability ≥ threshold → High Risk / Likely to Churn."
            )

            st.write(
                "This is a statistical estimate from a trained Logistic "
                "Regression model based on patterns in historical data — "
                "it is **not a guarantee** of this specific customer's "
                "future behavior."
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with chart_col:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("**Customer Profile Summary**")
            st.write(f"Tenure: {tenure} months")
            st.write(f"Contract: {contract}")
            st.write(f"Monthly Charges: ${monthly_charges:.2f}")
            st.write(f"Internet Service: {internet_service}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Retention Recommendations")
        st.caption(
            "These are simple, rule-based suggestions derived from the "
            "entered details — they are NOT generated by the machine "
            "learning model, and following them is not guaranteed to "
            "prevent churn."
        )
        recommendations = generate_retention_recommendations(input_dict)
        for rec in recommendations:
            st.write(f"- {rec}")
