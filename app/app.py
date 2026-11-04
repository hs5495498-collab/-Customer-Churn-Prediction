"""
Customer Churn Prediction System — Streamlit demo application.

Loads the Stage 5 final trained pipeline (models/final_churn_model.joblib)
and uses it, unmodified, to score a single customer entered through this
form. No model is retrained here, and no preprocessing step is
reimplemented -- the saved pipeline performs all encoding/scaling exactly
as it did during training, so the only new code in this file is the UI
(a multi-page dashboard layout), simple rule-based retention suggestions,
and the display of already-verified Stage 2/5 project statistics.

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

APP_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(APP_DIR, "..", "models", "final_churn_model.joblib")
FIGURES_DIR = os.path.join(APP_DIR, "..", "reports", "figures")

# ---------------------------------------------------------------------
# Verified project statistics (Stage 1-5). These are NOT fabricated --
# every figure here is taken from reports/final_project_report.md and
# reports/figures/, produced by the actual EDA (src/preprocess.py) and
# validation (src/validation.py) stages. They are displayed as static
# facts, never recomputed or guessed at by this UI layer.
# ---------------------------------------------------------------------
DATASET_SIZE = 7043
CHURN_COUNT = 1869
RETAINED_COUNT = 5174
CHURN_RATE = 26.54
CV_ROC_AUC = 0.8463
BEST_C = 10

TEST_METRICS = {
    "accuracy": 0.7622,
    "precision": 0.5399,
    "recall": 0.7059,
    "f1": 0.6118,
    "roc_auc": 0.8409,
}
CONFUSION_MATRIX = {"TN": 810, "FP": 225, "FN": 110, "TP": 264}

CONTRACT_CHURN_RATES = {
    "Month-to-month": 42.71,
    "One year": 11.27,
    "Two year": 2.83,
}
AVG_TENURE_CHURNED = 18.0
AVG_TENURE_RETAINED = 37.6
AVG_MONTHLY_CHURNED = 74.44
AVG_MONTHLY_RETAINED = 61.27
CORR_TENURE_CHURN = -0.35
CORR_MONTHLY_CHARGES_CHURN = 0.19

FIGURES = {
    "churn_pie": "churn_distribution_pie.png",
    "contract_churn": "churn_rate_by_contract.png",
    "tenure_dist": "tenure_distribution.png",
    "correlation_heatmap": "correlation_heatmap.png",
    "boxplot_tenure": "boxplot_tenure.png",
    "boxplot_monthly": "boxplot_MonthlyCharges.png",
    "boxplot_total": "boxplot_TotalCharges.png",
    "confusion_matrix": "confusion_matrix.png",
    "roc_curve": "roc_curve.png",
    "threshold_analysis": "threshold_analysis.png",
    "final_eval": "final_model_evaluation.png",
}

NAV_OVERVIEW = "🏠 Overview"
NAV_PREDICT = "🎯 Predict Churn"
NAV_INSIGHTS = "📊 Customer Insights"
NAV_PERFORMANCE = "📈 Model Performance"
NAV_ABOUT = "ℹ️ About Project"
NAV_PAGES = [NAV_OVERVIEW, NAV_PREDICT, NAV_INSIGHTS, NAV_PERFORMANCE, NAV_ABOUT]

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📉",
    layout="wide",
)

CUSTOM_CSS = """
<style>
:root {
    --navy: #0f1b33;
    --navy-light: #16233f;
    --charcoal: #1e293b;
    --cyan: #22d3ee;
    --blue: #3b82f6;
    --teal: #14b8a6;
    --green: #22c55e;
    --amber: #f59e0b;
    --surface: #ffffff;
    --border: #e5e7eb;
    --muted: #6b7280;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--navy);
}
section[data-testid="stSidebar"] * { color: #e5e7eb; }
section[data-testid="stSidebar"] hr { border-color: #2a3a5c; }

.brand-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.25rem;
}
.brand-icon {
    width: 2.4rem;
    height: 2.4rem;
    border-radius: 10px;
    background: linear-gradient(135deg, var(--cyan), var(--blue));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    flex-shrink: 0;
}
.brand-title {
    font-size: 1.05rem;
    font-weight: 700;
    line-height: 1.15;
}
.brand-subtitle {
    font-size: 0.75rem;
    color: #94a3b8;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    background-color: rgba(34, 197, 94, 0.15);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.35);
}
.status-dot {
    width: 0.5rem; height: 0.5rem; border-radius: 50%;
    background-color: #4ade80;
    display: inline-block;
}

.sidebar-fact {
    font-size: 0.85rem;
    margin: 0.15rem 0;
}
.sidebar-fact b { color: #f1f5f9; }

/* Header */
.app-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.25rem;
}
.main-title { font-size: 2rem; font-weight: 800; margin-bottom: 0.1rem; color: #0f1b33; }
.main-subtitle { font-size: 1rem; color: var(--muted); margin-top: 0; }
.page-heading {
    font-size: 1.3rem;
    font-weight: 700;
    color: #0f1b33;
    margin: 0.5rem 0 0.75rem 0;
}

/* Cards */
.metric-card {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    transition: box-shadow 0.15s ease, transform 0.15s ease;
    height: 100%;
}
.metric-card:hover {
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.10);
    transform: translateY(-1px);
}
.metric-card .metric-label {
    font-size: 0.8rem;
    color: var(--muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.metric-card .metric-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #0f1b33;
    margin-top: 0.15rem;
}
.metric-card .metric-accent { height: 4px; border-radius: 4px; margin-bottom: 0.6rem; }
.accent-cyan { background: linear-gradient(90deg, var(--cyan), var(--blue)); }
.accent-teal { background: linear-gradient(90deg, var(--teal), var(--green)); }
.accent-amber { background: linear-gradient(90deg, var(--amber), #fb923c); }

.section-card {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
}

.risk-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1.05rem;
}
.risk-high { background-color: #fff7ed; color: #b45309; border: 1px solid #fed7aa; }
.risk-low { background-color: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }

.rec-card {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--cyan);
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.65rem;
}
.rec-card .rec-title { font-weight: 700; color: #0f1b33; margin-bottom: 0.15rem; }
.rec-card .rec-body { font-size: 0.9rem; color: #374151; }

.disclaimer {
    font-size: 0.85rem;
    color: var(--muted);
    border-top: 1px solid var(--border);
    padding-top: 0.75rem;
    margin-top: 1rem;
}

figure, .stImage { margin-bottom: 0.25rem; }
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


def figure_path(key):
    path = os.path.join(FIGURES_DIR, FIGURES[key])
    return path if os.path.exists(path) else None


def metric_card(label, value, accent="accent-cyan"):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-accent {accent}"></div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_figure(key, caption):
    path = figure_path(key)
    if path:
        st.image(path, caption=caption, use_column_width=True)
    else:
        st.info(f"Figure not available: {FIGURES[key]} (run the Stage 2/5 notebook to regenerate it).")


# ---------------------------------------------------------------------
# Sidebar: branding, navigation, model status, project facts
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-icon">📉</div>
            <div>
                <div class="brand-title">Customer Churn<br/>Prediction</div>
            </div>
        </div>
        <div class="brand-subtitle">CSE3141 — Predictive Analytics</div>
        <br/>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<span class="status-pill"><span class="status-dot"></span> Model loaded &amp; ready</span>',
        unsafe_allow_html=True,
    )

    st.markdown("<br/>", unsafe_allow_html=True)
    nav_choice = st.radio("Navigation", NAV_PAGES, key="nav_page", label_visibility="collapsed")

    st.markdown("---")
    st.markdown("#### Model Status")
    st.markdown(
        f"""
        <p class="sidebar-fact">Algorithm: <b>Logistic Regression</b></p>
        <p class="sidebar-fact">Selected model: <b>{model_name}</b></p>
        <p class="sidebar-fact">Regularization (C): <b>10</b></p>
        <p class="sidebar-fact">Classification threshold: <b>{threshold:.0%}</b></p>
        <p class="sidebar-fact">Dataset: <b>Telco Customer Churn</b> (7,043 customers)</p>
        """,
        unsafe_allow_html=True,
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
# Header
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <div>
            <p class="main-title">Customer Churn Prediction System</p>
            <p class="main-subtitle">AI-powered customer retention and churn risk analysis.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------
def render_overview():
    st.markdown('<p class="page-heading">Project Overview</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Dataset Size", f"{DATASET_SIZE:,}", "accent-cyan")
    with c2:
        metric_card("Churn Rate", f"{CHURN_RATE:.2f}%", "accent-amber")
    with c3:
        metric_card("Retained Customers", f"{RETAINED_COUNT:,}", "accent-teal")
    with c4:
        metric_card("Churned Customers", f"{CHURN_COUNT:,}", "accent-amber")

    st.markdown("<br/>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Model", model_name, "accent-cyan")
    with c2:
        metric_card("Test ROC-AUC", f"{TEST_METRICS['roc_auc']:.4f}", "accent-teal")
    with c3:
        metric_card("Best C", str(BEST_C), "accent-cyan")

    st.markdown("<br/>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        show_figure("churn_pie", "Churn distribution — 26.54% churned vs. 73.46% retained")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        show_figure("contract_churn", "Churn rate by contract type")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    show_figure("tenure_dist", "Tenure distribution across the dataset")
    st.markdown('</div>', unsafe_allow_html=True)

    st.caption(
        "All figures above are verified Stage 1-2 results from "
        "reports/final_project_report.md and reports/figures/, computed "
        "once from the real Telco Customer Churn dataset — none of these "
        "numbers are generated by this UI."
    )


# ---------------------------------------------------------------------
# Page: Predict Churn
# ---------------------------------------------------------------------
def render_predict():
    st.markdown('<p class="page-heading">Predict Customer Churn</p>', unsafe_allow_html=True)

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
            tenure = st.slider(
                "Tenure (months)", min_value=TENURE_MIN, max_value=TENURE_MAX, value=12,
                help="Number of months the customer has been with the company.",
            )
        with c2:
            monthly_charges = st.number_input(
                "Monthly Charges ($)", min_value=MONTHLY_CHARGES_MIN, max_value=MONTHLY_CHARGES_MAX,
                value=65.0, step=0.5, help="Current monthly billing amount.",
            )
        with c3:
            total_charges = st.number_input(
                "Total Charges ($)", min_value=TOTAL_CHARGES_MIN, max_value=TOTAL_CHARGES_MAX,
                value=float(round(monthly_charges * max(tenure, 1), 2)), step=1.0,
                help="Total amount billed to the customer so far.",
            )
        c1, c2, c3 = st.columns(3)
        with c1:
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        with c2:
            partner = st.selectbox("Partner", categorical_options["Partner"])
        with c3:
            dependents = st.selectbox("Dependents", categorical_options["Dependents"])

        st.markdown("#### Contract Information")
        c1, c2, c3 = st.columns(3)
        with c1:
            contract = st.selectbox(
                "Contract Type", categorical_options["Contract"],
                help="Longer contracts are strongly associated with lower churn.",
            )
        with c2:
            paperless_billing = st.selectbox("Paperless Billing", categorical_options["PaperlessBilling"])
        with c3:
            payment_method = st.selectbox("Payment Method", categorical_options["PaymentMethod"])

        st.markdown("#### Internet and Services")
        c1, c2, c3 = st.columns(3)
        with c1:
            internet_service = st.selectbox("Internet Service", categorical_options["InternetService"])
            online_security = st.selectbox("Online Security", categorical_options["OnlineSecurity"])
            phone_service = st.selectbox("Phone Service", categorical_options["PhoneService"])
        with c2:
            online_backup = st.selectbox("Online Backup", categorical_options["OnlineBackup"])
            device_protection = st.selectbox("Device Protection", categorical_options["DeviceProtection"])
            multiple_lines = st.selectbox("Multiple Lines", categorical_options["MultipleLines"])
        with c3:
            tech_support = st.selectbox("Tech Support", categorical_options["TechSupport"])
            streaming_tv = st.selectbox("Streaming TV", categorical_options["StreamingTV"])
        streaming_movies = st.selectbox("Streaming Movies", categorical_options["StreamingMovies"])

        submitted = st.form_submit_button("Predict Churn Risk", type="primary", use_container_width=True)

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
            st.markdown('<p class="page-heading">Prediction Result</p>', unsafe_allow_html=True)

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
                    "The model estimates the likelihood of customer churn "
                    "based on the supplied customer information. This is a "
                    "statistical prediction, not a guarantee."
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
            icons = ["💡", "📋", "🛡️", "🎧", "🤝", "📶", "🎯"]
            for i, rec in enumerate(recommendations):
                icon = icons[i % len(icons)]
                st.markdown(
                    f"""
                    <div class="rec-card">
                        <div class="rec-title">{icon} Rule-Based Suggestion</div>
                        <div class="rec-body">{rec}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------
# Page: Customer Insights
# ---------------------------------------------------------------------
def render_insights():
    st.markdown('<p class="page-heading">Customer Insights (Exploratory Data Analysis)</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Overall Churn", f"{CHURN_RATE:.2f}%", "accent-amber")
    with c2:
        metric_card("Month-to-Month Churn", f"{CONTRACT_CHURN_RATES['Month-to-month']:.2f}%", "accent-amber")
    with c3:
        metric_card("One-Year Churn", f"{CONTRACT_CHURN_RATES['One year']:.2f}%", "accent-teal")
    with c4:
        metric_card("Two-Year Churn", f"{CONTRACT_CHURN_RATES['Two year']:.2f}%", "accent-teal")

    st.markdown("<br/>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Average Tenure**")
        st.write(f"Churned customers: ~{AVG_TENURE_CHURNED:.1f} months")
        st.write(f"Retained customers: ~{AVG_TENURE_RETAINED:.1f} months")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Average Monthly Charges**")
        st.write(f"Churned customers: ~${AVG_MONTHLY_CHURNED:.2f}")
        st.write(f"Retained customers: ~${AVG_MONTHLY_RETAINED:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**Numeric Correlations with Churn**")
    st.write(f"Tenure vs. Churn: {CORR_TENURE_CHURN:+.2f} (strongest numeric relationship — longer tenure, lower churn)")
    st.write(f"Monthly Charges vs. Churn: {CORR_MONTHLY_CHARGES_CHURN:+.2f} (higher charges, slightly higher churn)")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        show_figure("contract_churn", "Churn rate by contract type")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        show_figure("correlation_heatmap", "Correlation heatmap of numeric features")
        st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        show_figure("boxplot_tenure", "Tenure by churn status")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        show_figure("boxplot_monthly", "Monthly Charges by churn status")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        show_figure("boxplot_total", "Total Charges by churn status")
        st.markdown('</div>', unsafe_allow_html=True)

    st.caption(
        "All figures and statistics above are the verified Stage 2 EDA "
        "results from reports/final_project_report.md and "
        "reports/figures/ — computed once from the real dataset, not "
        "generated by this dashboard."
    )


# ---------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------
def render_performance():
    st.markdown('<p class="page-heading">Model Performance (Stage 5)</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Algorithm", "Logistic Regression", "accent-cyan")
    with c2:
        metric_card("Regularization (C)", str(BEST_C), "accent-cyan")
    with c3:
        metric_card("Threshold", f"{threshold:.2f}", "accent-teal")
    with c4:
        metric_card("CV ROC-AUC", f"{CV_ROC_AUC:.4f}", "accent-teal")

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("#### Held-Out Test Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Accuracy", f"{TEST_METRICS['accuracy']:.4f}", "accent-cyan")
    with c2:
        metric_card("Precision", f"{TEST_METRICS['precision']:.4f}", "accent-cyan")
    with c3:
        metric_card("Recall", f"{TEST_METRICS['recall']:.4f}", "accent-teal")
    with c4:
        metric_card("F1-Score", f"{TEST_METRICS['f1']:.4f}", "accent-teal")
    with c5:
        metric_card("ROC-AUC", f"{TEST_METRICS['roc_auc']:.4f}", "accent-amber")

    with st.expander("What do these metrics mean?"):
        st.write(
            "**Accuracy** is the share of all predictions (churn and "
            "non-churn) that were correct. **Precision** is, among "
            "customers predicted to churn, the share that actually did. "
            "**Recall** is, among customers who actually churned, the "
            "share the model correctly flagged. **F1-score** balances "
            "precision and recall. **ROC-AUC** measures ranking quality "
            "across all thresholds — the probability the model ranks a "
            "random churner above a random non-churner."
        )

    st.markdown("<br/>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Confusion Matrix**")
        cm = CONFUSION_MATRIX
        cm_cols = st.columns(2)
        with cm_cols[0]:
            metric_card("True Negatives", cm["TN"], "accent-teal")
            metric_card("False Negatives", cm["FN"], "accent-amber")
        with cm_cols[1]:
            metric_card("False Positives", cm["FP"], "accent-amber")
            metric_card("True Positives", cm["TP"], "accent-teal")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        show_figure("confusion_matrix", "Confusion matrix (held-out test set)")
        st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        show_figure("roc_curve", "ROC curve (held-out test set)")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        show_figure("threshold_analysis", "Threshold analysis (out-of-fold CV probabilities)")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    show_figure("final_eval", "Final model evaluation summary")
    st.markdown('</div>', unsafe_allow_html=True)

    st.caption(
        "All metrics and figures above are the verified Stage 5 results "
        "from reports/final_project_report.md — computed exactly once on "
        "the held-out test set, never re-estimated by this UI."
    )


# ---------------------------------------------------------------------
# Page: About Project
# ---------------------------------------------------------------------
def render_about():
    st.markdown('<p class="page-heading">About This Project</p>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Project Title")
    st.write("Customer Churn Prediction Using Logistic Regression")

    st.markdown("#### Problem Statement")
    st.write(
        "Given a customer's account information and subscribed services, "
        "predict whether that customer will churn (Churn = Yes) or not "
        "(Churn = No), and produce a probability estimate that can be "
        "used to flag at-risk customers for proactive retention outreach. "
        "This is formulated as a binary classification problem."
    )

    st.markdown("#### Dataset")
    st.write(
        "Telco Customer Churn dataset (originally published by IBM, "
        "widely distributed via Kaggle): 7,043 customer records, 21 "
        "columns. Source: "
        "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Algorithm & Preprocessing")
        st.write(
            "- **Algorithm:** Logistic Regression (C=10, selected via "
            "GridSearchCV)\n"
            "- **One-Hot Encoding:** `OneHotEncoder(handle_unknown=\"ignore\")` "
            "for categorical columns\n"
            "- **Scaling:** `StandardScaler` for numeric columns\n"
            "- **Pipeline:** an unfitted `ColumnTransformer` + `Pipeline`, "
            "fit only inside the training split or a CV fold — never on "
            "the full dataset, to avoid data leakage\n"
            "- `gender` and `customerID` are excluded from predictive "
            "inputs"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Validation & Evaluation")
        st.write(
            "- **Cross-validation:** `StratifiedKFold(n_splits=5, "
            "shuffle=True, random_state=42)`\n"
            "- **Hyperparameter tuning:** `GridSearchCV` over "
            "`C ∈ {0.001, 0.01, 0.1, 1, 10, 100}`, scored on ROC-AUC\n"
            "- **Threshold selection:** evaluated on out-of-fold CV "
            "probabilities (`cross_val_predict`), never the test set\n"
            "- **Final evaluation:** the tuned model is evaluated exactly "
            "once on a held-out test set (1,409 customers) never used "
            "for fitting, tuning, or threshold selection"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Prediction Workflow")
    st.write(
        "1. A customer's details are entered on the **Predict Churn** page.\n"
        "2. The input is assembled into a single-row DataFrame with the "
        "exact raw column names the trained pipeline expects.\n"
        "3. The saved pipeline's `predict_proba()` produces a churn "
        "probability — the same preprocessing and model used in Stage 5, "
        "unmodified.\n"
        f"4. The probability is compared against the Stage 5-selected "
        f"threshold ({threshold:.2f}) to classify the customer as "
        "High Risk / Low Risk.\n"
        "5. Simple, rule-based retention suggestions are generated from "
        "the entered details — these are not produced by the model."
    )

    st.markdown("#### Technologies Used")
    st.write("Python, pandas, NumPy, scikit-learn, Streamlit, Matplotlib, Seaborn, joblib, pytest")

    st.markdown("#### Limitations")
    st.write(
        "- The threshold search was limited to 0.30-0.60; F1 was still "
        "rising at the upper bound, so a wider search might find a "
        "different optimum.\n"
        "- The best threshold for a real business depends on the actual "
        "relative cost of missing a churner vs. a false alarm — not "
        "resolved here.\n"
        "- Final evaluation rests on a single held-out test split."
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------
if nav_choice == NAV_OVERVIEW:
    render_overview()
elif nav_choice == NAV_PREDICT:
    render_predict()
elif nav_choice == NAV_INSIGHTS:
    render_insights()
elif nav_choice == NAV_PERFORMANCE:
    render_performance()
else:
    render_about()
