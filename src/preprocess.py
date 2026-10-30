"""
Data cleaning and preprocessing functions for the Telco Customer Churn dataset.

These functions are intentionally kept small and single-purpose so each step
can be understood, tested, and explained on its own (useful for a viva).

IMPORTANT (data leakage): the functions in this file that build the
preprocessing pipeline (OneHotEncoder + StandardScaler) return an UNFITTED
ColumnTransformer. It must only be .fit() on the training split, never on
the full dataset, otherwise information from the test set would leak into
training (see build_preprocessing_pipeline docstring).

IMPORTANT (ethics): 'gender' is deliberately excluded from the list of
predictive features returned by identify_feature_types(), so it never
reaches the model or any retention decision. It is still kept in the raw
dataframe so it can be used for descriptive analysis in the notebook.
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(csv_path):
    """Load the raw Telco Customer Churn CSV into a DataFrame."""
    return pd.read_csv(csv_path)


def inspect_data(df):
    """
    Return a dictionary summarizing the dataset: shape, dtypes, missing
    value counts, duplicate row count, and Churn distribution.

    This does not print anything itself so it can be reused in both the
    notebook (for display) and in tests (for assertions).
    """
    return {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "dtypes": df.dtypes.to_dict(),
        "missing_per_column": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "churn_counts": df["Churn"].value_counts().to_dict() if "Churn" in df.columns else None,
    }


def remove_duplicates(df):
    """Drop fully duplicate rows, if any. Returns a new DataFrame."""
    return df.drop_duplicates().reset_index(drop=True)


def convert_total_charges(df):
    """
    Convert the 'TotalCharges' column from text to numeric.

    In the raw CSV, TotalCharges is stored as text, and 11 rows contain a
    blank string (just whitespace) instead of a number. pd.to_numeric with
    errors='coerce' turns those blank strings into NaN so they can be
    detected and handled explicitly in handle_missing_total_charges(),
    instead of silently becoming 0 or crashing the conversion.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def handle_missing_total_charges(df):
    """
    Fill missing TotalCharges values (created by convert_total_charges)
    with 0, but ONLY after verifying every affected row has tenure == 0.

    Reasoning: TotalCharges is the cumulative amount billed to a customer.
    A customer with tenure == 0 months has not yet completed a single
    billing cycle, so a TotalCharges of 0 is not a guess -- it is the
    logically correct value. If any missing row had tenure > 0, filling
    with 0 would misrepresent a customer who has actually been charged,
    so this function raises an error in that case rather than silently
    filling the wrong value.
    """
    df = df.copy()
    missing_mask = df["TotalCharges"].isnull()

    if missing_mask.any():
        non_zero_tenure_missing = df.loc[missing_mask, "tenure"] != 0
        if non_zero_tenure_missing.any():
            raise ValueError(
                "Found missing TotalCharges rows where tenure != 0 — "
                "filling with 0 would not be justified for these rows. "
                "Investigate manually before proceeding."
            )
        df.loc[missing_mask, "TotalCharges"] = 0.0

    return df


def drop_customer_id(df):
    """
    Drop the 'customerID' column. It is a unique identifier with no
    predictive value (every customer has a different one), so keeping it
    as a model input would add noise, not signal.
    """
    return df.drop(columns=["customerID"])


def detect_outliers_iqr(df, columns):
    """
    Investigate outliers in the given numeric columns using the IQR
    (Interquartile Range) method, WITHOUT removing anything.

    For each column:
        Q1 = 25th percentile, Q3 = 75th percentile, IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    Any value outside [lower_bound, upper_bound] is flagged as a
    statistical outlier.

    Returns a dict keyed by column name, each containing the bounds, the
    outlier count, and the row indices flagged — so the caller can inspect
    and report them rather than blindly dropping rows.
    """
    report = {}
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        report[col] = {
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": int(outlier_mask.sum()),
            "outlier_indices": df.index[outlier_mask].tolist(),
        }
    return report


def plot_boxplots(df, columns, output_dir="reports/figures"):
    """
    Generate and save one box plot per numeric column to output_dir as a
    PNG file. Returns the list of file paths written.

    A box plot visualizes the same IQR statistics computed in
    detect_outliers_iqr(): the box spans Q1-Q3, the line inside is the
    median, the whiskers extend to the IQR bounds, and any point beyond
    the whiskers is a statistical outlier.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    for col in columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(df[col].dropna(), vert=True)
        ax.set_title(f"Box Plot of {col}")
        ax.set_ylabel(col)
        fig.tight_layout()

        out_path = os.path.join(output_dir, f"boxplot_{col}.png")
        fig.savefig(out_path)
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def identify_feature_types(df, target_col="Churn", exclude_cols=None):
    """
    Split the dataframe's columns into categorical and numerical feature
    lists, ready to be handed to a ColumnTransformer.

    exclude_cols defaults to ['gender'] for the ethical reason documented
    at the top of this file: gender must never be used as a predictive
    input or influence a retention decision, even though it stays in the
    raw dataframe for descriptive analysis in the notebook.

    The target column is also excluded, since it is the label, not a
    feature.
    """
    if exclude_cols is None:
        exclude_cols = ["gender"]

    feature_df = df.drop(columns=[target_col] + exclude_cols, errors="ignore")

    categorical_cols = feature_df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = feature_df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    return categorical_cols, numerical_cols


def build_preprocessing_pipeline(categorical_cols, numerical_cols):
    """
    Build (but do NOT fit) a ColumnTransformer that:
      - one-hot encodes the categorical columns (handle_unknown='ignore'
        so that a category never seen during training, e.g. a brand new
        PaymentMethod value, does not crash prediction -- it is encoded
        as all zeros instead of raising an error)
      - standard-scales the numerical columns (zero mean, unit variance)

    DATA LEAKAGE WARNING: this function only constructs the transformer.
    The caller (Stage 4, src/train.py) must call .fit() or
    .fit_transform() on the TRAINING split only, then use .transform()
    (never .fit_transform() again) on the test split and on any future
    customer data. Fitting on the full dataset before splitting would let
    statistics from the test set (e.g. the scaler's mean/std) leak into
    training, producing an overly optimistic evaluation.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numerical", StandardScaler(), numerical_cols),
        ]
    )
    return preprocessor


def clean_dataset(csv_path):
    """
    Run the full cleaning sequence (load -> remove duplicates -> convert
    TotalCharges -> handle missing TotalCharges -> drop customerID) and
    return the cleaned DataFrame.

    This does NOT encode or scale anything -- that is deliberately left to
    build_preprocessing_pipeline(), fitted only on the training split in
    Stage 4, so this function stays reusable for both EDA (which wants
    human-readable categories) and modeling.
    """
    df = load_data(csv_path)
    df = remove_duplicates(df)
    df = convert_total_charges(df)
    df = handle_missing_total_charges(df)
    df = drop_customer_id(df)
    return df


if __name__ == "__main__":
    DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    raw_df = load_data(DATA_PATH)
    print("Raw dataset shape:", raw_df.shape)

    summary = inspect_data(raw_df)
    print("Duplicate rows:", summary["duplicate_rows"])
    print("Churn distribution:", summary["churn_counts"])

    cleaned_df = clean_dataset(DATA_PATH)
    print("Cleaned dataset shape:", cleaned_df.shape)
    print("Remaining missing values:", cleaned_df.isnull().sum().sum())

    outlier_report = detect_outliers_iqr(cleaned_df, ["tenure", "MonthlyCharges", "TotalCharges"])
    for col, stats in outlier_report.items():
        print(f"{col}: {stats['outlier_count']} outliers "
              f"(bounds {stats['lower_bound']:.2f} to {stats['upper_bound']:.2f})")

    saved = plot_boxplots(cleaned_df, ["tenure", "MonthlyCharges", "TotalCharges"])
    print("Saved box plots:", saved)

    cat_cols, num_cols = identify_feature_types(cleaned_df)
    print("Categorical feature columns:", cat_cols)
    print("Numerical feature columns:", num_cols)
    print("'gender' excluded from features:", "gender" not in cat_cols)
