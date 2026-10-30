"""
Model validation, hyperparameter tuning, and model comparison for the
Telco Customer Churn Logistic Regression project (Stage 5).

This module builds on src/preprocess.py and src/train.py rather than
duplicating their logic:
  - src/preprocess.py owns cleaning and the raw ColumnTransformer.
  - src/train.py owns the Logistic Regression pipeline, the single
    train/test split, and basic prediction/coefficient helpers.
  - This module (src/validation.py) adds everything Stage 4 deliberately
    left out: cross-validation, GridSearchCV tuning, an LDA pipeline for
    comparison, threshold analysis, and final model selection.

DATA LEAKAGE: every function here that fits anything (cross_validate,
GridSearchCV, cross_val_predict) is called with X_train/y_train ONLY --
never with the held-out test set. Because a scikit-learn Pipeline is
passed as the estimator in every case, preprocessing (OneHotEncoder +
StandardScaler) is refit from scratch inside EACH cross-validation fold,
using only that fold's training portion -- this is what makes
cross-validation itself leakage-safe, not just the outer train/test split.
"""

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline

from src.preprocess import build_preprocessing_pipeline
from src.train import build_model_pipeline, RANDOM_STATE

SCORING_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def make_cv(n_splits=5, random_state=RANDOM_STATE):
    """
    Build a StratifiedKFold splitter.

    StratifiedKFold (vs. plain KFold) makes every one of the 5 folds keep
    the same churn/non-churn ratio as the full training set, which matters
    here because churn is imbalanced (~26.5%). shuffle=True randomizes
    row order before splitting (rows aren't already shuffled in the raw
    CSV); random_state=42 makes the exact fold assignment reproducible.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def cross_validate_pipeline(pipeline, X_train, y_train, cv=None, scoring=None):
    """
    Run 5-fold (by default) stratified cross-validation for a given
    pipeline, using the TRAINING data only.

    Returns the raw dict from sklearn.model_selection.cross_validate
    (per-fold scores for every metric in `scoring`), plus a small summary
    DataFrame of each metric's mean and standard deviation across folds.

    Because `pipeline` is a full Pipeline (preprocessing + classifier),
    cross_validate refits preprocessing from scratch on each fold's
    training portion internally -- the validation fold in each split is
    never used to fit the encoder/scaler, only to score the already-fit
    model. This is what makes cross-validation here leakage-safe.
    """
    if cv is None:
        cv = make_cv()
    if scoring is None:
        scoring = SCORING_METRICS

    cv_results = cross_validate(
        pipeline, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False
    )

    summary_rows = []
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        summary_rows.append({
            "metric": metric,
            "mean": scores.mean(),
            "std": scores.std(),
        })
    summary = pd.DataFrame(summary_rows)

    return cv_results, summary


def tune_logistic_regression(categorical_cols, numerical_cols, X_train, y_train,
                              param_grid=None, cv=None, scoring="roc_auc"):
    """
    Run GridSearchCV over the Logistic Regression regularization strength
    C, using the full preprocessing-plus-classifier pipeline as the
    estimator so that preprocessing is refit inside every fold (never on
    the complete training set before splitting into folds).

    C controls the INVERSE of regularization strength:
      - very small C -> strong regularization -> the model is pushed
        toward smaller coefficients, risking UNDERFITTING (too simple to
        capture real patterns).
      - very large C -> weak regularization -> coefficients are freer to
        grow, risking OVERFITTING (fitting noise in the training folds).

    scoring="roc_auc" is used as the primary metric for selecting the
    best C because it evaluates the model's ranking ability across all
    thresholds, which matters more than a single-threshold accuracy
    number for an imbalanced churn dataset.
    """
    if param_grid is None:
        param_grid = {"classifier__C": [0.001, 0.01, 0.1, 1, 10, 100]}
    if cv is None:
        cv = make_cv()

    pipeline = build_model_pipeline(categorical_cols, numerical_cols)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def build_lda_pipeline(categorical_cols, numerical_cols):
    """
    Build (unfitted) a full Pipeline: preprocessing + Linear Discriminant
    Analysis.

    Compatibility check performed before writing this function: LDA (unlike
    LogisticRegression) cannot fit on a SciPy sparse matrix. The existing
    build_preprocessing_pipeline() from src/preprocess.py was tested
    directly -- its ColumnTransformer has a default sparse_threshold=0.3,
    and because this dataset's one-hot encoded matrix is dense enough
    (~42% non-zero, verified empirically), the ColumnTransformer already
    returns a dense numpy array, not a sparse matrix, for this specific
    dataset. LDA was fit-tested on that exact output with no errors and no
    collinearity warnings. This means the SAME preprocessing pipeline used
    for Logistic Regression is safe to reuse for LDA here -- no separate,
    incompatible configuration is needed.
    """
    preprocessor = build_preprocessing_pipeline(categorical_cols, numerical_cols)
    model = LinearDiscriminantAnalysis()

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", model),
    ])
    return pipeline


def compare_models_cv(models, X_train, y_train, cv=None, scoring=None):
    """
    Run cross-validation (training data only) for each named pipeline in
    `models` (a dict of {name: unfitted pipeline}), and return a
    comparison DataFrame of mean cross-validated scores per metric, one
    row per model.
    """
    if cv is None:
        cv = make_cv()
    if scoring is None:
        scoring = SCORING_METRICS

    rows = []
    for name, pipeline in models.items():
        cv_results, _ = cross_validate_pipeline(pipeline, X_train, y_train, cv=cv, scoring=scoring)
        row = {"model": name}
        for metric in scoring:
            row[metric] = cv_results[f"test_{metric}"].mean()
        rows.append(row)

    return pd.DataFrame(rows)


def get_out_of_fold_probabilities(pipeline, X_train, y_train, cv=None):
    """
    Return out-of-fold predicted probabilities for the positive class
    (churn = 1), using cross_val_predict with the TRAINING data only.

    "Out-of-fold" means: for each row, its probability was predicted by a
    copy of the pipeline that was fit on the OTHER folds, never on the
    fold containing that row. This gives a leakage-safe stand-in for
    "unseen data" predictions without touching the real held-out test
    set, making it the right basis for threshold selection.
    """
    if cv is None:
        cv = make_cv()

    oof_proba = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    return oof_proba


def evaluate_thresholds(y_true, y_proba, thresholds):
    """
    For each candidate classification threshold, compute precision,
    recall, F1-score, and confusion matrix counts, by applying
    `y_proba >= threshold` instead of the default 0.5 cutoff.

    Returns a DataFrame with one row per threshold.
    """
    rows = []
    for threshold in thresholds:
        y_pred_at_threshold = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_at_threshold).ravel()

        rows.append({
            "threshold": threshold,
            "precision": precision_score(y_true, y_pred_at_threshold, zero_division=0),
            "recall": recall_score(y_true, y_pred_at_threshold, zero_division=0),
            "f1_score": f1_score(y_true, y_pred_at_threshold, zero_division=0),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        })

    return pd.DataFrame(rows)


def select_best_threshold(threshold_table, metric="f1_score"):
    """
    Select the threshold that maximizes the given metric (default:
    F1-score, which balances precision and recall) in `threshold_table`
    (as produced by evaluate_thresholds()).

    This is a single, reproducible, documented default. A real business
    could instead pick a lower threshold (favoring recall, if missing a
    churner is costlier than a false alarm) or a higher one (favoring
    precision, if outreach itself is expensive) -- that trade-off is
    reported alongside this selection, not hidden by it.
    """
    best_row = threshold_table.loc[threshold_table[metric].idxmax()]
    return float(best_row["threshold"])


def final_evaluate(pipeline, X_test, y_test, threshold=0.5):
    """
    Evaluate an already-fitted pipeline ONCE on the held-out test set,
    using a (possibly non-default) classification threshold applied to
    predict_proba() rather than pipeline.predict()'s built-in 0.5 cutoff.

    This function must only ever be called with the true held-out test
    set, and only after the model, hyperparameters, and threshold have
    already been chosen using cross-validation / training data alone.

    Returns a dict of metrics plus the raw predictions, probabilities,
    and confusion matrix for plotting.
    """
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    cm = confusion_matrix(y_test, y_pred)

    return {
        "metrics": metrics,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "confusion_matrix": cm,
        "threshold": threshold,
    }


def select_final_model(comparison_table, candidates, priority_metric="roc_auc"):
    """
    Choose the final model from `candidates` (a dict of {name: pipeline})
    based on `comparison_table` (as produced by compare_models_cv()),
    using cross-validated `priority_metric` (default: roc_auc) as the
    primary selection criterion -- NOT raw accuracy, since accuracy alone
    can be misleadingly high on an imbalanced dataset.

    Returns (selected_name, selected_pipeline, selection_reason_str).
    """
    best_row = comparison_table.loc[comparison_table[priority_metric].idxmax()]
    selected_name = best_row["model"]
    selected_pipeline = candidates[selected_name]

    reason = (
        f"Selected '{selected_name}' because it had the highest cross-validated "
        f"{priority_metric} ({best_row[priority_metric]:.4f}) among the candidates "
        f"compared in {sorted(candidates.keys())}. Accuracy alone was NOT used as "
        f"the selection criterion, since it can be misleadingly high on this "
        f"imbalanced dataset (a model predicting 'No churn' for everyone would "
        f"already score ~73.5% accuracy)."
    )
    return selected_name, selected_pipeline, reason


def save_final_model(pipeline, model_path, model_name, threshold, metrics, feature_columns):
    """
    Save the complete fitted pipeline (preprocessing + classifier) ALONG
    WITH the metadata the Streamlit application (Stage 6) will need:
    which model was selected, the chosen classification threshold, the
    metrics it achieved, and the exact feature column names/order it
    expects as input.

    Saved as a single dict via joblib so the Streamlit app only needs one
    joblib.load() call and never has to guess the threshold or feature
    order separately.
    """
    import joblib
    import os
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    artifact = {
        "pipeline": pipeline,
        "model_name": model_name,
        "threshold": threshold,
        "metrics": metrics,
        "feature_columns": list(feature_columns),
    }
    joblib.dump(artifact, model_path)
    return model_path


if __name__ == "__main__":
    # Standalone, reproducible regeneration of the Stage 5 final model
    # artifact (models/final_churn_model.joblib), for a fresh clone where
    # that gitignored file doesn't exist yet. This runs the exact same
    # steps, in the exact same order, as notebooks/churn_analysis.ipynb
    # Section 11 -- no different model, no shortcuts, no test-set tuning.
    from src.preprocess import identify_feature_types
    from src.train import (
        load_and_clean_data, prepare_features_and_target, split_data,
        build_model_pipeline, train_model,
    )

    DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    MODEL_PATH = "models/final_churn_model.joblib"

    print("Loading and cleaning data...")
    df = load_and_clean_data(DATA_PATH)
    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    categorical_cols, numerical_cols = identify_feature_types(df)
    cv = make_cv()

    print("Tuning Logistic Regression (GridSearchCV, training data only)...")
    grid_search = tune_logistic_regression(categorical_cols, numerical_cols, X_train, y_train, cv=cv)
    print(f"  Best C: {grid_search.best_params_['classifier__C']}, "
          f"best CV ROC-AUC: {grid_search.best_score_:.4f}")

    print("Comparing baseline LR, tuned LR, and LDA (5-fold CV, training data only)...")
    candidates = {
        "logistic_regression_baseline": build_model_pipeline(categorical_cols, numerical_cols),
        "logistic_regression_tuned": grid_search.best_estimator_,
        "lda": build_lda_pipeline(categorical_cols, numerical_cols),
    }
    comparison_table = compare_models_cv(candidates, X_train, y_train, cv=cv)
    print(comparison_table.to_string(index=False))

    selected_name, selected_pipeline, reason = select_final_model(comparison_table, candidates, priority_metric="roc_auc")
    print(f"Selected model: {selected_name}")

    print("Selecting classification threshold (out-of-fold CV predictions, training data only)...")
    oof_proba = get_out_of_fold_probabilities(selected_pipeline, X_train, y_train, cv=cv)
    threshold_table = evaluate_thresholds(y_train, oof_proba, [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    selected_threshold = select_best_threshold(threshold_table, metric="f1_score")
    print(f"Selected threshold: {selected_threshold}")

    print("Fitting selected model on full training set and evaluating ONCE on the held-out test set...")
    final_pipeline = train_model(selected_pipeline, X_train, y_train)
    final_result = final_evaluate(final_pipeline, X_test, y_test, threshold=selected_threshold)
    print("Final held-out test metrics:", {k: round(v, 4) for k, v in final_result["metrics"].items()})

    saved_path = save_final_model(
        pipeline=final_pipeline,
        model_path=MODEL_PATH,
        model_name=selected_name,
        threshold=selected_threshold,
        metrics=final_result["metrics"],
        feature_columns=X.columns,
    )
    print(f"\nFinal model artifact saved to: {saved_path}")
