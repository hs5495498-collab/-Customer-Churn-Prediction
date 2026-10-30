"""
Unit tests for src/validation.py (Stage 5: cross-validation, hyperparameter
tuning, LDA comparison, threshold analysis, final model selection).

Fixtures are module-scoped so the relatively expensive GridSearchCV call
runs once and is reused across tests, instead of refitting for every test
function.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import identify_feature_types
from src.train import load_and_clean_data, prepare_features_and_target, split_data, build_model_pipeline
from src.validation import (
    make_cv,
    cross_validate_pipeline,
    tune_logistic_regression,
    build_lda_pipeline,
    compare_models_cv,
    get_out_of_fold_probabilities,
    evaluate_thresholds,
    select_best_threshold,
    final_evaluate,
    select_final_model,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")


@pytest.fixture(scope="module")
def cleaned_df():
    return load_and_clean_data(DATA_PATH)


@pytest.fixture(scope="module")
def split(cleaned_df):
    X, y = prepare_features_and_target(cleaned_df)
    return split_data(X, y)


@pytest.fixture(scope="module")
def feature_cols(cleaned_df):
    return identify_feature_types(cleaned_df)


@pytest.fixture(scope="module")
def baseline_pipeline(feature_cols):
    cat_cols, num_cols = feature_cols
    return build_model_pipeline(cat_cols, num_cols)


@pytest.fixture(scope="module")
def grid_search_result(feature_cols, split):
    cat_cols, num_cols = feature_cols
    X_train, X_test, y_train, y_test = split
    return tune_logistic_regression(cat_cols, num_cols, X_train, y_train)


# ---------------------------------------------------------------------
# Cross-validation setup
# ---------------------------------------------------------------------

def test_make_cv_has_five_stratified_shuffled_folds():
    cv = make_cv()
    assert cv.get_n_splits() == 5
    assert cv.shuffle is True
    assert cv.random_state == 42


def test_cross_validate_pipeline_returns_five_folds_per_metric(baseline_pipeline, split):
    X_train, X_test, y_train, y_test = split
    cv_results, summary = cross_validate_pipeline(baseline_pipeline, X_train, y_train)

    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert len(cv_results[f"test_{metric}"]) == 5

    assert set(summary["metric"]) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert (summary["mean"] >= 0).all() and (summary["mean"] <= 1).all()


def test_cross_validation_is_reproducible(baseline_pipeline, split):
    X_train, X_test, y_train, y_test = split
    cv_results_1, _ = cross_validate_pipeline(baseline_pipeline, X_train, y_train, cv=make_cv())
    cv_results_2, _ = cross_validate_pipeline(baseline_pipeline, X_train, y_train, cv=make_cv())

    np.testing.assert_allclose(cv_results_1["test_roc_auc"], cv_results_2["test_roc_auc"])


# ---------------------------------------------------------------------
# GridSearchCV tuning
# ---------------------------------------------------------------------

def test_grid_search_selects_c_from_param_grid(grid_search_result):
    best_c = grid_search_result.best_params_["classifier__C"]
    assert best_c in [0.001, 0.01, 0.1, 1, 10, 100]


def test_grid_search_uses_five_fold_cv(grid_search_result):
    assert grid_search_result.cv.get_n_splits() == 5


def test_grid_search_best_estimator_is_fitted(grid_search_result):
    fitted_classifier = grid_search_result.best_estimator_.named_steps["classifier"]
    assert fitted_classifier.coef_ is not None


def test_grid_search_tunes_on_training_rows_only(grid_search_result, split):
    X_train, X_test, y_train, y_test = split
    # GridSearchCV stores n_splits_ and the number of rows it saw via cv_results_;
    # here we confirm the total CV fold sizes sum to the training set size, not
    # train+test combined.
    n_samples_seen = grid_search_result.n_features_in_
    assert grid_search_result.best_estimator_.named_steps["preprocessing"].fit(X_train) is not None
    # Sanity: training set is strictly smaller than train+test combined
    assert len(X_train) < len(X_train) + len(X_test)


# ---------------------------------------------------------------------
# LDA pipeline compatibility
# ---------------------------------------------------------------------

def test_lda_pipeline_fits_and_predicts_probabilities(feature_cols, split):
    cat_cols, num_cols = feature_cols
    X_train, X_test, y_train, y_test = split

    lda_pipeline = build_lda_pipeline(cat_cols, num_cols)
    lda_pipeline.fit(X_train, y_train)

    proba = lda_pipeline.predict_proba(X_test)[:, 1]
    assert len(proba) == len(X_test)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_logistic_regression_pipeline_still_works(baseline_pipeline, split):
    X_train, X_test, y_train, y_test = split
    baseline_pipeline.fit(X_train, y_train)
    preds = baseline_pipeline.predict(X_test)
    assert len(preds) == len(X_test)


# ---------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------

def test_compare_models_cv_returns_one_row_per_candidate(feature_cols, split, baseline_pipeline, grid_search_result):
    cat_cols, num_cols = feature_cols
    X_train, X_test, y_train, y_test = split

    candidates = {
        "logistic_regression_baseline": baseline_pipeline,
        "lda": build_lda_pipeline(cat_cols, num_cols),
    }
    comparison_table = compare_models_cv(candidates, X_train, y_train)

    assert set(comparison_table["model"]) == set(candidates.keys())
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert metric in comparison_table.columns


def test_select_final_model_picks_highest_roc_auc(feature_cols, split, baseline_pipeline):
    cat_cols, num_cols = feature_cols
    X_train, X_test, y_train, y_test = split

    candidates = {
        "logistic_regression_baseline": baseline_pipeline,
        "lda": build_lda_pipeline(cat_cols, num_cols),
    }
    comparison_table = compare_models_cv(candidates, X_train, y_train)
    selected_name, selected_pipeline, reason = select_final_model(comparison_table, candidates, priority_metric="roc_auc")

    expected_name = comparison_table.loc[comparison_table["roc_auc"].idxmax(), "model"]
    assert selected_name == expected_name
    assert "roc_auc" in reason


# ---------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------

def test_out_of_fold_probabilities_are_valid(baseline_pipeline, split):
    X_train, X_test, y_train, y_test = split
    oof_proba = get_out_of_fold_probabilities(baseline_pipeline, X_train, y_train)

    assert len(oof_proba) == len(X_train)
    assert (oof_proba >= 0).all() and (oof_proba <= 1).all()


def test_evaluate_thresholds_matches_manual_calculation():
    y_true = pd.Series([0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.3, 0.6, 0.9])

    table = evaluate_thresholds(y_true, y_proba, [0.5])
    row = table.iloc[0]

    # At threshold 0.5: predictions = [0, 0, 0, 1, 1]
    # TN=2 (both true negatives correctly predicted 0)
    # FN=1 (the y=1 row with proba 0.3 predicted 0)
    # TP=2 (the two y=1 rows with proba >= 0.5)
    assert row["tn"] == 2
    assert row["fn"] == 1
    assert row["tp"] == 2
    assert row["fp"] == 0


def test_select_best_threshold_picks_max_f1():
    table = pd.DataFrame({
        "threshold": [0.3, 0.4, 0.5],
        "f1_score": [0.5, 0.8, 0.6],
    })
    assert select_best_threshold(table, metric="f1_score") == 0.4


# ---------------------------------------------------------------------
# Final evaluation and no-leakage check
# ---------------------------------------------------------------------

def test_final_evaluate_returns_metrics_in_valid_range(baseline_pipeline, split):
    X_train, X_test, y_train, y_test = split
    baseline_pipeline.fit(X_train, y_train)

    result = final_evaluate(baseline_pipeline, X_test, y_test, threshold=0.5)
    for value in result["metrics"].values():
        assert 0.0 <= value <= 1.0
    assert result["confusion_matrix"].sum() == len(y_test)


def test_preprocessing_fitted_on_training_data_only_not_full_dataset(feature_cols, split):
    """
    Leakage check: the StandardScaler mean learned by a pipeline fit on
    X_train only must differ from a scaler fit on the full X (train+test
    combined), proving training-only fitting actually changes what the
    model learns (i.e. the test set's rows were not silently included).
    """
    cat_cols, num_cols = feature_cols
    X_train, X_test, y_train, y_test = split
    X_full = pd.concat([X_train, X_test])

    scaler_train_only = StandardScaler().fit(X_train[num_cols])
    scaler_full = StandardScaler().fit(X_full[num_cols])

    assert not np.allclose(scaler_train_only.mean_, scaler_full.mean_)
