"""
Unit tests for diagnostics, explainability, error analysis, and failure mode exploration.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.loader import get_project_root
from src.features.schema import (
    FAILURE_TYPE_COLUMNS,
    PROHIBITED_COLUMNS,
    TARGET,
    validate_feature_matrix_columns,
)
from src.features.engineering import engineer_features
from src.diagnostics import (
    categorize_error,
    run_permutation_importance,
    run_error_analysis,
    run_failure_mode_exploration,
)
from src.train import get_candidate_models


@pytest.fixture
def sample_diagnostic_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, Pipeline]:
    """Generates a small synthetic fixture for fast diagnostic testing."""
    np.random.seed(42)
    n = 40
    raw_df = pd.DataFrame({
        "UDI": list(range(1, n + 1)),
        "Product ID": [f"L{i:05d}" for i in range(1, n + 1)],
        "Type": ["L"] * (n // 2) + ["M"] * (n // 2),
        "Air temperature [K]": np.linspace(298, 302, n),
        "Process temperature [K]": np.linspace(308, 312, n),
        "Rotational speed [rpm]": np.random.randint(1300, 1600, n),
        "Torque [Nm]": np.random.uniform(30, 60, n),
        "Tool wear [min]": np.random.randint(0, 220, n),
        "Machine failure": [0] * 36 + [1] * 4,
        "TWF": [0] * 38 + [1] * 2,
        "HDF": [0] * 37 + [1] * 3,
        "PWF": [0] * 39 + [1] * 1,
        "OSF": [0] * 38 + [1] * 2,
        "RNF": [0] * 39 + [1] * 1,
    })

    df_eng = engineer_features(raw_df)
    features_to_drop = PROHIBITED_COLUMNS
    X = df_eng.drop(columns=[c for c in features_to_drop if c in df_eng.columns])
    y = df_eng[TARGET]

    # Fit a fast pipeline
    pipe = get_candidate_models(random_state=42)["Random_Forest_Balanced"]
    pipe.fit(X, y)

    return raw_df, X, y, pipe


def test_categorize_error_mapping():
    """Verify exact mapping of actual and predicted binary flags to confusion quadrants."""
    row_tp = pd.Series({TARGET: 1, "predicted_failure": 1})
    row_tn = pd.Series({TARGET: 0, "predicted_failure": 0})
    row_fp = pd.Series({TARGET: 0, "predicted_failure": 1})
    row_fn = pd.Series({TARGET: 1, "predicted_failure": 0})

    assert categorize_error(row_tp) == "True Positive"
    assert categorize_error(row_tn) == "True Negative"
    assert categorize_error(row_fp) == "False Positive"
    assert categorize_error(row_fn) == "False Negative"


def test_permutation_importance_output_schema_and_determinism(sample_diagnostic_fixture):
    """Verify that permutation importance returns the expected schema and honors random seeds."""
    raw_df, X, y, pipe = sample_diagnostic_fixture

    perm_df1 = run_permutation_importance(pipe, X, y, random_state=42, n_repeats=3, save_artifacts=False)
    perm_df2 = run_permutation_importance(pipe, X, y, random_state=42, n_repeats=3, save_artifacts=False)

    assert set(perm_df1.columns) == {"feature", "importance_mean", "importance_std"}
    assert len(perm_df1) == X.shape[1]
    pd.testing.assert_frame_equal(perm_df1, perm_df2)


def test_error_analysis_preserves_records_and_labels(sample_diagnostic_fixture):
    """Verify that error analysis retains original UDIs, preserves labels, and computes categories."""
    raw_df, X, y, pipe = sample_diagnostic_fixture

    analysis_df, summary = run_error_analysis(pipe, raw_df, X, y, threshold=0.5, save_artifacts=False)

    assert len(analysis_df) == len(raw_df)
    assert "UDI" in analysis_df.columns
    assert "error_category" in analysis_df.columns
    assert "failure_probability" in analysis_df.columns

    # Verify original labels were not modified
    pd.testing.assert_series_equal(analysis_df[TARGET], raw_df[TARGET])
    for col in FAILURE_TYPE_COLUMNS:
        pd.testing.assert_series_equal(analysis_df[col], raw_df[col])

    # Check summary structure
    assert "confusion_breakdown" in summary
    assert "false_negatives_audit" in summary
    assert "false_positives_audit" in summary


def test_failure_mode_targets_cannot_enter_features(sample_diagnostic_fixture):
    """Verify that failure mode targets and Machine failure are strictly banned from model features."""
    raw_df, X, y, pipe = sample_diagnostic_fixture

    # Assert no prohibited columns in feature matrix X
    for col in PROHIBITED_COLUMNS:
        assert col not in X.columns

    # Explicit check that attempting to include any failure mode flag raises ValueError
    for col in FAILURE_TYPE_COLUMNS:
        with pytest.raises(ValueError) as exc_info:
            validate_feature_matrix_columns(["Air temperature [K]", col])
        assert "Prediction contract violation" in str(exc_info.value)


def test_rare_class_limitation_detected_and_reported(sample_diagnostic_fixture):
    """Verify that RNF is flagged as unviable for classifier training due to extreme sparsity and randomness."""
    raw_df, X, y, pipe = sample_diagnostic_fixture

    # Split into train/test fixtures
    X_train, X_test = X.iloc[:30], X.iloc[30:]

    comparison_df, summary = run_failure_mode_exploration(
        raw_df, X_train, X_test, random_state=42, save_artifacts=False
    )

    rnf_summary = summary["failure_modes_summary"]["RNF"]
    assert rnf_summary["viable_for_modeling"] is False
    assert "NOT VIABLE FOR CLASSIFICATION" in rnf_summary["rationale"]

    rnf_record = comparison_df[comparison_df["failure_mode"] == "RNF"].iloc[0]
    assert rnf_record["modeling_status"] == "Descriptive Only (Unviable)"
    assert pd.isna(rnf_record["mean_cv_avg_precision"])

