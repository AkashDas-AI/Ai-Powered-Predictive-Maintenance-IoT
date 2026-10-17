"""
Unit tests for baseline model training, evaluation metrics, pipeline contracts, and artifact loading.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.data.loader import get_project_root
from src.features.schema import (
    PROHIBITED_COLUMNS,
    validate_feature_matrix_columns,
)
from src.features.engineering import engineer_features
from src.features.split import split_dataset
from src.evaluate import (
    compute_classification_metrics,
    find_optimal_threshold_oof,
)
from src.train import (
    get_candidate_models,
    run_cross_validation_comparison,
)


@pytest.fixture
def synthetic_training_fixture() -> tuple[pd.DataFrame, pd.Series]:
    """Generates a small synthetic fixture (30 samples, 3 failures) for fast unit testing."""
    np.random.seed(42)
    n = 30
    df = pd.DataFrame({
        "Type": np.random.choice(["L", "M", "H"], size=n),
        "Air temperature [K]": np.random.normal(300, 2, size=n),
        "Process temperature [K]": np.random.normal(310, 2, size=n),
        "Rotational speed [rpm]": np.random.randint(1200, 1800, size=n),
        "Torque [Nm]": np.random.uniform(20, 60, size=n),
        "Tool wear [min]": np.random.randint(0, 220, size=n),
        "Machine failure": [0] * 27 + [1] * 3,
    })
    df_eng = engineer_features(df)
    X = df_eng.drop(columns=["Machine failure"])
    y = df_eng["Machine failure"]
    return X, y


def test_candidate_pipelines_can_fit_and_predict(synthetic_training_fixture):
    """Verify that all candidate baseline pipelines fit and produce valid predictions on a fixture."""
    X, y = synthetic_training_fixture
    candidates = get_candidate_models(random_state=42)

    assert "Dummy_Majority" in candidates
    assert "Logistic_Regression_Balanced" in candidates
    assert "Random_Forest_Balanced" in candidates

    for name, pipeline in candidates.items():
        assert isinstance(pipeline, Pipeline)
        pipeline.fit(X, y)

        preds = pipeline.predict(X)
        assert len(preds) == len(y)

        probs = pipeline.predict_proba(X)
        assert probs.shape[0] == len(y)
        assert probs.shape[1] in (1, 2)
        if probs.shape[1] == 2:
            assert np.all((probs >= 0.0) & (probs <= 1.0))
            np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_feature_contract_excludes_prohibited_columns(synthetic_training_fixture):
    """Verify that prohibited columns (identifiers, target, failure modes) cannot enter pipelines."""
    X, _ = synthetic_training_fixture

    # Assert no prohibited columns in predictive features
    for col in PROHIBITED_COLUMNS:
        assert col not in X.columns

    # Verify that passing any prohibited column triggers a ValueError
    for col in ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF", "UDI", "Product ID"]:
        with pytest.raises(ValueError) as exc_info:
            validate_feature_matrix_columns(["Air temperature [K]", col])
        assert "Prediction contract violation" in str(exc_info.value)


def test_preprocessing_is_inside_cross_validation_pipeline():
    """Verify that the preprocessor is embedded within the Pipeline to guarantee fold isolation."""
    candidates = get_candidate_models(random_state=42)

    for name, pipeline in candidates.items():
        assert "preprocessor" in pipeline.named_steps
        assert isinstance(pipeline.named_steps["preprocessor"], ColumnTransformer)
        assert "classifier" in pipeline.named_steps


def test_cross_validation_does_not_consume_test_set():
    """Verify that cross-validation operates exclusively on training data, keeping test set untouched."""
    # Create small dataset
    n = 40
    raw_df = pd.DataFrame({
        "Type": ["L"] * 20 + ["M"] * 20,
        "Air temperature [K]": np.linspace(298, 302, n),
        "Process temperature [K]": np.linspace(308, 312, n),
        "Rotational speed [rpm]": [1500] * n,
        "Torque [Nm]": [40.0] * n,
        "Tool wear [min]": list(range(n)),
        "Machine failure": [0] * 36 + [1] * 4,
    })
    df_eng = engineer_features(raw_df)
    X_train, X_test, y_train, y_test = split_dataset(df_eng, test_size=0.25, random_state=42)

    # Snapshot of test set
    test_snapshot = X_test.copy(deep=True)
    test_target_snapshot = y_test.copy(deep=True)

    # Run CV comparison strictly on X_train, y_train
    comparison_df, oof_probs, best_model = run_cross_validation_comparison(
        X_train, y_train, random_state=42, n_splits=2
    )

    # Test set must remain completely unchanged
    pd.testing.assert_frame_equal(X_test, test_snapshot)
    pd.testing.assert_series_equal(y_test, test_target_snapshot)
    assert len(comparison_df) == 3


def test_evaluation_metrics_positive_class_correctness():
    """Verify diagnostic metric calculations on a known ground truth case."""
    # 2 true negatives, 1 false positive, 1 false negative, 2 true positives
    # Ground truth: [0, 0, 0, 1, 1, 1]
    # Predictions:  [0, 0, 1, 0, 1, 1]
    y_true = [0, 0, 0, 1, 1, 1]
    y_prob = [0.1, 0.2, 0.7, 0.3, 0.8, 0.9]

    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["true_negatives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_positives"] == 2

    # Precision = 2 / (2 + 1) = 0.6667
    assert np.isclose(metrics["precision"], 2 / 3, atol=1e-3)
    # Recall = 2 / (2 + 1) = 0.6667
    assert np.isclose(metrics["recall"], 2 / 3, atol=1e-3)
    # F1 = 2 * (2/3 * 2/3) / (4/3) = 0.6667
    assert np.isclose(metrics["f1_score"], 2 / 3, atol=1e-3)


def test_saved_pipeline_can_be_loaded_and_produces_probabilities():
    """Verify that the serialized joblib artifact loads and predicts probabilities correctly."""
    root = get_project_root()
    model_path = root / "models" / "selected_model.joblib"

    assert model_path.exists(), f"Saved model not found at {model_path}"

    loaded_pipeline = joblib.load(model_path)
    assert isinstance(loaded_pipeline, Pipeline)

    # Create a representative single-sample DataFrame
    sample = pd.DataFrame({
        "Type": ["M"],
        "Air temperature [K]": [300.0],
        "Process temperature [K]": [310.5],
        "Rotational speed [rpm]": [1500],
        "Torque [Nm]": [42.0],
        "Tool wear [min]": [100],
        "temperature_difference": [10.5],
        "mechanical_power_w": [6597.34],
    })

    probs = loaded_pipeline.predict_proba(sample)
    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0, 1] <= 1.0


def test_deterministic_reproducibility(synthetic_training_fixture):
    """Verify that running cross-validation twice with the same seed yields identical scores."""
    X, y = synthetic_training_fixture

    df1, _, _ = run_cross_validation_comparison(X, y, random_state=42, n_splits=3)
    df2, _, _ = run_cross_validation_comparison(X, y, random_state=42, n_splits=3)

    pd.testing.assert_frame_equal(df1, df2)

