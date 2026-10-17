"""
Unit tests for feature engineering, prediction contracts, and preprocessing pipeline.
"""

import math
import pytest
import pandas as pd
import numpy as np

from src.features.schema import (
    TARGET,
    PROHIBITED_COLUMNS,
    FEATURE_TEMP_DIFF,
    FEATURE_MECH_POWER,
    RAW_NUMERIC_FEATURES,
    CATEGORICAL_PREDICTORS,
    ALL_NUMERIC_PREDICTORS,
    validate_feature_matrix_columns,
)
from src.features.engineering import (
    compute_temperature_difference,
    compute_mechanical_power,
    engineer_features,
)
from src.features.preprocessing import build_preprocessor
from src.features.split import split_dataset


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Creates a small synthetic DataFrame mimicking the raw AI4I schema for unit testing."""
    return pd.DataFrame({
        "UDI": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Product ID": ["L01", "M02", "H03", "L04", "M05", "L06", "L07", "H08", "M09", "L10"],
        "Type": ["L", "M", "H", "L", "M", "L", "L", "H", "M", "L"],
        "Air temperature [K]": [298.1, 298.2, 298.3, 298.4, 298.5, 298.6, 298.7, 298.8, 298.9, 299.0],
        "Process temperature [K]": [308.6, 308.7, 308.8, 308.9, 309.0, 309.1, 309.2, 309.3, 309.4, 309.5],
        "Rotational speed [rpm]": [1500, 1400, 1600, 1550, 1450, 1500, 1350, 1650, 1420, 1510],
        "Torque [Nm]": [40.0, 45.0, 35.0, 42.0, 48.0, 38.0, 52.0, 33.0, 46.0, 39.0],
        "Tool wear [min]": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "Machine failure": [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        "TWF": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        "HDF": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "PWF": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "OSF": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "RNF": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    })


def test_temperature_difference_calculation(sample_raw_data: pd.DataFrame):
    """Verify that temperature_difference is calculated as Process - Air in Kelvin."""
    temp_diff = compute_temperature_difference(sample_raw_data)
    # 308.6 - 298.1 = 10.5
    assert np.isclose(temp_diff.iloc[0], 10.5)
    # Check all rows match exact difference
    expected = sample_raw_data["Process temperature [K]"] - sample_raw_data["Air temperature [K]"]
    pd.testing.assert_series_equal(temp_diff, expected)


def test_mechanical_power_calculation(sample_raw_data: pd.DataFrame):
    """
    Verify mechanical_power_w = Torque * Rotational speed * (2 * pi / 60)
    and verify unit consistency (Watts).
    """
    power = compute_mechanical_power(sample_raw_data)
    # Row 0: Torque = 40.0 Nm, Speed = 1500 rpm
    # Angular velocity = 1500 * 2 * pi / 60 = 50 * pi rad/s = 157.07963268 rad/s
    # Power = 40 * 50 * pi = 2000 * pi = 6283.185307 W
    expected_row0 = 40.0 * 1500.0 * (2.0 * math.pi / 60.0)
    assert np.isclose(power.iloc[0], expected_row0)
    assert np.isclose(power.iloc[0], 6283.1853, atol=1e-3)


def test_input_dataframe_not_mutated(sample_raw_data: pd.DataFrame):
    """Verify that calling engineer_features does not mutate the input DataFrame."""
    original_cols = sample_raw_data.columns.tolist()
    original_shape = sample_raw_data.shape
    original_copy = sample_raw_data.copy(deep=True)

    engineered = engineer_features(sample_raw_data)

    # Original DataFrame must be completely unchanged
    assert sample_raw_data.columns.tolist() == original_cols
    assert sample_raw_data.shape == original_shape
    pd.testing.assert_frame_equal(sample_raw_data, original_copy)

    # Engineered DataFrame must have the 2 new columns
    assert FEATURE_TEMP_DIFF in engineered.columns
    assert FEATURE_MECH_POWER in engineered.columns
    assert engineered.shape == (original_shape[0], original_shape[1] + 2)


def test_missing_required_columns_raise_error():
    """Verify that missing input columns produce clear, informative KeyError."""
    incomplete_df = pd.DataFrame({"Air temperature [K]": [300.0]})
    with pytest.raises(KeyError) as exc_info:
        compute_temperature_difference(incomplete_df)
    assert "Process temperature [K]" in str(exc_info.value)

    with pytest.raises(KeyError) as exc_info:
        compute_mechanical_power(incomplete_df)
    assert "Torque [Nm]" in str(exc_info.value)


def test_prohibited_columns_exclusion_contract(sample_raw_data: pd.DataFrame):
    """Verify that target and failure mode columns are strictly excluded from predictive features."""
    engineered = engineer_features(sample_raw_data)
    X_train, X_test, y_train, y_test = split_dataset(engineered)

    # Check prohibited columns are completely absent from model inputs
    for prohibited in PROHIBITED_COLUMNS:
        assert prohibited not in X_train.columns
        assert prohibited not in X_test.columns

    # Explicit check that attempting to pass prohibited columns raises ValueError
    with pytest.raises(ValueError) as exc_info:
        validate_feature_matrix_columns(["Air temperature [K]", "Machine failure"])
    assert "Prediction contract violation" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        validate_feature_matrix_columns(["Air temperature [K]", "PWF"])
    assert "Prediction contract violation" in str(exc_info.value)


def test_categorical_unknown_values_handled(sample_raw_data: pd.DataFrame):
    """Verify that unknown/unseen categories in test data do not crash the pipeline."""
    engineered = engineer_features(sample_raw_data)
    X_train, X_test, _, _ = split_dataset(engineered, test_size=0.3, random_state=42)

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    # Introduce unseen category 'UnknownType' in test input
    X_test_unseen = X_test.copy()
    X_test_unseen.iloc[0, X_test_unseen.columns.get_loc("Type")] = "UnknownType"

    # Transforming must succeed without exception (one-hot columns for unseen category set to 0)
    transformed_test = preprocessor.transform(X_test_unseen)
    assert transformed_test.shape[0] == len(X_test)
    assert not np.isnan(transformed_test).any()


def test_reproducible_split_and_stratification(sample_raw_data: pd.DataFrame):
    """Verify split determinism and preservation of class balance."""
    engineered = engineer_features(sample_raw_data)

    # Run split twice with the same seed
    X_tr1, X_te1, y_tr1, y_te1 = split_dataset(engineered, test_size=0.3, random_state=42)
    X_tr2, X_te2, y_tr2, y_te2 = split_dataset(engineered, test_size=0.3, random_state=42)

    pd.testing.assert_frame_equal(X_tr1, X_tr2)
    pd.testing.assert_series_equal(y_tr1, y_tr2)

    # Check target class proportions
    # In sample_raw_data: 2 failures out of 10 samples (20%)
    # In train (7 samples): 1 or 2 failures; in test (3 samples): 1 or 0 failures
    assert y_tr1.sum() > 0
    assert y_te1.sum() > 0


def test_preprocessing_fit_on_train_only(sample_raw_data: pd.DataFrame):
    """Verify that preprocessor is fitted exclusively on train data and isolates test data."""
    engineered = engineer_features(sample_raw_data)
    X_train, X_test, _, _ = split_dataset(engineered, test_size=0.3, random_state=42)

    preprocessor = build_preprocessor()

    # Preprocessor must initially be unfitted
    with pytest.raises(Exception):
        preprocessor.transform(X_test)

    # Fit on training data
    preprocessor.fit(X_train)

    # Verify fitted scaler statistics match training data means, not test data
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    train_numeric_means = X_train[ALL_NUMERIC_PREDICTORS].mean().values
    np.testing.assert_allclose(scaler.mean_, train_numeric_means)

    # Apply transform to test data
    X_test_trans = preprocessor.transform(X_test)
    assert X_test_trans.shape[0] == len(X_test)
    assert not np.isnan(X_test_trans).any()

