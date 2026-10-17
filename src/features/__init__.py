"""
Feature engineering, prediction contracts, and preprocessing pipeline for AI-Predictive-Maintenance-IoT.
"""

from src.features.schema import (
    TARGET,
    PROHIBITED_COLUMNS,
    RAW_NUMERIC_FEATURES,
    CATEGORICAL_PREDICTORS,
    FEATURE_TEMP_DIFF,
    FEATURE_MECH_POWER,
    ENGINEERED_NUMERIC_FEATURES,
    ALL_NUMERIC_PREDICTORS,
    ALL_PREDICTORS,
    validate_feature_matrix_columns,
)
from src.features.engineering import (
    compute_temperature_difference,
    compute_mechanical_power,
    engineer_features,
)
from src.features.preprocessing import (
    build_preprocessor,
)
from src.features.split import (
    split_dataset,
)

__all__ = [
    "TARGET",
    "PROHIBITED_COLUMNS",
    "RAW_NUMERIC_FEATURES",
    "CATEGORICAL_PREDICTORS",
    "FEATURE_TEMP_DIFF",
    "FEATURE_MECH_POWER",
    "ENGINEERED_NUMERIC_FEATURES",
    "ALL_NUMERIC_PREDICTORS",
    "ALL_PREDICTORS",
    "validate_feature_matrix_columns",
    "compute_temperature_difference",
    "compute_mechanical_power",
    "engineer_features",
    "build_preprocessor",
    "split_dataset",
]

