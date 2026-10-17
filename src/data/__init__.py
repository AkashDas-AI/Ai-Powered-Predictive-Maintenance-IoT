"""
Data ingestion, validation, and loading package for AI-Predictive-Maintenance-IoT.
"""

from src.data.loader import get_project_root, load_ai4i_dataset
from src.data.schema import (
    CATEGORICAL_COLUMNS,
    FAILURE_TYPE_COLUMNS,
    IDENTIFIER_COLUMNS,
    NUMERICAL_FEATURES,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    VALID_TARGET_VALUES,
    VALID_TYPES,
    ValidationReport,
)
from src.data.validator import validate_ai4i_dataframe

__all__ = [
    "get_project_root",
    "load_ai4i_dataset",
    "validate_ai4i_dataframe",
    "ValidationReport",
    "REQUIRED_COLUMNS",
    "IDENTIFIER_COLUMNS",
    "CATEGORICAL_COLUMNS",
    "NUMERICAL_FEATURES",
    "TARGET_COLUMN",
    "FAILURE_TYPE_COLUMNS",
    "VALID_TYPES",
    "VALID_TARGET_VALUES",
]

