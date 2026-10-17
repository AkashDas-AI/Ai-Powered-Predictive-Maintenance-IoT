"""
Centralized prediction contract, feature definitions, and column exclusions
for the AI4I 2020 Predictive Maintenance system.
"""

from typing import List, Set
from src.data.schema import (
    IDENTIFIER_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    FAILURE_TYPE_COLUMNS,
)

# Target Variable
TARGET: str = TARGET_COLUMN

# Prohibited Columns: Columns that MUST NOT enter the predictive feature matrix.
# - Identifiers (UDI, Product ID): Non-generalizable row indices and serial tags.
# - Target (Machine failure): The prediction objective.
# - Failure Modes (TWF, HDF, PWF, OSF, RNF): Post-failure diagnostic flags that cause severe target leakage.
PROHIBITED_COLUMNS: List[str] = (
    IDENTIFIER_COLUMNS + [TARGET] + FAILURE_TYPE_COLUMNS
)

# Raw operating condition features available prior to failure
RAW_NUMERIC_FEATURES: List[str] = list(NUMERICAL_FEATURES)
CATEGORICAL_PREDICTORS: List[str] = list(CATEGORICAL_COLUMNS)

# Engineered feature names
FEATURE_TEMP_DIFF: str = "temperature_difference"
FEATURE_MECH_POWER: str = "mechanical_power_w"
ENGINEERED_NUMERIC_FEATURES: List[str] = [FEATURE_TEMP_DIFF, FEATURE_MECH_POWER]

# Complete set of predictive features entering the preprocessor
ALL_NUMERIC_PREDICTORS: List[str] = RAW_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
ALL_PREDICTORS: List[str] = CATEGORICAL_PREDICTORS + ALL_NUMERIC_PREDICTORS


def validate_feature_matrix_columns(columns: List[str]) -> None:
    """
    Enforces the prediction contract by verifying that no prohibited columns
    are present in the provided list of feature column names.

    Raises:
        ValueError: If any prohibited column (identifier, target, or failure mode) is detected.
    """
    detected_prohibited = [col for col in columns if col in PROHIBITED_COLUMNS]
    if detected_prohibited:
        raise ValueError(
            f"Prediction contract violation: Prohibited columns detected in feature matrix: "
            f"{detected_prohibited}. Target, failure mode flags, and identifier columns must be excluded."
        )

