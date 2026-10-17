"""
Validation engine for the AI4I 2020 Predictive Maintenance dataset.
"""

from typing import Dict, List
import pandas as pd
import numpy as np

from src.data.schema import (
    REQUIRED_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    FAILURE_TYPE_COLUMNS,
    VALID_TYPES,
    VALID_TARGET_VALUES,
    ValidationReport,
)


def validate_ai4i_dataframe(
    df: pd.DataFrame, raise_on_error: bool = True
) -> ValidationReport:
    """
    Validates an in-memory pandas DataFrame against the AI4I 2020 schema.

    Performs strict contract checking without altering, dropping, or modifying
    any rows, outliers, or labels.

    Args:
        df: The pandas DataFrame to validate.
        raise_on_error: If True, raises a ValueError on fatal schema violations.

    Returns:
        ValidationReport: Detailed validation summary and findings.

    Raises:
        ValueError: If required columns are missing or target values are invalid (when raise_on_error=True).
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Check for missing required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        msg = f"Schema violation: Missing required columns: {missing_cols}"
        errors.append(msg)
        if raise_on_error:
            raise ValueError(msg)

    # 2. Check row and column counts
    row_count, col_count = df.shape
    if row_count == 0:
        msg = "Dataset validation error: DataFrame contains 0 rows."
        errors.append(msg)
        if raise_on_error:
            raise ValueError(msg)

    # 3. Missing values analysis
    missing_values: Dict[str, int] = df.isnull().sum().to_dict()
    total_missing = sum(missing_values.values())
    if total_missing > 0:
        warnings.append(
            f"Dataset contains {total_missing} missing values across {sum(1 for v in missing_values.values() if v > 0)} columns."
        )

    # 4. Duplicate rows detection
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows > 0:
        warnings.append(f"Detected {duplicate_rows} duplicate rows in dataset.")

    # 5. Target validation
    target_distribution: Dict[int, int] = {}
    target_percentages: Dict[int, float] = {}
    if TARGET_COLUMN in df.columns:
        # Check target data type and uniqueness
        unique_targets = set(df[TARGET_COLUMN].dropna().unique())
        invalid_targets = unique_targets - VALID_TARGET_VALUES
        if invalid_targets:
            msg = (
                f"Target validation error: Column '{TARGET_COLUMN}' contains invalid values "
                f"{invalid_targets}. Expected binary set {VALID_TARGET_VALUES}."
            )
            errors.append(msg)
            if raise_on_error:
                raise ValueError(msg)

        counts = df[TARGET_COLUMN].value_counts().to_dict()
        target_distribution = {int(k): int(v) for k, v in counts.items()}
        target_percentages = {
            int(k): round(float(v) / row_count * 100.0, 3)
            for k, v in counts.items()
        }

    # 6. Failure types validation
    for ft_col in FAILURE_TYPE_COLUMNS:
        if ft_col in df.columns:
            invalid_ft = set(df[ft_col].dropna().unique()) - {0, 1}
            if invalid_ft:
                msg = f"Failure mode column '{ft_col}' contains invalid values {invalid_ft}."
                errors.append(msg)
                if raise_on_error:
                    raise ValueError(msg)

    # 7. Numerical features type validation
    for num_col in NUMERICAL_FEATURES:
        if num_col in df.columns and not pd.api.types.is_numeric_dtype(df[num_col]):
            msg = f"Data type mismatch: Column '{num_col}' expected numeric, got {df[num_col].dtype}."
            errors.append(msg)
            if raise_on_error:
                raise ValueError(msg)

    # 8. Type column categories validation
    if "Type" in df.columns:
        observed_types = set(df["Type"].dropna().unique())
        invalid_types = observed_types - VALID_TYPES
        if invalid_types:
            warnings.append(
                f"Unexpected categorical values in 'Type': {invalid_types}. Expected subset of {VALID_TYPES}."
            )

    is_valid = len(errors) == 0

    return ValidationReport(
        is_valid=is_valid,
        row_count=row_count,
        column_count=col_count,
        column_names=df.columns.tolist(),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        missing_values=missing_values,
        duplicate_rows=duplicate_rows,
        target_distribution=target_distribution,
        target_percentages=target_percentages,
        errors=errors,
        warnings=warnings,
    )

