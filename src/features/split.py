"""
Reproducible dataset splitting utilities for AI4I 2020 Predictive Maintenance.
Isolates test data and guarantees strictly stratified, deterministic partitions.
"""

from typing import List, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.schema import (
    TARGET,
    PROHIBITED_COLUMNS,
    validate_feature_matrix_columns,
)


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into training and test partitions using stratified sampling.

    Enforces the prediction contract:
        - Automatically excludes all PROHIBITED_COLUMNS (identifiers, failure modes, target).
        - Isolates the test partition completely so transformers/scalers cannot fit on it.
        - Preserves rare-event class balance (3.39% failure rate) via stratification.

    Important Methodology Note:
        A random stratified split on the AI4I 2020 dataset evaluates generalization
        performance across the distribution of independent simulated milling trials.
        Because AI4I 2020 consists of independent trial records rather than sequential
        time-series degradation paths for tracked physical assets, this split does NOT
        represent a temporal backtest or cross-machine fleet deployment validation.

    Args:
        df: Input DataFrame containing features and target.
        test_size: Proportion of the dataset to allocate to the test partition (default: 0.2).
        random_state: Seed for deterministic, reproducible splitting (default: 42).
        stratify: Whether to perform stratified sampling on the target (default: True).
        feature_cols: Optional explicit list of feature column names to extract.
            If None, all columns except PROHIBITED_COLUMNS will be selected.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            (X_train, X_test, y_train, y_test)

    Raises:
        KeyError: If the target column is missing from df.
        ValueError: If input dimensions are invalid, classes are insufficient, or
            prohibited columns are included in feature_cols.
    """
    if TARGET not in df.columns:
        raise KeyError(
            f"Cannot split dataset. Required target column '{TARGET}' is missing from DataFrame."
        )

    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be strictly between 0 and 1, got {test_size}")

    if len(df) < 2:
        raise ValueError(f"DataFrame must contain at least 2 records to split, got {len(df)}")

    # Extract target series
    y = df[TARGET].copy()

    # Determine predictive feature columns
    if feature_cols is None:
        extracted_cols = [c for c in df.columns if c not in PROHIBITED_COLUMNS]
    else:
        extracted_cols = list(feature_cols)

    # Validate that no prohibited columns are in the feature set
    validate_feature_matrix_columns(extracted_cols)

    if not extracted_cols:
        raise ValueError("Feature column list is empty after excluding prohibited columns.")

    X = df[extracted_cols].copy()

    # Target class validation
    unique_classes = y.nunique()
    if unique_classes < 2:
        raise ValueError(
            f"Target '{TARGET}' must have at least 2 distinct classes for classification, "
            f"found only {unique_classes} unique value(s): {y.unique().tolist()}"
        )

    stratify_target = y if stratify else None

    # Perform deterministic train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    return X_train, X_test, y_train, y_test

