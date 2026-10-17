"""
Reusable scikit-learn preprocessing pipeline for AI4I 2020 Predictive Maintenance.
Handles numeric scaling, categorical encoding, and missing-value imputation.
"""

from typing import List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.schema import (
    ALL_NUMERIC_PREDICTORS,
    CATEGORICAL_PREDICTORS,
    validate_feature_matrix_columns,
)


def build_preprocessor(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Constructs an unfitted scikit-learn ColumnTransformer for predictive features.

    Transformations:
        - Numeric Features:
            1. Median imputation (SimpleImputer(strategy='median'))
            2. Standardization (StandardScaler() to zero mean and unit variance)
        - Categorical Features:
            1. Most-frequent imputation (SimpleImputer(strategy='most_frequent'))
            2. One-hot encoding (OneHotEncoder(handle_unknown='ignore', sparse_output=False))
               Safely handles unseen/unknown categories in test or serving environments.

    Safety & Leakage Prevention:
        - The returned ColumnTransformer is completely UNFITTER.
        - remainder='drop' ensures any unlisted or prohibited columns that enter the input
          are discarded and cannot contaminate transformed representations.
        - Checks column lists against prohibited columns before building the transformer.

    Args:
        numeric_features: List of numeric column names. Defaults to ALL_NUMERIC_PREDICTORS.
        categorical_features: List of categorical column names. Defaults to CATEGORICAL_PREDICTORS.

    Returns:
        ColumnTransformer: An unfitted scikit-learn ColumnTransformer.
    """
    num_cols = list(numeric_features) if numeric_features is not None else list(ALL_NUMERIC_PREDICTORS)
    cat_cols = list(categorical_features) if categorical_features is not None else list(CATEGORICAL_PREDICTORS)

    # Validate that no prohibited columns are configured into transformers
    validate_feature_matrix_columns(num_cols + cat_cols)

    # 1. Numeric pipeline: Median imputation + Standard scaling
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # 2. Categorical pipeline: Most-frequent imputation + OneHotEncoding
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # 3. Combined ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor

