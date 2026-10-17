"""
Data schema definitions and validation structures for the AI4I 2020 Predictive Maintenance dataset.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

# Expected column definitions
IDENTIFIER_COLUMNS: List[str] = ["UDI", "Product ID"]
CATEGORICAL_COLUMNS: List[str] = ["Type"]
NUMERICAL_FEATURES: List[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET_COLUMN: str = "Machine failure"
FAILURE_TYPE_COLUMNS: List[str] = ["TWF", "HDF", "PWF", "OSF", "RNF"]

REQUIRED_COLUMNS: List[str] = (
    IDENTIFIER_COLUMNS
    + CATEGORICAL_COLUMNS
    + NUMERICAL_FEATURES
    + [TARGET_COLUMN]
    + FAILURE_TYPE_COLUMNS
)

VALID_TYPES: Set[str] = {"L", "M", "H"}
VALID_TARGET_VALUES: Set[int] = {0, 1}


@dataclass
class ValidationReport:
    """Structured report detailing validation findings for the dataset."""
    is_valid: bool
    row_count: int
    column_count: int
    column_names: List[str]
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    duplicate_rows: int
    target_distribution: Dict[int, int]
    target_percentages: Dict[int, float]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Returns a formatted human-readable summary of the validation report."""
        lines = [
            f"Validation Status: {'PASSED' if self.is_valid else 'FAILED'}",
            f"Dimensions: {self.row_count} rows x {self.column_count} columns",
            f"Duplicate Rows: {self.duplicate_rows}",
            f"Total Missing Values: {sum(self.missing_values.values())}",
            f"Target Distribution ('{TARGET_COLUMN}'): {self.target_distribution}",
            f"Target Percentages: {self.target_percentages}",
        ]
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}): {self.errors}")
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}): {self.warnings}")
        return "\n".join(lines)

