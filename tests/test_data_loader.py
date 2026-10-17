"""
Unit tests for data loading and validation modules in src/data.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data.loader import get_project_root, load_ai4i_dataset
from src.data.validator import validate_ai4i_dataframe
from src.data.schema import (
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    ValidationReport,
)


def test_project_root_resolution():
    """Verify get_project_root returns an existing directory containing expected files."""
    root = get_project_root()
    assert root.exists()
    assert (root / "requirements.txt").exists()
    assert (root / "src").exists()


def test_valid_dataset_loads_successfully():
    """Verify that the official AI4I dataset loads and passes all validation checks."""
    df, report = load_ai4i_dataset()

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (10000, 14)
    assert report is not None
    assert report.is_valid is True
    assert report.row_count == 10000
    assert report.column_count == 14
    assert report.duplicate_rows == 0
    assert sum(report.missing_values.values()) == 0
    assert report.target_distribution == {0: 9661, 1: 339}
    assert len(report.errors) == 0


def test_missing_file_raises_filenotfound():
    """Verify that attempting to load a non-existent file raises a clear FileNotFoundError."""
    non_existent_path = "data/raw/does_not_exist_12345.csv"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_ai4i_dataset(filepath=non_existent_path)

    assert "Dataset not found at" in str(exc_info.value)
    assert "https://archive.ics.uci.edu" in str(exc_info.value)


def test_missing_required_columns_detected(tmp_path: Path):
    """Verify that missing required columns are flagged with a ValueError."""
    # Create a DataFrame missing several required columns
    incomplete_df = pd.DataFrame({
        "UDI": [1, 2],
        "Product ID": ["L001", "L002"],
        "Type": ["L", "L"],
        "Machine failure": [0, 1],
    })

    # Test validator with raise_on_error=True
    with pytest.raises(ValueError) as exc_info:
        validate_ai4i_dataframe(incomplete_df, raise_on_error=True)
    assert "Missing required columns" in str(exc_info.value)

    # Test validator with raise_on_error=False
    report = validate_ai4i_dataframe(incomplete_df, raise_on_error=False)
    assert report.is_valid is False
    assert any("Missing required columns" in err for err in report.errors)


def test_missing_values_and_duplicates_reported(tmp_path: Path):
    """Verify that missing values and duplicate rows are accurately reported."""
    # Base dummy DataFrame satisfying required schema
    data = {col: [1, 1, 2] for col in REQUIRED_COLUMNS}
    data["Product ID"] = ["L1", "L1", "L2"]
    data["Type"] = ["L", "L", "L"]
    data["Machine failure"] = [0, 0, 1]
    data["Air temperature [K]"] = [298.1, np.nan, 298.5]  # 1 missing value

    df = pd.DataFrame(data)

    # Rows 0 and 1 are identical except for the NaN in Air temperature
    # Let's create an exact duplicate
    df.iloc[1] = df.iloc[0]  # Now row 1 is an exact copy of row 0

    report = validate_ai4i_dataframe(df, raise_on_error=False)
    assert report.row_count == 3
    assert report.duplicate_rows == 1
    assert report.missing_values["Air temperature [K]"] == 0  # overwritten by iloc
    assert any("duplicate rows" in w for w in report.warnings)

    # Add missing value to test missing reporting
    df.loc[2, "Torque [Nm]"] = np.nan
    report2 = validate_ai4i_dataframe(df, raise_on_error=False)
    assert report2.missing_values["Torque [Nm]"] == 1
    assert any("missing values" in w for w in report2.warnings)


def test_invalid_target_values_detected():
    """Verify that non-binary target values trigger validation failure."""
    # Create sample with target values outside {0, 1}
    data = {col: [1, 2] for col in REQUIRED_COLUMNS}
    data["Product ID"] = ["L1", "L2"]
    data["Type"] = ["L", "M"]
    data["Machine failure"] = [0, 99]  # 99 is invalid

    df = pd.DataFrame(data)

    with pytest.raises(ValueError) as exc_info:
        validate_ai4i_dataframe(df, raise_on_error=True)
    assert "contains invalid values" in str(exc_info.value)
    assert "99" in str(exc_info.value)

    report = validate_ai4i_dataframe(df, raise_on_error=False)
    assert report.is_valid is False
    assert any("contains invalid values" in err for err in report.errors)


def test_empty_dataframe_detected():
    """Verify that an empty DataFrame triggers a validation error."""
    empty_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    with pytest.raises(ValueError) as exc_info:
        validate_ai4i_dataframe(empty_df, raise_on_error=True)
    assert "0 rows" in str(exc_info.value)

