"""
Reusable data loader for the AI4I 2020 Predictive Maintenance dataset.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import pandas as pd

from src.data.schema import ValidationReport
from src.data.validator import validate_ai4i_dataframe


def get_project_root() -> Path:
    """
    Returns the absolute path to the project root directory.

    Determined relative to the location of this source file:
    `src/data/loader.py` -> `src/data` -> `src` -> project root.
    """
    return Path(__file__).resolve().parents[2]


def load_ai4i_dataset(
    filepath: Optional[Union[str, Path]] = None,
    validate: bool = True,
    raise_on_error: bool = True,
) -> Tuple[pd.DataFrame, Optional[ValidationReport]]:
    """
    Loads and optionally validates the AI4I 2020 Predictive Maintenance dataset.

    File paths are resolved relative to the project root when relative paths or
    None are provided.

    Args:
        filepath: Custom path to the dataset CSV file. Defaults to `data/raw/ai4i2020.csv`.
        validate: Whether to run schema and data validation on the loaded DataFrame.
        raise_on_error: Whether to raise ValueError on fatal validation errors.

    Returns:
        Tuple[pd.DataFrame, Optional[ValidationReport]]: The loaded DataFrame and
        its corresponding validation report (if validate=True).

    Raises:
        FileNotFoundError: If the specified or default file does not exist.
        ValueError: If validation fails and raise_on_error is True.
    """
    if filepath is None:
        target_path = get_project_root() / "data" / "raw" / "ai4i2020.csv"
    else:
        target_path = Path(filepath)
        if not target_path.is_absolute():
            target_path = get_project_root() / target_path

    if not target_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {target_path}\n"
            f"Please ensure the official AI4I 2020 dataset is downloaded into 'data/raw/ai4i2020.csv'.\n"
            f"Source URL: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset"
        )

    # Load dataset using pandas
    df = pd.read_csv(target_path)

    report: Optional[ValidationReport] = None
    if validate:
        report = validate_ai4i_dataframe(df, raise_on_error=raise_on_error)

    return df, report

