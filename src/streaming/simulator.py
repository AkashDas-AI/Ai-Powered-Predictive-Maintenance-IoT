"""
Virtual IoT Telemetry Simulator for AI-Predictive-Maintenance-IoT.

Emits simulated sensor telemetry records incrementally from the UCI AI4I 2020 dataset.
Designed for reproducible local demonstration and unit testing without network services
or external message brokers.

IMPORTANT ARCHITECTURAL DISCLAIMER:
    This module is a software simulation. The AI4I 2020 dataset contains independent
    synthetic experimental observations. Stepping or streaming these records sequentially
    is an ingestion demonstration only; it does NOT represent the continuous physical
    wear evolution or time-series dynamics of an actual factory milling machine.
"""

from copy import deepcopy
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    SENSOR_PHYSICAL_BOUNDS,
    VALID_PRODUCT_TYPES,
)
from src.features.schema import (
    RAW_NUMERIC_FEATURES,
    CATEGORICAL_PREDICTORS,
)

REQUIRED_TELEMETRY_FIELDS: List[str] = CATEGORICAL_PREDICTORS + RAW_NUMERIC_FEATURES


def validate_telemetry_record(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates a raw telemetry reading against structural and physical sanity constraints.

    Validation Rules:
        1. All required sensor predictors must be present ('Type', and 5 numeric sensor fields).
        2. Values must be non-null and numeric (or valid string for Type).
        3. Numeric fields must be finite (no NaN, inf, or -inf).
        4. Categorical 'Type' must be one of {'L', 'M', 'H'}.
        5. Numeric readings must fall within plausible domain boundaries (SENSOR_PHYSICAL_BOUNDS).

    Args:
        record: Dictionary representing a single sensor observation.

    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_validation_error_messages)
    """
    errors: List[str] = []

    # 1. Missing keys check
    for field in REQUIRED_TELEMETRY_FIELDS:
        if field not in record:
            errors.append(f"Missing required sensor field: '{field}'")

    if errors:
        return False, errors

    # 2. Type column validation
    val_type = record.get("Type")
    if val_type not in VALID_PRODUCT_TYPES:
        errors.append(
            f"Invalid product Type '{val_type}'. Must be one of {sorted(VALID_PRODUCT_TYPES)}"
        )

    # 3. Numeric validity and physical range checks
    for field in RAW_NUMERIC_FEATURES:
        val = record.get(field)
        if val is None:
            errors.append(f"Field '{field}' has null value.")
            continue

        try:
            val_float = float(val)
        except (ValueError, TypeError):
            errors.append(f"Field '{field}' has non-numeric value: {val!r}")
            continue

        if not math.isfinite(val_float):
            errors.append(f"Field '{field}' has non-finite value: {val_float}")
            continue

        # Physical range check
        if field in SENSOR_PHYSICAL_BOUNDS:
            min_bound, max_bound = SENSOR_PHYSICAL_BOUNDS[field]
            if val_float < min_bound or val_float > max_bound:
                errors.append(
                    f"Field '{field}' value {val_float} is outside plausible physical bounds "
                    f"[{min_bound}, {max_bound}]."
                )

    is_valid = len(errors) == 0
    return is_valid, errors


class TelemetrySimulator:
    """
    Deterministic telemetry generator replaying benchmark records record-by-record.

    Features:
        - Manual step-by-step stepping via `step()` for deterministic unit testing.
        - Configurable generator streaming via `stream(max_records)`.
        - Reproducible shuffling or exact sequential playback using a random seed.
        - Embedded validation and sequence tracking.
    """

    def __init__(
        self,
        data_source: Optional[Union[Path, str, pd.DataFrame, List[Dict[str, Any]]]] = None,
        random_state: Optional[int] = 42,
        shuffle: bool = False,
        interval_seconds: float = 0.0,
    ) -> None:
        """
        Initializes the telemetry simulator.

        Args:
            data_source: File path, pandas DataFrame, or list of dicts.
                         Defaults to `data/raw/ai4i2020.csv`.
            random_state: Seed for reproducible record shuffling.
            shuffle: Whether to shuffle records or replay in original dataset order.
            interval_seconds: Optional emission delay between records in stream mode.
        """
        self.random_state = random_state
        self.shuffle = shuffle
        self.interval_seconds = max(0.0, float(interval_seconds))
        self._current_index: int = 0
        self._sequence_counter: int = 0

        # Load records into memory
        self._records: List[Dict[str, Any]] = self._load_data(data_source)

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            perm = rng.permutation(len(self._records))
            self._records = [self._records[i] for i in perm]

    def _load_data(
        self,
        data_source: Optional[Union[Path, str, pd.DataFrame, List[Dict[str, Any]]]],
    ) -> List[Dict[str, Any]]:
        """Parses the data source into a list of native Python dictionaries."""
        if data_source is None:
            data_source = RAW_DATA_PATH

        if isinstance(data_source, list):
            return [deepcopy(r) for r in data_source]

        if isinstance(data_source, pd.DataFrame):
            return data_source.to_dict(orient="records")

        path = Path(data_source)
        if not path.exists():
            raise FileNotFoundError(
                f"Telemetry data source file not found at: {path}. "
                "Ensure dataset is ingested into data/raw/ai4i2020.csv."
            )

        df = pd.read_csv(path)
        return df.to_dict(orient="records")

    def __len__(self) -> int:
        """Returns total number of records available in the simulator."""
        return len(self._records)

    @property
    def current_index(self) -> int:
        """Index of the next record to be emitted."""
        return self._current_index

    @property
    def has_next(self) -> bool:
        """True if there are remaining records in the stream."""
        return self._current_index < len(self._records)

    def step(self) -> Optional[Dict[str, Any]]:
        """
        Emits the next telemetry record in the sequence.

        Returns:
            Dict[str, Any] with sensor fields, 'sequence_id', and 'simulated_timestamp',
            or None if the dataset has been fully consumed.
        """
        if not self.has_next:
            return None

        raw_record = deepcopy(self._records[self._current_index])
        self._current_index += 1
        self._sequence_counter += 1

        # Attach telemetry stream metadata
        record = {
            "sequence_id": self._sequence_counter,
            "simulated_timestamp": datetime.now(timezone.utc).isoformat(),
            **raw_record,
        }
        return record

    def stream(
        self,
        max_records: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Yields records incrementally up to `max_records` (or until end of data).

        Args:
            max_records: Optional upper limit on the number of records to emit.

        Yields:
            Dict[str, Any]: Simulated telemetry records.
        """
        emitted = 0
        while self.has_next:
            if max_records is not None and emitted >= max_records:
                break

            record = self.step()
            if record is None:
                break

            emitted += 1
            yield record

    def reset(self) -> None:
        """Resets the simulator sequence pointer to the beginning."""
        self._current_index = 0
        self._sequence_counter = 0

