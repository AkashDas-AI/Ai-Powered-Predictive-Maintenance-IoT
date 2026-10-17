"""
Bounded Sliding-Window Telemetry Buffer for AI-Predictive-Maintenance-IoT.

Maintains a fixed-capacity FIFO buffer of recent sensor readings, exposing
window snapshots and rolling diagnostic statistics.

IMPORTANT ARCHITECTURAL RULE:
    Rolling statistics (mean, min, max, std) computed by this buffer are
    DIAGNOSTIC TELEMETRY INDICATORS ONLY.
    They must NEVER be fed into the Phase 4 trained model, which was strictly
    trained on instantaneous 8-feature snapshots. Feeding window features
    into the model would violate the trained model's feature contract.
"""

from collections import deque
from copy import deepcopy
import math
from typing import Any, Dict, List, Optional
import numpy as np

from src.features.schema import RAW_NUMERIC_FEATURES


class SlidingWindowBuffer:
    """
    Fixed-capacity FIFO buffer storing recent telemetry records.

    Features:
        - Bounded capacity (deque with maxlen): automatically evicts oldest records.
        - Deep-copy guarantees: callers cannot mutate internal state.
        - Diagnostics computation: rolling mean, min, max, and sample std.
        - Predictable handling of empty windows and malformed records.
    """

    def __init__(self, window_size: int = 10) -> None:
        """
        Initializes the sliding window buffer.

        Args:
            window_size: Maximum number of records to retain (must be >= 1).
        """
        if window_size < 1:
            raise ValueError(f"window_size must be a positive integer, got: {window_size}")

        self.window_size: int = int(window_size)
        self._buffer: deque = deque(maxlen=self.window_size)

    def __len__(self) -> int:
        """Returns the current number of records stored in the buffer."""
        return len(self._buffer)

    @property
    def capacity(self) -> int:
        """Maximum capacity of the buffer."""
        return self.window_size

    @property
    def is_full(self) -> bool:
        """True if the buffer has reached maximum capacity."""
        return len(self._buffer) == self.window_size

    @property
    def is_empty(self) -> bool:
        """True if the buffer contains zero records."""
        return len(self._buffer) == 0

    def append(self, record: Dict[str, Any]) -> None:
        """
        Appends a telemetry record to the buffer.
        Stores an isolated deep copy to prevent external mutation.

        Args:
            record: Dictionary of sensor readings and metadata.
        """
        if not isinstance(record, dict):
            raise TypeError(f"Buffer record must be a dict, got: {type(record).__name__}")

        self._buffer.append(deepcopy(record))

    def latest_record(self) -> Optional[Dict[str, Any]]:
        """
        Returns a deep copy of the most recently appended telemetry record.

        Returns:
            Dict[str, Any] if buffer is not empty; otherwise None.
        """
        if not self._buffer:
            return None
        return deepcopy(self._buffer[-1])

    def get_window(self) -> List[Dict[str, Any]]:
        """
        Returns an immutable list snapshot (deep copies) of all records in the window.

        Returns:
            List[Dict[str, Any]]: Ordered from oldest [0] to newest [-1].
        """
        return [deepcopy(item) for item in self._buffer]

    def clear(self) -> None:
        """Empties all records from the buffer."""
        self._buffer.clear()

    def compute_rolling_stats(
        self,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Computes rolling summary statistics across records currently in the buffer.

        Statistics Computed:
            - 'mean': Arithmetic average
            - 'min': Minimum observed value in window
            - 'max': Maximum observed value in window
            - 'std': Sample standard deviation (ddof=1; 0.0 if count==1)
            - 'count': Number of valid numeric observations

        Note on Empty Buffers & Diagnostics:
            - If buffer is empty, returns an empty dictionary {}.
            - These statistics are for operations monitoring only and must NOT
              be passed into the ML model feature matrix.

        Args:
            columns: List of numeric column names to analyze.
                     Defaults to RAW_NUMERIC_FEATURES.

        Returns:
            Dict[str, Dict[str, float]]: Mapping of column name to statistics dict.
        """
        if not self._buffer:
            return {}

        target_columns = columns if columns is not None else list(RAW_NUMERIC_FEATURES)
        results: Dict[str, Dict[str, float]] = {}

        for col in target_columns:
            values: List[float] = []
            for record in self._buffer:
                if col in record:
                    raw_val = record[col]
                    try:
                        val_float = float(raw_val)
                        if math.isfinite(val_float):
                            values.append(val_float)
                    except (ValueError, TypeError):
                        # Skip malformed/non-numeric entries gracefully
                        continue

            if not values:
                continue

            arr = np.array(values, dtype=float)
            count = len(arr)
            mean_val = float(np.mean(arr))
            min_val = float(np.min(arr))
            max_val = float(np.max(arr))
            std_val = float(np.std(arr, ddof=1)) if count > 1 else 0.0

            results[col] = {
                "count": count,
                "mean": round(mean_val, 4),
                "min": round(min_val, 4),
                "max": round(max_val, 4),
                "std": round(std_val, 4),
            }

        return results

