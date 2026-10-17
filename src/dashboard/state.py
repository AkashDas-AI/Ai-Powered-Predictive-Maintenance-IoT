"""
Dashboard Session State & Simulation Orchestrator.

Encapsulates the state of the Streamlit dashboard simulation, ensuring that:
1. Advancing records is strictly controlled by explicit user actions.
2. The model and preprocessing pipeline are loaded only once.
3. Ordinary Streamlit reruns (e.g., tab navigation) do not accidentally ingest duplicate records.
4. History, buffer snapshots, and alert counters stay synchronized.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_DECISION_THRESHOLD,
    ALERT_NORMAL,
    ALERT_WATCH,
    ALERT_WARNING,
    ALERT_CRITICAL,
)
from src.features.engineering import compute_temperature_difference, compute_mechanical_power
from src.streaming.buffer import SlidingWindowBuffer
from src.streaming.inference import StreamingInferenceService
from src.streaming.simulator import TelemetrySimulator


class DashboardSession:
    """
    State manager for the Streamlit predictive maintenance simulation.

    Maintains:
        - TelemetrySimulator instance
        - SlidingWindowBuffer instance
        - StreamingInferenceService instance (reused, zero retraining)
        - Chronological history of inference results
        - Alert distribution counters
    """

    def __init__(
        self,
        window_size: int = 10,
        shuffle: bool = False,
        random_state: int = 42,
        model_path: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> None:
        """
        Initializes the simulation state.

        Args:
            window_size: Maximum capacity of the bounded sliding buffer (default: 10).
            shuffle: Whether to shuffle benchmark records reproducibly (default: False).
            random_state: Seed for reproducible record shuffling (default: 42).
            model_path: Optional explicit path to model artifact.
            threshold: Optional explicit decision threshold.
        """
        self.window_size: int = max(1, int(window_size))
        self.shuffle: bool = shuffle
        self.random_state: int = int(random_state)

        # 1. Initialize Simulator
        self.simulator: TelemetrySimulator = TelemetrySimulator(
            random_state=self.random_state,
            shuffle=self.shuffle,
        )

        # 2. Initialize Bounded Sliding Buffer
        self.buffer: SlidingWindowBuffer = SlidingWindowBuffer(window_size=self.window_size)

        # 3. Initialize Streaming Inference Service (Loaded Once)
        self.service: StreamingInferenceService = StreamingInferenceService(
            model_path=model_path,
            threshold=threshold,
        )

        # 4. History and Alert Tracking
        self.history: List[Dict[str, Any]] = []
        self.alert_counts: Dict[str, int] = {
            ALERT_NORMAL: 0,
            ALERT_WATCH: 0,
            ALERT_WARNING: 0,
            ALERT_CRITICAL: 0,
        }
        self.processed_count: int = 0

    @property
    def latest_result(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent inference result dict, or None if no records processed."""
        if not self.history:
            return None
        return self.history[-1]

    @property
    def latest_record(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent raw sensor record from the buffer."""
        return self.buffer.latest_record()

    @property
    def decision_threshold(self) -> float:
        """Active decision threshold loaded from configuration."""
        return self.service.threshold

    def step(self) -> Optional[Dict[str, Any]]:
        """
        Advances the simulation by exactly one record.

        Steps:
            1. Pulls next record from TelemetrySimulator.
            2. Ingests record into SlidingWindowBuffer.
            3. Scores record via StreamingInferenceService.
            4. Updates chronological history and alert-state counters.

        Returns:
            Dict[str, Any] result payload, or None if simulator stream is exhausted.
        """
        if not self.simulator.has_next:
            return None

        record = self.simulator.step()
        if record is None:
            return None

        # 1. Bounded buffer ingestion
        self.buffer.append(record)

        # 2. Inference scoring
        result = self.service.predict_record(record, sequence_id=record.get("sequence_id"))

        # 3. Append to history and update counters
        self.history.append(result)
        self.processed_count += 1

        alert = result.get("alert_state", "INVALID")
        if alert in self.alert_counts:
            self.alert_counts[alert] += 1

        return result

    def run_batch(self, batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        Advances the simulation by multiple records sequentially.

        Args:
            batch_size: Number of records to step (default: 10).

        Returns:
            List[Dict[str, Any]]: List of inference result payloads produced.
        """
        results: List[Dict[str, Any]] = []
        for _ in range(batch_size):
            if not self.simulator.has_next:
                break
            res = self.step()
            if res is not None:
                results.append(res)
        return results

    def reset(
        self,
        window_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Completely resets the simulation state back to step 0.

        Args:
            window_size: Optional new window size.
            shuffle: Optional new shuffle setting.
            random_state: Optional new random seed.
        """
        if window_size is not None:
            self.window_size = max(1, int(window_size))
        if shuffle is not None:
            self.shuffle = shuffle
        if random_state is not None:
            self.random_state = int(random_state)

        # Re-initialize components
        self.simulator = TelemetrySimulator(
            random_state=self.random_state,
            shuffle=self.shuffle,
        )
        self.buffer = SlidingWindowBuffer(window_size=self.window_size)
        self.history.clear()
        self.alert_counts = {
            ALERT_NORMAL: 0,
            ALERT_WATCH: 0,
            ALERT_WARNING: 0,
            ALERT_CRITICAL: 0,
        }
        self.processed_count = 0

    def update_window_size(self, new_size: int) -> None:
        """
        Updates buffer capacity, preserving existing history records in the new window.

        Args:
            new_size: New maximum capacity of sliding buffer.
        """
        target_size = max(1, int(new_size))
        if target_size == self.window_size:
            return

        self.window_size = target_size
        new_buf = SlidingWindowBuffer(window_size=self.window_size)

        # Re-populate new buffer with up to target_size most recent raw records
        recent_records = [item["input_record"] for item in self.history[-self.window_size:]]
        for rec in recent_records:
            new_buf.append(rec)

        self.buffer = new_buf

    def get_recent_history(self, limit: int = 25) -> List[Dict[str, Any]]:
        """
        Returns a slice of the most recent inference results.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List[Dict[str, Any]]: Ordered from newest [0] to oldest [-1].
        """
        if not self.history:
            return []
        slice_items = self.history[-limit:]
        return list(reversed(slice_items))

    def to_history_dataframe(self) -> pd.DataFrame:
        """
        Flattens history into a clean pandas DataFrame for charts and table display.

        Returns:
            pd.DataFrame: Structured table of processed records and outputs.
        """
        if not self.history:
            return pd.DataFrame(columns=[
                "sequence_id",
                "simulated_timestamp",
                "Type",
                "Air temperature [K]",
                "Process temperature [K]",
                "temperature_difference",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]",
                "mechanical_power_w",
                "failure_probability",
                "decision_threshold",
                "predicted_failure",
                "alert_state",
                "demonstration_health_index",
                "inference_latency_ms",
            ])

        rows = []
        for h in self.history:
            rec = h.get("input_record", {})
            air_t = rec.get("Air temperature [K]")
            proc_t = rec.get("Process temperature [K]")
            rpm = rec.get("Rotational speed [rpm]")
            torque = rec.get("Torque [Nm]")

            # Physical derived indicators for visualization
            temp_diff = (proc_t - air_t) if (proc_t is not None and air_t is not None) else None
            mech_pow = (
                torque * rpm * (2.0 * np.pi / 60.0)
                if (torque is not None and rpm is not None)
                else None
            )

            rows.append({
                "sequence_id": h.get("sequence_id"),
                "simulated_timestamp": h.get("timestamp"),
                "Type": rec.get("Type"),
                "Air temperature [K]": air_t,
                "Process temperature [K]": proc_t,
                "temperature_difference": round(temp_diff, 2) if temp_diff is not None else None,
                "Rotational speed [rpm]": rpm,
                "Torque [Nm]": torque,
                "Tool wear [min]": rec.get("Tool wear [min]"),
                "mechanical_power_w": round(mech_pow, 2) if mech_pow is not None else None,
                "failure_probability": h.get("failure_probability"),
                "decision_threshold": h.get("decision_threshold"),
                "predicted_failure": h.get("predicted_failure"),
                "alert_state": h.get("alert_state"),
                "demonstration_health_index": h.get("demonstration_health_index"),
                "inference_latency_ms": h.get("inference_latency_ms"),
            })

        return pd.DataFrame(rows)

