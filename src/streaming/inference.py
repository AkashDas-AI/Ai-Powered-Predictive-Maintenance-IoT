"""
Streaming Inference Service for AI-Predictive-Maintenance-IoT.

Loads the Phase 4 trained model pipeline once, applies domain feature engineering,
and evaluates incoming telemetry records with high-resolution latency tracking.
"""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    DEFAULT_MODEL_PATH,
    load_decision_threshold,
)
from src.features.engineering import engineer_features
from src.features.schema import (
    PROHIBITED_COLUMNS,
    validate_feature_matrix_columns,
)
from src.streaming.alerts import (
    determine_alert_state,
    compute_demonstration_health_index,
)
from src.streaming.simulator import validate_telemetry_record


class StreamingInferenceService:
    """
    Low-latency inference service for streaming IoT telemetry records.

    Guarantees:
        - Single Model Load: Serialized pipeline is loaded once during service initialization.
        - Zero Retraining: Model weights and scalers are frozen; no training happens during serving.
        - Exact Preprocessor: Uses the saved scikit-learn Pipeline and ColumnTransformer.
        - Strict Column Exclusion: Prohibited target and failure-mode columns are strictly banned.
        - Latency Isolation: Inference timer measures only feature prep and scoring.
    """

    def __init__(
        self,
        model_path: Optional[Union[Path, str]] = None,
        threshold: Optional[float] = None,
    ) -> None:
        """
        Initializes the streaming inference service.

        Args:
            model_path: Path to serialized scikit-learn pipeline (.joblib).
                        Defaults to `models/selected_model.joblib`.
            threshold: Decision threshold for binary classification.
                       Defaults to optimal training-OOF threshold (0.6732).

        Raises:
            FileNotFoundError: If the model artifact cannot be found.
            RuntimeError: If the model cannot be deserialized or is invalid.
        """
        resolved_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {resolved_path}. "
                "Ensure Phase 4 model training was executed."
            )

        try:
            self._model: Pipeline = joblib.load(resolved_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model pipeline from {resolved_path}: {e}") from e

        if not hasattr(self._model, "predict_proba"):
            raise ValueError(f"Loaded object from {resolved_path} lacks 'predict_proba' method.")

        # Centralized threshold resolution
        if threshold is not None:
            self.threshold: float = float(threshold)
        else:
            self.threshold = load_decision_threshold()

        self._model_path: Path = resolved_path
        self._inference_count: int = 0

    @property
    def model_path(self) -> Path:
        """Path to the loaded model artifact."""
        return self._model_path

    @property
    def inference_count(self) -> int:
        """Total number of inference requests scored by this service instance."""
        return self._inference_count

    def predict_record(
        self,
        record: Dict[str, Any],
        sequence_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Processes a single telemetry record and returns structured inference results.

        Steps:
            1. Validate input record structure and domain physical ranges.
            2. Isolate predictive features, strictly omitting prohibited target & mode columns.
            3. Apply domain feature engineering (temperature differential, mechanical power).
            4. Execute pipeline prediction (ColumnTransformer -> RandomForestClassifier).
            5. Measure monotonic inference latency (t1 - t0 in ms).
            6. Evaluate alert state and demonstration Health Index.

        Args:
            record: Dictionary of sensor observations.
            sequence_id: Optional tracking identifier for stream ordering.

        Returns:
            Dict[str, Any]: Comprehensive structured inference result payload.
        """
        safe_record = deepcopy(record)
        seq_id = sequence_id if sequence_id is not None else safe_record.get("sequence_id", 0)
        timestamp = safe_record.get("simulated_timestamp", datetime.now(timezone.utc).isoformat())

        # 1. Validation
        is_valid, validation_errors = validate_telemetry_record(safe_record)
        if not is_valid:
            return {
                "sequence_id": seq_id,
                "timestamp": timestamp,
                "input_record": safe_record,
                "failure_probability": None,
                "decision_threshold": self.threshold,
                "predicted_failure": None,
                "alert_state": "INVALID",
                "demonstration_health_index": None,
                "inference_latency_ms": 0.0,
                "validation_status": {
                    "is_valid": False,
                    "errors": validation_errors,
                },
            }

        # 2. Prepare single-row DataFrame for pipeline
        # Build clean input DataFrame without prohibited columns
        filtered_input = {
            k: v for k, v in safe_record.items()
            if k not in PROHIBITED_COLUMNS
        }
        df_input = pd.DataFrame([filtered_input])

        # Verify no prohibited columns crept in
        validate_feature_matrix_columns(df_input.columns.tolist())

        # 3. High-resolution inference latency timer
        # Measures feature engineering + preprocessor transformation + model scoring
        t_start = time.perf_counter()

        df_engineered = engineer_features(df_input)
        prob_matrix = self._model.predict_proba(df_engineered)
        prob_pos = float(prob_matrix[0, 1])

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        self._inference_count += 1

        # 4. Binary classification decision and Alert / Health indicators
        predicted_class = int(prob_pos >= self.threshold)
        alert = determine_alert_state(prob_pos, threshold=self.threshold)
        health_index = compute_demonstration_health_index(prob_pos)

        return {
            "sequence_id": seq_id,
            "timestamp": timestamp,
            "input_record": safe_record,
            "failure_probability": round(prob_pos, 4),
            "decision_threshold": round(self.threshold, 4),
            "predicted_failure": predicted_class,
            "alert_state": alert,
            "demonstration_health_index": health_index,
            "inference_latency_ms": round(latency_ms, 3),
            "validation_status": {
                "is_valid": True,
                "errors": [],
            },
        }

    def predict_batch(
        self,
        records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Sequentially scores a list of telemetry records.

        Args:
            records: List of sensor dictionaries.

        Returns:
            List[Dict[str, Any]]: List of structured inference results.
        """
        return [self.predict_record(rec) for rec in records]

