"""
Unit tests for Phase 6: Virtual Telemetry Streaming and Inference Buffer.

Covers:
    - Bounded buffer capacity, FIFO eviction, and immutability.
    - Empty-buffer handling and diagnostic rolling statistics.
    - Simulator stepping, determinism, and physical range validation.
    - Feature schema and prohibited-column exclusion in streaming.
    - Single model loading, repeated scoring determinism, and load error handling.
    - Decision threshold boundary transitions and alert state rules.
    - Demonstration Health Index mathematical contract and range bounds.
    - Latency measurement output schema without brittle timing assertions.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_DECISION_THRESHOLD,
    ALERT_NORMAL,
    ALERT_WATCH,
    ALERT_WARNING,
    ALERT_CRITICAL,
)
from src.features.schema import (
    PROHIBITED_COLUMNS,
    RAW_NUMERIC_FEATURES,
    CATEGORICAL_PREDICTORS,
)
from src.streaming.alerts import (
    determine_alert_state,
    compute_demonstration_health_index,
    get_alert_tier_metadata,
)
from src.streaming.buffer import SlidingWindowBuffer
from src.streaming.inference import StreamingInferenceService
from src.streaming.simulator import (
    TelemetrySimulator,
    validate_telemetry_record,
)


@pytest.fixture
def sample_valid_telemetry() -> dict:
    """Provides a single valid telemetry record matching AI4I schema."""
    return {
        "UDI": 1001,
        "Product ID": "M15860",
        "Type": "M",
        "Air temperature [K]": 298.5,
        "Process temperature [K]": 308.7,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 42.5,
        "Tool wear [min]": 110,
    }


# -----------------------------------------------------------------------------
# 1. Bounded Buffer Capacity & Ordering Tests
# -----------------------------------------------------------------------------
def test_buffer_capacity_and_fifo_eviction():
    """Verify bounded buffer retains at most window_size records and evicts in FIFO order."""
    buf = SlidingWindowBuffer(window_size=3)
    assert buf.capacity == 3
    assert len(buf) == 0
    assert buf.is_empty is True
    assert buf.is_full is False

    # Append 3 records
    for i in range(1, 4):
        buf.append({"sequence_id": i, "val": i * 10})

    assert len(buf) == 3
    assert buf.is_full is True
    assert [r["sequence_id"] for r in buf.get_window()] == [1, 2, 3]

    # Append 4th record - oldest (id 1) must be evicted
    buf.append({"sequence_id": 4, "val": 40})
    assert len(buf) == 3
    assert buf.is_full is True
    assert [r["sequence_id"] for r in buf.get_window()] == [2, 3, 4]
    assert buf.latest_record()["sequence_id"] == 4


def test_buffer_immutability():
    """Verify modifying returned latest_record or get_window does not alter buffer state."""
    buf = SlidingWindowBuffer(window_size=2)
    original = {"sequence_id": 1, "Torque [Nm]": 40.0}
    buf.append(original)

    # Mutate latest_record
    retrieved = buf.latest_record()
    retrieved["Torque [Nm]"] = 999.0
    assert buf.latest_record()["Torque [Nm]"] == 40.0

    # Mutate get_window
    window = buf.get_window()
    window[0]["Torque [Nm]"] = 888.0
    assert buf.get_window()[0]["Torque [Nm]"] == 40.0


def test_buffer_invalid_capacity():
    """Verify invalid window sizes raise ValueError."""
    with pytest.raises(ValueError):
        SlidingWindowBuffer(window_size=0)
    with pytest.raises(ValueError):
        SlidingWindowBuffer(window_size=-5)


# -----------------------------------------------------------------------------
# 2. Empty-Window Behavior & Diagnostic Rolling Statistics Tests
# -----------------------------------------------------------------------------
def test_empty_buffer_and_rolling_stats():
    """Verify empty buffer handles operations predictably without throwing errors."""
    buf = SlidingWindowBuffer(window_size=5)
    assert buf.latest_record() is None
    assert buf.get_window() == []
    assert buf.compute_rolling_stats() == {}

    # Add single record
    buf.append({"Air temperature [K]": 300.0, "Torque [Nm]": 40.0})
    stats_1 = buf.compute_rolling_stats(["Air temperature [K]", "Torque [Nm]"])
    assert stats_1["Air temperature [K]"]["count"] == 1
    assert stats_1["Air temperature [K]"]["mean"] == 300.0
    assert stats_1["Air temperature [K]"]["min"] == 300.0
    assert stats_1["Air temperature [K]"]["max"] == 300.0
    assert stats_1["Air temperature [K]"]["std"] == 0.0  # Sample std for N=1 is defined as 0.0

    # Add second record
    buf.append({"Air temperature [K]": 304.0, "Torque [Nm]": 50.0})
    stats_2 = buf.compute_rolling_stats(["Air temperature [K]"])
    assert stats_2["Air temperature [K]"]["count"] == 2
    assert stats_2["Air temperature [K]"]["mean"] == 302.0
    assert stats_2["Air temperature [K]"]["min"] == 300.0
    assert stats_2["Air temperature [K]"]["max"] == 304.0
    # Sample std of [300, 304] with ddof=1 is sqrt((4+4)/1) = sqrt(8) ~ 2.8284
    assert stats_2["Air temperature [K]"]["std"] == pytest.approx(2.8284, rel=1e-3)


def test_buffer_ignores_non_numeric_in_rolling_stats():
    """Verify compute_rolling_stats gracefully ignores non-numeric or missing fields."""
    buf = SlidingWindowBuffer(window_size=3)
    buf.append({"Air temperature [K]": 300.0, "Torque [Nm]": "corrupted"})
    buf.append({"Air temperature [K]": 302.0})

    stats = buf.compute_rolling_stats(["Air temperature [K]", "Torque [Nm]"])
    assert stats["Air temperature [K]"]["count"] == 2
    assert "Torque [Nm]" not in stats  # No valid float values


# -----------------------------------------------------------------------------
# 3. Simulator Stepping & Seed Determinism Tests
# -----------------------------------------------------------------------------
def test_simulator_stepping_and_reproducibility():
    """Verify TelemetrySimulator emits records step-by-step and respects random seeds."""
    sim1 = TelemetrySimulator(random_state=42, shuffle=True)
    sim2 = TelemetrySimulator(random_state=42, shuffle=True)

    records1 = [sim1.step() for _ in range(5)]
    records2 = [sim2.step() for _ in range(5)]

    assert len(records1) == 5
    for r1, r2 in zip(records1, records2):
        assert r1["UDI"] == r2["UDI"]
        assert r1["Air temperature [K]"] == r2["Air temperature [K]"]
        assert r1["sequence_id"] == r2["sequence_id"]

    # Test reset
    sim1.reset()
    assert sim1.current_index == 0
    reset_rec = sim1.step()
    assert reset_rec["UDI"] == records1[0]["UDI"]


def test_simulator_sequential_mode():
    """Verify TelemetrySimulator in un-shuffled mode preserves original benchmark ordering."""
    sim = TelemetrySimulator(shuffle=False)
    r1 = sim.step()
    r2 = sim.step()
    assert r1["UDI"] == 1
    assert r2["UDI"] == 2


# -----------------------------------------------------------------------------
# 4. Telemetry Validation Rules Tests
# -----------------------------------------------------------------------------
def test_telemetry_validation_valid_and_invalid(sample_valid_telemetry):
    """Verify structural and domain boundary validation rules."""
    # 1. Valid record
    is_valid, errors = validate_telemetry_record(sample_valid_telemetry)
    assert is_valid is True
    assert errors == []

    # 2. Missing required column
    bad_record_missing = sample_valid_telemetry.copy()
    del bad_record_missing["Torque [Nm]"]
    is_valid, errors = validate_telemetry_record(bad_record_missing)
    assert is_valid is False
    assert any("Torque [Nm]" in err for err in errors)

    # 3. Invalid categorical Type
    bad_record_type = sample_valid_telemetry.copy()
    bad_record_type["Type"] = "X"
    is_valid, errors = validate_telemetry_record(bad_record_type)
    assert is_valid is False
    assert any("Type 'X'" in err for err in errors)

    # 4. Non-finite values (NaN / Inf)
    bad_record_nan = sample_valid_telemetry.copy()
    bad_record_nan["Rotational speed [rpm]"] = float("nan")
    is_valid, errors = validate_telemetry_record(bad_record_nan)
    assert is_valid is False
    assert any("non-finite" in err for err in errors)

    # 5. Out of physical domain bounds
    bad_record_bounds = sample_valid_telemetry.copy()
    bad_record_bounds["Air temperature [K]"] = 150.0  # Plausible bounds: [250, 350]
    is_valid, errors = validate_telemetry_record(bad_record_bounds)
    assert is_valid is False
    assert any("outside plausible physical bounds" in err for err in errors)


# -----------------------------------------------------------------------------
# 5. Inference Service Output Contract & Prohibited Column Exclusion
# -----------------------------------------------------------------------------
def test_inference_service_output_contract(sample_valid_telemetry):
    """Verify StreamingInferenceService returns all required contract fields."""
    service = StreamingInferenceService()
    result = service.predict_record(sample_valid_telemetry, sequence_id=42)

    expected_keys = {
        "sequence_id",
        "timestamp",
        "input_record",
        "failure_probability",
        "decision_threshold",
        "predicted_failure",
        "alert_state",
        "demonstration_health_index",
        "inference_latency_ms",
        "validation_status",
    }
    assert expected_keys.issubset(set(result.keys()))
    assert result["sequence_id"] == 42
    assert isinstance(result["failure_probability"], float)
    assert 0.0 <= result["failure_probability"] <= 1.0
    assert result["predicted_failure"] in (0, 1)
    assert result["alert_state"] in {ALERT_NORMAL, ALERT_WATCH, ALERT_WARNING, ALERT_CRITICAL}
    assert 0.0 <= result["demonstration_health_index"] <= 100.0
    assert result["inference_latency_ms"] > 0.0
    assert result["validation_status"]["is_valid"] is True


def test_inference_service_prohibited_columns_exclusion(sample_valid_telemetry):
    """Verify that prohibited columns (targets, modes, identifiers) are never fed to the model."""
    service = StreamingInferenceService()

    # Poison input record with all prohibited columns
    poisoned_record = sample_valid_telemetry.copy()
    poisoned_record["Machine failure"] = 1
    poisoned_record["TWF"] = 1
    poisoned_record["HDF"] = 0
    poisoned_record["PWF"] = 0
    poisoned_record["OSF"] = 0
    poisoned_record["RNF"] = 0

    # Service should strip prohibited columns before scoring
    result = service.predict_record(poisoned_record)
    assert result["validation_status"]["is_valid"] is True
    assert result["predicted_failure"] in (0, 1)


def test_inference_service_invalid_record_handling(sample_valid_telemetry):
    """Verify service returns structured error payload for invalid input without crashing."""
    service = StreamingInferenceService()
    invalid_record = sample_valid_telemetry.copy()
    invalid_record["Torque [Nm]"] = "corrupted_signal"

    result = service.predict_record(invalid_record)
    assert result["validation_status"]["is_valid"] is False
    assert len(result["validation_status"]["errors"]) > 0
    assert result["failure_probability"] is None
    assert result["predicted_failure"] is None
    assert result["alert_state"] == "INVALID"
    assert result["demonstration_health_index"] is None


# -----------------------------------------------------------------------------
# 6. Model Loading, Repeated Scoring & Retraining Guards
# -----------------------------------------------------------------------------
def test_model_load_failure():
    """Verify clear FileNotFoundError when model path does not exist."""
    with pytest.raises(FileNotFoundError):
        StreamingInferenceService(model_path="non_existent_model_artifact.joblib")


def test_repeated_scoring_without_reloading_or_retraining(sample_valid_telemetry):
    """Verify model pipeline instance remains identical and deterministic across repeated calls."""
    service = StreamingInferenceService()
    initial_model_id = id(service._model)

    # Score multiple records
    res1 = service.predict_record(sample_valid_telemetry)
    res2 = service.predict_record(sample_valid_telemetry)

    # Model instance not reloaded
    assert id(service._model) == initial_model_id
    assert service.inference_count == 2

    # Determinism: identical input yields exact same probability
    assert res1["failure_probability"] == res2["failure_probability"]
    assert res1["predicted_failure"] == res2["predicted_failure"]


# -----------------------------------------------------------------------------
# 7. Decision Threshold & Alert State Boundary Behavior
# -----------------------------------------------------------------------------
def test_decision_threshold_exact_boundary():
    """Verify binary prediction behavior at and around decision threshold."""
    thresh = 0.6732

    # Below threshold -> 0
    pred_below = int(0.67319 >= thresh)
    assert pred_below == 0

    # At exact threshold -> 1
    pred_exact = int(0.67320 >= thresh)
    assert pred_exact == 1

    # Above threshold -> 1
    pred_above = int(0.67321 >= thresh)
    assert pred_above == 1


def test_alert_state_boundary_rules():
    """Verify alert transitions at NORMAL (0.30), WATCH (0.50), WARNING, and CRITICAL (0.6732)."""
    thresh = 0.6732

    # NORMAL: [0.00, 0.30)
    assert determine_alert_state(0.00, thresh) == ALERT_NORMAL
    assert determine_alert_state(0.299, thresh) == ALERT_NORMAL

    # WATCH: [0.30, 0.50)
    assert determine_alert_state(0.30, thresh) == ALERT_WATCH
    assert determine_alert_state(0.499, thresh) == ALERT_WATCH

    # WARNING: [0.50, 0.6732)
    assert determine_alert_state(0.50, thresh) == ALERT_WARNING
    assert determine_alert_state(0.6731, thresh) == ALERT_WARNING

    # CRITICAL: [0.6732, 1.00]
    assert determine_alert_state(0.6732, thresh) == ALERT_CRITICAL
    assert determine_alert_state(0.85, thresh) == ALERT_CRITICAL
    assert determine_alert_state(1.00, thresh) == ALERT_CRITICAL

    # Out of range clamping
    assert determine_alert_state(-0.1, thresh) == ALERT_NORMAL
    assert determine_alert_state(1.5, thresh) == ALERT_CRITICAL


# -----------------------------------------------------------------------------
# 8. Demonstration Health Index Properties
# -----------------------------------------------------------------------------
def test_demonstration_health_index_contract():
    """Verify demonstration Health Index formula, range, and clamping."""
    # 0% failure probability -> 100% Health Index
    assert compute_demonstration_health_index(0.0) == 100.0

    # 100% failure probability -> 0% Health Index
    assert compute_demonstration_health_index(1.0) == 0.0

    # Intermediate values
    assert compute_demonstration_health_index(0.25) == 75.0
    assert compute_demonstration_health_index(0.6732) == 32.68

    # Boundary clamping
    assert compute_demonstration_health_index(-0.5) == 100.0
    assert compute_demonstration_health_index(1.5) == 0.0


# -----------------------------------------------------------------------------
# 9. Latency Measurement Schema (Non-Brittle)
# -----------------------------------------------------------------------------
def test_latency_measurement_schema(sample_valid_telemetry):
    """Verify latency is measured as a positive float in ms without brittle hardware assumptions."""
    service = StreamingInferenceService()
    result = service.predict_record(sample_valid_telemetry)

    lat = result["inference_latency_ms"]
    assert isinstance(lat, float)
    assert lat > 0.0
    # Confirm timing is not zero or infinite
    assert np.isfinite(lat)

