"""
Streaming Telemetry Ingestion and Inference Runtime Package.
"""

from src.streaming.alerts import (
    determine_alert_state,
    compute_demonstration_health_index,
    get_alert_tier_metadata,
)
from src.streaming.buffer import SlidingWindowBuffer
from src.streaming.demo import run_streaming_demo
from src.streaming.inference import StreamingInferenceService
from src.streaming.simulator import (
    TelemetrySimulator,
    validate_telemetry_record,
)

__all__ = [
    "TelemetrySimulator",
    "validate_telemetry_record",
    "SlidingWindowBuffer",
    "StreamingInferenceService",
    "determine_alert_state",
    "compute_demonstration_health_index",
    "get_alert_tier_metadata",
    "run_streaming_demo",
]

