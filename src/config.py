"""
Centralized Configuration & Constants for AI-Predictive-Maintenance-IoT.

Defines project directory paths, model artifact locations, decision thresholds,
operational sensor boundaries, and alert tier rules.
"""

from pathlib import Path
import json
from typing import Any, Dict, Optional, Tuple, Set


# -----------------------------------------------------------------------------
# 1. Project Directory Structure
# -----------------------------------------------------------------------------
def get_project_root() -> Path:
    """Returns the absolute root directory of the project."""
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT: Path = get_project_root()
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_PATH: Path = DATA_DIR / "raw" / "ai4i2020.csv"
MODELS_DIR: Path = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH: Path = MODELS_DIR / "selected_model.joblib"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
TEST_METRICS_PATH: Path = OUTPUTS_DIR / "test_metrics.json"
IMAGES_DIR: Path = PROJECT_ROOT / "images"
DOCS_DIR: Path = PROJECT_ROOT / "docs"


# -----------------------------------------------------------------------------
# 2. Decision Threshold Configuration
# -----------------------------------------------------------------------------
# Optimal decision threshold derived in Phase 4 via out-of-fold training F1 maximization
# (Held-out test set was NOT peeked or used to select this threshold).
DEFAULT_DECISION_THRESHOLD: float = 0.6732


def load_decision_threshold(metrics_path: Optional[Path] = None) -> float:
    """
    Loads the optimal decision threshold from outputs/test_metrics.json.
    Falls back to DEFAULT_DECISION_THRESHOLD (0.6732) if the file is unavailable.

    Returns:
        float: The decision threshold.
    """
    path = metrics_path or TEST_METRICS_PATH
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            threshold = data.get("threshold_tuning_oof", {}).get("selected_threshold")
            if threshold is not None:
                return float(threshold)
        except Exception:
            pass
    return DEFAULT_DECISION_THRESHOLD


# -----------------------------------------------------------------------------
# 3. Alert Tier Definitions & Boundaries
# -----------------------------------------------------------------------------
# Alert states are distinct operational categories defined independently of the
# raw binary classifier label:
#   - NORMAL: P(failure) < 0.30 -> Machine operates in nominal parameter envelopes.
#   - WATCH:  0.30 <= P(failure) < 0.50 -> Early operational anomaly or moderate tool wear;
#             inspect operating parameters during next scheduled shift.
#   - WARNING: 0.50 <= P(failure) < threshold (0.6732) -> Elevated failure probability approaching
#              the decision threshold; prioritize preventative inspection.
#   - CRITICAL: P(failure) >= threshold (0.6732) -> Failure predicted by classifier;
#               immediate machine shutdown or tooling replacement required.
ALERT_NORMAL: str = "NORMAL"
ALERT_WATCH: str = "WATCH"
ALERT_WARNING: str = "WARNING"
ALERT_CRITICAL: str = "CRITICAL"

ALERT_CUTOFF_WATCH: float = 0.30
ALERT_CUTOFF_WARNING: float = 0.50

# Rationale documentation for auditability
ALERT_TIER_DESCRIPTIONS: Dict[str, str] = {
    ALERT_NORMAL: "Nominal operational telemetry. Failure risk is negligible (<30%).",
    ALERT_WATCH: "Elevated risk envelope (30-50%). Increased logging and routine inspection advised.",
    ALERT_WARNING: "High failure probability (50% to decision threshold). Proactive maintenance advised.",
    ALERT_CRITICAL: "Actionable failure alert (>= decision threshold). Immediate intervention required.",
}


# -----------------------------------------------------------------------------
# 4. Physical Operating Bounds for Telemetry Validation
# -----------------------------------------------------------------------------
# Domain sensor boundaries derived from machine tooling specifications in AI4I benchmark:
# Any telemetry record falling outside these physical limits represents sensor malfunction,
# packet corruption, or severe instrumentation failure.
SENSOR_PHYSICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Air temperature [K]": (250.0, 350.0),       # Ambient room/hall temperature (~-23C to 77C)
    "Process temperature [K]": (250.0, 360.0),   # Machining process temperature
    "Rotational speed [rpm]": (500.0, 4000.0),   # Spindle rotation speed
    "Torque [Nm]": (0.0, 150.0),                 # Spindle shaft torque
    "Tool wear [min]": (0.0, 500.0),             # Cumulative tool cutting time
}

VALID_PRODUCT_TYPES: Set[str] = {"L", "M", "H"}

