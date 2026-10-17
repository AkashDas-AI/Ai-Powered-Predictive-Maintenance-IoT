"""
Alert-State Rules and Demonstration Health Index Calculation.

Defines the 4-tier operational alert classification:
    NORMAL, WATCH, WARNING, CRITICAL

Also provides a demonstration Health Index (0-100%).
IMPORTANT DISCLAIMER:
    The Health Index is a synthetic demonstration indicator derived solely from
    instantaneous classifier failure probability. It is NOT a validated estimate
    of physical remaining useful life (RUL), machine degradation curve, or certified
    industrial safety envelope.
"""

from typing import Dict
from src.config import (
    ALERT_NORMAL,
    ALERT_WATCH,
    ALERT_WARNING,
    ALERT_CRITICAL,
    ALERT_CUTOFF_WATCH,
    ALERT_CUTOFF_WARNING,
    DEFAULT_DECISION_THRESHOLD,
    ALERT_TIER_DESCRIPTIONS,
)


def determine_alert_state(
    failure_probability: float,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> str:
    """
    Categorizes the predicted failure probability into a structured operational alert state.

    Alert Tiers and Operational Meaning:
        - NORMAL (P < 0.30):
            Nominal operating regime. Telemetry aligns with standard baseline runs.
        - WATCH (0.30 <= P < 0.50):
            Slightly elevated failure probability. Suggests sensor drift, higher thermal gradient,
            or moderate tool wear. Operators should monitor trend on subsequent windows.
        - WARNING (0.50 <= P < threshold):
            High failure risk approaching the optimal decision boundary. Preventative inspection
            or maintenance scheduling is recommended.
        - CRITICAL (P >= threshold):
            Classified as failure (predicted_failure == 1). Immediate diagnostic intervention,
            workpiece inspection, or tool replacement required.

    Design Rationale:
        Separating alert levels from the binary classifier prediction gives operations
        early warning stages (WATCH, WARNING) before the critical threshold is crossed,
        reducing catastrophic line stoppages while keeping the primary decision boundary
        rigorously aligned with the Phase 4 optimal threshold (0.6732).

    Disclaimer:
        These alert thresholds are demonstration boundaries designed for educational and
        monitoring workflow illustration; they do NOT constitute certified industrial
        safety cutoffs or ISO machine vibration standards.

    Args:
        failure_probability: Estimated failure probability in [0.0, 1.0].
        threshold: Decision threshold separating WARNING from CRITICAL (default: 0.6732).

    Returns:
        str: One of 'NORMAL', 'WATCH', 'WARNING', or 'CRITICAL'.
    """
    prob = float(failure_probability)
    # Ensure boundary clamping
    prob = max(0.0, min(1.0, prob))

    if prob < ALERT_CUTOFF_WATCH:
        return ALERT_NORMAL
    elif prob < ALERT_CUTOFF_WARNING:
        return ALERT_WATCH
    elif prob < threshold:
        return ALERT_WARNING
    else:
        return ALERT_CRITICAL


def compute_demonstration_health_index(failure_probability: float) -> float:
    """
    Calculates a demonstration Asset Health Index (HI) from failure probability.

    Formula:
        Health Index = 100.0 * (1.0 - failure_probability)

    Properties:
        - Range: [0.0, 100.0]
        - Direction: Higher is better (100.0 indicates 0% failure probability;
          0.0 indicates 100% failure probability).
        - Unit: Percent (%)

    Critical Engineering Limitations:
        1. Demonstration Only: This score is a synthetic inversion of instantaneous
           statistical model confidence, NOT a physics-based fatigue or Remaining
           Useful Life (RUL) estimation.
        2. Independence Assumption: AI4I benchmark records are independent operating runs;
           sequential replay does not represent the real continuous degradation dynamics
           of a physical cutting tool.
        3. Non-Linearity: A small jump in spindle torque or temperature difference
           can cause abrupt shifts in the Health Index rather than a smooth wear trajectory.

    Args:
        failure_probability: Estimated failure probability in [0.0, 1.0].

    Returns:
        float: Calculated health index in [0.0, 100.0], rounded to 2 decimal places.
    """
    prob = float(failure_probability)
    prob_clamped = max(0.0, min(1.0, prob))
    health_index = 100.0 * (1.0 - prob_clamped)
    return round(health_index, 2)


def get_alert_tier_metadata() -> Dict[str, Dict[str, str]]:
    """Returns documentation and rationale for each alert tier."""
    return {
        ALERT_NORMAL: {
            "range": f"0.00 <= P < {ALERT_CUTOFF_WATCH:.2f}",
            "action": "Continue regular production.",
            "description": ALERT_TIER_DESCRIPTIONS[ALERT_NORMAL],
        },
        ALERT_WATCH: {
            "range": f"{ALERT_CUTOFF_WATCH:.2f} <= P < {ALERT_CUTOFF_WARNING:.2f}",
            "action": "Increase monitoring cadence; inspect at next scheduled shift.",
            "description": ALERT_TIER_DESCRIPTIONS[ALERT_WATCH],
        },
        ALERT_WARNING: {
            "range": f"{ALERT_CUTOFF_WARNING:.2f} <= P < {DEFAULT_DECISION_THRESHOLD:.4f}",
            "action": "Plan preventative tool check or maintenance window.",
            "description": ALERT_TIER_DESCRIPTIONS[ALERT_WARNING],
        },
        ALERT_CRITICAL: {
            "range": f"P >= {DEFAULT_DECISION_THRESHOLD:.4f}",
            "action": "Trigger maintenance alarm; halt operation if indicated.",
            "description": ALERT_TIER_DESCRIPTIONS[ALERT_CRITICAL],
        },
    }

