"""
Dashboard Metrics & Static Evaluation Loader.

Reads persisted model evaluation artifacts and permutation importance tables
for display in the Streamlit Model & Limitations tab without re-running models.
"""

from pathlib import Path
import json
from typing import Any, Dict, List, Optional
import pandas as pd

from src.config import TEST_METRICS_PATH, OUTPUTS_DIR


def load_test_evaluation_metrics(metrics_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Loads held-out test evaluation metrics and model comparison metadata.

    Args:
        metrics_path: Path to test_metrics.json. Defaults to OUTPUTS_DIR / "test_metrics.json".

    Returns:
        Dict[str, Any]: Persisted metrics dictionary.
    """
    path = metrics_path or TEST_METRICS_PATH
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_permutation_importance_summary(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads the permutation feature importance table.

    Args:
        csv_path: Path to permutation_importance.csv.

    Returns:
        pd.DataFrame: Table with columns 'feature', 'importance_mean', 'importance_std'.
    """
    path = csv_path or (OUTPUTS_DIR / "permutation_importance.csv")
    if not path.exists():
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])


# Static local latency benchmark documented from Phase 6 empirical measurement
PHASE_6_LATENCY_BENCHMARK: Dict[str, Any] = {
    "sample_count": 25,
    "median_p50_ms": 19.33,
    "p95_ms": 53.84,
    "mean_ms": 25.30,
    "min_ms": 18.36,
    "max_ms": 81.42,
    "environment": "Python 3.11.0, Windows 11 Pro 64-bit, Local CPU",
    "disclaimer": (
        "Preliminary local benchmark isolating feature engineering and model scoring. "
        "Measured on single-core CPU execution. This is a reference measurement, not "
        "a guaranteed real-time SLA."
    ),
}

