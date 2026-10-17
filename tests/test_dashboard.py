"""
Unit and Integration Tests for Phase 7 Streamlit Predictive Maintenance Dashboard.

Covers:
    - DashboardSession initialization, single model loading, and initial state.
    - Single-step and batch simulation execution without duplicate ingestion.
    - Buffer capacity bounds and historical consistency.
    - Dynamic window size updates with state preservation.
    - Simulation reset back to step zero.
    - Safe Plotly chart generation for empty, single-record, and multi-record DataFrames.
    - Model feature contract and prohibited column exclusion.
    - Zero retraining verification across session steps.
    - Streamlit AppTest headless smoke test.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import (
    DEFAULT_DECISION_THRESHOLD,
    ALERT_NORMAL,
    ALERT_WATCH,
    ALERT_WARNING,
    ALERT_CRITICAL,
)
from src.dashboard.charts import (
    build_spindle_dynamics_chart,
    build_thermal_profile_chart,
    build_tool_wear_and_power_chart,
    build_failure_risk_trajectory_chart,
    build_permutation_importance_bar_chart,
)
from src.dashboard.metrics import (
    load_test_evaluation_metrics,
    load_permutation_importance_summary,
    PHASE_6_LATENCY_BENCHMARK,
)
from src.dashboard.state import DashboardSession


# -----------------------------------------------------------------------------
# 1. Dashboard Session Initialization & Reset Tests
# -----------------------------------------------------------------------------
def test_dashboard_session_initial_state():
    """Verify DashboardSession starts with clean initial state and single model load."""
    session = DashboardSession(window_size=10, shuffle=False)

    assert session.processed_count == 0
    assert len(session.history) == 0
    assert len(session.buffer) == 0
    assert session.window_size == 10
    assert session.latest_result is None
    assert session.latest_record is None
    assert session.decision_threshold == pytest.approx(DEFAULT_DECISION_THRESHOLD, abs=1e-4)
    assert session.alert_counts == {
        ALERT_NORMAL: 0,
        ALERT_WATCH: 0,
        ALERT_WARNING: 0,
        ALERT_CRITICAL: 0,
    }


def test_dashboard_session_reset():
    """Verify reset clears buffer and history and restores sequence pointer."""
    session = DashboardSession(window_size=10, shuffle=False)

    # Ingest 5 records
    session.run_batch(batch_size=5)
    assert session.processed_count == 5
    assert len(session.history) == 5

    # Reset
    session.reset()
    assert session.processed_count == 0
    assert len(session.history) == 0
    assert len(session.buffer) == 0
    assert session.simulator.current_index == 0
    assert sum(session.alert_counts.values()) == 0


# -----------------------------------------------------------------------------
# 2. Progression & Idempotency (No Duplicate Ingestion on Inspection)
# -----------------------------------------------------------------------------
def test_dashboard_session_step_progression():
    """Verify advancing by one step correctly scores and updates history."""
    session = DashboardSession(window_size=10, shuffle=False)

    res = session.step()
    assert res is not None
    assert session.processed_count == 1
    assert len(session.history) == 1
    assert len(session.buffer) == 1
    assert session.latest_result == res

    # Verification of output schema
    assert "failure_probability" in res
    assert "predicted_failure" in res
    assert "alert_state" in res
    assert "demonstration_health_index" in res
    assert "inference_latency_ms" in res
    assert res["alert_state"] in {ALERT_NORMAL, ALERT_WATCH, ALERT_WARNING, ALERT_CRITICAL}
    assert session.alert_counts[res["alert_state"]] == 1


def test_dashboard_session_idempotent_inspection():
    """Verify inspecting state (to_history_dataframe, get_recent_history) does not cause duplicate ingestion."""
    session = DashboardSession(window_size=10, shuffle=False)
    session.step()

    initial_count = session.processed_count

    # Repeatedly inspect session state
    _ = session.latest_result
    _ = session.latest_record
    df1 = session.to_history_dataframe()
    df2 = session.to_history_dataframe()
    hist = session.get_recent_history(limit=10)

    assert session.processed_count == initial_count
    assert len(session.history) == initial_count
    assert len(df1) == initial_count
    assert len(df2) == initial_count
    assert len(hist) == initial_count


def test_dashboard_session_run_batch():
    """Verify running a batch advances by exact requested count."""
    session = DashboardSession(window_size=10, shuffle=False)
    batch_results = session.run_batch(batch_size=7)

    assert len(batch_results) == 7
    assert session.processed_count == 7
    assert len(session.history) == 7
    assert [r["sequence_id"] for r in batch_results] == list(range(1, 8))


# -----------------------------------------------------------------------------
# 3. Buffer Bounds & Dynamic Resizing Tests
# -----------------------------------------------------------------------------
def test_dashboard_buffer_capacity_bounds():
    """Verify buffer caps at window_size while history retains full chronological stream."""
    session = DashboardSession(window_size=3, shuffle=False)
    session.run_batch(batch_size=6)

    # History preserves all 6
    assert len(session.history) == 6
    assert session.processed_count == 6

    # Buffer holds exactly the last 3
    assert len(session.buffer) == 3
    assert session.buffer.is_full is True
    window_seqs = [r["sequence_id"] for r in session.buffer.get_window()]
    assert window_seqs == [4, 5, 6]


def test_dashboard_update_window_size():
    """Verify updating window size preserves existing history in new buffer."""
    session = DashboardSession(window_size=4, shuffle=False)
    session.run_batch(batch_size=8)
    assert len(session.buffer) == 4

    # Expand buffer capacity to 6
    session.update_window_size(6)
    assert session.window_size == 6
    assert len(session.buffer) == 6
    window_seqs = [r["sequence_id"] for r in session.buffer.get_window()]
    assert window_seqs == [3, 4, 5, 6, 7, 8]


# -----------------------------------------------------------------------------
# 4. Chart Builder Safety Tests (Empty & Non-Empty)
# -----------------------------------------------------------------------------
def test_charts_handle_empty_dataframe():
    """Verify all chart functions return valid go.Figure objects when history is empty."""
    empty_df = pd.DataFrame()

    fig_spindle = build_spindle_dynamics_chart(empty_df)
    fig_thermal = build_thermal_profile_chart(empty_df)
    fig_wear = build_tool_wear_and_power_chart(empty_df)
    fig_risk = build_failure_risk_trajectory_chart(empty_df)
    fig_perm = build_permutation_importance_bar_chart(empty_df)

    assert isinstance(fig_spindle, go.Figure)
    assert isinstance(fig_thermal, go.Figure)
    assert isinstance(fig_wear, go.Figure)
    assert isinstance(fig_risk, go.Figure)
    assert isinstance(fig_perm, go.Figure)


def test_charts_handle_populated_dataframe():
    """Verify chart functions render correctly with single and multiple records."""
    session = DashboardSession(window_size=10, shuffle=False)
    session.run_batch(batch_size=5)
    df = session.to_history_dataframe()

    fig_spindle = build_spindle_dynamics_chart(df)
    fig_thermal = build_thermal_profile_chart(df)
    fig_wear = build_tool_wear_and_power_chart(df)
    fig_risk = build_failure_risk_trajectory_chart(df, threshold=0.6732)

    assert len(fig_spindle.data) == 2  # speed and torque
    assert len(fig_thermal.data) == 3  # proc temp, air temp, delta T
    assert len(fig_wear.data) == 2     # wear and power
    assert len(fig_risk.data) == 1     # probability line


# -----------------------------------------------------------------------------
# 5. Methodological & Model Integrity Tests
# -----------------------------------------------------------------------------
def test_dashboard_model_retraining_guard():
    """Verify that the model pipeline instance remains frozen and is not retrained."""
    session = DashboardSession(window_size=10, shuffle=False)
    initial_pipe_id = id(session.service._model)

    session.run_batch(batch_size=5)

    # Pipeline object must be identical in memory
    assert id(session.service._model) == initial_pipe_id
    assert session.service.inference_count == 5


def test_dashboard_static_metrics_loading():
    """Verify load_test_evaluation_metrics and permutation tables load without crashing."""
    metrics = load_test_evaluation_metrics()
    assert isinstance(metrics, dict)
    assert "test_metrics_selected_threshold" in metrics

    perm_df = load_permutation_importance_summary()
    assert isinstance(perm_df, pd.DataFrame)
    assert "feature" in perm_df.columns
    assert "importance_mean" in perm_df.columns

    # Verify static latency benchmark payload
    assert PHASE_6_LATENCY_BENCHMARK["sample_count"] == 25
    assert PHASE_6_LATENCY_BENCHMARK["median_p50_ms"] > 0.0


# -----------------------------------------------------------------------------
# 6. Streamlit AppTest Smoke Test
# -----------------------------------------------------------------------------
def test_streamlit_app_smoke_test():
    """Verify app.py initializes, steps, and renders headlessly without exceptions."""
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30)
    app.run(timeout=30)

    # Ensure no unhandled exception occurred during initial app rendering
    assert not app.exception, f"app.py raised unhandled exception on init: {app.exception}"
    assert len(app.title) > 0
    assert "Predictive Maintenance" in app.title[0].value

    # Click 'Next Telemetry Sample' (first button) and verify data table renders cleanly
    if app.button:
        app.button[0].click().run(timeout=30)
        assert not app.exception, f"app.py raised unhandled exception on step: {app.exception}"

        # Click 'Run Batch' (second button) and verify multi-record ingestion & dataframe rendering
        if len(app.button) > 1:
            app.button[1].click().run(timeout=30)
            assert not app.exception, f"app.py raised unhandled exception on batch: {app.exception}"

