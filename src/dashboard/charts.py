"""
Plotly Chart Builders for Streamlit Dashboard.

Creates clean, responsive sequence charts for:
1. Spindle Dynamics (Rotational speed vs. Torque)
2. Thermal Profile (Air temp, Process temp, Delta T)
3. Cumulative Tool Wear & Shaft Mechanical Power
4. Model Failure Probability & Alert Boundary Trajectory
5. Permutation Feature Importance Summary
"""

from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import (
    DEFAULT_DECISION_THRESHOLD,
    ALERT_CUTOFF_WATCH,
    ALERT_CUTOFF_WARNING,
    ALERT_NORMAL,
    ALERT_WATCH,
    ALERT_WARNING,
    ALERT_CRITICAL,
)

ALERT_COLOR_MAP = {
    ALERT_NORMAL: "#2ca02c",    # Green
    ALERT_WATCH: "#1f77b4",     # Blue
    ALERT_WARNING: "#ff7f0e",   # Amber / Orange
    ALERT_CRITICAL: "#d62728",  # Red
    "INVALID": "#7f7f7f",       # Grey
}


def _create_empty_figure(message: str = "No telemetry ingested yet. Advance simulation to view trends.") -> go.Figure:
    """Creates a placeholder Plotly figure with an informative message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="#6c757d"),
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=320,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def build_spindle_dynamics_chart(df: pd.DataFrame) -> go.Figure:
    """
    Renders dual-axis interactive chart for Spindle Rotational Speed and Torque.

    Args:
        df: History DataFrame.
    """
    if df.empty or "sequence_id" not in df.columns:
        return _create_empty_figure("Awaiting telemetry records for Spindle Dynamics...")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Rotational Speed [rpm]
    fig.add_trace(
        go.Scatter(
            x=df["sequence_id"],
            y=df["Rotational speed [rpm]"],
            name="Rotational Speed [rpm]",
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6),
        ),
        secondary_y=False,
    )

    # Torque [Nm]
    fig.add_trace(
        go.Scatter(
            x=df["sequence_id"],
            y=df["Torque [Nm]"],
            name="Torque [Nm]",
            mode="lines+markers",
            line=dict(color="#d62728", width=2, dash="dash"),
            marker=dict(size=6),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=dict(text="Spindle Dynamics (Speed vs. Torque)", font=dict(size=15, weight="bold")),
        xaxis_title="Telemetry Sequence ID",
        height=350,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=30),
    )
    fig.update_yaxes(title_text="Speed [rpm]", secondary_y=False)
    fig.update_yaxes(title_text="Torque [Nm]", secondary_y=True)

    return fig


def build_thermal_profile_chart(df: pd.DataFrame) -> go.Figure:
    """
    Renders thermal conditions (Process Temp, Air Temp, and Delta T).

    Args:
        df: History DataFrame.
    """
    if df.empty or "sequence_id" not in df.columns:
        return _create_empty_figure("Awaiting telemetry records for Thermal Profile...")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["sequence_id"],
            y=df["Process temperature [K]"],
            name="Process Temp [K]",
            mode="lines+markers",
            line=dict(color="#e377c2", width=2),
            marker=dict(size=5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["sequence_id"],
            y=df["Air temperature [K]"],
            name="Air Temp [K]",
            mode="lines+markers",
            line=dict(color="#17becf", width=2),
            marker=dict(size=5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["sequence_id"],
            y=df["temperature_difference"],
            name="ΔT (Process - Air) [K]",
            mode="lines+markers",
            line=dict(color="#ff7f0e", width=2.5, dash="dot"),
            marker=dict(size=6),
        )
    )

    # Reference threshold for HDF risk (<8.6 K)
    fig.add_hline(
        y=8.6,
        line_dash="dash",
        line_color="#e41a1c",
        annotation_text="HDF Danger Boundary (ΔT < 8.6 K)",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=dict(text="Thermal Profile & Gradient (ΔT)", font=dict(size=15, weight="bold")),
        xaxis_title="Telemetry Sequence ID",
        yaxis_title="Temperature [Kelvin, K]",
        height=350,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


def build_tool_wear_and_power_chart(df: pd.DataFrame) -> go.Figure:
    """
    Renders cumulative tool wear [min] and calculated shaft mechanical power [W].

    Args:
        df: History DataFrame.
    """
    if df.empty or "sequence_id" not in df.columns:
        return _create_empty_figure("Awaiting telemetry records for Tool Wear & Power...")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df["sequence_id"],
            y=df["Tool wear [min]"],
            name="Cumulative Tool Wear [min]",
            mode="lines+markers",
            line=dict(color="#8c564b", width=2),
            marker=dict(size=6),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["sequence_id"],
            y=df["mechanical_power_w"],
            name="Shaft Mechanical Power [W]",
            mode="lines+markers",
            line=dict(color="#2ca02c", width=2, dash="dash"),
            marker=dict(size=6),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=dict(text="Tool Wear & Shaft Mechanical Power", font=dict(size=15, weight="bold")),
        xaxis_title="Telemetry Sequence ID",
        height=350,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=30),
    )
    fig.update_yaxes(title_text="Tool Wear [min]", secondary_y=False)
    fig.update_yaxes(title_text="Mechanical Power [W]", secondary_y=True)

    return fig


def build_failure_risk_trajectory_chart(
    df: pd.DataFrame,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> go.Figure:
    """
    Renders model failure probability trajectory against alert boundaries.

    Args:
        df: History DataFrame.
        threshold: Decision threshold separating WARNING from CRITICAL.
    """
    if df.empty or "sequence_id" not in df.columns:
        return _create_empty_figure("Awaiting telemetry records for Risk Trajectory...")

    fig = go.Figure()

    # Determine point colors based on alert state
    point_colors = [ALERT_COLOR_MAP.get(a, "#7f7f7f") for a in df["alert_state"]]

    fig.add_trace(
        go.Scatter(
            x=df["sequence_id"],
            y=df["failure_probability"],
            name="P(Failure)",
            mode="lines+markers",
            line=dict(color="#333333", width=2),
            marker=dict(size=8, color=point_colors, line=dict(width=1, color="black")),
            hovertemplate=(
                "Seq: %{x}<br>"
                "P(Fail): %{y:.4f}<br>"
                "<extra></extra>"
            ),
        )
    )

    # Decision Threshold
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#d62728",
        line_width=2,
        annotation_text=f"Decision Threshold τ* = {threshold:.4f}",
        annotation_position="top right",
    )

    # Warning Cutoff
    fig.add_hline(
        y=ALERT_CUTOFF_WARNING,
        line_dash="dot",
        line_color="#ff7f0e",
        annotation_text="WARNING (P >= 0.50)",
        annotation_position="bottom right",
    )

    # Watch Cutoff
    fig.add_hline(
        y=ALERT_CUTOFF_WATCH,
        line_dash="dot",
        line_color="#1f77b4",
        annotation_text="WATCH (P >= 0.30)",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=dict(text="Model Risk Trajectory & Alert Thresholds", font=dict(size=15, weight="bold")),
        xaxis_title="Telemetry Sequence ID",
        yaxis_title="Failure Probability",
        yaxis=dict(range=[-0.05, 1.05]),
        height=380,
        margin=dict(l=20, r=20, t=50, b=30),
    )

    return fig


def build_permutation_importance_bar_chart(perm_df: pd.DataFrame) -> go.Figure:
    """
    Renders horizontal bar chart of held-out permutation feature importance.

    Args:
        perm_df: DataFrame with 'feature', 'importance_mean', 'importance_std'.
    """
    if perm_df.empty:
        return _create_empty_figure("Permutation importance data unavailable.")

    # Sort ascending for horizontal bar chart (top at top)
    sorted_df = perm_df.sort_values(by="importance_mean", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=sorted_df["importance_mean"],
            y=sorted_df["feature"],
            orientation="h",
            error_x=dict(type="data", array=sorted_df["importance_std"], visible=True),
            marker=dict(color="#2b5c8f"),
            hovertemplate="Feature: %{y}<br>Mean AP Drop: %{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="Permutation Feature Importance (Average Precision Drop)",
            font=dict(size=14, weight="bold"),
        ),
        xaxis_title="Mean Drop in Average Precision (10 Repeats on Held-Out Test)",
        yaxis_title="",
        height=380,
        margin=dict(l=10, r=20, t=40, b=30),
    )
    return fig

