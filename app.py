"""
AI-Powered Predictive Maintenance for IoT Devices
Interactive Streamlit Predictive Maintenance Dashboard (Phase 7).

Launch Command:
    .\\.venv\\Scripts\\python.exe -m streamlit run app.py
"""

from copy import deepcopy
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_DECISION_THRESHOLD,
    ALERT_NORMAL,
    ALERT_WATCH,
    ALERT_WARNING,
    ALERT_CRITICAL,
    ALERT_TIER_DESCRIPTIONS,
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
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Powered Predictive Maintenance | IoT Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Session State Initialization (Idempotent & Leakage Safe)
# -----------------------------------------------------------------------------
def get_session() -> DashboardSession:
    """Retrieves or initializes the singleton DashboardSession in Streamlit session_state."""
    if "dashboard_session" not in st.session_state:
        st.session_state.dashboard_session = DashboardSession(
            window_size=10,
            shuffle=False,
            random_state=42,
        )
    return st.session_state.dashboard_session


session: DashboardSession = get_session()


# -----------------------------------------------------------------------------
# Sidebar: Simulation Controls & Parameter Configuration
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Simulation Controls")

st.sidebar.markdown(
    "**Virtual Telemetry Source**: Replays records from the benchmark "
    "dataset record-by-record to simulate an edge ingestion stream."
)

# 1. Mode Configuration
replay_mode = st.sidebar.radio(
    "Playback Ordering",
    options=["Sequential AI4I Replay", "Reproducibly Shuffled"],
    index=1 if session.shuffle else 0,
    help="Sequential replays original dataset order. Shuffled uses a fixed random seed.",
)
is_shuffled = replay_mode == "Reproducibly Shuffled"

seed = session.random_state
if is_shuffled:
    seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=999999,
        value=session.random_state,
        step=1,
    )

# 2. Window Buffer Size
window_size = st.sidebar.slider(
    "Sliding Window Buffer Size (W)",
    min_value=5,
    max_value=50,
    value=session.window_size,
    step=5,
    help="Maximum number of recent telemetry records retained in the FIFO buffer.",
)

if window_size != session.window_size:
    session.update_window_size(window_size)

# Check if mode or seed changed requiring reset
if (is_shuffled != session.shuffle) or (is_shuffled and seed != session.random_state):
    session.reset(window_size=window_size, shuffle=is_shuffled, random_state=seed)
    st.sidebar.info("Simulation reset to apply new playback ordering.")

st.sidebar.divider()

# 3. Action Buttons
st.sidebar.subheader("Stream Progression")
col_s1, col_s2 = st.sidebar.columns(2)

with col_s1:
    if st.button("▶️ Next Sample", use_container_width=True, help="Ingest and score next record"):
        session.step()

with col_s2:
    batch_size = st.sidebar.selectbox("Batch Count", options=[5, 10, 25, 50], index=1)
    if st.button(f"⏩ Run Batch ({batch_size})", use_container_width=True):
        session.run_batch(batch_size=batch_size)

if st.sidebar.button("🔄 Reset Simulation", use_container_width=True, help="Clear buffer and reset sequence pointer"):
    session.reset(window_size=window_size, shuffle=is_shuffled, random_state=seed)
    st.sidebar.success("Simulation reset.")

st.sidebar.divider()

# 4. Runtime & Architecture Info Card
st.sidebar.subheader("Inference Runtime Info")
st.sidebar.markdown(f"""
- **Model**: `Random Forest (Balanced)`
- **Decision Threshold ($\\tau^*$)**: `{session.decision_threshold:.4f}`
- **Active Buffer Records**: `{len(session.buffer)} / {session.window_size}`
- **Total Ingested Samples**: `{session.processed_count}`
- **Feature Contract**: Instantaneous 8 Predictors (Prohibited columns excluded)
""")


# -----------------------------------------------------------------------------
# Main Header & Disclaimer Callout
# -----------------------------------------------------------------------------
st.title("⚙️ AI-Powered Predictive Maintenance for IoT Devices")
st.caption(
    "Virtual Telemetry Streaming, Bounded Sliding-Window Ingestion & Low-Latency Inference Runtime"
)

# Portfolio simulation disclaimer banner
st.info(
    "🛡️ **Demonstration & Simulation Notice**: This system is an educational software simulation "
    "built upon the UCI AI4I 2020 Predictive Maintenance benchmark dataset. It does **NOT** connect "
    "to physical factory hardware, does **NOT** present validated industrial safety cutoffs, and does "
    "**NOT** estimate remaining useful life (RUL). Operational alert states and health scores are illustrative.",
    icon="ℹ️",
)


# -----------------------------------------------------------------------------
# Dashboard Navigation Tabs
# -----------------------------------------------------------------------------
tab_overview, tab_sim, tab_trends, tab_model = st.tabs([
    "📊 Overview",
    "📡 Live Simulation",
    "📈 Sensor Trends & Buffer Diagnostics",
    "🔬 Model & Benchmark Limitations",
])


# =============================================================================
# TAB 1: OVERVIEW
# =============================================================================
with tab_overview:
    st.subheader("Simulated Machine Health & Operational Status")

    latest = session.latest_result
    raw_rec = session.latest_record

    if latest is None or raw_rec is None:
        st.warning(
            "No telemetry records have been ingested yet. Advance the simulation using "
            "**'▶️ Next Sample'** or **'⏩ Run Batch'** in the sidebar to begin monitoring.",
            icon="⚠️",
        )
    else:
        prob = latest.get("failure_probability", 0.0)
        alert = latest.get("alert_state", "NORMAL")
        hi = latest.get("demonstration_health_index", 100.0)
        pred = latest.get("predicted_failure", 0)
        thresh = session.decision_threshold

        # Define alert badge color & icon
        alert_badges = {
            ALERT_NORMAL: ("#2ca02c", "🟢 NORMAL", "Nominal operating envelope. Failure risk is negligible."),
            ALERT_WATCH: ("#1f77b4", "🔵 WATCH", "Elevated operational variation. Review parameters during routine checks."),
            ALERT_WARNING: ("#ff7f0e", "🟠 WARNING", "High risk approaching threshold. Preventative inspection advised."),
            ALERT_CRITICAL: ("#d62728", "🔴 CRITICAL", "Model failure threshold crossed (Predicted Failure = 1). Diagnostic review advised."),
        }
        badge_color, badge_text, badge_desc = alert_badges.get(alert, ("#7f7f7f", alert, "Unknown"))

        # Row 1: Key Metric Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            st.markdown(f"""
            <div style="background-color: {badge_color}18; border-left: 5px solid {badge_color}; padding: 12px 16px; border-radius: 6px;">
                <span style="font-size: 13px; color: #555; text-transform: uppercase; font-weight: bold;">Operational Alert Tier</span>
                <div style="font-size: 24px; font-weight: bold; color: {badge_color}; margin-top: 4px;">{badge_text}</div>
                <span style="font-size: 11px; color: #666;">Demonstration alert state</span>
            </div>
            """, unsafe_allow_html=True)

        with col_m2:
            st.metric(
                label="Model Failure Probability",
                value=f"{prob:.4f}",
                delta=f"Threshold: {thresh:.4f}",
                delta_color="inverse" if pred == 1 else "normal",
                help="Instantaneous statistical failure probability estimated by Random Forest.",
            )
            # Probability indicator bar
            st.progress(float(min(1.0, max(0.0, prob))))

        with col_m3:
            st.metric(
                label="Demonstration Health Index",
                value=f"{hi:.1f} %",
                delta=f"{hi - 100:.1f} % vs Nominal",
                help="Synthetic demonstration indicator: 100 * (1 - failure probability). Not physical RUL.",
            )
            st.caption("Illustrative confidence metric only")

        with col_m4:
            st.metric(
                label="Processed Ingestion Count",
                value=f"{session.processed_count:,}",
                delta=f"Sequence ID: #{raw_rec.get('sequence_id', 0)}",
                help="Total records processed since simulation initialization.",
            )
            st.caption(f"Product ID: {raw_rec.get('Product ID', 'N/A')} (Variant: {raw_rec.get('Type', 'N/A')})")

        st.markdown(
            f"**Recommended Action Guidance**: *{badge_desc}* "
            "(Review the model score and simulated sensor readings.)"
        )

        st.divider()

        # Row 2: Ingestion & Alert Distribution Summary
        st.markdown("##### Cumulative Alert Distribution")
        col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns(5)
        col_c1.metric("🟢 Normal Records", f"{session.alert_counts.get(ALERT_NORMAL, 0)}")
        col_c2.metric("🔵 Watch Alerts", f"{session.alert_counts.get(ALERT_WATCH, 0)}")
        col_c3.metric("🟠 Warning Alerts", f"{session.alert_counts.get(ALERT_WARNING, 0)}")
        col_c4.metric("🔴 Critical Alerts", f"{session.alert_counts.get(ALERT_CRITICAL, 0)}")
        col_c5.metric("⚡ Latest Latency", f"{latest.get('inference_latency_ms', 0):.2f} ms")

        st.divider()

        # Row 3: Instantaneous Sensor Telemetry Grid
        st.markdown("##### Latest Ingested Telemetry Readings (Instantaneous Features)")
        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)

        air_temp = raw_rec.get("Air temperature [K]", 0.0)
        proc_temp = raw_rec.get("Process temperature [K]", 0.0)
        speed = raw_rec.get("Rotational speed [rpm]", 0)
        torque = raw_rec.get("Torque [Nm]", 0.0)
        wear = raw_rec.get("Tool wear [min]", 0)
        temp_diff = proc_temp - air_temp
        mech_power = torque * speed * (2.0 * 3.1415926535 / 60.0)

        col_s1.metric("Rotational Speed", f"{speed:,} rpm")
        col_s2.metric("Torque", f"{torque:.1f} Nm")
        col_s3.metric("Tool Wear", f"{wear} min")
        col_s4.metric("Thermal Gradient (ΔT)", f"{temp_diff:.2f} K", help="Process Temp - Air Temp")
        col_s5.metric("Shaft Power", f"{mech_power:.1f} W", help="Mechanical shaft power output")


# =============================================================================
# TAB 2: LIVE SIMULATION
# =============================================================================
with tab_sim:
    st.subheader("Live Telemetry Ingestion & Scoring Stream")

    col_info1, col_info2 = st.columns([2, 1])
    with col_info1:
        st.markdown(
            "Step through simulated sensor packets emitted by the virtual telemetry simulator. "
            "Each packet is validated for structural integrity and physical range limits, appended "
            "to the bounded FIFO buffer, and scored through the frozen scikit-learn pipeline."
        )
    with col_info2:
        st.info(
            f"**Playback Mode**: {replay_mode} | **Buffer Cap**: {session.window_size} | "
            f"**Threshold**: `{session.decision_threshold:.4f}`"
        )

    st.divider()

    # Telemetry Stream History Table
    st.markdown("##### Recent Ingested Telemetry History")
    history_df = session.to_history_dataframe()

    if history_df.empty:
        st.info("No records in history. Advance the simulation from the sidebar.")
    else:
        # Display formatted slice (newest records first)
        display_df = history_df.tail(25).iloc[::-1].copy()

        # Format columns for display
        display_cols = [
            "sequence_id",
            "Type",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "temperature_difference",
            "mechanical_power_w",
            "failure_probability",
            "alert_state",
            "demonstration_health_index",
            "inference_latency_ms",
        ]
        available_cols = [c for c in display_cols if c in display_df.columns]

        try:
            st.dataframe(
                display_df[available_cols],
                use_container_width=True,
                hide_index=True,
            )
        except Exception:
            st.table(display_df[available_cols])
        st.caption("Showing up to 25 most recent telemetry events (most recent at top).")

    # Contract Verification Summary
    st.divider()
    st.markdown("##### Data Integrity & Prediction Contract Guards")
    col_g1, col_g2, col_g3 = st.columns(3)

    col_g1.success("✅ **Prohibited Columns Excluded**: Target (`Machine failure`) and failure modes (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) never enter predictive features.")
    col_g2.success("✅ **Single Model Load**: `models/selected_model.joblib` loaded once at session startup; zero online retraining.")
    col_g3.success("✅ **Deterministic Scoring**: Optimal threshold $\\tau^* = 0.6732$ applied consistently across all incoming records.")


# =============================================================================
# TAB 3: SENSOR TRENDS & BUFFER DIAGNOSTICS
# =============================================================================
with tab_trends:
    st.subheader("Sliding-Window Sensor Dynamics & Diagnostic Statistics")

    st.markdown(
        "> ℹ️ **Diagnostic Separation Notice**: The charts and rolling statistics below are derived "
        "from the active bounded FIFO buffer (`SlidingWindowBuffer`) strictly for **operator diagnostic visibility**. "
        "The machine learning model was trained strictly on instantaneous 8-feature snapshots. "
        "**Rolling window statistics are NEVER passed to the predictive model.**"
    )

    df_hist = session.to_history_dataframe()

    # 1. Plotly Interactive Sequence Charts
    col_ch1, col_ch2 = st.columns(2)
    with col_ch1:
        st.plotly_chart(build_spindle_dynamics_chart(df_hist), use_container_width=True)
    with col_ch2:
        st.plotly_chart(build_thermal_profile_chart(df_hist), use_container_width=True)

    col_ch3, col_ch4 = st.columns(2)
    with col_ch3:
        st.plotly_chart(build_tool_wear_and_power_chart(df_hist), use_container_width=True)
    with col_ch4:
        st.plotly_chart(
            build_failure_risk_trajectory_chart(df_hist, threshold=session.decision_threshold),
            use_container_width=True,
        )

    st.divider()

    # 2. Bounded Buffer Rolling Statistics Table
    st.markdown("##### Active Bounded Buffer Rolling Statistics")
    st.caption(
        f"Calculated across the {len(session.buffer)} records currently residing in the sliding window buffer "
        f"(Capacity: {session.window_size}). Sample standard deviation uses Bessel's correction (ddof=1)."
    )

    rolling_stats = session.buffer.compute_rolling_stats()
    if not rolling_stats:
        st.info("Buffer is currently empty. Advance the simulation to view window statistics.")
    else:
        stats_rows = []
        for feature_name, metrics_dict in rolling_stats.items():
            stats_rows.append({
                "Sensor Feature": feature_name,
                "Window Count": metrics_dict.get("count", 0),
                "Mean": metrics_dict.get("mean", 0.0),
                "Min": metrics_dict.get("min", 0.0),
                "Max": metrics_dict.get("max", 0.0),
                "Sample Std (ddof=1)": metrics_dict.get("std", 0.0),
            })
        try:
            st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)
        except Exception:
            st.table(pd.DataFrame(stats_rows))


# =============================================================================
# TAB 4: MODEL & BENCHMARK LIMITATIONS
# =============================================================================
with tab_model:
    st.subheader("Model Evaluation, Feature Importance & Scientific Integrity")

    metrics_payload = load_test_evaluation_metrics()
    perm_df = load_permutation_importance_summary()

    # Section 1: Benchmark Overview & Held-Out Test Evaluation
    st.markdown("##### 1. Benchmark & Evaluation Protocol")
    st.markdown("""
    - **Benchmark Source**: UCI AI4I 2020 Predictive Maintenance Dataset ($10,000$ synthetic observations, $14$ columns).
    - **Class Balance**: 339 failure records ($3.39\\%$ prevalence) vs 9,661 normal records ($96.61\\%$, imbalance ratio $28.5:1$).
    - **Split Protocol**: Deterministic Stratified 80/20 train/test split ($8,000$ train / $2,000$ test). The test partition was isolated and evaluated once.
    - **Threshold Selection Protocol**: Decision threshold ($\\tau^* = 0.6732$) was optimized strictly on **training out-of-fold (OOF)** predictions to maximize F1 score. Test set labels were never peeked or used to adjust thresholds.
    """)

    if metrics_payload:
        test_m = metrics_payload.get("test_metrics_selected_threshold", {})
        col_tm1, col_tm2, col_tm3, col_tm4, col_tm5 = st.columns(5)
        col_tm1.metric("Average Precision", f"{test_m.get('average_precision', 0):.4f}")
        col_tm2.metric("ROC-AUC", f"{test_m.get('roc_auc', 0):.4f}")
        col_tm3.metric("F1-Score", f"{test_m.get('f1_score', 0):.4f}")
        col_tm4.metric("Recall (Sensitivity)", f"{test_m.get('recall', 0):.4f}")
        col_tm5.metric("Precision", f"{test_m.get('precision', 0):.4f}")

        st.caption(
            f"Held-out test performance (2,000 samples, 68 failures) at optimal threshold τ* = {session.decision_threshold:.4f}. "
            f"Confusion matrix: TP={test_m.get('true_positives')}, TN={test_m.get('true_negatives')}, "
            f"FP={test_m.get('false_positives')}, FN={test_m.get('false_negatives')}."
        )

    st.divider()

    # Section 2: Permutation Feature Importance & Collinearity
    st.markdown("##### 2. Permutation Feature Importance (Predictive Reliance)")
    col_p1, col_p2 = st.columns([3, 2])

    with col_p1:
        st.plotly_chart(build_permutation_importance_bar_chart(perm_df), use_container_width=True)

    with col_p2:
        st.markdown("""
        **Interpretation & Correlation Dynamics**:
        - **`Rotational speed [rpm]`** (0.4575) and **`temperature_difference`** (0.2675) are the strongest individual predictive drivers.
        - **Predictive Reliance vs. Physical Causality**: Permutation importance measures how much discrimination degrades when feature values are shuffled. It reflects *predictive association*, not mechanical causality.
        - **Collinearity**: Raw temperatures have low marginal importance because their joint information is already captured by the domain-engineered $\\Delta T = T_{\\text{process}} - T_{\\text{air}}$.
        """)

    st.divider()

    # Section 3: Empirical Latency Benchmark
    st.markdown("##### 3. Empirical Inference Latency Benchmark")
    lat_bench = PHASE_6_LATENCY_BENCHMARK
    col_l1, col_l2, col_l3, col_l4 = st.columns(4)
    col_l1.metric("Sample Count", f"{lat_bench['sample_count']}")
    col_l2.metric("Median (P50) Latency", f"{lat_bench['median_p50_ms']} ms")
    col_l3.metric("95th Percentile (P95)", f"{lat_bench['p95_ms']} ms")
    col_l4.metric("Mean Latency", f"{lat_bench['mean_ms']} ms")

    st.caption(
        f"**Environment**: {lat_bench['environment']} | **Note**: {lat_bench['disclaimer']}"
    )

    st.divider()

    # Section 4: Known Limitations & Scientific Honesty
    st.markdown("##### 4. Known Methodological Limitations & Integrity Principles")
    st.markdown("""
    1. **Independent Experimental Trials**: The AI4I dataset contains independent static runs, not continuous degradation time series of a physical machine. Replaying them sequentially is a software demonstration only.
    2. **Tool Wear Failure (`TWF`) Detection Ceiling**: Phase 5 error analysis revealed that 9 of 15 false negatives correspond to Tool Wear Failure, where machine operating parameters remain within nominal bands while cumulative wear triggers probabilistic failure. Instantaneous sensor snapshots cannot detect these events without high-frequency vibration or acoustic emission channels.
    3. **Overlapping Multi-Label Modes**: In the AI4I benchmark, 24 machines have multiple failure mode flags asserted simultaneously (e.g. `TWF` and `OSF`). They are non-mutually exclusive, and mode counts cannot be summed to reconstruct `Machine failure`.
    4. **Unviability of `RNF` Modeling**: Random Failure (`RNF`) is simulated as an independent Bernoulli random trial ($P=0.001$, only 15 training positives) and represents pure stochastic noise unviable for ML classification.
    5. **Demonstration Health Index**: The Health Index formula ($100 \\times (1 - P(\\text{fail}))$) is an illustrative inversion of statistical model confidence. It does not estimate remaining operational cycles or physical fatigue state.
    """)

