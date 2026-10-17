# System Architecture: AI-Powered Predictive Maintenance for IoT Devices

This document specifies the system architecture, component topologies, and runtime pipelines for the **AI-Powered Predictive Maintenance for IoT Devices** platform.

---

## 1. High-Level Architecture Overview

The system is architected as two decoupled sub-systems:
1. **Offline Training & Validation Pipeline**: A reproducible batch workflow for data validation, stratified splitting, leakage-safe feature engineering, cross-validated model selection, and out-of-fold threshold optimization.
2. **Real-Time Streaming & Monitoring Runtime**: A lightweight, memory-resident inference engine that simulates edge IoT sensor streaming, sliding-window buffering, low-latency scoring, and interactive web dashboard visualization.

```
+-------------------------------------------------------------------------------------------------+
|                                 1. OFFLINE BATCH TRAINING PIPELINE                              |
+-------------------------------------------------------------------------------------------------+
|                                                                                                 |
|   [Raw Dataset]                                                                                 |
|   data/raw/ai4i2020.csv (10,000 samples, 14 columns)                                            |
|         │                                                                                       |
|         ▼                                                                                       |
|   [Data Ingestion & Validation Layer] (src.data.loader, src.data.validator)                     |
|   - Reusable loader with project-root relative resolution                                       |
|   - Schema validation, null checks, duplicate audits, physical range assertions                 |
|         │                                                                                       |
|         ▼                                                                                       |
|   [Stratified Split Layer] (src.features.split)                                                 |
|   - Stratified 80/20 train/test split (random_state=42)                                         |
|   - Training: N=8,000 (271 failures) | Held-Out Test: N=2,000 (68 failures)                     |
|         │                                                                                       |
|         ▼                                                                                       |
|   [Domain Feature Engineering] (src.features.engineering)                                       |
|   - Temperature Differential: ΔT = Process Temp [K] - Air Temp [K]                              |
|   - Shaft Mechanical Power: P_mech = Torque [Nm] * (Rotational speed [rpm] * 2π / 60) [W]       |
|         │                                                                                       |
|         ▼                                                                                       |
|   [Scikit-Learn Pipeline Encapsulation] (src.features.preprocessing, src.train)                 |
|   - Numeric: SimpleImputer(median) + StandardScaler                                             |
|   - Categorical (Type): SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown='ignore')   |
|   - Classifier: Balanced Random Forest (100 estimators, max_depth=10, class_weight='balanced')  |
|         │                                                                                       |
|         ▼                                                                                       |
|   [Evaluation & Threshold Optimization] (src.evaluate)                                          |
|   - Stratified 5-Fold Cross-Validation on training partition (Mean CV Avg Precision = 0.8521)   |
|   - Out-of-fold (OOF) threshold grid search to maximize F1 (Optimal: τ* = 0.6732)               |
|         │                                                                                       |
|         ├───────────────────────────────────────────┐                                           |
|         ▼                                           ▼                                           |
|   [Serialized Artifact]                       [Evaluation Reports]                              |
|   models/selected_model.joblib (1.62 MB)      outputs/test_metrics.json                         |
|                                               outputs/permutation_importance.csv                |
|                                               outputs/test_predictions.csv                      |
+-------------------------------------------------------------------------------------------------+

+-------------------------------------------------------------------------------------------------+
|                               2. REAL-TIME STREAMING & MONITORING RUNTIME                       |
+-------------------------------------------------------------------------------------------------+
|                                                                                                 |
|   [Virtual Telemetry Simulator] (src.streaming.simulator)                                       |
|   - Replays AI4I benchmark records sequentially or with reproducible seeded shuffle             |
|   - Validates physical boundary constraints per record (SENSOR_PHYSICAL_BOUNDS)                 |
|         │ Incremental Telemetry Record                                                          |
|         ▼                                                                                       |
|   [Bounded Sliding-Window Buffer] (src.streaming.buffer)                                        |
|   - Bounded FIFO deque (default capacity W = 10 samples)                                        |
|   - Deep-copy guarantees protecting buffer immutability                                         |
|   - Computes rolling statistics (mean, min, max, sample std with Bessel's correction ddof=1)   |
|   * DIAGNOSTIC SEPARATION: Window statistics are for display only; NEVER fed to model.          |
|         │ Instantaneous 8-Feature Vector                                                        |
|         ▼                                                                                       |
|   [Streaming Inference Service] (src.streaming.inference)                                       |
|   - Preloads models/selected_model.joblib ONCE at initialization into memory                    |
|   - Applies engineer_features() deterministically                                               |
|   - Enforces prohibited column exclusions (UDI, Product ID, Target, Failure modes)              |
|   - Evaluates predict_proba() -> failure probability                                            |
|   - Measures pure scoring latency with high-resolution monotonic timer (time.perf_counter)      |
|         │ Inference Output Payload                                                              |
|         ▼                                                                                       |
|   [Operational Alert & Health Classifier] (src.streaming.alerts)                                |
|   - 4-Tier Operational Alerts: NORMAL (<0.30), WATCH (0.30-0.50), WARNING (0.50-0.6732),        |
|     CRITICAL (≥0.6732)                                                                          |
|   - Demonstration Health Index = 100.0 * (1.0 - P(failure))                                     |
|         │                                                                                       |
|         ▼                                                                                       |
|   [Interactive Streamlit Dashboard] (app.py)                                                    |
|   - Orchestrated by src.dashboard.state.DashboardSession                                        |
|   - Tab 1: Fleet & Machine Overview (Status badge, risk gauge, health index, metric cards)      |
|   - Tab 2: Live Telemetry Simulation (Step sample, batch runner, replay modes, history table)   |
|   - Tab 3: Sensor Trends & Diagnostics (4 synchronized Plotly charts, rolling stats table)      |
|   - Tab 4: Model Diagnostics & Limitations (Test metrics scorecard, permutation chart, notes)  |
+-------------------------------------------------------------------------------------------------+
```

---

## 2. Component Subsystems and Design Principles

### 1. Ingestion & Validation Subsystem
- **Modules**: `src/data/loader.py`, `src/data/validator.py`, `src/data/schema.py`
- **Design Principle**: Fail-early schema validation. Ingests raw data via project-root relative resolution, performs strict type and value range checks, asserts zero nulls, and verifies target column binary integrity before data reaches feature engineering.

### 2. Feature Engineering & Preprocessing Subsystem
- **Modules**: `src/features/engineering.py`, `src/features/preprocessing.py`, `src/features/schema.py`, `src/features/split.py`
- **Design Principle**: Strict leakage prevention and immutability.
  - Excludes identifiers (`UDI`, `Product ID`), the target (`Machine failure`), and post-hoc failure mode flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`).
  - Engineers physically grounded features:
    $$\Delta T = T_{\text{process}} - T_{\text{air}}$$
    $$P_{\text{mech}} = \tau \cdot \left(\omega_{\text{rpm}} \cdot \frac{2\pi}{60}\right)$$
  - Encapsulates transformations within a Scikit-Learn `ColumnTransformer` to ensure fitting occurs strictly on training folds.

### 3. Training & Evaluation Subsystem
- **Modules**: `src/train.py`, `src/evaluate.py`, `src/diagnostics.py`
- **Design Principle**: Asymmetric risk evaluation and test set isolation.
  - Compares candidate models via stratified 5-fold cross-validation using **Average Precision (PR-AUC)** as the primary selection criterion.
  - Balances class weights (`class_weight='balanced'`) to address the 28.5:1 class imbalance.
  - Tunes decision threshold ($\tau^* = 0.6732$) strictly on training out-of-fold predictions.
  - Evaluates the held-out test set (N=2,000) exactly once, followed by permutation feature importance and forensic false-negative auditing.

### 4. Streaming Runtime Subsystem
- **Modules**: `src/streaming/simulator.py`, `src/streaming/buffer.py`, `src/streaming/inference.py`, `src/streaming/alerts.py`
- **Design Principle**: Decoupled, stateful edge simulation with low latency.
  - Simulates edge sensor packets with physical range checks.
  - Maintains a bounded sliding-window FIFO buffer for operator rolling statistics.
  - Preloads model pipeline once in memory, delivering sub-50 ms median inference latency.
  - Maps failure probabilities into 4 actionable operational alert tiers.

### 5. Presentation Subsystem
- **Modules**: `app.py`, `src/dashboard/state.py`, `src/dashboard/charts.py`, `src/dashboard/metrics.py`
- **Design Principle**: Session state idempotency and reactive safety.
  - Stores application state in `DashboardSession`, ensuring that Streamlit script re-executions do not advance telemetry or trigger duplicate scoring without explicit user button clicks.
  - Employs defensive rendering fallbacks from `st.dataframe()` to `st.table()` to guarantee resilience across diverse host environments.

---

## 3. Detailed Data Flow and Interface Contracts

For complete tracing of data transformations through every stage, see [Data Pipeline & Flow](data_pipeline_and_flow.md). For formal class and function signatures, see [Modules & Components](modules_and_components.md).
