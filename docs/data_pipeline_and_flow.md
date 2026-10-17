# Data Pipeline & Flow Specification

This document details the complete end-to-end flow of data, mathematical transformations, leakage prevention boundaries, and state transitions across the **AI-Powered Predictive Maintenance for IoT Devices** platform.

---

## 1. End-to-End Telemetry Lifecycle

Data flows through two operational pathways: the **Batch Training & Diagnostics Pipeline** and the **Live Streaming Telemetry & Inference Runtime**.

```
===================================================================================================
1. BATCH TRAINING & DIAGNOSTICS PIPELINE
===================================================================================================
Raw Dataset File [data/raw/ai4i2020.csv] (10,000 rows x 14 columns)
       │
       ▼ (1) Data Loading & Schema Validation (src.data.loader, src.data.validator)
       ├─ Assert all 14 required columns are present
       ├─ Verify 0 null / NaN / infinite values
       ├─ Confirm Machine failure target values are strictly binary {0, 1}
       └─ Audit physical ranges (temperatures 250-350 K, speed 500-4000 rpm, torque 0-150 Nm)
       │
       ▼ (2) Deterministic Stratified Split (src.features.split)
       ├─ Stratify on target: df['Machine failure'] (3.39% positive class prevalence)
       ├─ Split 80% Training (N=8,000, 271 failures) vs. 20% Held-Out Test (N=2,000, 68 failures)
       └─ ISOLATE TEST SET: Placed aside immediately; evaluated exactly ONCE at the end.
       │
       ▼ (3) Domain Physical Feature Engineering (src.features.engineering)
       ├─ ΔT = Process temperature [K] - Air temperature [K]
       ├─ P_mech = Torque [Nm] * (Rotational speed [rpm] * 2π / 60) [W]
       └─ Deep-copy input DataFrame to guarantee immutability
       │
       ▼ (4) Feature Matrix Construction & Leakage Filtering (src.features.schema)
       ├─ Retained Categorical Feature: Type (L, M, H)
       ├─ Retained Numerical Features: Air temp, Process temp, Speed, Torque, Tool wear, ΔT, P_mech
       └─ STRICT PROHIBITED EXCLUSIONS: UDI, Product ID, Machine failure, TWF, HDF, PWF, OSF, RNF
       │
       ▼ (5) Preprocessing Pipeline Encapsulation (src.features.preprocessing)
       ├─ Numeric Features: SimpleImputer(median) -> StandardScaler()
       └─ Categorical Feature: SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')
       │
       ▼ (6) Stratified 5-Fold Cross-Validation (src.train)
       ├─ Fit ColumnTransformer exclusively on 4 folds, transform validation fold (NO FOLD LEAKAGE)
       ├─ Evaluate candidate models: Dummy, Balanced Logistic Regression, Balanced Random Forest
       └─ Model Selection: Balanced Random Forest achieves highest Mean CV Avg Precision (0.8521)
       │
       ▼ (7) Out-of-Fold (OOF) Threshold Optimization (src.evaluate)
       ├─ Generate out-of-fold predicted probability vector for all 8,000 training samples
       ├─ Evaluate F1 score across threshold grid [0.10, 0.90] in steps of 0.0001
       └─ Selected Optimal Threshold: τ* = 0.6732 (Max Training OOF F1 = 0.7984)
       │
       ▼ (8) Final Fit & Serialization
       ├─ Fit winning pipeline on all 8,000 training samples
       └─ Serialize pipeline to models/selected_model.joblib (1.62 MB)
       │
       ▼ (9) Single Held-Out Test Set Evaluation & Diagnostics (src.diagnostics)
       ├─ Transform test features through fitted pipeline; score probabilities
       ├─ Evaluate metrics at baseline (0.50) and selected threshold (0.6732)
       ├─ Results: Avg Precision 0.8394, ROC-AUC 0.9754, F1 0.8346, Precision 0.8983, Recall 0.7794
       ├─ Permutation feature importance on test set (10 repeats)
       └─ Audit 15 false negatives and 6 false positives against secondary failure modes

===================================================================================================
2. LIVE STREAMING TELEMETRY & INFERENCE RUNTIME
===================================================================================================
[Virtual Telemetry Simulator] (src.streaming.simulator)
  - Loads data/raw/ai4i2020.csv as benchmark stream source
  - Emits telemetry packets step-by-step (.step()) or as generator (.stream())
  - Replay modes: Sequential benchmark progression or Seeded random shuffle
  - Enforces physical boundary validation per record
       │ Incremental Telemetry Record: {sequence_id, timestamp, Type, Temps, Speed, Torque, Wear}
       ▼
[Bounded Sliding-Window Buffer] (src.streaming.buffer) (Bounded FIFO, maxlen = 10)
  - Pushes deep copy of packet into internal collections.deque
  - Evicts oldest sample when buffer reaches capacity (FIFO)
  - Computes rolling statistics (mean, min, max, sample std ddof=1) across numerical channels
  * DIAGNOSTIC SEPARATION: Window statistics are for dashboard display only; NEVER passed to model.
       │ Single Instantaneous Record (8 Raw + Engineered Features)
       ▼
[Streaming Inference Service] (src.streaming.inference)
  - Loads models/selected_model.joblib ONCE into memory at session initialization
  - Applies engineer_features() deterministically to compute ΔT and P_mech
  - Asserts prohibited columns are absent
  - Evaluates predict_proba() -> failure probability
  - Records monotonic execution latency via time.perf_counter()
       │ Inference Payload: {probability, alert_state, health_index, latency_ms}
       ▼
[Alert & Health Classifier] (src.streaming.alerts)
  - Evaluates probability against decision threshold τ* = 0.6732
  - Categorizes operational state: NORMAL (<0.30), WATCH (0.30-0.50), WARNING (0.50-0.6732), CRITICAL (≥0.6732)
  - Computes synthetic demonstration Health Index = 100.0 * (1.0 - failure_probability)
       │
       ▼
[Streamlit Dashboard] (app.py via src.dashboard.state.DashboardSession)
  - Records result in session history log
  - Updates Fleet Overview status cards, risk gauge bar, and instantaneous telemetry grid
  - Appends recent sample to Live Simulation inspection table
  - Plots updated time series in Plotly charts (Spindle Dynamics, Thermal Profile, Wear/Power, Risk Trajectory)
  - Renders active buffer rolling statistics table
```

---

## 2. Feature Engineering Mathematics

The machine learning model operates strictly on instantaneous operational features. Two domain-engineered physical indicators are derived mathematically:

### 1. Thermal Differential ($\Delta T$)
$$\Delta T = T_{\text{process}} - T_{\text{air}} \quad [\text{Kelvin, K}]$$
- **Physical Motivation**: Heat Dissipation Failure (HDF) occurs when the heat generated by tool-workpiece cutting friction cannot be effectively transferred into the ambient air. In the AI4I benchmark simulation, HDF failure is triggered whenever $\Delta T < 8.6\text{ K}$ while spindle rotational speed is below $1,380\text{ rpm}$.
- **Engineered Benefit**: Exposing $\Delta T$ directly allows the tree ensemble to isolate the HDF failure regime with a single split, eliminating collinearity between raw process and air temperatures.

### 2. Spindle Shaft Mechanical Power ($P_{\text{mech}}$)
$$P_{\text{mech}} = \tau \cdot \omega = \text{Torque [Nm]} \cdot \left(\text{Rotational speed [rpm]} \cdot \frac{2\pi}{60}\right) \quad [\text{Watts, W}]$$
- **Physical Motivation**: Power Failure (PWF) occurs when the mechanical power delivered by the rotating drive spindle falls outside the allowable operating envelope ($P_{\text{mech}} < 3,500\text{ W}$ or $P_{\text{mech}} > 9,000\text{ W}$).
- **Engineered Benefit**: Without $P_{\text{mech}}$, a machine learning model would have to approximate a non-linear hyperbolic curve ($\tau \cdot \omega = C$). Providing the product directly creates a linear boundary that decision trees and linear models can split cleanly.

---

## 3. Strict Leakage Prevention Architecture

Data leakage invalidates predictive maintenance models in real-world deployments. The pipeline enforces three strict architectural boundaries:

1. **Prediction Contract Isolation**:
   - The predictive feature matrix is strictly constructed prior to model scoring.
   - Prohibited columns (`UDI`, `Product ID`, `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are formally blocked by schema assertions in `src/features/schema.py` and filtered out in `StreamingInferenceService`.
2. **Train/Test Partitioning Independence**:
   - The dataset is split into training (80%) and held-out test (20%) sets before fitting any transformer.
   - The test set is completely isolated and evaluated exactly once. It is never used to select models, tune hyperparameters, or optimize decision thresholds.
3. **Cross-Validation Fold Isolation**:
   - Preprocessing steps (median imputation, standard scaling, one-hot encoding) are encapsulated within a unified Scikit-Learn `Pipeline`.
   - Within 5-fold cross-validation, transformers are fitted strictly on the 4 training folds and transform the 5th validation fold, preventing validation fold leakage.

---

## 4. Sliding-Window Buffer State Management

In edge IoT systems, temporal buffering is required to compute rolling descriptive statistics without storing infinite history:

```
Buffer Ingestion (Capacity W = 10):
Incoming Record t_11 ──► [t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10, t_11] ──► Evicted Record t_1
```

- **Data Structure**: `collections.deque(maxlen=window_size)` providing $O(1)$ amortized append and eviction complexity.
- **Deep-Copy Immutability**: `copy.deepcopy()` is applied on all incoming and outgoing records to prevent external callers or UI elements from mutating stored history.
- **Rolling Statistics Formulation**:
  - Sample Mean: $\bar{x} = \frac{1}{N} \sum_{i=1}^N x_i$
  - Sample Standard Deviation: $s = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (x_i - \bar{x})^2}$ (Bessel's correction $ddof=1$)
  - Minimum: $\min(X)$, Maximum: $\max(X)$
- **Decoupling Guarantee**: The buffer's rolling statistics are strictly for operator visual inspection in Tab 3 of the dashboard. They are **never** passed into `StreamingInferenceService`. The model feature contract expects exactly 8 instantaneous inputs.
