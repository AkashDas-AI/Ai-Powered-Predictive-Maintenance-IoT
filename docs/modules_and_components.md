# Modules & Component Architecture

This document specifies the modular decomposition, package hierarchies, class responsibilities, and API contracts for the **AI-Powered Predictive Maintenance for IoT Devices** codebase.

---

## 1. Package Directory Hierarchy

```
AI-Powered Predictive Maintenance for IoT Devices/
├── main.py                             # Unified CLI workflow entry point
├── app.py                              # Streamlit interactive web dashboard
├── src/                                # Core package source code
│   ├── __init__.py
│   ├── config.py                       # Centralized configuration, thresholds, bounds & paths
│   ├── train.py                        # Baseline model training & cross-validation engine
│   ├── evaluate.py                     # Diagnostic metrics & out-of-fold threshold search
│   ├── diagnostics.py                  # Permutation importance, error analysis & failure modes
│   ├── data/                           # Data loading and schema validation
│   │   ├── __init__.py
│   │   ├── schema.py                   # Data schemas & ValidationReport dataclass
│   │   ├── validator.py                # Schema validation & anomaly audit
│   │   └── loader.py                   # Reusable loader with path resolution
│   ├── features/                       # Feature engineering & preprocessing
│   │   ├── __init__.py
│   │   ├── schema.py                   # Prediction contract & column exclusions
│   │   ├── engineering.py              # Domain feature calculation (ΔT, P_mech)
│   │   ├── preprocessing.py            # Reusable scikit-learn ColumnTransformer
│   │   └── split.py                    # Deterministic stratified train/test split
│   ├── streaming/                      # Virtual telemetry streaming & inference runtime
│   │   ├── __init__.py
│   │   ├── simulator.py                # TelemetrySimulator & physical bounds validation
│   │   ├── buffer.py                   # SlidingWindowBuffer & diagnostic rolling statistics
│   │   ├── inference.py                # StreamingInferenceService & latency measurement
│   │   ├── alerts.py                   # 4-tier alert classifier & demonstration Health Index
│   │   └── demo.py                     # CLI demonstration runner & latency statistics
│   └── dashboard/                      # Dashboard support modules
│       ├── __init__.py
│       ├── state.py                    # DashboardSession state orchestrator & rerun guard
│       ├── charts.py                   # Interactive Plotly time-series & importance charts
│       └── metrics.py                  # Static held-out test scorecard & latency loaders
└── tests/                              # Automated test suite (56 tests)
    ├── __init__.py
    ├── test_data_loader.py             # 7 tests: loader, schema, missing file, validations
    ├── test_features_and_preprocessing.py # 8 tests: feature formulas, immutability, split
    ├── test_modeling.py                # 7 tests: pipeline fitting, leakage guards, serialization
    ├── test_diagnostics_and_explainability.py # 5 tests: permutation, errors, failure modes
    ├── test_streaming.py               # 17 tests: simulator, buffer FIFO, latency, alerts
    └── test_dashboard.py               # 12 tests: session, stepping, buffer, charts, AppTest
```

---

## 2. Core Module Contracts & Interfaces

### 1. Centralized Configuration: `src/config.py`
- **Location**: `src/config.py`
- **Role**: Single source of truth for all paths, decision thresholds, operating limits, and alert cutoffs.
- **Key Constants & Functions**:
  - `DEFAULT_DECISION_THRESHOLD: float = 0.6732`: Optimal threshold derived in Phase 4 via training out-of-fold F1 maximization.
  - `ALERT_CUTOFF_WATCH: float = 0.30`: Probability threshold for `WATCH` tier.
  - `ALERT_CUTOFF_WARNING: float = 0.50`: Probability threshold for `WARNING` tier.
  - `SENSOR_PHYSICAL_BOUNDS: Dict[str, Tuple[float, float]]`: Expected physical operating ranges for validation (e.g. Air temp: 250-350 K, Speed: 500-4000 rpm, Torque: 0-150 Nm, Tool wear: 0-400 min).
  - `load_decision_threshold(metrics_path=None) -> float`: Safely loads threshold from `outputs/test_metrics.json`, falling back to `0.6732` if missing.

### 2. Data Loading & Validation: `src/data/`
- **`src/data/loader.py`**:
  - `load_ai4i_dataset(filepath=None, validate=True, raise_on_error=True) -> Tuple[pd.DataFrame, Optional[ValidationReport]]`:
    Loads the AI4I 2020 dataset from `data/raw/ai4i2020.csv` using project-root relative path resolution. Optionally validates schema.
- **`src/data/validator.py`**:
  - `validate_ai4i_dataframe(df: pd.DataFrame) -> ValidationReport`:
    Checks column presence, verifies zero nulls, confirms `Machine failure` is binary `{0, 1}`, audits duplicate rows, and verifies physical value ranges.
- **`src/data/schema.py`**:
  - `ValidationReport`: Dataclass capturing `is_valid`, `missing_columns`, `null_counts`, `duplicate_count`, `invalid_target_values`, and `out_of_bounds_counts`.
  - `REQUIRED_COLUMNS`: List of all 14 expected AI4I columns.

### 3. Feature Engineering & Preprocessing: `src/features/`
- **`src/features/engineering.py`**:
  - `engineer_features(df: pd.DataFrame) -> pd.DataFrame`:
    Computes $\Delta T = T_{\text{process}} - T_{\text{air}}$ [K] and $P_{\text{mech}} = \tau \cdot (\omega_{\text{rpm}} \cdot 2\pi / 60)$ [W]. Deep-copies input DataFrame to guarantee immutability.
- **`src/features/schema.py`**:
  - `PROHIBITED_COLUMNS`: `['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']`.
  - `NUMERIC_FEATURES`: `['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'temperature_difference', 'mechanical_power_w']`.
  - `CATEGORICAL_FEATURES`: `['Type']`.
- **`src/features/preprocessing.py`**:
  - `create_preprocessor() -> ColumnTransformer`:
    Assembles a Scikit-Learn `ColumnTransformer` with `SimpleImputer(median)` and `StandardScaler` for numeric columns, and `SimpleImputer(most_frequent)` and `OneHotEncoder(handle_unknown='ignore')` for `Type`.
- **`src/features/split.py`**:
  - `stratified_train_test_split(df: pd.DataFrame, test_size=0.2, random_state=42) -> Tuple[pd.DataFrame, pd.DataFrame]`:
    Executes reproducible stratified partition on `Machine failure`, preserving the 3.39% failure ratio across train (N=8,000) and held-out test (N=2,000) sets.

### 4. Training, Evaluation & Diagnostics: `src/`
- **`src/train.py`**:
  - `build_model_candidates() -> Dict[str, Pipeline]`: Constructs Dummy, Logistic Regression, and Balanced Random Forest pipelines.
  - `train_and_evaluate_pipeline() -> Tuple[Pipeline, pd.DataFrame, Dict[str, Any]]`: Executes stratified 5-fold cross-validation on training partition, evaluates out-of-fold threshold, fits winning pipeline on full training data, and serializes to `models/selected_model.joblib`.
- **`src/evaluate.py`**:
  - `compute_classification_metrics(y_true, y_prob, threshold=0.5) -> Dict[str, Any]`: Computes Average Precision, ROC-AUC, F1, precision, recall, and confusion matrix counts.
  - `find_optimal_threshold_oof(y_true, y_prob, metric='f1') -> Tuple[float, float]`: Searches threshold grid [0.10, 0.90] to maximize training out-of-fold F1.
- **`src/diagnostics.py`**:
  - `compute_permutation_importance(pipeline, X_test, y_test, n_repeats=10) -> pd.DataFrame`: Calculates test set Average Precision drop across 10 random feature permutations.
  - `run_error_analysis(pipeline, X_test, y_test, threshold) -> Tuple[pd.DataFrame, Dict[str, Any]]`: Audits false negatives and false positives against secondary failure modes (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`).
  - `evaluate_failure_modes_cv(df_train) -> pd.DataFrame`: Trains exploratory baselines across individual failure modes.

### 5. Streaming Ingestion & Inference Runtime: `src/streaming/`
- **`src/streaming/simulator.py`**:
  - `TelemetrySimulator(data_source=None, random_state=42, shuffle=False)`:
    - `.step() -> Optional[Dict[str, Any]]`: Advances simulation by 1 record, returning dictionary with `sequence_id` and timestamp.
    - `.stream(max_records=None) -> Iterator[Dict[str, Any]]`: Yields records incrementally.
    - `.reset() -> None`: Resets pointer to sequence start.
  - `validate_telemetry_record(record: Dict[str, Any]) -> Tuple[bool, List[str]]`: Enforces non-null, finite values and checks against `SENSOR_PHYSICAL_BOUNDS`.
- **`src/streaming/buffer.py`**:
  - `SlidingWindowBuffer(window_size=10)`:
    - `.append(record: Dict[str, Any]) -> None`: Adds deep copy of record into bounded FIFO deque.
    - `.latest_record() -> Optional[Dict[str, Any]]`: Returns copy of most recent reading.
    - `.get_window() -> List[Dict[str, Any]]`: Returns copies of all records in window.
    - `.compute_rolling_stats(columns=None) -> Dict[str, Dict[str, float]]`: Calculates `mean`, `min`, `max`, and sample `std` ($ddof=1$) for operator visual diagnostics.
- **`src/streaming/inference.py`**:
  - `StreamingInferenceService(model_path=None, threshold=None)`:
    - Preloads serialized Scikit-Learn pipeline once into memory at initialization.
    - `.predict_record(record: Dict[str, Any], sequence_id=None) -> Dict[str, Any]`: Applies feature engineering, excludes prohibited columns, scores probability via `predict_proba`, measures pure inference latency via `time.perf_counter()`, and returns structured payload.
- **`src/streaming/alerts.py`**:
  - `determine_alert_state(failure_probability: float, threshold: float = 0.6732) -> str`: Maps probability to `NORMAL`, `WATCH`, `WARNING`, or `CRITICAL`.
  - `compute_demonstration_health_index(failure_probability: float) -> float`: Calculates synthetic health indicator $100.0 \times (1.0 - P(\text{fail}))$.
- **`src/streaming/demo.py`**:
  - `run_streaming_demo(num_samples=25, window_size=10, verbose=True) -> Dict[str, Any]`: Standalone CLI demonstration runner measuring latency statistics ($P_{50}, P_{95}$, mean, min, max).

### 6. Dashboard State & Presentation: `src/dashboard/` and `app.py`
- **`src/dashboard/state.py`**:
  - `DashboardSession(window_size=10, shuffle=False, random_state=42)`:
    - Wraps `TelemetrySimulator`, `SlidingWindowBuffer`, and `StreamingInferenceService`.
    - Maintains `history: List[Dict[str, Any]]` and `alert_counts: Dict[str, int]`.
    - `.step() -> Optional[Dict[str, Any]]`: Ingests 1 sample, buffers it, scores it, and logs result.
    - `.run_batch(batch_size: int) -> List[Dict[str, Any]]`: Ingests multiple records sequentially.
    - `.reset(shuffle=None, random_state=None) -> None`: Resets pointer and clears buffer.
    - `.update_window_size(new_size: int) -> None`: Dynamically resizes buffer while retaining recent records up to new capacity.
    - `.to_history_dataframe() -> pd.DataFrame`: Flattens history records into a pandas DataFrame for Plotly charts and tables.
- **`src/dashboard/charts.py`**:
  - `build_spindle_dynamics_chart(df)`: Dual-axis Plotly chart (Rotational Speed vs. Torque).
  - `build_thermal_profile_chart(df)`: Multi-trace Plotly chart (Air Temp, Process Temp, $\Delta T$).
  - `build_tool_wear_and_power_chart(df)`: Dual-axis Plotly chart (Tool Wear vs. Mechanical Power).
  - `build_failure_risk_trajectory_chart(df, threshold)`: Failure probability curve with threshold and alert level reference lines.
  - `build_permutation_importance_bar_chart(df)`: Horizontal bar chart of test permutation importance.
- **`src/dashboard/metrics.py`**:
  - `load_test_evaluation_metrics()`: Loads held-out test evaluation report from `outputs/test_metrics.json`.
  - `load_permutation_importance_summary()`: Loads permutation importance table from `outputs/permutation_importance.csv`.
  - `PHASE_6_LATENCY_BENCHMARK`: Static dictionary documenting Phase 6 local CPU inference benchmark ($P_{50} = 19.33\text{ ms}$, $P_{95} = 53.84\text{ ms}$, Mean = $25.30\text{ ms}$).
- **`app.py`**:
  - Main Streamlit web application orchestrating the 4 functional tabs (Fleet Overview, Live Simulation, Sensor Trends & Diagnostics, Model Limitations). Features defensive `try/except` fallbacks from `st.dataframe()` to `st.table()` to ensure robust rendering.
