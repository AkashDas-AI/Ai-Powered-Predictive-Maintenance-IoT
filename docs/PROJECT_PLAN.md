# AI-Predictive-Maintenance-IoT: Project Plan & Roadmap

This document outlines the phased roadmap for building, training, evaluating, and deploying the **AI-Powered Predictive Maintenance for IoT Devices** platform.

---

## Phased Roadmap

### Phase 1: Project Initialization & Virtual Environment (Current)
- Initialize project structure, directories, and Git configuration.
- Set up dedicated Python 3.11 virtual environment (`.venv`).
- Install and verify foundational dependencies (`numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `jupyter`, `streamlit`, `plotly`, `pytest`).
- Establish project tracking, documentation, and verification protocol.

### Phase 2: Data Ingestion & Exploratory Data Analysis (EDA) (Completed)
- Ingested official UCI AI4I 2020 Predictive Maintenance benchmark dataset (`10,000` records x `14` columns) into `data/raw/`.
- Authored comprehensive [DATA_CARD.md](DATA_CARD.md).
- Developed reusable, project-root relative data loader and validator in `src/data/` (`load_ai4i_dataset`, `validate_ai4i_dataframe`).
- Executed end-to-end EDA notebook [01_dataset_exploration.ipynb](../notebooks/01_dataset_exploration.ipynb).
- Generated and saved publication-quality analytical plots into `images/` (class distribution, sensor KDEs, correlation heatmap, failure modes, power boundary).
- Implemented and passed automated test suite in `tests/test_data_loader.py` (7/7 tests passing).

### Phase 3: Preprocessing, Feature Engineering & Pipeline Construction (Completed)
- Audited target-to-failure mode discrepancies, preserving original labels and formalizing prediction contracts in `src/features/schema.py`.
- Formulated strict exclusion of prohibited columns (`UDI`, `Product ID`, `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`).
- Implemented immutable physical feature engineering in `src/features/engineering.py`:
  - `temperature_difference = Process temperature [K] - Air temperature [K]` (Kelvin [K])
  - `mechanical_power_w = Torque [Nm] * Rotational speed [rpm] * (2 * pi / 60)` (Watts [W], shaft mechanical power)
- Built reusable scikit-learn preprocessing pipeline via `ColumnTransformer` in `src/features/preprocessing.py`:
  - Numeric features: median imputation + standard scaling.
  - Categorical feature (`Type`): most-frequent imputation + one-hot encoding with unknown handling.
- Implemented deterministic stratified train/test split utility in `src/features/split.py` guaranteeing test set isolation.
- Created and passed comprehensive test suite in `tests/test_features_and_preprocessing.py` (15/15 tests passing across project).

### Phase 4: Baseline Model Training & Evaluation (Completed)
- Embedded preprocessing inside scikit-learn `Pipeline` to prevent cross-validation fold leakage.
- Implemented and evaluated three baseline models on training partition (8,000 samples) via stratified 5-fold CV:
  1. `DummyClassifier` (majority strategy): baseline floor (Mean CV Avg Precision: 0.0339).
  2. `LogisticRegression` (`class_weight='balanced'`): linear baseline (Mean CV Avg Precision: 0.4835).
  3. `RandomForestClassifier` (`class_weight='balanced'`, `max_depth=10`, `n_estimators=100`): non-linear ensemble (Mean CV Avg Precision: 0.8521).
- Selected top model using primary metric: mean cross-validated Average Precision.
- Optimized decision threshold ($\tau^* = 0.6732$) strictly on training out-of-fold (OOF) probabilities without test set leakage.
- Evaluated selected model once on held-out test set (2,000 samples): Average Precision 0.8394, ROC-AUC 0.9754, F1 0.8346, Recall 0.7794, Precision 0.8983.
- Persisted serialized pipeline (`models/selected_model.joblib`), comparison table (`outputs/model_comparison.csv`), test metrics (`outputs/test_metrics.json`), confusion matrix (`outputs/confusion_matrix.csv`), and model card (`docs/MODEL_CARD.md`).
- Added 7 unit tests in `tests/test_modeling.py` (22/22 total passing tests).

### Phase 5: Model Explainability, Error Analysis & Failure-Mode Exploration (Completed)
- Corrected evaluation documentation to clarify single test set evaluation at predetermined reference thresholds ($0.50$ default, $\tau^* = 0.6732$ training-OOF derived).
- Implemented permutation importance on held-out test data (`scoring='average_precision'`), documenting predictive reliance vs physical causality and collinearity between engineered and raw features (`Rotational speed`: 0.4575, `temperature_difference`: 0.2675, `Tool wear`: 0.2361).
- Conducted granular error analysis on test predictions:
  - Audited 15 False Negatives: dominated by Tool Wear Failure (9 cases) and unattributed failures (2 cases).
  - Audited 6 False Positives: concentrated at high-torque boundary regimes and one simulated random failure (`RNF=1`).
- Executed exploratory failure mode analysis across `TWF`, `HDF`, `PWF`, `OSF`, `RNF`:
  - Documented why `RNF` is unviable for classifier modeling (0.1% independent random Bernoulli trial, 15 training positives).
  - Evaluated exploratory CV baselines for viable modes (`PWF`: AP 0.9992, `HDF`: AP 0.9841, `OSF`: AP 0.8225, `TWF`: AP 0.0613).
- Persisted diagnostic artifacts: `permutation_importance.csv`, `permutation_importance.png`, `test_predictions.csv`, `error_analysis_summary.json`, `error_analysis.png`, `failure_mode_comparison.csv`, and `failure_mode_analysis.json`.
- Added 5 unit tests in `tests/test_diagnostics_and_explainability.py` (27/27 total tests passing).

### Phase 6: Real-Time Stream Ingestion & Inference Runtime (Completed)
- Implemented modular, virtual telemetry simulator (`src/streaming/simulator.py`) replaying AI4I benchmark records with deterministic seeding, sequential or shuffled playback, and domain-informed physical sanity checks (`SENSOR_PHYSICAL_BOUNDS`).
- Implemented stateful, bounded sliding-window FIFO buffer (`src/streaming/buffer.py`) with deep-copy immutability and diagnostic rolling statistics (`mean`, `min`, `max`, sample `std` with `ddof=1`). Enforced rule that diagnostic rolling statistics are for operator visibility only and must never contaminate instantaneous model features.
- Implemented single-load low-latency streaming inference service (`src/streaming/inference.py`), reusing Phase 4 `selected_model.joblib` pipeline and domain feature engineering without model reloading or serving-time retraining.
- Formulated 4-tier operational alert classification (`src/streaming/alerts.py`): `NORMAL` ($<0.30$), `WATCH` ($0.30 - 0.50$), `WARNING` ($0.50 - 0.6732$), and `CRITICAL` ($\ge 0.6732$), documented separately from the raw classifier probability.
- Formulated synthetic demonstration Health Index ($100 \times (1 - P(\text{failure}))$), with explicit documentation that it is an operational demonstration indicator, NOT a certified remaining useful life or physical fatigue estimator.
- Implemented high-resolution monotonic inference latency measurement (`time.perf_counter()`), measuring median P50 latency of `19.78 ms` and P95 latency of `32.93 ms`.
- Created centralized configuration (`src/config.py`) establishing a single source of truth for decision thresholds, operational bounds, and alert cutoffs.
- Implemented comprehensive automated test suite (`tests/test_streaming.py`) with 17 unit tests (44/44 total project tests passing).
- Created finite demonstration CLI runner in `src/streaming/demo.py` and integrated into `main.py`.

### Phase 7: Interactive Dashboard & Alerting System (Completed)
- Developed an interactive, beginner-friendly Streamlit application in [`app.py`](../app.py) demonstrating virtual telemetry streaming, sliding-window buffer diagnostics, saved-model inference, and demonstration operational alerts.
- Structured into 4 distinct functional tabs:
  1. *Fleet / Machine Overview*: Real-time status badge (`NORMAL`, `WATCH`, `WARNING`, `CRITICAL`), model failure probability gauge, synthetic demonstration Health Index ($100 \times (1 - P(\text{fail}))$), cumulative processed count, alert distribution breakdown, and instantaneous sensor telemetry metric cards.
  2. *Live Telemetry Simulation*: Step-by-step single sample stepping, batch advancement (5, 10, 25 samples), replay mode toggling (sequential vs. reproducibly shuffled with seed), reset control, and recent telemetry inspection table (last 25 records).
  3. *Sensor Trends & Buffer Diagnostics*: Four synchronized Plotly charts (Spindle Dynamics, Thermal Profile, Tool Wear & Mechanical Power, and Model Risk Trajectory with alert/threshold reference lines) plus rolling window summary statistics table (`mean`, `min`, `max`, `std`). Explicitly warns operators that window statistics are for visual diagnostics only and are not fed to the model.
  4. *Model Diagnostics & Benchmark Limitations*: AI4I 2020 benchmark specifications, held-out test evaluation metrics ($\tau^* = 0.6732$), permutation feature importance chart, empirical inference latency benchmarks ($P_{50} = 19.33\text{ ms}$, $P_{95} = 53.84\text{ ms}$), and transparent scientific and physical limitations.
- Built reusable dashboard architecture in `src/dashboard/`:
  - [`src/dashboard/state.py`](../src/dashboard/state.py): Implemented `DashboardSession` state orchestrator wrapping `TelemetrySimulator`, `SlidingWindowBuffer`, and `StreamingInferenceService`. Prevents duplicate ingestion on Streamlit UI reruns.
  - [`src/dashboard/charts.py`](../src/dashboard/charts.py): Implemented 5 interactive Plotly chart builders handling empty and single-sample states safely.
  - [`src/dashboard/metrics.py`](../src/dashboard/metrics.py): Loads static held-out test evaluation metrics, permutation importance tables, and Phase 6 empirical latency benchmark dictionary.
  - [`src/dashboard/__init__.py`](../src/dashboard/__init__.py): Dashboard package exports.
- Maintained zero retraining or resaving of `models/selected_model.joblib`; loaded threshold strictly via `src/config.py`.
- Implemented comprehensive automated test suite in [`tests/test_dashboard.py`](../tests/test_dashboard.py) with 12 tests covering session initialization, stepping idempotency, batch execution, buffer capacity, dynamic window resizing, empty/populated chart generation, retraining guards, and headless `AppTest` smoke execution.
- Total test suite count increased to 56/56 passing tests.

### Phase 8: End-to-End Testing, Packaging & Final Verification
- Execute comprehensive `pytest` test suite covering data pipeline, models, inference, and dashboard.
- Finalize documentation, run-books, and architecture diagrams.
- Verify production readiness.


