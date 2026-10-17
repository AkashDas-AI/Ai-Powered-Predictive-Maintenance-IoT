# Project Execution Log

A living record documenting each phase of implementation, architectural decisions, and validation results.

---

## Phase 0: Documentation & Technical Blueprint Initialization
- **Timestamp**: 2026-10-17
- **Implemented Components**:
  - Initialized dedicated technical documentation directory `docs/`.
  - Authored [Index & Overview](README.md).
  - Authored [System Architecture](architecture.md).
  - Authored [Data Pipeline & Flow](data_pipeline_and_flow.md).
  - Authored [Dataset Specifications](dataset_specifications.md).
  - Authored [Modules & Components](modules_and_components.md).
  - Authored [Model Training & Learning Mechanics](model_training_and_learning.md).
  - Authored [Validation & Testing Protocol](validation_and_testing_protocol.md).
- **Validation**:
  - Structural validation completed: all documentation paths and cross-references verified.
- **Status**: Completed.

---

## Phase 1: Project Initialization & Virtual Environment
- **Timestamp**: 2026-10-17
- **Environment**:
  - Operating System: Windows 11 (Build 10.0.26100)
  - Python: 3.11.0 (64-bit) at `.venv\Scripts\python.exe`
  - Shell: PowerShell 5.1
  - Package Manager: pip
- **Implemented Components**:
  - Root directory structure established (`data/`, `notebooks/`, `src/`, `models/`, `outputs/`, `images/`, `docs/`).
  - Added `.gitkeep` placeholders to preserve empty directories in version control.
  - Initialized `src/__init__.py`.
  - Authored comprehensive project [Roadmap](PROJECT_PLAN.md).
  - Authored root [README.md](../README.md).
  - Configured [.gitignore](../.gitignore) protecting against virtual environments, caches, notebooks, and models.
  - Specified direct dependencies in [requirements.txt](../requirements.txt).
  - Implemented initial entry point [main.py](../main.py).
  - Created `.venv` virtual environment and installed dependencies.
- **Validation**:
  - Environment activation & path verification (`.venv\Scripts\python.exe` -> Python 3.11.0, pip 26.3).
  - Dependency import verification: all 10 target packages (`numpy`, `pandas`, `sklearn`, `matplotlib`, `seaborn`, `joblib`, `jupyter`, `streamlit`, `plotly`, `pytest`) verified successfully.
  - Entry-point verification: `main.py` executed cleanly with zero errors.
- **Status**: Completed & Verified. Ready for Phase 2.


---

## Phase 2: Dataset Ingestion & Exploratory Data Analysis (EDA)
- **Timestamp**: 2026-10-17
- **Dataset**: UCI AI4I 2020 Predictive Maintenance Dataset (`https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset`)
- **Implemented Components**:
  - Downloaded official raw archive into [`data/raw/ai4i2020.zip`](../data/raw/ai4i2020.zip) and extracted [`data/raw/ai4i2020.csv`](../data/raw/ai4i2020.csv) (10,000 rows x 14 columns).
  - Authored detailed dataset card at [`docs/DATA_CARD.md`](DATA_CARD.md).
  - Implemented schema definitions & validation report dataclass in [`src/data/schema.py`](../src/data/schema.py).
  - Implemented strict data contract validator in [`src/data/validator.py`](../src/data/validator.py).
  - Implemented reusable data loader with project-root relative path resolution in [`src/data/loader.py`](../src/data/loader.py).
  - Created and fully executed Jupyter notebook [`notebooks/01_dataset_exploration.ipynb`](../notebooks/01_dataset_exploration.ipynb).
  - Generated and saved visual analytical artifacts into `images/` (`target_distribution.png`, `sensor_distributions.png`, `correlation_matrix.png`, `failure_modes_breakdown.png`, `torque_vs_rotational_speed.png`).
  - Persisted structured metrics summary into [`outputs/eda_summary.json`](../outputs/eda_summary.json).
  - Developed test suite in [`tests/test_data_loader.py`](../tests/test_data_loader.py) with 7 test cases covering valid loading, missing files, missing columns, missing values, duplicate detection, and invalid target handling.
  - Updated [`main.py`](../main.py) to execute data ingestion and validation workflow.
- **Validation**:
  - `pytest tests/ -v`: 7 passed in 0.79s.
  - `python main.py`: Executed cleanly, reporting 10,000 rows, 14 columns, 0 nulls, 0 duplicates, 9661 normal (96.61%), 339 failures (3.39%).
  - Notebook execution: Ran through `nbconvert --execute` with zero errors, writing all evaluated outputs and plots.
- **Status**: Completed & Verified. Ready for Phase 3.

---

## Phase 3: Preprocessing, Feature Engineering & Pipeline Construction
- **Timestamp**: 2026-10-17
- **Implemented Components**:
  - Audited target-to-failure mode discrepancies (`339` failures vs `373` mode flags; 9 unattributed failures, 18 random non-failure triggers). Confirmed original labels preserved untouched without artificial synthesis.
  - Formalized prediction contract and column exclusion policy in [`src/features/schema.py`](../src/features/schema.py). Prohibited columns: `['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']`.
  - Implemented immutable feature engineering in [`src/features/engineering.py`](../src/features/engineering.py):
    - `temperature_difference = Process temperature [K] - Air temperature [K]` (Kelvin [K])
    - `mechanical_power_w = Torque [Nm] * Rotational speed [rpm] * (2 * pi / 60)` (Watts [W], shaft mechanical power)
  - Built reusable, unfitted scikit-learn preprocessing pipeline via `ColumnTransformer` in [`src/features/preprocessing.py`](../src/features/preprocessing.py) (median imputation + standard scaling for numeric; most-frequent + one-hot encoding with unknown category ignoring for categorical).
  - Developed reproducible, deterministic stratified splitting utility in [`src/features/split.py`](../src/features/split.py) preserving the 3.39% class balance (80/20 train/test split: 8,000 train / 2,000 test) while isolating the test set.
  - Implemented 8 dedicated unit tests in [`tests/test_features_and_preprocessing.py`](../tests/test_features_and_preprocessing.py) covering formula accuracy, unit consistency, DataFrame immutability, prohibited column exclusions, unknown category handling, reproducibility, and leakage isolation.
- **Validation**:
  - Full test suite: `15 passed in 2.34s` via `pytest tests/ -v`.
  - Feature transformation verified: 8 input predictive features mapped to 10 scaled & encoded numeric features.
  - Zero data leakage verified: scalers fitted strictly on training data; test set isolated.
- **Status**: Completed & Verified. Ready for Phase 4 (Model Development).

---

## Phase 4: Baseline Model Training and Evaluation
- **Timestamp**: 2026-10-17
- **Implemented Components**:
  - Implemented evaluation metrics engine in [`src/evaluate.py`](../src/evaluate.py) evaluating Average Precision, ROC-AUC, F1-Score, Precision, Recall, and Confusion Matrices for the positive failure class (`1`), along with training-only out-of-fold threshold search (`find_optimal_threshold_oof`).
  - Constructed baseline model pipelines in [`src/train.py`](../src/train.py) embedding preprocessors inside scikit-learn `Pipeline` to prevent cross-validation fold contamination.
  - Implemented and evaluated 3 baseline candidates on training partition (8,000 samples) via stratified 5-fold CV:
    1. `DummyClassifier` (majority strategy): Floor baseline.
    2. `LogisticRegression` (`class_weight='balanced'`): Linear baseline.
    3. `RandomForestClassifier` (`class_weight='balanced'`, `n_estimators=100`, `max_depth=10`): Non-linear ensemble baseline.
  - Selected `RandomForestClassifier` as top model using primary metric: mean cross-validated Average Precision (0.8521 vs 0.4835 for Logistic Regression and 0.0339 for Dummy).
  - Optimized decision threshold ($\tau^* = 0.6732$) on training out-of-fold probabilities to balance precision and recall without peeking at the test set.
  - Evaluated the selected model once on held-out test partition (2,000 samples, 68 failures): Average Precision 0.8394, ROC-AUC 0.9754, F1 0.8346, Recall 0.7794, Precision 0.8983.
  - Persisted serialized pipeline to [`models/selected_model.joblib`](../models/selected_model.joblib).
  - Saved model comparison results to [`outputs/model_comparison.csv`](../outputs/model_comparison.csv).
  - Saved test evaluation report to [`outputs/test_metrics.json`](../outputs/test_metrics.json) and confusion matrix to [`outputs/confusion_matrix.csv`](../outputs/confusion_matrix.csv).
  - Authored comprehensive Model Card in [`docs/MODEL_CARD.md`](MODEL_CARD.md).
  - Added 7 dedicated unit tests in [`tests/test_modeling.py`](../tests/test_modeling.py) verifying pipeline fitting, prohibited column guards, CV isolation, metric correctness, joblib reloading, and seed determinism.
- **Validation**:
  - Full test suite: `22 passed in 6.13s` via `pytest tests/ -v`.
  - Smoke test: `python main.py` executed cleanly end-to-end.
- **Status**: Completed & Verified. Ready for Phase 5.

---

## Phase 5: Model Explainability, Error Analysis & Failure-Mode Exploration
- **Timestamp**: 2026-10-17
- **Implemented Components**:
  - Corrected and clarified evaluation documentation in [`docs/MODEL_CARD.md`](MODEL_CARD.md), confirming that cross-validation metrics were evaluated at the standard default threshold ($0.50$), decision threshold tuning ($\tau^* = 0.6732$) was performed strictly on training out-of-fold predictions, and the held-out test partition was evaluated once on the single generated prediction vector at both predetermined reference thresholds without post-hoc tuning.
  - Implemented permutation importance module in [`src/diagnostics.py`](../src/diagnostics.py) using Average Precision drop across 10 repeats on the test set. Top features: `Rotational speed [rpm]` (0.4575), `temperature_difference` (0.2675), `Tool wear [min]` (0.2361), `Torque [Nm]` (0.1929), `mechanical_power_w` (0.1708). Documented predictive reliance vs. physical causality and collinearity dynamics.
  - Conducted granular error analysis on test predictions at $\tau^* = 0.6732$ (53 TP, 1,926 TN, 15 FN, 6 FP). Audited all 15 False Negatives, demonstrating that 9 correspond to Tool Wear Failure (probabilistic failure under normal operational sensor readings) and 2 correspond to unattributed failures. Audited the 6 False Positives, noting extreme mechanical load regimes and one simulated random failure flag (`RNF=1`, UDI 5490).
  - Executed exploratory failure mode analysis across all 5 modes (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`), keeping them strictly isolated from predictive features. Documented why `RNF` is unviable for classifier modeling (0.1% independent Bernoulli noise, 15 training samples). Evaluated 5-fold CV baselines for viable modes (`PWF`: AP 0.9992, `HDF`: AP 0.9841, `OSF`: AP 0.8225, `TWF`: AP 0.0613).
  - Saved diagnostic artifacts: [`outputs/permutation_importance.csv`](../outputs/permutation_importance.csv), [`images/permutation_importance.png`](../images/permutation_importance.png), [`outputs/test_predictions.csv`](../outputs/test_predictions.csv), [`outputs/error_analysis_summary.json`](../outputs/error_analysis_summary.json), [`images/error_analysis.png`](../images/error_analysis.png), [`outputs/failure_mode_comparison.csv`](../outputs/failure_mode_comparison.csv), and [`outputs/failure_mode_analysis.json`](../outputs/failure_mode_analysis.json).
  - Added 5 unit tests in [`tests/test_diagnostics_and_explainability.py`](../tests/test_diagnostics_and_explainability.py) verifying permutation schema, error categorization, record preservation, prohibited column exclusions, and rare-class sample sufficiency guards.
- **Validation**:
  - Full test suite: `27 passed in 16.97s` via `pytest tests/ -v`.
  - Diagnostics workflow: `python src/diagnostics.py` executed cleanly end-to-end.
- **Status**: Completed & Verified. Ready for Phase 6.

---

## Phase 6: Virtual Telemetry Streaming and Inference Buffer
- **Timestamp**: 2026-10-17
- **Implemented Components**:
  - Centralized project configurations, threshold loaders, sensor operating boundaries, and alert tier cutoffs into [`src/config.py`](../src/config.py). Eliminates duplicated hardcoded constants.
  - Formulated 4-tier operational alert classifier and synthetic demonstration Health Index in [`src/streaming/alerts.py`](../src/streaming/alerts.py):
    - `NORMAL`: $P < 0.30$
    - `WATCH`: $0.30 \le P < 0.50$
    - `WARNING`: $0.50 \le P < 0.6732$
    - `CRITICAL`: $P \ge 0.6732$
    - $\text{Health Index} = 100.0 \times (1.0 - P(\text{failure}))$, with explicit disclaimers regarding demonstration nature and non-RUL semantics.
  - Implemented modular, virtual telemetry generator in [`src/streaming/simulator.py`](../src/streaming/simulator.py) replaying AI4I benchmark records with manual stepping (`step()`), generator streaming (`stream()`), reproducible seeding, and domain physical boundary checks (`validate_telemetry_record`).
  - Implemented bounded sliding-window FIFO buffer in [`src/streaming/buffer.py`](../src/streaming/buffer.py) (`window_size=10`), with immutable snapshots and diagnostic rolling statistics (`mean`, `min`, `max`, sample `std` with `ddof=1`). Enforced isolation preventing window stats from contaminating instantaneous model features.
  - Implemented low-latency streaming inference service in [`src/streaming/inference.py`](../src/streaming/inference.py), loading [`models/selected_model.joblib`](../models/selected_model.joblib) once at initialization, applying domain feature engineering (`temperature_difference`, `mechanical_power_w`), enforcing prohibited-column exclusion, and measuring monotonic inference latency via `time.perf_counter()`.
  - Built finite CLI demonstration runner in [`src/streaming/demo.py`](../src/streaming/demo.py) and integrated into [`main.py`](../main.py) with `--stream`, `--samples`, `--window`, and `--full-pipeline` CLI options.
  - Created streaming package init in [`src/streaming/__init__.py`](../src/streaming/__init__.py).
  - Added 17 unit tests in [`tests/test_streaming.py`](../tests/test_streaming.py) covering buffer capacity, FIFO eviction, immutability, empty windows, simulator determinism, validation rules, output contracts, prohibited column exclusions, threshold boundaries, alert state transitions, Health Index bounds, single model loading, and latency schema.
  - Updated [`src/diagnostics.py`](../src/diagnostics.py) and [`tests/test_diagnostics_and_explainability.py`](../tests/test_diagnostics_and_explainability.py) to load the centralized threshold and prevent test runs from clobbering production artifacts.
- **Validation**:
  - Full test suite: `44 passed in 20.76s` via `pytest tests/ -v` (100% pass rate, all 27 Phase 1-5 tests + 17 Phase 6 tests).
  - Inference Latency: Sample count 25, Median $P_{50} = 19.78\text{ ms}$, $P_{95} = 32.93\text{ ms}$, Mean = $26.45\text{ ms}$ on Windows 11 / Python 3.11.0.
  - Model and Phase 4 evaluation artifacts confirmed untouched with zero retraining.
- **Status**: Completed & Verified. Ready for Phase 7 (Dashboard & Alerting System).

---

## Phase 7: Interactive Streamlit Predictive Maintenance Dashboard
- **Timestamp**: 2026-10-17
- **Implemented Components**:
  - Implemented standalone interactive Streamlit web dashboard in [`app.py`](../app.py) with 4 functional tabs:
    1. *Fleet / Machine Overview*: Status badge (`NORMAL`, `WATCH`, `WARNING`, `CRITICAL`), model failure probability bar, synthetic demonstration Health Index ($100 \times (1 - P(\text{fail}))$), cumulative processed sample counter, alert distribution tally, and instantaneous telemetry grid.
    2. *Live Telemetry Simulation*: Manual step button (`Next Telemetry Sample`), batch execution (`Run Batch` for 5, 10, or 25 samples), replay mode selector (sequential vs. reproducibly shuffled with seed), simulation reset, and recent history table (last 25 records).
    3. *Sensor Trends & Buffer Diagnostics*: Synchronized Plotly time-series charts (Spindle Dynamics, Thermal Profile, Tool Wear & Mechanical Power, and Model Risk Trajectory with threshold $\tau^* = 0.6732$ and alert reference lines) alongside rolling window summary statistics table (`mean`, `min`, `max`, `std`). Explicitly warns that window statistics are for visual diagnostics only and are not fed to the ML model.
    4. *Model Diagnostics & Benchmark Limitations*: AI4I 2020 benchmark specifications, held-out test evaluation metrics table, interactive permutation feature importance bar chart, empirical inference latency benchmark table ($P_{50} = 19.33\text{ ms}$, $P_{95} = 53.84\text{ ms}$), and transparent scientific and physical limitations.
  - Created modular dashboard package in [`src/dashboard/`](../src/dashboard):
    - [`src/dashboard/state.py`](../src/dashboard/state.py): Implemented `DashboardSession` state orchestrator wrapping `TelemetrySimulator`, `SlidingWindowBuffer`, and `StreamingInferenceService`. Prevents duplicate ingestion on UI reruns.
    - [`src/dashboard/charts.py`](../src/dashboard/charts.py): Implemented 5 interactive Plotly chart builders handling empty and single-record DataFrames safely.
    - [`src/dashboard/metrics.py`](../src/dashboard/metrics.py): Loads static held-out test evaluation metrics, permutation importance tables, and Phase 6 empirical latency benchmark dictionary.
    - [`src/dashboard/__init__.py`](../src/dashboard/__init__.py): Dashboard package exports.
  - Added 12 unit and integration tests in [`tests/test_dashboard.py`](../tests/test_dashboard.py) covering session initialization, stepping, idempotent inspection, batch execution, buffer capacity, dynamic window resizing, empty/populated charts, model retraining guards, and headless `AppTest` smoke execution.
- **Validation & Compatibility Fix**:
  - Full test suite: `56 passed in 18.70s` via `pytest tests/ -v` (100% pass rate across all 6 test modules).
  - Headless Streamlit smoke test: `AppTest` initialized, stepped through tabs, verified widgets, and verified absence of unhandled exceptions.
  - Zero retraining or resaving of `models/selected_model.joblib`; loaded threshold strictly via centralized [`src/config.py`](../src/config.py).
  - **PyArrow / NumPy Dependency Resolution**: Resolved `ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4` triggered by Streamlit's `st.dataframe()` serialization. Downgraded `pyarrow 26.0.0` to `pyarrow 18.1.0` (fully compatible with NumPy 1.26.4), pinned `pyarrow>=14.0.0,<19.0.0` in [`requirements.txt`](../requirements.txt), wrapped UI dataframes in [`app.py`](../app.py) with defensive `try/except` fallbacks to `st.table()`, and hardened [`tests/test_dashboard.py`](../tests/test_dashboard.py) to simulate active button clicks (`Next Telemetry Sample` and `Run Batch`) verifying error-free table rendering under live stream ingestion.
- **Status**: Completed, Hardened & Verified. Ready for Phase 8 (Final Verification & Packaging).



