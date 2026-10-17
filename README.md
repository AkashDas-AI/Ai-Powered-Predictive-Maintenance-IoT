# AI-Powered Predictive Maintenance for IoT Devices

An end-to-end machine learning system for equipment failure prediction, virtual telemetry streaming, sliding-window diagnostic buffering, and interactive monitoring built on the UCI AI4I 2020 Predictive Maintenance benchmark dataset.

---

## 1. Project Overview

Industrial machinery failures cause costly unplanned downtime, safety hazards, and reactive maintenance expenses. Predictive maintenance leverages sensor telemetry to identify conditions preceding failure before catastrophic breakdown occurs.

This project formulates predictive maintenance as a binary classification task: given machine operational parameters, predict the probability that a machine failure will occur. The repository covers the full lifecycle from data validation and leakage-safe preprocessing through ensemble modeling, error analysis, virtual streaming ingestion, and an interactive Streamlit monitoring dashboard.

This is an educational and engineering simulation. It demonstrates how to decouple streaming ingestion from machine learning inference, handle severe class imbalance, and evaluate models without data leakage.

---

## 2. Key Features

- Leakage-Safe Feature Engineering: Derives physical indicators (temperature differential, shaft mechanical power) with strict exclusion of post-hoc failure mode flags and arbitrary identifiers.
- Production-Grade ML Pipeline: Encapsulates median imputation, standard scaling, and one-hot encoding inside Scikit-Learn pipelines to prevent cross-validation leakage.
- Resampled Cross-Validation Comparison: Compares Dummy, Logistic Regression, and Balanced Random Forest candidates across stratified 5-fold cross-validation.
- Out-of-Fold Threshold Optimization: Optimizes the decision threshold on training out-of-fold predictions to maximize F1 score, keeping the test set isolated.
- Model Explainability and Error Audit: Uses permutation feature importance on held-out test data and audits false negatives against underlying mechanical failure modes.
- Virtual Telemetry Streaming: Simulates edge sensor ingestion with physical range checks and sequence tracking.
- Bounded Sliding-Window Buffer: Maintains a FIFO buffer with diagnostic rolling statistics (mean, min, max, standard deviation) for operator visibility. Rolling stats are strictly decoupled from model inputs.
- Low-Latency Inference Service: Preloads the serialized model pipeline into memory, delivering sub-50 ms median inference latency.
- Interactive Streamlit Dashboard: Multi-tab web application providing live simulation stepping, risk gauges, synchronized Plotly sensor charts, and model diagnostics.

---

## 3. Technology Stack

- Language: Python 3.11
- Core Data Science: NumPy 1.26.4, Pandas 2.2.3, SciPy 1.14.1
- Machine Learning: Scikit-Learn 1.5.2, Joblib 1.4.2
- Interactive Dashboard: Streamlit 1.39.0, PyArrow 18.1.0, Plotly 5.24.1
- Testing: Pytest 8.3.3
- Documentation: Markdown, KaTeX, Git

---

## 4. Architecture and Workflow

The system implements a dual-plane architecture that strictly decouples offline batch model development from real-time streaming telemetry and low-latency inference:

```
========================================================================================================================
                                      HIGH-LEVEL SYSTEM ARCHITECTURE OVERVIEW
========================================================================================================================

  [ PLANE 1: OFFLINE BATCH TRAINING & EVALUATION PIPELINE ]
  +------------------------------------------------------------------------------------------------------------------+
  | 1. Raw Dataset Ingestion: UCI AI4I 2020 Predictive Maintenance Benchmark (10,000 samples x 14 columns)           |
  |    |-- Validation Layer (src/data/loader.py, validator.py, schema.py): Null/Duplicate/Range Assertions          |
  |    \-- Target: Machine failure (96.61% Normal [0] vs. 3.39% Failure [1] | Imbalance Ratio: 28.5 to 1)            |
  |                                                                                                                  |
  | 2. Deterministic Stratified Split (src/features/split.py): random_state=42                                       |
  |    |-- Training Partition: N = 8,000 (80% | 271 Failures) ---------------------+                                 |
  |    \-- Held-Out Test Partition: N = 2,000 (20% | 68 Failures) -------------+   | (Fold/Test Isolation Guard)     |
  |                                                                           |   |                                  |
  | 3. Domain Physics Feature Engineering (src/features/engineering.py)       |   |                                  |
  |    |-- Thermal Differential: Delta T = Process Temp [K] - Air Temp [K]     |   |                                  |
  |    \-- Shaft Mechanical Power: P_mech = Torque [Nm] * (RPM * 2*pi / 60)    |   |                                  |
  |                                                                           |   |                                  |
  | 4. Preprocessing & Model Pipeline (src/features/preprocessing.py)         |   |                                  |
  |    |-- ColumnTransformer: SimpleImputer(median) + StandardScaler (Numeric)|   |                                  |
  |    |                      SimpleImputer(frequent) + OneHotEncoder (Type)  |   |                                  |
  |    |-- Classifier: BalancedRandomForestClassifier(n_estimators=100)       |   |                                  |
  |    \-- Feature Contract: 8 Instantaneous Inputs (Prohibited: UDI,         |   |                                  |
  |                          Product ID, Target, and 5 Failure Flags)         |   |                                  |
  |                                                                           |   |                                  |
  | 5. Model Selection & Threshold Optimization (src/train.py)                |   |                                  |
  |    |-- Stratified 5-Fold Cross-Validation on Train:                       |   |                                  |
  |    |   Balanced Random Forest (Mean CV AP = 0.8521, ROC-AUC = 0.9786)     |   |                                  |
  |    \-- Out-of-Fold Threshold Tuning: Maximize F1 without test leak        |   |                                  |
  |        Optimal Decision Threshold: tau* = 0.6732 (OOF F1 = 0.7984)        |   |                                  |
  |                                                                           |   |                                  |
  | 6. Production Model Serialization: models/selected_model.joblib <---------+---+                                  |
  |                                                                           |                                      |
  | 7. Single Held-Out Evaluation (src/evaluate.py, diagnostics.py): <--------+                                      |
  |    |-- Evaluated ONCE at tau* = 0.6732 (outputs/test_metrics.json, confusion_matrix.csv)                         |
  |    \-- Results: AP: 0.8394 | ROC-AUC: 0.9754 | F1: 0.8346 | Precision: 89.83% | Recall: 77.94% | FPR: 0.31%    |
  +--------------------------------------------+---------------------------------------------------------------------+
                                               | Pipeline Serialized (.joblib)
                                               v Preloaded ONCE into Memory
  [ PLANE 2: REAL-TIME STREAMING TELEMETRY & MONITORING RUNTIME ]
  +--------------------------------------------+---------------------------------------------------------------------+
  |                                                                                                                  |
  |   [Virtual Telemetry Simulator] (src/streaming/simulator.py)                                                     |
  |   - Replays benchmark test partition records sequentially or with reproducible seeded shuffle                    |
  |   - Generates simulated point-in-time sensor packets with physical range and sensor validity assertions          |
  |                                    | Stream Packet: {seq_id, timestamp, Type, 5 raw sensor readings}             |
  |                                    v                                                                             |
  |   [Bounded Sliding-Window Buffer] (src/streaming/buffer.py)                                                      |
  |   - Thread-safe FIFO circular buffer (capacity W = 10, collections.deque) with deep-copy immutability           |
  |   - Computes rolling window statistics (mean, min, max, sample std ddof=1) across numerical channels            |
  |   * DECOUPLING GUARD: Rolling statistics are for dashboard display only; NEVER passed to model feature matrix.  |
  |                                    | Instantaneous 8-Feature Vector (Type, 5 Raw Sensors, Delta T, P_mech)        |
  |                                    v                                                                             |
  |   [Streaming Inference Service] (src/streaming/inference.py)                                                     |
  |   - Low-latency inference wrapper preloading serialized pipeline in memory                                       |
  |   - Deterministic row-wise physics calculation; asserts all prohibited columns excluded                         |
  |   - Evaluates predict_proba() -> failure probability | Latency benchmark: < 30 ms median (sub-50 ms target met)  |
  |                                    | Inference Payload: {probability, alert_state, health_index, latency_ms}     |
  |                                    v                                                                             |
  |   [Alert Engine & Health Classifier] (src/streaming/alerts.py)                                                   |
  |   - Operational Alert Tiers: NORMAL (<0.30) | WATCH (0.30-0.50) | WARNING (0.50-0.6732) | CRITICAL (>=0.6732)   |
  |   - Hysteresis logic and cooldown counters to suppress alert fatigue                                             |
  |   - Demonstration Health Index = 100.0 * (1.0 - P(failure))                                                      |
  |                                    |                                                                             |
  |                                    v                                                                             |
  |   [Interactive Streamlit Monitoring Dashboard] (app.py)                                                          |
  |   - Orchestrated by session state manager (src/dashboard/state.py) with defensive table fallbacks                |
  |   - Tab 1: Fleet & Machine Overview (Operational badge, failure probability gauge, instantaneous sensor cards)   |
  |   - Tab 2: Live Telemetry Simulation (Step single, batch runner 5/10/25, replay ordering, event history)         |
  |   - Tab 3: Sensor Trends & Buffer Diagnostics (Synchronized Plotly charts, active rolling statistics)            |
  |   - Tab 4: Model Diagnostics & Scorecard (Held-out test metrics, permutation importance, scientific limits)      |
  |                                                                                                                  |
  +------------------------------------------------------------------------------------------------------------------+
========================================================================================================================
```

For detailed specifications, see the documentation in the `docs/` directory:
- [System Architecture](docs/architecture.md)
- [Data Pipeline and Flow](docs/data_pipeline_and_flow.md)
- [Module and Component Contracts](docs/modules_and_components.md)

---

## 5. Dataset Source and Limitations

- Benchmark: UCI AI4I 2020 Predictive Maintenance Dataset (UCI ID: 601)
- Source: Donated by Stephan Matzka, School of Engineering, HTW Berlin
- Dimensions: 10,000 rows by 14 columns
- Target: `Machine failure` (0 = Normal, 1 = Failure)
- Class Distribution: 9,661 normal records (96.61%) and 339 failure records (3.39%), an imbalance ratio of 28.5 to 1.
- Input Features Used: Machine Type (L, M, H), Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], and Tool wear [min].
- Engineered Predictors:
  - Thermal Differential: `temperature_difference = Process temperature [K] - Air temperature [K]`
  - Shaft Mechanical Power: `mechanical_power_w = Torque [Nm] * (Rotational speed [rpm] * 2 * pi / 60)`
- Excluded Columns (Leakage Guards):
  - Identifiers: `UDI` and `Product ID` are excluded as non-generalizable indices.
  - Failure Mode Flags: `TWF`, `HDF`, `PWF`, `OSF`, `RNF` are post-hoc failure attribution flags and are strictly excluded to prevent label leakage.
- Dataset Limitations:
  - The dataset consists of simulated experimental runs rather than continuous factory sensor time series.
  - Tool Wear Failure (TWF) is simulated probabilistically between 200 and 240 minutes of wear while instantaneous sensor readings remain nominal, making TWF difficult to detect with point-in-time classifiers.
  - For full details, see the [Dataset Card](docs/DATA_CARD.md).

---

## 6. Model Approach and Key Evaluation Results

Due to severe class imbalance, accuracy is misleading (a majority classifier scores 96.61% accuracy while finding zero failures). The primary selection metric is **Average Precision (PR-AUC)**.

### Model Candidates Comparison (5-Fold CV on 8,000 Training Samples)

| Model Candidate | Mean CV Avg Precision | Std CV Avg Precision | OOF ROC-AUC | OOF F1-Score | OOF Recall | OOF Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random_Forest_Balanced** (Selected) | **0.8521** | 0.0328 | **0.9762** | **0.7365** | 0.8044 | 0.6791 |
| Logistic_Regression_Balanced | 0.4835 | 0.0254 | 0.9187 | 0.2738 | 0.8229 | 0.1642 |
| Dummy_Majority (Floor Baseline) | 0.0339 | 0.0002 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |

### Threshold Selection
The decision threshold was tuned strictly on training out-of-fold (OOF) predicted probabilities to maximize F1 score, yielding `tau* = 0.6732`. The test set was never used to select or tune thresholds.

### Held-Out Test Set Performance (2,000 Samples, 68 Failures)

| Metric | Baseline Threshold (0.50) | Selected Threshold (0.6732) |
| :--- | :---: | :---: |
| **Average Precision (PR-AUC)** | 0.8394 | 0.8394 |
| **ROC-AUC** | 0.9754 | 0.9754 |
| **F1-Score** | 0.7317 | **0.8346** |
| **Precision** | 0.6250 | **0.8983** |
| **Recall (Sensitivity)** | 0.8824 | 0.7794 |
| **True Positives (TP)** | 60 | 53 |
| **True Negatives (TN)** | 1,896 | 1,926 |
| **False Positives (FP)** | 36 | **6** |
| **False Negatives (FN)** | 8 | 15 |

At the selected threshold of 0.6732, false positives dropped from 36 to 6, increasing precision to 89.8% while maintaining 77.9% recall. For deeper analysis, see the [Model Card](docs/MODEL_CARD.md).

---

## 7. Visual Evidence and System Showcase

This section compiles visual evidence from dataset exploration, model evaluation, the live dashboard, and terminal telemetry streaming.

### Dataset and Exploratory Analysis
![Dataset Class Imbalance](images/target_distribution.png)

*Figure 1: Binary target distribution in the UCI AI4I 2020 dataset showing severe class imbalance (96.61% normal vs. 3.39% failure).*

![Correlation Matrix Heatmap](images/correlation_matrix.png)

*Figure 2: Correlation heatmap across continuous sensor channels and failure indicators.*

![Physics-Informed Feature Engineering](images/feature_engineering.png)

*Figure 3: First-principles physical derivations for thermal differential and shaft mechanical power alongside strict leakage exclusion boundaries.*

### Model Evaluation and Explainability
![Held-Out Test Confusion Matrix](images/confusion_matrix.png)

*Figure 4: Held-out test set confusion matrix (N=2,000) at optimized threshold tau* = 0.6732, achieving 89.83% precision, 77.94% recall, and 0.31% false positive rate.*

![Permutation Feature Importance](images/permutation_importance.png)

*Figure 5: Permutation feature importance on test data (scoring: average precision), identifying rotational speed and thermal differential as dominant predictors.*

![Error Analysis by Failure Mode](images/error_analysis.png)

*Figure 6: Forensic audit of test false negatives and false positives broken down across mechanical failure modes.*

### Streamlit Dashboard
The interactive Streamlit dashboard (`app.py`) provides an operator interface with 4 dedicated functional tabs:

#### Fleet and Machine Overview
Real-time operational status badge (NORMAL, WATCH, WARNING, CRITICAL), failure probability gauge bar, synthetic demonstration health index, cumulative processed sample counts, and instantaneous sensor metrics:

![Dashboard Overview](images/dashboard_overview.png)

#### Live Telemetry Simulation
Interactive stepping controls (Next Sample, Run Batch of 5, 10, or 25), replay ordering toggle (sequential vs. reproducibly shuffled), dynamic buffer capacity slider, and recent event history:

![Live Telemetry Simulation](images/dashboard_live_simulation.png)

#### Sensor Trends and Buffer Diagnostics
Synchronized Plotly charts tracking Spindle Dynamics (Speed vs. Torque), Thermal Profile (Air, Process, Delta T), Tool Wear vs. Mechanical Power, and Model Risk Trajectory alongside rolling window summary statistics:

![Sensor Trends and Diagnostics](images/dashboard_sensor_trends.png)

#### Model Diagnostics and Limitations
Held-out test scorecard, permutation feature importance bar chart, empirical inference latency benchmarks, and transparent discussion of scientific limitations:

![Model Diagnostics and Limitations](images/dashboard_model_diagnostics.png)

### Virtual Telemetry Demonstration
Execution of the streaming simulator CLI (`python main.py --samples 5`):

```text
==========================================================================================
  IoT Telemetry Streaming & Low-Latency Inference Demonstration (Phase 6)
==========================================================================================
Total Stream Samples: 5 | Buffer Capacity: 10 | Decision Threshold: 0.6732
Simulation Mode: Replaying AI4I benchmark records (Software simulation only)
------------------------------------------------------------------------------------------
Seq   | Type | Temp Diff [K] | Speed [rpm] | Torque [Nm] | P(Fail)  | Alert    | HI (%)  | Latency  
------------------------------------------------------------------------------------------
1     | M    | 10.50         | 1551        | 42.8        | 0.0019   | NORMAL   | 99.8    | 141.83 ms
2     | L    | 10.50         | 1408        | 46.3        | 0.0141   | NORMAL   | 98.6    | 32.15 ms
3     | L    | 10.40         | 1498        | 49.4        | 0.0377   | NORMAL   | 96.2    | 20.32 ms
4     | L    | 10.40         | 1433        | 39.5        | 0.0076   | NORMAL   | 99.2    | 21.68 ms
5     | L    | 10.50         | 1408        | 40.0        | 0.0076   | NORMAL   | 99.2    | 29.98 ms
------------------------------------------------------------------------------------------
Final Buffer Rolling Diagnostics (Last 10 Records - For Operator Display Only):
  - Air temperature [K]: Mean=298.16, Min=298.1, Max=298.2, Std=0.0548
  - Process temperature [K]: Mean=308.62, Min=308.5, Max=308.7, Std=0.0837
  - Rotational speed [rpm]: Mean=1459.6, Min=1408.0, Max=1551.0, Std=62.9706
  - Torque [Nm]: Mean=43.6, Min=39.5, Max=49.4, Std=4.2232
  - Tool wear [min]: Mean=4.8, Min=0.0, Max=9.0, Std=3.4928
------------------------------------------------------------------------------------------
Performance Measurement (Local Inference Latency):
  - Sample Count: 5
  - Median (P50) Latency: 29.977 ms
  - 95th Percentile (P95): 119.891 ms
  - Mean Latency:         49.190 ms (Min: 20.321 ms, Max: 141.827 ms)
  - Alert Distribution:   {'NORMAL': 5}
==========================================================================================
```

---

## 8. Installation and Quickstart

### Prerequisites
- Python 3.11 installed.
- Git installed.

### Step 1: Clone the Repository
```bash
git clone https://github.com/AkashDas-AI/Ai-Powered-Predictive-Maintenance-IoT.git
cd Ai-Powered-Predictive-Maintenance-IoT
```

### Step 2: Create and Activate Virtual Environment

On Windows (PowerShell):
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify dependency consistency:
```bash
python -m pip check
```
Expected output: `No broken requirements found.`

---

## 9. Verification and Execution Commands

### Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Or with explicit virtual environment Python:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Run CLI Telemetry Streaming Demo
```bash
# Default (25 samples, window 10)
python main.py

# Custom stream count and buffer capacity
python main.py --stream --samples 50 --window 10
```

### Run Complete Test Suite (56 tests)
```bash
pytest tests/ -v
```

### Run Model Training and Diagnostics Workflows
```bash
# Rerun cross-validation training pipeline
python src/train.py

# Rerun permutation importance and error analysis
python src/diagnostics.py
```

---

## 10. Important Limitations and Future Scope

### Project Limitations
1. Software Simulation: This is an educational demonstration using benchmark data. It does not connect to physical factory PLCs or industrial control loops.
2. Independent Experimental Runs: The dataset consists of independent static trials rather than continuous degradation time series of a physical cutting tool.
3. Demonstration Health Index: The Health Index formula `100 * (1 - P(failure))` is an inverted model probability. It is not an engineered or physics-certified Remaining Useful Life (RUL) estimator.
4. Tool Wear Failure Ceiling: Instantaneous sensor values remain normal during TWF events. High-frequency vibration or acoustic sensing is required for physical tool wear detection.
5. Random Failures (RNF): Simulated as an independent random trial (0.1% probability), making it statistically unviable for predictive modeling.

### Future Scope
- Sequence Prognostics: Implement LSTM and Transformer regression models on NASA C-MAPSS turbofan run-to-failure degradation trajectories.
- Multimodal Sensing: Ingest high-frequency vibration and accelerometer data to overcome the tool wear detection ceiling.
- Unsupervised Novelty Detection: Add Autoencoder or Isolation Forest pipelines to catch operational drift.
- Edge Containerization: Package inference runtime into Docker containers with ONNX Runtime for edge IoT deployment.

---

## 11. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for the complete license text:

```text
MIT License
Copyright (c) 2026 Akash Das
```

*Note on Third-Party Data: The UCI AI4I 2020 dataset is distributed under Creative Commons Attribution 4.0 International (CC BY 4.0). The MIT License covers project code, pipeline architecture, and documentation, and does not override original dataset licensing.*

---

## 12. References and Dataset Attribution

- Stephan Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," Third International Conference on Artificial Intelligence for Industries (AI4I), 2020.
- UCI Machine Learning Repository: AI4I 2020 Predictive Maintenance Dataset (https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).
- Scikit-Learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
- Streamlit Documentation: https://docs.streamlit.io/
- Project Implementation History: See [Execution Log](docs/execution_log.md) and [Project Plan](docs/PROJECT_PLAN.md).
