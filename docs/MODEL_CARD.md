# Model Card: Predictive Maintenance Diagnostics & Explainability (Random_Forest_Balanced)

## 1. Model Overview
- **Selected Architecture**: `Random_Forest_Balanced` (100 estimators, `max_depth=10`, `min_samples_split=5`, `class_weight='balanced'`)
- **Target Variable**: `Machine failure` (Binary: `0` = Normal, `1` = Failure)
- **Positive Class Prevalence**: ~3.39% across dataset (28.5 : 1 class imbalance)
- **Primary Selection Metric**: Mean Cross-Validated **Average Precision**
- **Model Pipeline**: Scikit-Learn `Pipeline` embedding median/most-frequent imputation, standard scaling, one-hot encoding with unknown category handling, and the class-weighted classifier.

---

## 2. Evaluation Protocol & Methodology Clarification

> [!IMPORTANT]
> **Threshold and Test Set Integrity Policy**:
> 1. **Cross-Validation Table Threshold**: In the cross-validation comparison table below, all threshold-dependent metrics (`mean_cv_f1`, `oof_f1`, `oof_precision`, `oof_recall`) were evaluated at the **standard default threshold of 0.50** for all models.
> 2. **Threshold Tuning Isolation**: Decision threshold optimization was conducted **strictly on training out-of-fold (OOF) predicted probabilities**. Test set labels were **never** used to tune, select, or adjust the threshold.
> 3. **Single Test Evaluation**: The held-out test set ($N=2,000$) was evaluated **once**. The resulting predicted probability vector was examined at two predetermined reference points:
>    - The unadjusted standard default threshold ($\tau = 0.50$).
>    - The training-derived optimal threshold ($\tau^* = 0.6732$).
>    No model retraining, hyperparameter adjustment, or threshold re-selection was performed after viewing test results.

### Cross-Validation Model Comparison (Training Partition Only, $N=8,000$)
Evaluated across 5 stratified folds (threshold = 0.50 for F1/Precision/Recall):

| Model Candidate | Mean CV Avg Precision | Std CV Avg Precision | OOF ROC-AUC | OOF F1-Score | OOF Recall | OOF Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Random_Forest_Balanced`** (Selected) | **0.8521** | 0.0328 | **0.9762** | **0.7365** | 0.8044 | 0.6791 |
| **`Logistic_Regression_Balanced`** | 0.4835 | 0.0254 | 0.9187 | 0.2738 | 0.8229 | 0.1642 |
| **`Dummy_Majority`** (Floor Baseline) | 0.0339 | 0.0002 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |

---

## 3. Permutation Feature Importance

Permutation feature importance was measured on the held-out test partition using **Average Precision** drop across 10 repeats (`random_state=42`).

| Predictive Feature | Mean AP Drop | Std AP Drop | Physical Role & Collinearity Dynamics |
| :--- | :---: | :---: | :--- |
| **`Rotational speed [rpm]`** | **0.4575** | 0.0265 | Primary speed indicator; captures stall, low-speed overheating, and power limit boundaries. |
| **`temperature_difference`** | **0.2675** | 0.0380 | Engineered thermal gradient ($T_{\text{process}} - T_{\text{air}}$); captures Heat Dissipation Failure (`HDF`). |
| **`Tool wear [min]`** | **0.2361** | 0.0199 | Cumulative cutting time; captures Tool Wear Failure (`TWF`) and Overstrain Failure (`OSF`). |
| **`Torque [Nm]`** | **0.1929** | 0.0159 | Spindle mechanical load; crucial component of power and overstrain boundaries. |
| **`mechanical_power_w`** | **0.1708** | 0.0200 | Engineered shaft mechanical power ($\tau \cdot \omega$); isolates Power Failure (`PWF`) window. |
| **`Type`** | **0.0359** | 0.0189 | Machine quality variant (`L`, `M`, `H`); controls variant-specific stress thresholds. |
| **`Air temperature [K]`** | **0.0058** | 0.0187 | Raw ambient temperature; low marginal importance due to collinearity with `temperature_difference`. |
| **`Process temperature [K]`** | **-0.0209** | 0.0144 | Raw process temperature; redundant once thermal differential is explicitly engineered. |

### Predictive Reliance vs. Physical Causality
- **Predictive Reliance**: Permutation importance quantifies how heavily the model relies on a feature to sustain its Average Precision score on test records.
- **Not Physical Causality**: A high importance score does not mean that altering the sensor value directly causes machine breakdown in a factory.
- **Collinearity Distribution**: Because `mechanical_power_w` is a direct mathematical combination of `Torque` and `Rotational speed`, predictive importance is distributed across all three features rather than concentrated in a single column. Similarly, `temperature_difference` captures the thermal dissipation signal, rendering raw `Process temperature` and `Air temperature` marginally redundant.

---

## 4. Error Analysis & Diagnostics

Evaluated on the held-out test set ($N=2,000$, 68 failures) at the pre-selected threshold $\tau^* = 0.6732$:
- **True Positives (TP)**: **53**
- **True Negatives (TN)**: **1,926**
- **False Negatives (FN)**: **15** (Missed failures)
- **False Positives (FP)**: **6** (False alarms)

```
                       Predicted Normal (0)    Predicted Failure (1)
Actual Normal (0)             1,926                     6  (FP)
Actual Failure (1)               15  (FN)              53  (TP)
```

### Audit of False Negatives (15 Missed Failures)
Joining the post-hoc failure mode annotations reveals:
1. **Tool Wear Failure (`TWF`) Dominance (9 out of 15 FNs)**:
   - In the benchmark simulation, `TWF` triggers probabilistically when tool wear is between 200 and 240 minutes.
   - At the time of failure, instantaneous temperatures, speed, and torque remain within completely normal operating envelopes.
   - The model assigns these cases moderate failure probabilities (typically $0.35$ to $0.55$), which fall just below the decision threshold ($\tau^* = 0.6732$).
2. **Unattributed Failures (2 out of 15 FNs)**:
   - UDIs `8507` and `9016` have `Machine failure == 1` but all 5 failure mode flags are `0`. The model struggles to detect failures where physical failure indicators were not asserted in the benchmark generator.
3. **Overstrain / Thermal / Power Failures (4 FNs)**:
   - 2 `OSF`, 1 `HDF`, 1 `PWF` cases occurred right at the borderline of failure boundaries where sensor noise slightly masked the threshold crossing.

### Audit of False Positives (6 False Alarms)
1. **Operating Boundary Stress**: 5 of the 6 false alarms occurred in machines operating under extreme torque (>50 Nm) or high cumulative tool wear (>200 min), which closely resemble failure-imminent profiles.
2. **Random Mode Non-Failure (UDI 5490)**: One false positive (UDI `5490`) had random failure flagged (`RNF == 1`) in the generator, but `Machine failure` was labeled `0`. The model recognized the unusual state and flagged it.

---

## 5. Exploratory Failure Mode Analysis

The five failure modes were evaluated independently as separate exploratory targets, maintaining strict isolation from the primary binary classifier:

| Failure Mode | Total Count (10k) | Train Count (8k) | Prevalence | Modeling Status | Mean CV Avg Precision | Mean CV ROC-AUC | Mean CV F1 |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **`PWF`** (Power Failure) | 95 | 82 | 0.95% | Exploratory Baseline | **0.9992** | 1.0000 | 0.9878 |
| **`HDF`** (Heat Dissipation) | 115 | 86 | 1.15% | Exploratory Baseline | **0.9841** | 0.9998 | 0.9266 |
| **`OSF`** (Overstrain Failure) | 98 | 82 | 0.98% | Exploratory Baseline | **0.8225** | 0.9964 | 0.6733 |
| **`TWF`** (Tool Wear Failure) | 46 | 36 | 0.46% | Exploratory Baseline | **0.0613** | 0.9304 | 0.1250 |
| **`RNF`** (Random Failure) | 19 | 15 | 0.19% | **Descriptive Only (Unviable)** | *N/A* | *N/A* | *N/A* |

### Why `RNF` Modeling Is Not Justified
- **Stochastic Mechanism**: `RNF` is simulated as an independent Bernoulli random trial ($P = 0.001$) independent of all operating parameters.
- **Extreme Sparsity**: With only 15 positive training examples (~3 per 5-fold CV split), fitting a machine learning model is statistically invalid and models pure stochastic noise.

### Multi-Target Attribution Caution
- Because failure modes frequently co-occur (e.g. `PWF` and `OSF`), and because 9 failures lack any failure mode label, individual failure mode predictions cannot be summed to reconstruct `Machine failure`.

---

## 6. Operating Constraints & Benchmark Limitations
1. **Synthetic Generation Artifacts**: The model is evaluated on AI4I 2020 simulated data and does not represent live industrial shop-floor telemetry.
2. **Independent Trial Nature**: The dataset represents single-point trials rather than continuous time-series degradation trajectories.
3. **No Production Deployment Claim**: These models serve as reproducible diagnostic baselines and do not warrant unsupervised industrial deployment.

---

## 7. Streaming Inference & Runtime Deployment (Phase 6)

1. **Instantaneous Feature Contract**:
   - The trained pipeline (`models/selected_model.joblib`) expects exactly 8 instantaneous features (`Type` categorical + 5 raw sensor features + 2 engineered features $\Delta T, P_{\text{mech}}$).
   - Although the runtime maintains a stateful `SlidingWindowBuffer` to compute diagnostic rolling statistics (`mean`, `min`, `max`, `std`), **these window aggregates must NEVER be passed to the model**.
2. **Serving Optimization & Threshold Invariance**:
   - The pipeline is loaded once into memory via `StreamingInferenceService`, ensuring local median inference latency under 20 ms without per-sample disk I/O.
   - The optimal decision threshold ($\tau^* = 0.6732$) is strictly immutable at runtime; it was chosen on training out-of-fold predictions and is never adapted or retuned online.
3. **Alert Tiers vs. Demonstration Health Index**:
   - The operational alert tiers (`NORMAL`, `WATCH`, `WARNING`, `CRITICAL`) provide early operator visibility prior to threshold crossing.
   - The synthetic demonstration Health Index ($100 \times (1 - P(\text{failure}))$) reflects instantaneous model confidence on an inverted 0-100% scale. It is explicitly **not** a validated estimator of Remaining Useful Life (RUL) or physical fatigue state.

---

## 8. Interactive Dashboard Serving Context (Phase 7)

1. **Dashboard Architecture & State Management**:
   - The interactive Streamlit dashboard ([`app.py`](../app.py)) integrates the frozen model via [`src/dashboard/state.py`](../src/dashboard/state.py) (`DashboardSession`).
   - Session state isolates user interaction (single-sample stepping, batch execution, replay mode selection, simulation reset) to prevent duplicate inference triggers or state mutation during UI re-renders.
2. **Probability Interpretation & Alert Thresholds**:
   - Model failure probability $P(\text{Machine failure} = 1)$ is evaluated against the centralized threshold $\tau^* = 0.6732$.
   - Operator alert tiers provide color-coded early-warning indicators (`NORMAL` $<0.30$, `WATCH` $0.30-0.50$, `WARNING` $0.50-0.6732$, `CRITICAL` $\ge 0.6732$) on the main fleet overview card and risk trajectory chart.
3. **Strict Separation of Diagnostic Visuals and ML Features**:
   - Sliding-window statistics (`mean`, `min`, `max`, `std`) computed by `SlidingWindowBuffer` are rendered exclusively in Tab 3 for operator situational awareness.
   - The dashboard enforces that rolling statistics are never passed into `StreamingInferenceService.predict_record()`.
4. **Offline Benchmark Limitations**:
   - Displayed in Tab 4 to ensure transparent communication: AI4I 2020 consists of synthetic benchmark runs rather than time-series telemetry from a live physical plant.
   - Predictions demonstrate software simulation and low-latency inference mechanics rather than industrial safety-critical prognostics.