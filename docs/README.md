# Technical Documentation Index: AI-Powered Predictive Maintenance for IoT Devices

Welcome to the technical documentation repository for the **AI-Powered Predictive Maintenance for IoT Devices** platform. This directory contains detailed system architecture designs, data flow patterns, module contracts, model specifications, and execution logs.

---

## Authoritative Documentation Suite

| Document | Primary Focus | Key Contents |
| :--- | :--- | :--- |
| [**System Architecture**](architecture.md) | Architectural Topologies & Subsystems | Component layout, offline batch training vs. streaming runtime pipelines, subsystem design principles, and latency boundaries. |
| [**Data Pipeline & Flow**](data_pipeline_and_flow.md) | Telemetry Lifecycle & Data Movement | Step-by-step tracing from raw CSV ingestion through stratified splitting, feature engineering, pipeline transformation, streaming FIFO buffering, and dashboard presentation. |
| [**Modules & Components**](modules_and_components.md) | Codebase Structure & API Contracts | Complete functional breakdown of all modules in `src/`, `src/data/`, `src/features/`, `src/streaming/`, `src/dashboard/`, and `app.py`. |
| [**Dataset Card**](DATA_CARD.md) | Benchmark Dataset Specifications | UCI AI4I 2020 dataset provenance, class imbalance (28.5:1), column definitions, physical sensor bounds, and known data discrepancies. |
| [**Model Card**](MODEL_CARD.md) | Machine Learning Model Specifications | Balanced Random Forest architecture, 5-fold CV comparison, out-of-fold threshold selection ($\tau^* = 0.6732$), held-out test evaluation scorecard, permutation importance, and error audit. |
| [**Project Roadmap**](PROJECT_PLAN.md) | Phased Implementation Milestones | Phased engineering progression covering initialization, EDA, feature engineering, baseline training, diagnostics, streaming runtime, dashboard, and final verification. |
| [**Execution Log**](execution_log.md) | Chronological Implementation Record | Detailed milestone log recording implemented components, validation commands, latency benchmarks, and test results across all phases. |

---

## Core Engineering Objectives

1. **Leakage-Safe Telemetry Learning**: Preventing target and post-hoc failure mode leakage through formal prediction contracts and pipeline encapsulation.
2. **Asymmetric Risk Optimization**: Optimizing the decision threshold on training out-of-fold predictions to reduce false alarms by 83.3% while maintaining high failure recall.
3. **Decoupled Streaming Simulation**: Simulating edge IoT sensor ingestion with bounded FIFO buffering and low-latency memory-resident inference (sub-50 ms).
4. **Transparent Explainability**: Auditing physical failure mechanisms through permutation feature importance and forensic error analysis.
