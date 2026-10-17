"""
AI-Powered Predictive Maintenance for IoT Devices
Entry-point script and workflow runner.

Usage:
    python main.py                 # Runs the Phase 6 Telemetry Streaming & Inference Demo (default)
    python main.py --stream        # Explicitly run streaming inference demonstration
    python main.py --samples 50    # Stream 50 telemetry samples
    python main.py --full-pipeline # Retrospectively run full Phase 1-6 pipeline (ingestion -> training -> diagnostics -> streaming)
"""

import argparse
import sys
from pathlib import Path

from src.data import load_ai4i_dataset
from src.features import (
    engineer_features,
    split_dataset,
    PROHIBITED_COLUMNS,
)
from src.train import train_and_evaluate
from src.diagnostics import run_full_diagnostics
from src.streaming import run_streaming_demo


def run_pipeline() -> None:
    """Executes the full Phase 1-5 data, training, and diagnostics pipeline."""
    print("=" * 80)
    print("  AI-Powered Predictive Maintenance for IoT Devices: Full Pipeline")
    print("=" * 80)
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version.split()[0]}")
    print("-" * 80)

    # 1. Ingestion & Validation
    print("[1/5] Data Ingestion & Schema Validation:")
    try:
        df, report = load_ai4i_dataset()
        print(f"  - Dataset Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"  - Target Distribution ('Machine failure'): {report.target_distribution}")
        print(f"  - Validation Status: {'PASSED' if report.is_valid else 'FAILED'}")
    except Exception as e:
        print(f"Validation Error: {e}")
        sys.exit(1)

    print("-" * 80)
    # 2. Feature Engineering
    print("[2/5] Physical Domain Feature Engineering:")
    df_eng = engineer_features(df)
    print(f"  - Features Added: 'temperature_difference' [K], 'mechanical_power_w' [W]")
    print(f"  - Engineered Shape: {df_eng.shape[0]} rows x {df_eng.shape[1]} columns")

    print("-" * 80)
    # 3. Stratified Split & Leakage-Safe Preprocessing
    print("[3/5] Stratified Dataset Split & Preprocessing Fit:")
    X_train, X_test, y_train, y_test = split_dataset(df_eng, test_size=0.2, random_state=42)
    print(f"  - Training Set: {X_train.shape[0]} samples (Failures: {int(y_train.sum())}, Rate: {y_train.mean():.4f})")
    print(f"  - Test Set:     {X_test.shape[0]} samples (Failures: {int(y_test.sum())}, Rate: {y_test.mean():.4f})")

    # Check prohibited columns
    leaked = [col for col in PROHIBITED_COLUMNS if col in X_train.columns or col in X_test.columns]
    print(f"  - Leakage Guard Check: {'PASSED (Zero Prohibited Columns in Features)' if not leaked else f'FAILED: {leaked}'}")

    print("-" * 80)
    # 4. Training, Model Comparison, and Held-out Evaluation
    print("[4/5] Baseline Training & Evaluation Workflow:")
    results = train_and_evaluate(random_state=42, test_size=0.2)
    print(f"  - Selected Model: {results['selected_model']}")
    print(f"  - Out-of-Fold Tuned Threshold: {results['optimal_threshold']}")
    m_tune = results["metrics_tuned"]
    print(f"  - Held-out Test Average Precision: {m_tune['average_precision']:.4f}")
    print(f"  - Held-out Test ROC-AUC:           {m_tune['roc_auc']:.4f}")
    print(f"  - Held-out Test F1-Score:          {m_tune['f1_score']:.4f}")
    print(f"  - Confusion Matrix: TP={m_tune['true_positives']}, FN={m_tune['false_negatives']}, FP={m_tune['false_positives']}, TN={m_tune['true_negatives']}")

    print("-" * 80)
    # 5. Diagnostics, Explainability & Failure Mode Exploration
    print("[5/5] Diagnostics, Explainability & Failure-Mode Analysis:")
    diag_results = run_full_diagnostics()
    perm_top = diag_results["permutation_importance"].iloc[0]
    print(f"  - Top Permutation Feature: '{perm_top['feature']}' (Mean AP Drop: {perm_top['importance_mean']:.4f})")
    err_bd = diag_results["error_summary"]["confusion_breakdown"]
    print(f"  - Error Breakdown: {err_bd}")
    fn_audit = diag_results["error_summary"]["false_negatives_audit"]
    print(f"  - False Negatives Dominant Mode: Tool Wear Failure (TWF = {fn_audit['failure_mode_breakdown']['TWF']}/{fn_audit['total_false_negatives']})")
    print(f"  - RNF Viability Status: Marked as Descriptive Only (pure stochastic noise, unviable for modeling)")

    print("-" * 80)
    # 6. Streaming Ingestion & Inference
    print("[6/6] Phase 6 Virtual Telemetry Streaming Demonstration:")
    run_streaming_demo(num_records=25, window_size=10, shuffle=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI-Powered Predictive Maintenance for IoT Devices runner."
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Run virtual telemetry streaming and inference buffer demonstration (Phase 6).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=25,
        help="Number of simulated telemetry samples to process (default: 25).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Capacity of sliding-window buffer (default: 10).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle benchmark records with random seed instead of sequential replay.",
    )
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Execute full training, evaluation, diagnostics, and streaming pipeline.",
    )

    args = parser.parse_args()

    if args.full_pipeline:
        run_pipeline()
    else:
        # Default action: Run Phase 6 Virtual Telemetry Streaming Demonstration
        run_streaming_demo(
            num_records=args.samples,
            window_size=args.window,
            shuffle=args.shuffle,
            verbose=True,
        )


if __name__ == "__main__":
    main()
