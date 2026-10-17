import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

# Ensure project root is in sys.path when executed directly as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.data.loader import get_project_root, load_ai4i_dataset
from src.evaluate import (
    compute_classification_metrics,
    find_optimal_threshold_oof,
    confusion_matrix_to_dataframe,
)
from src.features.engineering import engineer_features
from src.features.preprocessing import build_preprocessor
from src.features.schema import (
    ALL_PREDICTORS,
    PROHIBITED_COLUMNS,
    TARGET,
)
from src.features.split import split_dataset


def get_candidate_models(random_state: int = 42) -> Dict[str, Pipeline]:
    """
    Constructs the candidate baseline model pipelines.

    Every pipeline embeds the unfitted preprocessor as its initial stage,
    ensuring all scaling and encoding are refitted strictly within each cross-validation fold.

    Candidates:
        1. DummyClassifier (majority-class strategy): Floor baseline.
        2. LogisticRegression (class_weight='balanced'): Linear baseline.
        3. RandomForestClassifier (class_weight='balanced', max_depth=10, n_estimators=100):
           Non-linear ensemble baseline.

    Args:
        random_state: Seed for reproducibility.

    Returns:
        Dict[str, Pipeline]: Mapping of candidate names to scikit-learn Pipelines.
    """
    candidates = {
        "Dummy_Majority": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "Logistic_Regression_Balanced": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_state,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "Random_Forest_Balanced": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    return candidates


def run_cross_validation_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    n_splits: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], str]:
    """
    Executes stratified 5-fold cross-validation on the training set to compare baseline candidates.

    Guarantees:
        - Cross-validation and comparison are performed strictly on X_train.
        - The held-out test set is completely isolated and untouched.
        - Preprocessing is fitted inside each fold.
        - Primary selection metric: Mean Out-Of-Fold / CV Average Precision.

    Args:
        X_train: Training features DataFrame.
        y_train: Training binary targets Series.
        random_state: Seed for fold shuffling.
        n_splits: Number of stratified folds.

    Returns:
        Tuple[pd.DataFrame, Dict[str, np.ndarray], str]:
            (comparison_df, oof_probabilities_dict, best_model_name)
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    candidates = get_candidate_models(random_state=random_state)

    comparison_records = []
    oof_probs: Dict[str, np.ndarray] = {}

    for name, pipeline in candidates.items():
        # Generate out-of-fold predicted probabilities for training partition
        # For DummyClassifier predicting constant, predict_proba returns shape (N, 1) or (N, 2)
        probs_2d = cross_val_predict(
            pipeline, X_train, y_train, cv=cv, method="predict_proba"
        )
        if probs_2d.shape[1] == 2:
            prob_pos = probs_2d[:, 1]
        else:
            # Handle majority dummy predicting single class 0
            prob_pos = np.zeros(len(y_train), dtype=float)

        oof_probs[name] = prob_pos

        # Compute fold-by-fold metrics to capture variability (std)
        fold_aps = []
        fold_rocs = []
        fold_f1s = []
        fold_precs = []
        fold_recs = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            fold_pipe = get_candidate_models(random_state=random_state)[name]
            fold_pipe.fit(X_tr, y_tr)

            p_val_2d = fold_pipe.predict_proba(X_val)
            p_val = p_val_2d[:, 1] if p_val_2d.shape[1] == 2 else np.zeros(len(y_val))

            m = compute_classification_metrics(y_val, p_val, threshold=0.5)
            fold_aps.append(m["average_precision"])
            fold_rocs.append(m["roc_auc"])
            fold_f1s.append(m["f1_score"])
            fold_precs.append(m["precision"])
            fold_recs.append(m["recall"])

        # OOF aggregate metrics
        oof_metrics = compute_classification_metrics(y_train, prob_pos, threshold=0.5)

        record = {
            "model": name,
            "mean_cv_avg_precision": round(float(np.mean(fold_aps)), 4),
            "std_cv_avg_precision": round(float(np.std(fold_aps)), 4),
            "oof_avg_precision": oof_metrics["average_precision"],
            "mean_cv_roc_auc": round(float(np.mean(fold_rocs)), 4),
            "std_cv_roc_auc": round(float(np.std(fold_rocs)), 4),
            "oof_roc_auc": oof_metrics["roc_auc"],
            "mean_cv_f1": round(float(np.mean(fold_f1s)), 4),
            "oof_f1": oof_metrics["f1_score"],
            "oof_precision": oof_metrics["precision"],
            "oof_recall": oof_metrics["recall"],
        }
        comparison_records.append(record)

    comparison_df = pd.DataFrame(comparison_records)
    # Sort by primary selection metric: mean_cv_avg_precision descending
    comparison_df = comparison_df.sort_values(by="mean_cv_avg_precision", ascending=False).reset_index(drop=True)
    best_model_name = comparison_df.iloc[0]["model"]

    return comparison_df, oof_probs, best_model_name


def train_and_evaluate(
    random_state: int = 42,
    test_size: float = 0.2,
) -> Dict[str, Any]:
    """
    Executes the full end-to-end training and evaluation protocol.

    Steps:
        1. Ingest raw AI4I 2020 dataset and apply row-wise feature engineering.
        2. Perform stratified train/test split (80% train / 20% test).
        3. Run 5-fold cross-validation comparison on train partition ONLY.
        4. Select top baseline using mean cross-validated Average Precision.
        5. Tune decision threshold on out-of-fold training probabilities ONLY.
        6. Fit selected pipeline on entire training partition.
        7. Evaluate ONCE on held-out test partition at both default (0.5) and tuned thresholds.
        8. Persist artifacts: model (.joblib), comparison (.csv), test metrics (.json),
           confusion matrix (.csv), and model card (.md).

    Returns:
        Dict[str, Any]: Comprehensive execution summary and metrics.
    """
    root = get_project_root()
    models_dir = root / "models"
    outputs_dir = root / "outputs"
    docs_dir = root / "docs"

    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest & Feature Engineering
    df_raw, _ = load_ai4i_dataset()
    df_eng = engineer_features(df_raw)

    # 2. Stratified Split (8,000 train / 2,000 test)
    X_train, X_test, y_train, y_test = split_dataset(
        df_eng,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
    )

    # 3. Cross-Validation Comparison on Train Set ONLY
    comparison_df, oof_probs, best_model_name = run_cross_validation_comparison(
        X_train, y_train, random_state=random_state, n_splits=5
    )

    # Save model comparison table
    comparison_csv_path = outputs_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)

    # 4. Out-of-fold threshold tuning on Train partition ONLY
    best_oof_prob = oof_probs[best_model_name]
    optimal_threshold, best_oof_f1 = find_optimal_threshold_oof(
        y_train, best_oof_prob, metric="f1"
    )

    # 5. Fit selected model on full training set
    selected_pipeline = get_candidate_models(random_state=random_state)[best_model_name]
    selected_pipeline.fit(X_train, y_train)

    # Save fitted model
    model_path = models_dir / "selected_model.joblib"
    joblib.dump(selected_pipeline, model_path)

    # 6. Evaluate ONCE on held-out test set
    y_test_prob = selected_pipeline.predict_proba(X_test)[:, 1]

    metrics_default_thresh = compute_classification_metrics(
        y_test, y_test_prob, threshold=0.5
    )
    metrics_tuned_thresh = compute_classification_metrics(
        y_test, y_test_prob, threshold=optimal_threshold
    )

    # Confusion matrix at tuned threshold
    cm_df = confusion_matrix_to_dataframe(
        tn=metrics_tuned_thresh["true_negatives"],
        fp=metrics_tuned_thresh["false_positives"],
        fn=metrics_tuned_thresh["false_negatives"],
        tp=metrics_tuned_thresh["true_positives"],
    )
    cm_path = outputs_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)

    # Structure test metrics JSON
    test_metrics_payload = {
        "dataset": "UCI AI4I 2020 Predictive Maintenance",
        "random_state": random_state,
        "split_sizes": {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_failures": int(y_train.sum()),
            "test_failures": int(y_test.sum()),
            "test_failure_prevalence": round(float(y_test.mean()), 4),
        },
        "selected_model": best_model_name,
        "selection_metric": "mean_cv_avg_precision",
        "selected_model_cv_results": comparison_df[
            comparison_df["model"] == best_model_name
        ].iloc[0].to_dict(),
        "threshold_tuning_oof": {
            "selected_threshold": optimal_threshold,
            "training_oof_f1_at_threshold": best_oof_f1,
            "selection_strategy": "Maximized F1 on training out-of-fold predictions without test set leakage",
        },
        "test_metrics_default_threshold_0_5": metrics_default_thresh,
        "test_metrics_selected_threshold": metrics_tuned_thresh,
    }

    test_metrics_json_path = outputs_dir / "test_metrics.json"
    with open(test_metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics_payload, f, indent=2)

    # 7. Generate Model Card & Summary
    model_card_content = generate_model_card_markdown(test_metrics_payload, comparison_df)
    model_card_path = docs_dir / "MODEL_CARD.md"
    with open(model_card_path, "w", encoding="utf-8") as f:
        f.write(model_card_content)

    return {
        "selected_model": best_model_name,
        "comparison_df": comparison_df,
        "optimal_threshold": optimal_threshold,
        "metrics_default": metrics_default_thresh,
        "metrics_tuned": metrics_tuned_thresh,
        "paths": {
            "model": str(model_path),
            "comparison": str(comparison_csv_path),
            "test_metrics": str(test_metrics_json_path),
            "confusion_matrix": str(cm_path),
            "model_card": str(model_card_path),
        },
    }


def df_to_markdown_simple(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def generate_model_card_markdown(payload: Dict[str, Any], comparison_df: pd.DataFrame) -> str:
    """Generates comprehensive Model Card markdown documentation."""
    m_def = payload["test_metrics_default_threshold_0_5"]
    m_tune = payload["test_metrics_selected_threshold"]
    model_name = payload["selected_model"]

    lines = [
        f"# Model Card: Baseline Predictive Maintenance Classifier ({model_name})",
        "",
        "## 1. Model Overview",
        f"- **Selected Architecture**: `{model_name}`",
        "- **Target Variable**: `Machine failure` (Binary: `0` = Normal, `1` = Failure)",
        "- **Positive Class Prevalence**: ~3.39% across dataset (28.5 : 1 class imbalance)",
        "- **Primary Selection Metric**: Mean Cross-Validated **Average Precision**",
        "- **Model Pipeline**: Scikit-Learn `Pipeline` embedding median/most-frequent imputation, standard scaling, one-hot encoding, and class-weighted classifier.",
        "",
        "---",
        "",
        "## 2. Cross-Validation Model Comparison (Training Set Only)",
        "Models were evaluated on 5 stratified folds drawn exclusively from the training partition (8,000 samples).",
        "",
        df_to_markdown_simple(comparison_df),
        "",
        "---",
        "",
        "## 3. Decision Threshold Tuning & Tradeoffs",
        f"- **Default Threshold**: `0.50`",
        f"- **Tuned Threshold**: `{payload['threshold_tuning_oof']['selected_threshold']}`",
        f"- **Tuning Rationale**: {payload['threshold_tuning_oof']['selection_strategy']}.",
        "",
        "### Industrial Tradeoff Context",
        "- **False Negatives (Missed Failures)**: Lead to catastrophic unplanned machinery breakdown, secondary damage, and emergency downtime costs.",
        "- **False Positives (False Alarms)**: Trigger unnecessary preventative inspection, costing minimal labor time.",
        "- Lowering the decision threshold or optimizing F1 shifts operating sensitivity to capture rare failures without overwhelming operators with false alarms.",
        "",
        "---",
        "",
        "## 4. Held-Out Test Evaluation (2,000 Samples, 68 Failures)",
        "The final test partition was evaluated strictly once after model selection and threshold determination.",
        "",
        "| Metric | Default Threshold (0.50) | Selected Threshold (" + str(m_tune['threshold']) + ") |",
        "| :--- | :---: | :---: |",
        f"| **Average Precision** | **{m_def['average_precision']:.4f}** | **{m_tune['average_precision']:.4f}** |",
        f"| **ROC-AUC** | **{m_def['roc_auc']:.4f}** | **{m_tune['roc_auc']:.4f}** |",
        f"| **F1-Score** | {m_def['f1_score']:.4f} | **{m_tune['f1_score']:.4f}** |",
        f"| **Precision** | {m_def['precision']:.4f} | {m_tune['precision']:.4f} |",
        f"| **Recall** | {m_def['recall']:.4f} | **{m_tune['recall']:.4f}** |",
        f"| **True Positives (TP)** | {m_def['true_positives']} | {m_tune['true_positives']} |",
        f"| **False Negatives (FN)** | {m_def['false_negatives']} | {m_tune['false_negatives']} |",
        f"| **False Positives (FP)** | {m_def['false_positives']} | {m_tune['false_positives']} |",
        f"| **True Negatives (TN)** | {m_def['true_negatives']} | {m_tune['true_negatives']} |",
        "",
        "---",
        "",
        "## 5. Limitations & Operating Constraints",
        "- **Synthetic Benchmark**: Trained on AI4I 2020 simulated milling trials. It does not reflect live physical plant vibration or temporal wear degradation over time.",
        "- **Static Trial Sampling**: Does not model sequential sensor degradation trajectories. Temporal deep learning models (NASA C-MAPSS RUL) will address sequential dynamics in later phases.",
        "- **No Industrial Deployment Claim**: Model performance is validated for benchmark reproducibility only and is not validated for live manufacturing production.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print("Running Baseline Model Training & Evaluation Workflow...")
    results = train_and_evaluate()
    print("Execution Completed Successfully!")
    print(f"Selected Model: {results['selected_model']}")
    print(f"Optimal Threshold: {results['optimal_threshold']}")
    print(f"Test Metrics at Optimal Threshold: {results['metrics_tuned']}")
