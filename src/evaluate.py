"""
Evaluation metrics and diagnostic reporting for AI-Predictive-Maintenance-IoT.
Focuses on class-imbalanced failure detection metrics (Average Precision, F1, Recall, ROC-AUC).
"""

from typing import Any, Dict, Optional, Tuple, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes diagnostic classification metrics for class-imbalanced predictive maintenance.

    Metrics:
        - Average Precision: Area under the stepwise precision-recall curve.
        - ROC-AUC: Area under the receiver operating characteristic curve.
        - Precision, Recall, F1-Score: Evaluated at the specified decision threshold.
        - Confusion Matrix Breakdown: True Negatives (TN), False Positives (FP),
          False Negatives (FN), True Positives (TP).
        - Positive-Class Prevalence: Proportion of true failure instances in the sample.

    Args:
        y_true: Ground truth binary targets (0 = Normal, 1 = Failure).
        y_prob: Predicted probability of positive class (class 1).
        threshold: Decision threshold for positive classification (default: 0.5).

    Returns:
        Dict[str, Any]: Key-value mapping of evaluated metrics.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    # Validate inputs
    if len(y_true_arr) != len(y_prob_arr):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true_arr)}) vs y_prob ({len(y_prob_arr)})"
        )

    # Predictions at specified decision threshold
    y_pred = (y_prob_arr >= threshold).astype(int)

    # Positive class prevalence
    pos_prevalence = float(np.mean(y_true_arr))

    # Average Precision (area under stepwise PR curve)
    # Handle edge case where y_true has only one class or y_prob is constant
    if len(np.unique(y_true_arr)) > 1 and len(np.unique(y_prob_arr)) > 1:
        avg_precision = float(average_precision_score(y_true_arr, y_prob_arr))
        try:
            roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        except ValueError:
            roc_auc = 0.5
    else:
        # For dummy classifiers predicting constant probabilities
        avg_precision = pos_prevalence
        roc_auc = 0.5

    prec = float(precision_score(y_true_arr, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, pos_label=1, zero_division=0))

    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return {
        "threshold": round(float(threshold), 4),
        "positive_prevalence": round(pos_prevalence, 4),
        "average_precision": round(avg_precision, 4),
        "roc_auc": round(roc_auc, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def find_optimal_threshold_oof(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    metric: str = "f1",
    num_thresholds: int = 100,
) -> Tuple[float, float]:
    """
    Searches for the decision threshold that maximizes a target metric (default: F1-score)
    using training out-of-fold (OOF) predicted probabilities ONLY.

    Leakage Prevention:
        This search MUST be executed strictly on out-of-fold training predictions.
        The held-out test set must NEVER be used to select or tune the threshold.

    Args:
        y_true: Training out-of-fold ground truth labels.
        y_prob: Training out-of-fold predicted probabilities for positive class.
        metric: Metric to optimize ('f1', 'recall', 'precision'). Default: 'f1'.
        num_thresholds: Number of threshold candidate points between 0.01 and 0.99.

    Returns:
        Tuple[float, float]: (optimal_threshold, best_metric_score)
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    best_threshold = 0.5
    best_score = -1.0

    for t in thresholds:
        y_pred = (y_prob_arr >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true_arr, y_pred, pos_label=1, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true_arr, y_pred, pos_label=1, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true_arr, y_pred, pos_label=1, zero_division=0)
        else:
            raise ValueError(f"Unsupported threshold optimization metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = t

    return round(float(best_threshold), 4), round(float(best_score), 4)


def confusion_matrix_to_dataframe(
    tn: int, fp: int, fn: int, tp: int
) -> pd.DataFrame:
    """
    Formats confusion matrix values into a labeled pandas DataFrame.
    """
    return pd.DataFrame(
        data=[[tn, fp], [fn, tp]],
        index=["Actual_Normal (0)", "Actual_Failure (1)"],
        columns=["Predicted_Normal (0)", "Predicted_Failure (1)"],
    )

