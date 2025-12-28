from typing import Dict, Optional
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def compute_classification_metrics(y_true, y_pred, y_score: Optional[list] = None) -> Dict[str, float]:
    """Compute common classification metrics.

    Args:
        y_true: ground-truth labels
        y_pred: predicted labels
        y_score: optional predicted probability for positive class (required for ROC AUC)

    Returns:
        dict with precision, recall, f1, and roc_auc (if score provided)
    """
    metrics = {}
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    if y_score is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            metrics["roc_auc"] = float("nan")
    return metrics
