from src.metrics import compute_classification_metrics


def test_compute_metrics_basic():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    y_score = [0.1, 0.6, 0.8, 0.9]

    metrics = compute_classification_metrics(y_true, y_pred, y_score)
    assert round(metrics["precision"], 2) == 0.67
    assert round(metrics["recall"], 2) == 1.00
    assert "roc_auc" in metrics
