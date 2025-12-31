from src.metrics import compute_classification_metrics


def test_metrics_on_imbalanced_predictions():
    # Majority class 0, minority 1; predictions biased to majority
    y_true = [0] * 95 + [1] * 5
    y_pred = [0] * 100
    metrics = compute_classification_metrics(y_true, y_pred)

    # No positive predictions => precision 0, recall 0, f1 0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
