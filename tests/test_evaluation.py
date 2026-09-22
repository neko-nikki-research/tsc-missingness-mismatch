import numpy as np

from src.evaluation import metrics, selection_summary


def test_metrics_uses_balanced_accuracy_not_majority_class_accuracy():
    y_true = np.array(["A"] * 9 + ["B"])
    y_pred = np.array(["A"] * 10)
    result = metrics(y_true, y_pred)
    assert result["balanced_accuracy"] == 0.5


def test_selection_uses_validation_balanced_accuracy():
    common = {
        "dataset": "Demo", "seed": 1, "missing_rate": 0.2,
        "source_pattern": "point", "target_pattern": "block", "imputer": "linear",
    }
    rows = [
        common | {"classifier": "majority", "val_balanced_accuracy": 0.50, "test_balanced_accuracy": 0.50},
        common | {"classifier": "balanced", "val_balanced_accuracy": 0.80, "test_balanced_accuracy": 0.75},
    ]
    summary = selection_summary(rows)
    assert summary["selected_classifier"] == "balanced"
    assert summary["oracle_classifier"] == "balanced"
    assert summary["regret"] == 0.0
