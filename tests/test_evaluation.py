import numpy as np

from src.evaluation import calculate_classification_metrics, selection_summary


def test_metrics_uses_balanced_accuracy_not_majority_class_accuracy():
    y_true = np.array(["A"] * 9 + ["B"])
    y_pred = np.array(["A"] * 10)
    result = calculate_classification_metrics(y_true, y_pred)
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


def test_validation_ties_are_reproducible_and_do_not_use_test_performance():
    common = {
        "dataset": "Demo", "seed": 3, "missing_rate": 0.2,
        "source_pattern": "point", "target_pattern": "block", "imputer": "linear",
        "val_balanced_accuracy": 0.8,
    }
    rows = [
        common | {"classifier": "dtw", "test_balanced_accuracy": 0.6},
        common | {"classifier": "minirocket", "test_balanced_accuracy": 0.9},
    ]
    first = selection_summary(rows)
    second = selection_summary(rows)
    assert first["selected_classifier"] == second["selected_classifier"]
    assert first["selected_classifier"] in {"dtw", "minirocket"}
    assert first["n_validation_ties"] == 2
    assert first["oracle_classifier"] == "minirocket"


def test_test_oracle_tie_is_not_an_artificial_selection_error():
    common = {
        "dataset": "Demo", "seed": 4, "missing_rate": 0.2,
        "source_pattern": "point", "target_pattern": "block", "imputer": "linear",
    }
    rows = [
        common | {"classifier": "dtw", "val_balanced_accuracy": 0.9, "test_balanced_accuracy": 0.8},
        common | {"classifier": "minirocket", "val_balanced_accuracy": 0.7, "test_balanced_accuracy": 0.8},
    ]
    summary = selection_summary(rows)
    assert summary["oracle_classifier"] == "dtw|minirocket"
    assert summary["n_oracle_ties"] == 2
    assert summary["selection_error"] is False
