"""Metrics and validation-based selection."""

from sklearn.metrics import balanced_accuracy_score, f1_score


def metrics(y_true, y_pred) -> dict:
    """Return class-balanced selection metrics.

    Balanced accuracy gives equal weight to each class recall and is therefore
    robust to class imbalance. It is the only accuracy used in this benchmark.
    """
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def selection_summary(rows: list[dict]) -> dict:
    """Select by validation accuracy and evaluate regret against the test oracle."""
    selected = max(rows, key=lambda row: (row["val_balanced_accuracy"], row["classifier"]))
    oracle = max(rows, key=lambda row: (row["test_balanced_accuracy"], row["classifier"]))
    first = rows[0]
    return {
        key: first[key]
        for key in ("dataset", "seed", "missing_rate", "source_pattern", "target_pattern", "imputer")
    } | {
        "selected_classifier": selected["classifier"],
        "oracle_classifier": oracle["classifier"],
        "selection_error": bool(selected["classifier"] != oracle["classifier"]),
        "selected_test_balanced_accuracy": selected["test_balanced_accuracy"],
        "oracle_test_balanced_accuracy": oracle["test_balanced_accuracy"],
        "regret": float(
            oracle["test_balanced_accuracy"] - selected["test_balanced_accuracy"]
        ),
    }
