"""Balanced-accuracy metrics and validation-only method selection."""

import hashlib

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def calculate_classification_metrics(y_true, y_pred) -> dict:
    """Return the protocol metrics for one set of predictions.

    Balanced accuracy is the arithmetic mean of per-class recall. It is the
    only accuracy used for selection, oracle identification, and regret.
    """
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _validation_tie_seed(row: dict) -> int:
    """Derive a stable seed from validation-side information only."""
    fields = (
        row["dataset"], row["seed"], row["missing_rate"],
        row["source_pattern"], row["imputer"],
    )
    digest = hashlib.blake2b(repr(fields).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little")


def selection_summary(rows: list[dict]) -> dict:
    """Select by validation BA, then evaluate against the post-hoc test oracle.

    Exact validation ties are broken uniformly at random using a deterministic
    seed made only from validation-side experiment metadata. Test outcomes are
    never consulted for selection. Test-oracle ties are reported as all tied
    classifier names, and are not counted as a selection error.
    """
    first = rows[0]
    best_validation = max(row["val_balanced_accuracy"] for row in rows)
    validation_ties = [
        row for row in rows
        if np.isclose(row["val_balanced_accuracy"], best_validation, rtol=0, atol=1e-12)
    ]
    validation_ties = sorted(validation_ties, key=lambda row: row["classifier"])
    rng = np.random.default_rng(_validation_tie_seed(first))
    selected = validation_ties[int(rng.integers(len(validation_ties)))]

    best_test = max(row["test_balanced_accuracy"] for row in rows)
    oracle_ties = [
        row for row in rows
        if np.isclose(row["test_balanced_accuracy"], best_test, rtol=0, atol=1e-12)
    ]
    oracle_names = "|".join(sorted(row["classifier"] for row in oracle_ties))
    return {
        key: first[key]
        for key in (
            "dataset", "seed", "missing_rate", "n_timepoints",
            "n_missing_per_series", "realized_missing_rate", "source_pattern",
            "target_pattern", "imputer",
        )
    } | {
        "selected_classifier": selected["classifier"],
        "oracle_classifier": oracle_names,
        "n_validation_ties": len(validation_ties),
        "n_oracle_ties": len(oracle_ties),
        "selection_error": bool(selected["test_balanced_accuracy"] < best_test - 1e-12),
        "selected_test_balanced_accuracy": selected["test_balanced_accuracy"],
        "oracle_test_balanced_accuracy": best_test,
        "regret": float(
            best_test - selected["test_balanced_accuracy"]
        ),
    }
