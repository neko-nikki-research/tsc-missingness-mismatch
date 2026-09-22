"""Run a reproducible validation/deployment missingness benchmark."""

import argparse
import math
import time
from pathlib import Path

import pandas as pd
import yaml

from src.classifiers import make_classifier
from src.datasets import load_ucr_with_validation
from src.evaluation import calculate_classification_metrics, selection_summary
from src.imputation import impute
from src.masking import apply_mask


def run_config(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows, selection_rows = [], []
    params = config.get("classifier_params", {})
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            X_train, y_train, X_val, y_val, X_test, y_test = load_ucr_with_validation(
                dataset, config["validation_size"], seed
            )
            rates = config.get("missing_rates")
            if rates is None:
                rates = [config["missing_rate"]]
            for missing_rate in rates:
                n_timepoints = X_train.shape[-1]
                n_missing_per_series = math.floor(missing_rate * n_timepoints)
                realized_missing_rate = n_missing_per_series / n_timepoints
                for source_pattern in config["source_patterns"]:
                    # Training data stays complete. Source missingness is a validation-only
                    # selection condition, while target missingness is deployment-only.
                    X_val_masked, val_mask = apply_mask(X_val, source_pattern, missing_rate, seed + 1)
                    for target_pattern in config["target_patterns"]:
                        X_test_masked, test_mask = apply_mask(X_test, target_pattern, missing_rate, seed + 2)
                        for imputer_name in config["imputers"]:
                            X_train_ready = X_train
                            X_val_ready = impute(X_val_masked, val_mask, imputer_name)
                            X_test_ready = impute(X_test_masked, test_mask, imputer_name)
                            experiment_rows = []
                            for classifier_name in config["classifiers"]:
                                classifier = make_classifier(classifier_name, seed, params)
                                started = time.perf_counter()
                                classifier.fit(X_train_ready, y_train)
                                fit_seconds = time.perf_counter() - started
                                started = time.perf_counter()
                                val_pred = classifier.predict(X_val_ready)
                                test_pred = classifier.predict(X_test_ready)
                                predict_seconds = time.perf_counter() - started
                                val_metrics = calculate_classification_metrics(y_val, val_pred)
                                test_metrics = calculate_classification_metrics(y_test, test_pred)
                                row = {
                                    "dataset": dataset,
                                    "seed": seed,
                                    "missing_rate": missing_rate,
                                    "n_timepoints": n_timepoints,
                                    "n_missing_per_series": n_missing_per_series,
                                    "realized_missing_rate": realized_missing_rate,
                                    "source_pattern": source_pattern,
                                    "target_pattern": target_pattern,
                                    "imputer": imputer_name,
                                    "classifier": classifier_name,
                                    "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                                    "val_macro_f1": val_metrics["macro_f1"],
                                    "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                                    "test_macro_f1": test_metrics["macro_f1"],
                                    "fit_time_seconds": fit_seconds, "predict_time_seconds": predict_seconds,
                                }
                                raw_rows.append(row)
                                experiment_rows.append(row)
                            selection_rows.append(selection_summary(experiment_rows))
    return pd.DataFrame(raw_rows), pd.DataFrame(selection_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    raw, selection = run_config(config)
    results_dir = Path(config.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(results_dir / "raw_results.csv", index=False)
    selection.to_csv(results_dir / "selection_results.csv", index=False)
    print(f"Saved {len(raw)} rows to {results_dir / 'raw_results.csv'}")
    print(f"Saved {len(selection)} rows to {results_dir / 'selection_results.csv'}")


if __name__ == "__main__":
    main()
