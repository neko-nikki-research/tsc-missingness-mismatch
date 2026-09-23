"""Run a reproducible validation/deployment missingness benchmark."""

import argparse
import hashlib
import json
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


def _result_paths(results_dir: Path) -> tuple[Path, Path, Path]:
    """Return the checkpoint files for one fixed benchmark configuration."""
    return (
        results_dir / "raw_results.csv",
        results_dir / "selection_results.csv",
        results_dir / "run_manifest.json",
    )


def _config_hash(config: dict) -> str:
    """Make an identifier so results from different configurations are not mixed."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _expected_rows_per_dataset(config: dict) -> tuple[int, int]:
    """Return expected raw and selection row counts for one dataset."""
    rates = config.get("missing_rates", [config.get("missing_rate")])
    conditions = (
        len(config["seeds"])
        * len(rates)
        * len(config["source_patterns"])
        * len(config["target_patterns"])
        * len(config["imputers"])
    )
    return conditions * len(config["classifiers"]), conditions


def _completed_datasets(
    raw: pd.DataFrame, selection: pd.DataFrame, config: dict
) -> set[str]:
    """Identify datasets whose full set of conditions has already been saved."""
    if "dataset" not in raw.columns or "dataset" not in selection.columns:
        return set()
    expected_raw, expected_selection = _expected_rows_per_dataset(config)
    return {
        dataset
        for dataset in config["datasets"]
        if (raw["dataset"] == dataset).sum() == expected_raw
        and (selection["dataset"] == dataset).sum() == expected_selection
    }


def _write_checkpoint(raw: pd.DataFrame, selection: pd.DataFrame, results_dir: Path) -> None:
    """Atomically replace both CSV checkpoints after one dataset finishes."""
    raw_path, selection_path, _ = _result_paths(results_dir)
    for frame, path in ((raw, raw_path), (selection, selection_path)):
        temporary_path = path.with_suffix(".tmp")
        frame.to_csv(temporary_path, index=False)
        temporary_path.replace(path)


def _load_checkpoint(
    config: dict, results_dir: Path, resume: bool
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """Load saved work and discard an incomplete dataset, if a run was interrupted."""
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path, selection_path, manifest_path = _result_paths(results_dir)
    fingerprint = _config_hash(config)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_hash") != fingerprint:
            raise ValueError(
                "Existing checkpoint belongs to a different config. "
                "Use a new results_dir to avoid mixing experiments."
            )
    else:
        manifest_path.write_text(
            json.dumps({"config_hash": fingerprint, "config": config}, indent=2),
            encoding="utf-8",
        )

    if not resume:
        return pd.DataFrame(), pd.DataFrame(), set()

    raw = _read_csv_or_empty(raw_path)
    selection = _read_csv_or_empty(selection_path)
    completed = _completed_datasets(raw, selection, config)
    # A crash between the two file replacements can leave partial rows. Keeping
    # only fully completed datasets prevents duplicates when we retry.
    if "dataset" in raw.columns:
        raw = raw[raw["dataset"].isin(completed)].copy()
    if "dataset" in selection.columns:
        selection = selection[selection["dataset"].isin(completed)].copy()
    return raw, selection, completed


def run_config(
    config: dict, checkpoint_dir: Path | None = None, resume: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a benchmark, optionally saving a checkpoint after every dataset."""
    if checkpoint_dir is None:
        raw_rows, selection_rows, completed = [], [], set()
    else:
        saved_raw, saved_selection, completed = _load_checkpoint(config, checkpoint_dir, resume)
        raw_rows = saved_raw.to_dict("records")
        selection_rows = saved_selection.to_dict("records")
    params = config.get("classifier_params", {})
    for dataset in config["datasets"]:
        if dataset in completed:
            print(f"Skipping completed dataset: {dataset}", flush=True)
            continue
        print(f"Running dataset: {dataset}", flush=True)
        dataset_raw_rows, dataset_selection_rows = [], []
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
                                dataset_raw_rows.append(row)
                                experiment_rows.append(row)
                            dataset_selection_rows.append(selection_summary(experiment_rows))
        raw_rows.extend(dataset_raw_rows)
        selection_rows.extend(dataset_selection_rows)
        if checkpoint_dir is not None:
            _write_checkpoint(
                pd.DataFrame(raw_rows), pd.DataFrame(selection_rows), checkpoint_dir
            )
            print(f"Checkpoint saved after {dataset}.", flush=True)
    return pd.DataFrame(raw_rows), pd.DataFrame(selection_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument(
        "--no-resume", action="store_true", help="Ignore existing checkpoint rows."
    )
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    results_dir = Path(config.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    raw, selection = run_config(config, results_dir, resume=not args.no_resume)
    _write_checkpoint(raw, selection, results_dir)
    print(f"Saved {len(raw)} rows to {results_dir / 'raw_results.csv'}")
    print(f"Saved {len(selection)} rows to {results_dir / 'selection_results.csv'}")


if __name__ == "__main__":
    main()
