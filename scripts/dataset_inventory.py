"""Record the dataset files and train/validation splits used by a configuration.

For every dataset this writes the file that aeon 1.6.0 actually reads (bundled
copy first, otherwise the download cache), its SHA-256 and modification time,
the train/test sizes, series length and class counts, and for every seed a
SHA-256 of the validation indices and any class missing from validation.

The inventory describes the files present when it is run. It can show that they
match the archived runs only indirectly (the archived metrics do not include the
data files), so it must not be presented as a record made at run time.

Usage:
    python scripts/dataset_inventory.py configs/main_protocol_balanced.yaml results/dataset_inventory.csv
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import aeon
import numpy as np
import pandas as pd
import yaml
from aeon.datasets import load_classification
from aeon.datasets._data_loaders import _get_data_home
from sklearn.model_selection import train_test_split

BUNDLED = os.path.join(os.path.dirname(aeon.__file__), "datasets", "data")


def data_file(dataset: str, split: str) -> str:
    for root in (BUNDLED, _get_data_home()):
        path = os.path.join(root, dataset, f"{dataset}_{split}.ts")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No local file for {dataset} {split}")


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def inventory(config: dict) -> pd.DataFrame:
    rows = []
    for dataset in config["datasets"]:
        X_train, y_train = load_classification(dataset, split="train")
        X_test, y_test = load_classification(dataset, split="test")
        row = {
            "dataset": dataset,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "length": X_train.shape[2],
            "n_classes": len(np.unique(y_train)),
            "train_class_counts": json.dumps({str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))}),
            "test_class_counts": json.dumps({str(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))}),
        }
        for split in ("TRAIN", "TEST"):
            path = data_file(dataset, split)
            row[f"{split.lower()}_source"] = "bundled" if path.startswith(BUNDLED) else "cache"
            row[f"{split.lower()}_sha256"] = sha256(path)
            row[f"{split.lower()}_modified_utc"] = datetime.fromtimestamp(
                os.path.getmtime(path), timezone.utc).strftime("%Y-%m-%d %H:%M")
        indices = np.arange(len(y_train))
        missing = []
        for seed in config["seeds"]:
            fit_idx, val_idx = train_test_split(indices, test_size=config["validation_size"],
                                                random_state=seed, stratify=y_train)
            row[f"val_idx_sha256_seed{seed}"] = hashlib.sha256(np.sort(val_idx).astype(np.int64).tobytes()).hexdigest()
            if set(y_train) - set(y_train[fit_idx]):
                raise ValueError(f"{dataset} seed {seed}: fitting subset lacks a class")
            missing += [f"{seed}:{c}" for c in sorted(set(y_train) - set(y_train[val_idx]))]
        row["validation_missing_classes"] = ";".join(missing)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
    table = inventory(config)
    table.to_csv(sys.argv[2], index=False)
    n_missing = table.validation_missing_classes.str.count(":").sum()
    print(f"{len(table)} datasets; {n_missing} dataset-seed splits lack a validation class")
