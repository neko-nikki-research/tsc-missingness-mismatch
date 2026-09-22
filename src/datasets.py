"""Data loading and the train/validation split."""

import numpy as np
from aeon.datasets import load_classification
from sklearn.model_selection import train_test_split


def validate_main_experiment_data(X: np.ndarray, dataset: str) -> None:
    """Require the univariate, equal-length, complete format of the main study."""
    if X.ndim != 3:
        raise ValueError(f"{dataset} must be a 3D time-series array.")
    if X.shape[1] != 1:
        raise ValueError(f"{dataset} is multivariate; the main study is univariate only.")
    if X.shape[2] < 2:
        raise ValueError(f"{dataset} needs at least two timepoints.")
    if np.isnan(X).any():
        raise ValueError(f"{dataset} has original missing values; the main study requires complete data.")


def load_ucr_with_validation(dataset: str, validation_size: float, seed: int):
    """Load UCR's official train/test and split only the official train set."""
    X_official_train, y_official_train = load_classification(dataset, split="train")
    X_test, y_test = load_classification(dataset, split="test")
    validate_main_experiment_data(X_official_train, dataset)
    validate_main_experiment_data(X_test, dataset)
    indices = np.arange(len(y_official_train))
    train_idx, val_idx = train_test_split(
        indices, test_size=validation_size, random_state=seed, stratify=y_official_train
    )
    return (
        X_official_train[train_idx].copy(), y_official_train[train_idx].copy(),
        X_official_train[val_idx].copy(), y_official_train[val_idx].copy(),
        X_test.copy(), y_test.copy(),
    )
