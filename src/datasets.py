"""Data loading and the train/validation split."""

import numpy as np
from aeon.datasets import load_classification
from sklearn.model_selection import train_test_split


def load_ucr_with_validation(dataset: str, validation_size: float, seed: int):
    """Load UCR's official train/test and split only the official train set."""
    X_official_train, y_official_train = load_classification(dataset, split="train")
    X_test, y_test = load_classification(dataset, split="test")
    indices = np.arange(len(y_official_train))
    train_idx, val_idx = train_test_split(
        indices, test_size=validation_size, random_state=seed, stratify=y_official_train
    )
    return (
        X_official_train[train_idx].copy(), y_official_train[train_idx].copy(),
        X_official_train[val_idx].copy(), y_official_train[val_idx].copy(),
        X_test.copy(), y_test.copy(),
    )
