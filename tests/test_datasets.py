import numpy as np
import pytest

from src.datasets import validate_main_experiment_data


def test_main_data_validation_accepts_complete_univariate_equal_length_data():
    validate_main_experiment_data(np.zeros((3, 1, 5)), "Demo")


def test_main_data_validation_rejects_multivariate_data():
    with pytest.raises(ValueError, match="univariate"):
        validate_main_experiment_data(np.zeros((3, 2, 5)), "Demo")


def test_main_data_validation_rejects_original_missing_values():
    X = np.zeros((3, 1, 5))
    X[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="original missing"):
        validate_main_experiment_data(X, "Demo")
