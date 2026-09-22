import numpy as np
import pytest

from src.imputation import impute


def test_linear_interpolation_handles_both_ends_and_keeps_observations():
    X = np.array([[[np.nan, np.nan, 2.0, 4.0, np.nan, np.nan]]])
    mask = np.isnan(X)
    result = impute(X, mask, "linear")
    expected = np.array([[[2.0, 2.0, 2.0, 4.0, 4.0, 4.0]]])
    assert np.array_equal(result, expected)
    assert np.array_equal(result[~mask], X[~mask])
    assert not np.isnan(result).any()


@pytest.mark.parametrize("method, expected", [("zero", [0.0, 2.0, 0.0]), ("mean", [2.0, 2.0, 2.0])])
def test_simple_imputers(method, expected):
    X = np.array([[[np.nan, 2.0, np.nan]]])
    result = impute(X, np.isnan(X), method)
    assert np.array_equal(result[0, 0], expected)


def test_fully_missing_series_raises():
    X = np.full((1, 1, 3), np.nan)
    with pytest.raises(ValueError, match="fully missing"):
        impute(X, np.ones_like(X, dtype=bool), "linear")
