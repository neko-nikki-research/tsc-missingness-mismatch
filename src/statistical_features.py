"""Transparent per-series statistical features for the protocol RF baseline."""

import numpy as np


FEATURE_NAMES = (
    "mean",
    "std",
    "minimum",
    "maximum",
    "median",
    "interquartile_range",
    "linear_slope",
    "first_value",
    "last_value",
    "lag1_autocorrelation",
)


def extract_statistical_features(X: np.ndarray) -> np.ndarray:
    """Return fixed, interpretable features for each sample and channel.

    X must be complete: missing-value imputation happens before feature extraction.
    The output has ``n_channels * len(FEATURE_NAMES)`` columns.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError("X must have shape (n_samples, n_channels, n_timepoints).")
    if np.isnan(X).any():
        raise ValueError("Statistical features require imputed data with no NaN values.")

    n_samples, n_channels, n_timepoints = X.shape
    if n_timepoints < 2:
        raise ValueError("At least two timepoints are required for statistical features.")

    time = np.arange(n_timepoints, dtype=float)
    centered_time = time - time.mean()
    slope_denominator = np.sum(centered_time**2)
    rows = []
    for sample in X:
        sample_features = []
        for series in sample:
            centered_series = series - series.mean()
            lag_denominator = np.sum(centered_series**2)
            lag1 = (
                float(np.dot(centered_series[:-1], centered_series[1:]) / lag_denominator)
                if lag_denominator > 0
                else 0.0
            )
            slope = float(np.dot(centered_time, series) / slope_denominator)
            sample_features.extend(
                [
                    float(np.mean(series)),
                    float(np.std(series)),
                    float(np.min(series)),
                    float(np.max(series)),
                    float(np.median(series)),
                    float(np.percentile(series, 75) - np.percentile(series, 25)),
                    slope,
                    float(series[0]),
                    float(series[-1]),
                    lag1,
                ]
            )
        rows.append(sample_features)
    return np.asarray(rows, dtype=float).reshape(n_samples, n_channels * len(FEATURE_NAMES))
