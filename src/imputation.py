"""Imputers that operate independently on each time series."""

import numpy as np


def impute(X_masked: np.ndarray, mask: np.ndarray, method: str) -> np.ndarray:
    """Fill masked positions without changing observed positions.

    The imputer derives values only from the observed values in the same series,
    so no information is learned from the official UCR test set.
    """
    X = np.asarray(X_masked, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if X.shape != mask.shape or X.ndim != 3:
        raise ValueError("X_masked and mask must be equally shaped 3D arrays.")
    if method not in {"zero", "mean", "linear"}:
        raise ValueError("method must be 'zero', 'mean', or 'linear'.")
    if np.isnan(X[~mask]).any():
        raise ValueError("Unmasked positions must not contain NaN.")

    filled = X.copy()
    time = np.arange(X.shape[-1])
    for sample in range(X.shape[0]):
        for channel in range(X.shape[1]):
            series_mask = mask[sample, channel]
            if not series_mask.any():
                continue
            observed = ~series_mask
            if not observed.any():
                raise ValueError("Cannot impute a fully missing time series.")
            if method == "zero":
                filled[sample, channel, series_mask] = 0.0
            elif method == "mean":
                filled[sample, channel, series_mask] = X[sample, channel, observed].mean()
            else:
                # np.interp also extends the nearest observed value at both ends.
                filled[sample, channel, series_mask] = np.interp(
                    time[series_mask], time[observed], X[sample, channel, observed]
                )
    if np.isnan(filled).any():
        raise RuntimeError("Imputation left NaN values behind.")
    return filled
