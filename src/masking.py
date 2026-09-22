"""Synthetic temporal missingness mechanisms."""

import numpy as np


def apply_mask(X: np.ndarray, pattern: str, rate: float, seed: int):
    """Return a masked copy of X and its boolean missingness mask.

    X must have shape (n_samples, n_channels, n_timepoints). True means missing.
    Every sample-channel series receives floor(rate * n_timepoints) missing values.
    """
    X_array = np.asarray(X)
    if X_array.ndim != 3:
        raise ValueError("X must be a 3D array: (samples, channels, timepoints).")
    if pattern not in {"point", "block", "linear_block", "prefix", "suffix"}:
        raise ValueError("pattern must be point, block, linear_block, prefix, or suffix.")
    if not 0 <= rate <= 1:
        raise ValueError("rate must be between 0 and 1.")

    n_samples, n_channels, n_timepoints = X_array.shape
    n_missing = int(np.floor(rate * n_timepoints))
    rng = np.random.default_rng(seed)
    mask = np.zeros(X_array.shape, dtype=bool)
    if n_missing:
        for sample in range(n_samples):
            for channel in range(n_channels):
                if pattern == "point":
                    locations = rng.choice(n_timepoints, size=n_missing, replace=False)
                    mask[sample, channel, locations] = True
                elif pattern == "block":
                    # Main protocol: a circular contiguous block. A random
                    # start may wrap from the last timepoint back to the first.
                    start = rng.integers(0, n_timepoints)
                    locations = (start + np.arange(n_missing)) % n_timepoints
                    mask[sample, channel, locations] = True
                elif pattern == "linear_block":
                    # Supplementary robustness condition; never the main B.
                    start = rng.integers(0, n_timepoints - n_missing + 1)
                    mask[sample, channel, start : start + n_missing] = True
                elif pattern == "prefix":
                    mask[sample, channel, :n_missing] = True
                else:  # suffix
                    mask[sample, channel, n_timepoints - n_missing :] = True
    X_masked = X_array.astype(float, copy=True)
    X_masked[mask] = np.nan
    return X_masked, mask
