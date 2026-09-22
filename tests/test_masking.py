import numpy as np
import pytest

from src.masking import apply_mask


def test_point_mask_is_reproducible_and_does_not_mutate_input():
    X = np.arange(2 * 1 * 10).reshape(2, 1, 10)
    original = X.copy()
    masked_a, mask_a = apply_mask(X, "point", 0.2, seed=7)
    masked_b, mask_b = apply_mask(X, "point", 0.2, seed=7)
    assert np.array_equal(X, original)
    assert np.array_equal(mask_a, mask_b)
    assert np.array_equal(mask_a.sum(axis=-1), np.full((2, 1), 2))
    assert np.isnan(masked_a[mask_a]).all()
    assert np.array_equal(masked_a[~mask_a], original[~mask_a])


def test_block_mask_is_contiguous_and_has_floor_count():
    X = np.zeros((3, 2, 11))
    _, mask = apply_mask(X, "block", 0.28, seed=11)
    assert np.array_equal(mask.sum(axis=-1), np.full((3, 2), 3))
    for series_mask in mask.reshape(-1, 11):
        assert np.all(np.diff(np.flatnonzero(series_mask)) == 1)


def test_prefix_and_suffix_masks_are_at_the_expected_endpoints():
    X = np.zeros((1, 1, 10))
    _, prefix = apply_mask(X, "prefix", 0.3, seed=1)
    _, suffix = apply_mask(X, "suffix", 0.3, seed=1)
    assert np.array_equal(prefix[0, 0], [True, True, True, False, False, False, False, False, False, False])
    assert np.array_equal(suffix[0, 0], [False, False, False, False, False, False, False, True, True, True])


@pytest.mark.parametrize("rate", [-0.1, 1.1])
def test_invalid_rate_raises(rate):
    with pytest.raises(ValueError):
        apply_mask(np.zeros((1, 1, 4)), "point", rate, 1)
