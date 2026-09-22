import numpy as np
import pytest

from src.statistical_features import FEATURE_NAMES, extract_statistical_features


def test_statistical_features_have_one_fixed_vector_per_sample_and_channel():
    X = np.array([[[1.0, 2.0, 3.0]], [[3.0, 3.0, 3.0]]])
    features = extract_statistical_features(X)
    assert features.shape == (2, len(FEATURE_NAMES))
    assert features[0, 0] == 2.0
    assert features[0, 6] == 1.0
    assert features[1, -1] == 0.0


def test_statistical_features_reject_nan_input():
    with pytest.raises(ValueError, match="no NaN"):
        extract_statistical_features(np.array([[[1.0, np.nan]]]))
