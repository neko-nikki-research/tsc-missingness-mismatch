import numpy as np
from aeon.classification.convolution_based import MiniRocketClassifier
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from sklearn.linear_model import RidgeClassifierCV

from src.classifiers import StatisticalFeaturesRandomForestClassifier, make_classifier


def test_protocol_classifier_registry_uses_the_specified_implementations():
    dtw = make_classifier("dtw", seed=1, params={})
    minirocket = make_classifier("minirocket", seed=1, params={"minirocket_n_kernels": 1000})
    stat_rf = make_classifier("stat_rf", seed=1, params={"stat_rf_n_estimators": 20})
    assert isinstance(dtw, KNeighborsTimeSeriesClassifier)
    assert dtw.n_neighbors == 1
    assert dtw.distance == "dtw"
    assert isinstance(minirocket, MiniRocketClassifier)
    assert isinstance(minirocket.estimator, RidgeClassifierCV)
    assert minirocket.n_kernels == 1000
    assert isinstance(stat_rf, StatisticalFeaturesRandomForestClassifier)
    assert stat_rf.n_estimators == 20


def test_statistical_feature_random_forest_fits_time_series_input():
    X = np.array([[[0.0, 1.0, 2.0]], [[2.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]], [[2.0, 2.0, 1.0]]])
    y = np.array(["a", "b", "a", "b"])
    classifier = make_classifier("stat_rf", seed=1, params={"stat_rf_n_estimators": 10})
    assert classifier.fit(X, y).predict(X).shape == (4,)
