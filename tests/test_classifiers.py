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


def test_batched_dtw_predicts_exactly_like_aeon_including_distance_ties():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(30, 1, 40))
    # Duplicate series with different labels create exact distance ties, which
    # aeon breaks by the smaller training index.
    X_train[10] = X_train[3]
    X_train[20] = X_train[3]
    y_train = np.array(["a", "b", "c"] * 10)
    X_query = np.concatenate([rng.normal(size=(25, 1, 40)), X_train[[3, 10, 20]]])

    batched = make_classifier("dtw", seed=1, params={"n_jobs": 4}).fit(X_train, y_train)
    reference = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="dtw").fit(X_train, y_train)

    assert np.array_equal(batched.predict(X_query), reference.predict(X_query))
    assert list(batched.predict(X_train[[10, 20]])) == [y_train[3], y_train[3]]


def test_statistical_feature_random_forest_fits_time_series_input():
    X = np.array([[[0.0, 1.0, 2.0]], [[2.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]], [[2.0, 2.0, 1.0]]])
    y = np.array(["a", "b", "a", "b"])
    classifier = make_classifier("stat_rf", seed=1, params={"stat_rf_n_estimators": 10})
    assert classifier.fit(X, y).predict(X).shape == (4,)
