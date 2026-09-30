"""Classifier registry for the pre-specified protocol candidate methods."""

from aeon.classification.convolution_based import MiniRocketClassifier
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV

from src.statistical_features import extract_statistical_features


class BatchedKNeighborsTimeSeriesClassifier(KNeighborsTimeSeriesClassifier):
    """aeon's k-NN, with every query sent to the distance function in one call.

    aeon's ``_predict`` loops over one query series at a time. Its numba distance
    kernel parallelises over queries, so that loop runs on a single thread
    whatever ``n_jobs`` is. For uniform 1-NN this override computes the same
    distances with aeon's own ``_kneighbors`` (same (distance, index) tie rule)
    for all queries at once, so predictions are unchanged but use ``n_jobs``.
    """

    def _predict(self, X):
        if self.n_neighbors != 1 or self.weights != "uniform":
            return super()._predict(X)
        neigh_ind = self._kneighbors(X, n_neighbors=1, return_distance=False, query_is_train=False)
        return self.classes_[self.y_[neigh_ind[:, 0]]]


class StatisticalFeaturesRandomForestClassifier(BaseEstimator, ClassifierMixin):
    """Fixed statistical-feature extractor followed by a Random Forest."""

    def __init__(self, n_estimators: int = 500, random_state: int | None = None, n_jobs: int = 1):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(extract_statistical_features(X), y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(extract_statistical_features(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(extract_statistical_features(X))


def make_classifier(name: str, seed: int, params: dict):
    """Create one classifier; hyperparameters are fixed before test evaluation."""
    n_jobs = int(params.get("n_jobs", 1))
    if n_jobs < 1:
        raise ValueError("classifier_params.n_jobs must be at least 1.")
    if name == "dtw":
        return BatchedKNeighborsTimeSeriesClassifier(n_neighbors=1, distance="dtw", n_jobs=n_jobs)
    if name == "stat_rf":
        return StatisticalFeaturesRandomForestClassifier(
            n_estimators=int(params.get("stat_rf_n_estimators", 500)), random_state=seed, n_jobs=n_jobs
        )
    if name == "minirocket":
        return MiniRocketClassifier(
            n_kernels=int(params.get("minirocket_n_kernels", 10000)),
            # aeon applies its built-in sparse scaling before this linear head.
            estimator=RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
            random_state=seed,
            n_jobs=n_jobs,
        )
    raise ValueError(f"Unknown classifier: {name}")
