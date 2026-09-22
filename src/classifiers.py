"""A small, explicit classifier registry using aeon."""

from aeon.classification.convolution_based import MiniRocketClassifier
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.interval_based import TimeSeriesForestClassifier


def make_classifier(name: str, seed: int, params: dict):
    """Create one classifier; hyperparameters are fixed before test evaluation."""
    if name == "dtw":
        return KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="dtw", n_jobs=1)
    if name == "tsf":
        return TimeSeriesForestClassifier(
            n_estimators=int(params.get("tsf_n_estimators", 200)), random_state=seed, n_jobs=1
        )
    if name == "minirocket":
        return MiniRocketClassifier(
            n_kernels=int(params.get("minirocket_n_kernels", 10000)), random_state=seed, n_jobs=1
        )
    raise ValueError(f"Unknown classifier: {name}")
