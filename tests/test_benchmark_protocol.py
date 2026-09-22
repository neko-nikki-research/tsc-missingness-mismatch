import numpy as np

import src.run_benchmark as benchmark


class _PerfectClassifier:
    def fit(self, X, y):
        self.train_X = X.copy()
        return self

    def predict(self, X):
        return np.array(["a", "b"])


def test_source_mask_is_validation_only_and_training_stays_complete(monkeypatch):
    X_train = np.full((2, 1, 4), 10.0)
    X_val = np.full((2, 1, 4), 20.0)
    X_test = np.full((2, 1, 4), 30.0)
    y = np.array(["a", "b"])
    mask_inputs = []
    fitted = []

    monkeypatch.setattr(
        benchmark,
        "load_ucr_with_validation",
        lambda *args: (X_train, y, X_val, y, X_test, y),
    )

    def fake_mask(X, pattern, rate, seed):
        mask_inputs.append(X.copy())
        return X.copy(), np.zeros_like(X, dtype=bool)

    def fake_classifier(*args):
        classifier = _PerfectClassifier()
        fitted.append(classifier)
        return classifier

    monkeypatch.setattr(benchmark, "apply_mask", fake_mask)
    monkeypatch.setattr(benchmark, "make_classifier", fake_classifier)
    config = {
        "datasets": ["Demo"], "seeds": [1], "missing_rate": 0.2,
        "source_patterns": ["point"], "target_patterns": ["block"],
        "imputers": ["linear"], "classifiers": ["dtw"], "validation_size": 0.25,
    }

    benchmark.run_config(config)

    assert len(mask_inputs) == 2
    assert np.array_equal(mask_inputs[0], X_val)
    assert np.array_equal(mask_inputs[1], X_test)
    assert np.array_equal(fitted[0].train_X, X_train)
