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


def test_checkpoint_resume_skips_a_completed_dataset(monkeypatch, tmp_path):
    X = np.full((2, 1, 4), 1.0)
    y = np.array(["a", "b"])
    calls = []

    def fake_load(*args):
        calls.append(args[0])
        return X, y, X, y, X, y

    monkeypatch.setattr(benchmark, "load_ucr_with_validation", fake_load)
    monkeypatch.setattr(
        benchmark,
        "apply_mask",
        lambda X, *args: (X.copy(), np.zeros_like(X, dtype=bool)),
    )
    monkeypatch.setattr(benchmark, "make_classifier", lambda *args: _PerfectClassifier())
    config = {
        "datasets": ["Demo"], "seeds": [1], "missing_rate": 0.2,
        "source_patterns": ["point"], "target_patterns": ["block"],
        "imputers": ["linear"], "classifiers": ["dtw"], "validation_size": 0.25,
    }

    first_raw, first_selection = benchmark.run_config(config, checkpoint_dir=tmp_path)
    second_raw, second_selection = benchmark.run_config(config, checkpoint_dir=tmp_path)

    assert calls == ["Demo"]
    assert len(first_raw) == len(second_raw) == 1
    assert len(first_selection) == len(second_selection) == 1


def test_test_predictions_are_reused_across_source_patterns(monkeypatch):
    X_train = np.full((2, 1, 4), 10.0)
    X_val = np.full((2, 1, 4), 20.0)
    X_test = np.full((2, 1, 4), 30.0)
    y = np.array(["a", "b"])
    predictions = []

    class CountingClassifier(_PerfectClassifier):
        def predict(self, X):
            predictions.append(float(X[0, 0, 0]))
            return super().predict(X)

    monkeypatch.setattr(
        benchmark,
        "load_ucr_with_validation",
        lambda *args: (X_train, y, X_val, y, X_test, y),
    )
    def fake_mask(X, pattern, *args):
        masked = X.copy()
        if X is X_val:
            masked[:, :, 0] += 1 if pattern == "point" else 2
        return masked, np.zeros_like(X, dtype=bool)

    monkeypatch.setattr(benchmark, "apply_mask", fake_mask)
    monkeypatch.setattr(benchmark, "make_classifier", lambda *args: CountingClassifier())
    config = {
        "datasets": ["Demo"], "seeds": [1], "missing_rate": 0.2,
        "source_patterns": ["point", "block"], "target_patterns": ["point", "block"],
        "imputers": ["linear"], "classifiers": ["dtw"], "validation_size": 0.25,
    }

    raw, selection = benchmark.run_config(config)

    assert predictions == [21.0, 30.0, 30.0, 22.0]
    assert len(raw) == len(selection) == 4
    assert raw["test_balanced_accuracy"].nunique() == 1
