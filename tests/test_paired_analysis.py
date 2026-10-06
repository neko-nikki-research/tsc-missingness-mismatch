import numpy as np
import pandas as pd
import pytest

from src.evaluation import selection_summary
from src.paired_analysis import (
    holm,
    load_and_validate,
    pair_by_source,
    pair_by_target,
    rate_slope,
    run_analysis,
)

CONFIG = {
    "datasets": ["A", "B", "C"],
    "seeds": [1, 2],
    "missing_rates": [0.1, 0.2],
    "source_patterns": ["point", "block"],
    "target_patterns": ["point", "block"],
    "imputers": ["linear"],
    "classifiers": ["dtw", "minirocket", "stat_rf"],
}


def _write_results(path, datasets):
    """Synthetic results that respect the design: validation BA ignores the
    target pattern and test BA ignores the source pattern."""
    rng = np.random.default_rng(0)
    raw = []
    for dataset in datasets:
        for seed in CONFIG["seeds"]:
            for rate in CONFIG["missing_rates"]:
                val = {(s, c): rng.random() for s in ("point", "block") for c in CONFIG["classifiers"]}
                test = {(t, c): rng.random() for t in ("point", "block") for c in CONFIG["classifiers"]}
                for source in ("point", "block"):
                    for target in ("point", "block"):
                        for classifier in CONFIG["classifiers"]:
                            raw.append({
                                "dataset": dataset, "seed": seed, "missing_rate": rate,
                                "n_timepoints": 100, "n_missing_per_series": int(rate * 100),
                                "realized_missing_rate": rate, "source_pattern": source,
                                "target_pattern": target, "imputer": "linear",
                                "classifier": classifier,
                                "val_balanced_accuracy": val[source, classifier],
                                "test_balanced_accuracy": test[target, classifier],
                            })
    raw = pd.DataFrame(raw)
    keys = ["dataset", "seed", "missing_rate", "source_pattern", "target_pattern", "imputer"]
    selection = pd.DataFrame(
        [selection_summary(group.to_dict("records")) for _, group in raw.groupby(keys)]
    )
    path.mkdir(parents=True, exist_ok=True)
    raw.to_csv(path / "raw_results.csv", index=False)
    selection.to_csv(path / "selection_results.csv", index=False)
    return raw, selection


def test_target_pairing_shares_the_oracle_so_ba_and_regret_differences_agree(tmp_path):
    _write_results(tmp_path, CONFIG["datasets"])
    selection, _ = load_and_validate(tmp_path, CONFIG)
    pairs = pair_by_target(selection)
    # 3 datasets x 2 seeds x 2 rates x 2 target patterns
    assert len(pairs) == 24
    assert np.allclose(pairs.delta_selected_test_ba, pairs.delta_regret)


def test_source_pairing_keeps_the_selected_model_fixed(tmp_path):
    _write_results(tmp_path, CONFIG["datasets"])
    selection, _ = load_and_validate(tmp_path, CONFIG)
    pairs = pair_by_source(selection)
    assert len(pairs) == 24
    assert (pairs.selection_changed == 0).all()


def test_across_dataset_statistics_use_one_value_per_dataset(tmp_path):
    results, output = tmp_path / "results", tmp_path / "analysis"
    _write_results(results, CONFIG["datasets"])
    run_analysis(results, CONFIG, output, allow_incomplete=False, n_bootstrap=200)
    overall = pd.read_csv(output / "tables" / "target_paired_overall.csv")
    assert (overall.n_datasets == 3).all()
    by_rate = pd.read_csv(output / "tables" / "target_paired_by_rate.csv")
    assert set(by_rate.n_datasets) == {3}
    assert (output / "ANALYSIS_STATUS.txt").read_text(encoding="utf-8").startswith("COMPLETE")


def test_incomplete_results_are_refused_unless_marked_interim(tmp_path):
    results, output = tmp_path / "results", tmp_path / "analysis"
    _write_results(results, ["A", "B"])
    with pytest.raises(ValueError, match="no results yet"):
        run_analysis(results, CONFIG, output, allow_incomplete=False, n_bootstrap=50)
    run_analysis(results, CONFIG, output, allow_incomplete=True, n_bootstrap=50)
    assert "INTERIM" in (output / "ANALYSIS_STATUS.txt").read_text(encoding="utf-8")


def test_tampered_regret_is_detected(tmp_path):
    _, selection = _write_results(tmp_path, CONFIG["datasets"])
    selection.loc[0, "regret"] += 0.01
    selection.to_csv(tmp_path / "selection_results.csv", index=False)
    with pytest.raises(ValueError, match="regret"):
        load_and_validate(tmp_path, CONFIG)


def test_output_must_not_be_the_checkpoint_directory(tmp_path):
    _write_results(tmp_path, CONFIG["datasets"])
    with pytest.raises(ValueError, match="separate directory"):
        run_analysis(tmp_path, CONFIG, tmp_path, allow_incomplete=False, n_bootstrap=50)


RATES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def test_rate_slope_keeps_exact_zeros_and_ties_for_count_metrics():
    # Each rate value is a count over 10 pairs, as for selection error/change.
    # Contrast weights (-5, -3, -1, 1, 3, 5) give equal numerators for a and b,
    # and zero numerators for flat and symmetric.
    a = np.array([0, 0, 0, 0, 0, 1]) / 10
    b = np.array([1, 0, 0, 0, 0, 2]) / 10
    flat = np.array([3, 3, 3, 3, 3, 3]) / 10
    symmetric = np.array([1, 0, 0, 0, 0, 1]) / 10
    assert rate_slope(RATES, a) == rate_slope(RATES, b) == pytest.approx(5 / 1750)
    assert rate_slope(RATES, flat) == 0.0
    assert rate_slope(RATES, symmetric) == 0.0


def test_rate_slope_matches_least_squares_for_continuous_values():
    values = np.random.default_rng(1).random(6)
    expected = np.polyfit(np.array(RATES) * 100, values, 1)[0]
    assert rate_slope(RATES, values) == pytest.approx(expected, abs=1e-12)


def test_holm_adjustment():
    adjusted = holm(pd.Series([0.01, 0.04, 0.03], index=["a", "b", "c"]))
    assert adjusted.to_dict() == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})
