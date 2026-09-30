import numpy as np
import pandas as pd
import pytest
import yaml

import src.run_benchmark as benchmark
from src import plot_final_results
from src.assemble_linear_block import assemble
from src.evaluation import selection_summary
from src.paired_analysis import load_and_validate, pair_by_target

CLASSIFIERS = ["dtw", "minirocket", "stat_rf"]
KEYS = ["dataset", "seed", "missing_rate", "source_pattern", "target_pattern", "imputer"]
SUPPLEMENTARY = {
    "datasets": ["A", "B", "C"], "seeds": [1, 2], "missing_rates": [0.1, 0.2],
    "source_patterns": ["point", "linear_block"], "target_patterns": ["point", "linear_block"],
    "condition_pairs": [["point", "linear_block"], ["linear_block", "point"],
                        ["linear_block", "linear_block"]],
    "imputers": ["linear"], "classifiers": CLASSIFIERS,
}


def _results(block, pairs, point_scores, rng):
    """Synthetic results obeying the design: validation BA depends only on the
    source pattern and test BA only on the target pattern."""
    raw = []
    for dataset in SUPPLEMENTARY["datasets"]:
        for seed in SUPPLEMENTARY["seeds"]:
            for rate in SUPPLEMENTARY["missing_rates"]:
                point_val, point_test = point_scores[dataset, seed, rate]
                val = {("point", c): point_val[c] for c in CLASSIFIERS} | {
                    (block, c): rng.random() for c in CLASSIFIERS}
                test = {("point", c): point_test[c] for c in CLASSIFIERS} | {
                    (block, c): rng.random() for c in CLASSIFIERS}
                for source, target in pairs:
                    for classifier in CLASSIFIERS:
                        raw.append({
                            "dataset": dataset, "seed": seed, "missing_rate": rate,
                            "n_timepoints": 100, "n_missing_per_series": int(rate * 100),
                            "realized_missing_rate": rate, "source_pattern": source,
                            "target_pattern": target, "imputer": "linear", "classifier": classifier,
                            "val_balanced_accuracy": val[source, classifier],
                            "test_balanced_accuracy": test[target, classifier],
                        })
    raw = pd.DataFrame(raw)
    selection = pd.DataFrame([selection_summary(g.to_dict("records")) for _, g in raw.groupby(KEYS)])
    return raw, selection


def _write(path, raw, selection):
    path.mkdir(parents=True, exist_ok=True)
    raw.to_csv(path / "raw_results.csv", index=False)
    selection.to_csv(path / "selection_results.csv", index=False)


def _main_and_supplementary(tmp_path, tamper=False):
    rng = np.random.default_rng(0)
    point_scores = {
        (d, s, r): ({c: rng.random() for c in CLASSIFIERS}, {c: rng.random() for c in CLASSIFIERS})
        for d in SUPPLEMENTARY["datasets"] for s in SUPPLEMENTARY["seeds"]
        for r in SUPPLEMENTARY["missing_rates"]
    }
    full = [(s, t) for s in ("point", "block") for t in ("point", "block")]
    main_dir, new_dir = tmp_path / "main", tmp_path / "linear"
    _write(main_dir, *_results("block", full, point_scores, rng))
    new_raw, new_selection = _results("linear_block", [tuple(p) for p in SUPPLEMENTARY["condition_pairs"]],
                                      point_scores, rng)
    if tamper:
        row = new_raw.index[new_raw.source_pattern == "point"][0]
        new_raw.loc[row, "val_balanced_accuracy"] += 0.01
    _write(new_dir, new_raw, new_selection)
    return main_dir, new_dir


def test_condition_pairs_run_only_the_listed_conditions(monkeypatch, tmp_path):
    X = np.full((2, 1, 20), 1.0)
    y = np.array(["a", "b"])

    class Classifier:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.array(["a", "b"])

    monkeypatch.setattr(benchmark, "load_ucr_with_validation", lambda *args: (X, y, X, y, X, y))
    monkeypatch.setattr(benchmark, "make_classifier", lambda *args: Classifier())
    config = SUPPLEMENTARY | {"datasets": ["Demo"], "validation_size": 0.25}

    raw, selection = benchmark.run_config(config, checkpoint_dir=tmp_path)

    pairs = set(zip(selection.source_pattern, selection.target_pattern))
    assert pairs == {("point", "linear_block"), ("linear_block", "point"), ("linear_block", "linear_block")}
    assert len(selection) == 2 * 2 * 3 and len(raw) == 2 * 2 * 3 * 3
    assert benchmark._expected_rows_per_dataset(config) == (36, 12)
    # A resumed run recognises the three-condition dataset as complete.
    assert benchmark._completed_datasets(raw, selection, config) == {"Demo"}


def test_condition_pairs_must_be_part_of_the_design():
    config = SUPPLEMENTARY | {"condition_pairs": [["point", "block"]]}
    with pytest.raises(ValueError, match="condition_pairs"):
        benchmark._condition_pairs(config)


def test_assembled_results_reuse_pp_and_feed_the_paired_analysis(tmp_path):
    main_dir, new_dir = _main_and_supplementary(tmp_path)
    raw, selection = assemble(main_dir, new_dir)
    out = tmp_path / "assembled"
    _write(out, raw, selection)

    loaded, datasets = load_and_validate(out, SUPPLEMENTARY)
    assert datasets == ["A", "B", "C"]
    assert set(zip(loaded.source_pattern, loaded.target_pattern)) == {
        ("point", "point"), ("point", "linear_block"), ("linear_block", "point"),
        ("linear_block", "linear_block")}
    main_pp = pd.read_csv(main_dir / "selection_results.csv").query(
        "source_pattern == 'point' and target_pattern == 'point'")
    assert len(loaded.query("source_pattern == 'point' and target_pattern == 'point'")) == len(main_pp)
    pairs = pair_by_target(loaded)
    assert set(pairs.target_pattern) == {"point", "linear_block"}


def test_point_side_mismatch_blocks_pp_reuse(tmp_path):
    main_dir, new_dir = _main_and_supplementary(tmp_path, tamper=True)
    with pytest.raises(ValueError, match="PP cannot be reused"):
        assemble(main_dir, new_dir)


def test_figures_for_the_linear_block_analysis(tmp_path, monkeypatch):
    main_dir, new_dir = _main_and_supplementary(tmp_path)
    out = tmp_path / "assembled"
    _write(out, *assemble(main_dir, new_dir))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(SUPPLEMENTARY), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["plot", "--results-dir", str(out), "--config", str(config_path),
                                     "--figures-dir", str(tmp_path / "figures"), "--n-bootstrap", "50"])
    plot_final_results.main()
    assert len(list((tmp_path / "figures").glob("*.png"))) == 4
