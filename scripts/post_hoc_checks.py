"""Post hoc sensitivity checks reported in the paper (not prespecified).

Most checks use the target-paired loss in the selected candidate's test balanced
accuracy, averaged within datasets and summarized with the same bootstrap and
Wilcoxon procedure as the formal analysis (``summarize_datasets``):

* ``all``: every target pair (reproduces the formal overall result);
* ``point_minus_block``: per-dataset point-test minus block-test loss;
* ``no_missing_class_splits``: drops dataset-seed splits whose validation subset
  lacks a class, as listed in ``results/dataset_inventory.csv``;
* ``no_phalanx_tasks``: drops the nine phalanx outline tasks (one task family);
* ``leave_one_dataset_out``: range of the mean when one dataset is omitted;
* ``macro_f1_*``: the same contrast measured with the selected candidate's test
  macro-F1 instead of balanced accuracy (selection still uses balanced accuracy).

Two further rules are compared with the archived selections, in test balanced
accuracy averaged over both target patterns (rows named ``a_minus_b`` hold the
per-dataset value of a minus b):

* ``mixed`` selects by the mean of the point- and block-masked validation balanced
  accuracies, so it needs no knowledge of the deployment pattern; ties are broken
  as in the main study, with ``mixed`` in place of the validation pattern;
* ``fixed_minirocket`` always uses MiniRocket-Ridge, without any selection.

Two descriptive Spearman correlations are reported over datasets:

* MiniRocket-Ridge's share of the point-test oracle (in PP) against the
  point-minus-block loss;
* the validation-subset size against the absolute dataset-level loss.

Usage:
    python scripts/post_hoc_checks.py --results-dir results/final_ucr64_v1_4 \
        --config configs/main_protocol_balanced.yaml \
        --output results/final_ucr64_v1_4/analysis_v2/tables/post_hoc_checks.csv
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from src.evaluation import _validation_tie_seed
from src.paired_analysis import load_and_validate, pair_by_target, summarize_datasets

N_BOOTSTRAP, SEED = 10000, 20260928
PAIR_KEYS = ["dataset", "seed", "missing_rate", "imputer", "target_pattern"]
CONDITION_KEYS = ["dataset", "seed", "missing_rate", "imputer"]


def _row(name: str, per_dataset: pd.Series) -> dict:
    return {"check": name} | summarize_datasets(per_dataset, N_BOOTSTRAP, SEED)


def _spearman(name: str, x: pd.Series, y: pd.Series) -> dict:
    x, y = x.align(y, join="inner")
    rho, p_value = spearmanr(x, y)
    return {"check": name, "n_datasets": len(x), "spearman_rho": rho, "spearman_p": p_value}


def missing_class_splits(inventory: Path) -> set[tuple[str, int]]:
    table = pd.read_csv(inventory).dropna(subset=["validation_missing_classes"])
    return {(row.dataset, int(item.split(":")[0]))
            for row in table.itertuples() for item in row.validation_missing_classes.split(";")}


def selected_macro_f1_loss(raw: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    """Target-paired loss in the selected candidate's test macro-F1."""
    f1 = raw.set_index(["dataset", "seed", "missing_rate", "imputer", "source_pattern",
                        "target_pattern", "classifier"]).test_macro_f1
    keys = ["dataset", "seed", "missing_rate", "imputer", "source_pattern", "target_pattern",
            "selected_classifier"]
    selection = selection.assign(selected_f1=f1.reindex(pd.MultiIndex.from_frame(selection[keys])).to_numpy())
    matched = selection[selection.source_pattern == selection.target_pattern].set_index(PAIR_KEYS)
    mismatched = selection[selection.source_pattern != selection.target_pattern].set_index(PAIR_KEYS)
    loss = matched.selected_f1 - mismatched.selected_f1.reindex(matched.index)
    return loss.rename("delta_selected_test_macro_f1").reset_index()


def alternative_rules(raw: pd.DataFrame, selection: pd.DataFrame, block: str) -> pd.DataFrame:
    """Test BA of the matched, mismatched, mixed-validation and fixed-MiniRocket choices."""
    val = raw.groupby(CONDITION_KEYS + ["source_pattern", "classifier"]).val_balanced_accuracy.first()
    test = raw.groupby(CONDITION_KEYS + ["target_pattern", "classifier"]).test_balanced_accuracy.first()
    rows = []
    for key, scores in val.groupby(level=CONDITION_KEYS):
        scores = scores.droplevel(CONDITION_KEYS).unstack("source_pattern")
        mixed = (scores["point"] + scores[block]) / 2
        tied = sorted(mixed.index[np.isclose(mixed, mixed.max(), rtol=0, atol=1e-12)])
        seed = _validation_tie_seed(dict(zip(CONDITION_KEYS, key)) | {"source_pattern": "mixed"})
        pick = tied[int(np.random.default_rng(seed).integers(len(tied)))]
        for target in ("point", block):
            rows.append(dict(zip(CONDITION_KEYS, key)) | {
                "target_pattern": target,
                "mixed": test[key + (target, pick)],
                "fixed_minirocket": test[key + (target, "minirocket")],
            })
    rules = pd.DataFrame(rows).set_index(PAIR_KEYS)
    is_matched = selection.source_pattern == selection.target_pattern
    rules["matched"] = selection[is_matched].set_index(PAIR_KEYS).selected_test_balanced_accuracy
    rules["mismatched"] = selection[~is_matched].set_index(PAIR_KEYS).selected_test_balanced_accuracy
    return rules.reset_index()


def post_hoc_checks(results_dir: Path, config: dict, inventory: Path) -> pd.DataFrame:
    selection, _ = load_and_validate(results_dir, config)
    raw = pd.read_csv(results_dir / "raw_results.csv")
    pairs = pair_by_target(selection)
    loss = "delta_selected_test_ba"
    per_dataset = pairs.groupby("dataset")[loss].mean()
    by_target = pairs.groupby(["dataset", "target_pattern"])[loss].mean().unstack()
    block = next(pattern for pattern in by_target.columns if pattern != "point")

    split = list(zip(pairs.dataset, pairs.seed))
    excluded = missing_class_splits(inventory)
    kept = pairs[[key not in excluded for key in split]]
    phalanx = [name for name in per_dataset.index if "Phalan" in name]
    leave_one_out = [per_dataset.drop(name).mean() for name in per_dataset.index]

    f1 = selected_macro_f1_loss(raw, selection)
    f1_by_target = f1.groupby(["dataset", "target_pattern"]).delta_selected_test_macro_f1.mean().unstack()

    rules = alternative_rules(raw, selection, block)
    rule_means = rules.groupby("dataset")[["matched", "mismatched", "mixed", "fixed_minirocket"]].mean()
    point_rules = rules[rules.target_pattern == "point"].groupby("dataset")[["matched", "fixed_minirocket"]].mean()

    pp = selection[(selection.source_pattern == "point") & (selection.target_pattern == "point")]
    minirocket_share = pp.oracle_classifier.str.split("|").apply(
        lambda names: 1 / len(names) if "minirocket" in names else 0.0).groupby(pp.dataset).mean()
    sizes = pd.read_csv(inventory).set_index("dataset").n_train
    validation_size = sizes.apply(lambda n: math.ceil(config["validation_size"] * n))

    rows = [
        _row("all", per_dataset),
        _row("point_minus_block", by_target["point"] - by_target[block]),
        _row("no_missing_class_splits", kept.groupby("dataset")[loss].mean())
        | {"n_pairs": len(kept), "n_excluded_splits": len(excluded)},
        _row(f"no_phalanx_tasks ({len(phalanx)})", per_dataset.drop(phalanx)),
        {"check": "leave_one_dataset_out", "n_datasets": len(per_dataset) - 1,
         "mean_min": min(leave_one_out), "mean_max": max(leave_one_out)},
        _row("macro_f1_all", f1.groupby("dataset").delta_selected_test_macro_f1.mean()),
        _row("macro_f1_point_test", f1_by_target["point"]),
        _row("macro_f1_block_test", f1_by_target[block]),
        _row("matched_minus_mixed", rule_means.matched - rule_means.mixed),
        _row("mixed_minus_mismatched", rule_means.mixed - rule_means.mismatched),
        _row("fixed_minirocket_minus_matched", rule_means.fixed_minirocket - rule_means.matched),
        _row("fixed_minirocket_minus_matched_point_test", point_rules.fixed_minirocket - point_rules.matched),
        _spearman("spearman_minirocket_point_oracle_share_vs_point_minus_block",
                  minirocket_share, by_target["point"] - by_target[block]),
        _spearman("spearman_validation_size_vs_abs_loss", validation_size, per_dataset.abs()),
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--inventory", default="results/dataset_inventory.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    table = post_hoc_checks(Path(args.results_dir), config, Path(args.inventory))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
