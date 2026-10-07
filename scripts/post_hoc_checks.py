"""Post hoc sensitivity checks reported in the paper (not prespecified).

All checks use the target-paired loss in the selected candidate's test balanced
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

Usage:
    python scripts/post_hoc_checks.py --results-dir results/final_ucr64_v1_4 \
        --config configs/main_protocol_balanced.yaml \
        --output results/final_ucr64_v1_4/analysis_v2/tables/post_hoc_checks.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.paired_analysis import load_and_validate, pair_by_target, summarize_datasets

N_BOOTSTRAP, SEED = 10000, 20260928
PAIR_KEYS = ["dataset", "seed", "missing_rate", "imputer", "target_pattern"]


def _row(name: str, per_dataset: pd.Series) -> dict:
    return {"check": name} | summarize_datasets(per_dataset, N_BOOTSTRAP, SEED)


def missing_class_splits(inventory: Path) -> set[tuple[str, int]]:
    table = pd.read_csv(inventory).dropna(subset=["validation_missing_classes"])
    return {(row.dataset, int(item.split(":")[0]))
            for row in table.itertuples() for item in row.validation_missing_classes.split(";")}


def selected_macro_f1_loss(results_dir: Path, selection: pd.DataFrame) -> pd.DataFrame:
    """Target-paired loss in the selected candidate's test macro-F1."""
    raw = pd.read_csv(results_dir / "raw_results.csv")
    f1 = raw.set_index(["dataset", "seed", "missing_rate", "imputer", "source_pattern",
                        "target_pattern", "classifier"]).test_macro_f1
    keys = ["dataset", "seed", "missing_rate", "imputer", "source_pattern", "target_pattern",
            "selected_classifier"]
    selection = selection.assign(selected_f1=f1.reindex(pd.MultiIndex.from_frame(selection[keys])).to_numpy())
    matched = selection[selection.source_pattern == selection.target_pattern].set_index(PAIR_KEYS)
    mismatched = selection[selection.source_pattern != selection.target_pattern].set_index(PAIR_KEYS)
    loss = matched.selected_f1 - mismatched.selected_f1.reindex(matched.index)
    return loss.rename("delta_selected_test_macro_f1").reset_index()


def post_hoc_checks(results_dir: Path, config: dict, inventory: Path) -> pd.DataFrame:
    selection, _ = load_and_validate(results_dir, config)
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

    f1 = selected_macro_f1_loss(results_dir, selection)
    f1_by_target = f1.groupby(["dataset", "target_pattern"]).delta_selected_test_macro_f1.mean().unstack()
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
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
