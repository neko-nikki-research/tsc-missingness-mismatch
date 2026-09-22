"""Create figures, descriptive tables, and paired tests from selection results."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon


def add_condition(selection: pd.DataFrame) -> pd.DataFrame:
    result = selection.copy()
    result["condition"] = result.apply(
        lambda row: "matched" if row.source_pattern == row.target_pattern else "mismatched",
        axis=1,
    )
    return result


def paired_wilcoxon(selection: pd.DataFrame) -> pd.DataFrame:
    """Compare target mismatch against the same source-pattern matched run."""
    keys = ["dataset", "seed", "missing_rate", "imputer", "source_pattern"]
    matched = selection[selection.condition == "matched"].set_index(keys)["regret"]
    mismatched = selection[selection.condition == "mismatched"].set_index(keys)["regret"]
    paired = pd.concat({"matched": matched, "mismatched": mismatched}, axis=1).dropna()
    difference = paired["mismatched"] - paired["matched"]
    statistic, p_value = wilcoxon(difference, alternative="greater", zero_method="pratt")
    return pd.DataFrame([{
        "n_pairs": len(difference),
        "mean_matched_regret": paired["matched"].mean(),
        "mean_mismatched_regret": paired["mismatched"].mean(),
        "mean_difference_mismatch_minus_matched": difference.mean(),
        "wilcoxon_statistic": statistic,
        "one_sided_p_value": p_value,
        "alternative": "mismatched regret > matched regret",
    }])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    selection = add_condition(pd.read_csv(results_dir / "selection_results.csv"))
    figures_dir, tables_dir = results_dir / "figures", results_dir / "tables"
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    summary = (selection.groupby(["missing_rate", "condition"], as_index=False)
               .agg(n_runs=("regret", "size"), mean_regret=("regret", "mean"),
                    std_regret=("regret", "std"), selection_error_rate=("selection_error", "mean"))
               .round(6))
    summary.to_csv(tables_dir / "regret_by_rate_and_condition.csv", index=False)
    paired_wilcoxon(selection).round(6).to_csv(tables_dir / "paired_wilcoxon.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for condition, group in summary.groupby("condition"):
        group = group.sort_values("missing_rate")
        axis.errorbar(group["missing_rate"] * 100, group["mean_regret"], yerr=group["std_regret"],
                      marker="o", capsize=4, linewidth=2, label=condition)
    axis.set_xlabel("Missing rate (%)")
    axis.set_ylabel("Selection regret")
    axis.set_title("Selection regret by missing rate and pattern condition")
    axis.legend(title="Pattern condition")
    figure.tight_layout()
    figure.savefig(figures_dir / "regret_by_rate.png", dpi=180)
    plt.close(figure)
    print(f"Saved report files under {results_dir}")


if __name__ == "__main__":
    main()
