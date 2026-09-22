"""Summarise selection regret for matched and mismatched missingness."""

import argparse
from pathlib import Path

import pandas as pd


def add_condition(selection: pd.DataFrame) -> pd.DataFrame:
    """Label a run by whether validation and deployment patterns agree."""
    result = selection.copy()
    result["condition"] = result.apply(
        lambda row: "matched"
        if row["source_pattern"] == row["target_pattern"]
        else "mismatched",
        axis=1,
    )
    return result


def summarise(selection: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Compute transparent descriptive statistics for selection outcomes."""
    return (
        selection.groupby(groups, as_index=False)
        .agg(
            n_runs=("regret", "size"),
            mean_regret=("regret", "mean"),
            std_regret=("regret", "std"),
            selection_error_rate=("selection_error", "mean"),
            mean_selected_test_accuracy=("selected_test_accuracy", "mean"),
            mean_oracle_test_accuracy=("oracle_test_accuracy", "mean"),
        )
        .round(6)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Directory containing selection_results.csv")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    selection = pd.read_csv(results_dir / "selection_results.csv")
    labelled = add_condition(selection)
    summarise(labelled, ["condition"]).to_csv(
        results_dir / "summary_by_condition.csv", index=False
    )
    summarise(labelled, ["dataset", "condition"]).to_csv(
        results_dir / "summary_by_dataset_condition.csv", index=False
    )
    summarise(labelled, ["imputer", "condition"]).to_csv(
        results_dir / "summary_by_imputer_condition.csv", index=False
    )
    print(f"Saved summaries to {results_dir}")


if __name__ == "__main__":
    main()
