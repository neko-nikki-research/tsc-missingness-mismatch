"""Plot descriptive interim results for the point/block benchmark.

These figures are intentionally descriptive: a partial dataset run is not a
confirmatory statistical analysis of the full 64-dataset study.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PATTERNS = ("point", "block")
CONDITIONS = (
    ("point", "point"),
    ("point", "block"),
    ("block", "point"),
    ("block", "block"),
)


def load_and_check(results_dir: Path, config_path: Path) -> pd.DataFrame:
    """Require a complete, balanced checkpoint before drawing any figure."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    results = pd.read_csv(results_dir / "selection_results.csv")
    required = {
        "dataset", "seed", "missing_rate", "source_pattern", "target_pattern",
        "imputer", "selection_error", "regret",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if results.empty:
        raise ValueError("No selection results to plot")
    if set(config["source_patterns"]) != set(PATTERNS) or set(config["target_patterns"]) != set(PATTERNS):
        raise ValueError("This interim plot expects the point/block 2x2 design")
    if results.dataset.nunique() > len(config["datasets"]):
        raise ValueError("Snapshot includes datasets outside the configured study")
    if not set(results.dataset).issubset(config["datasets"]):
        raise ValueError("Snapshot includes an unconfigured dataset")
    for column, configured in (
        ("seed", config["seeds"]),
        ("missing_rate", config["missing_rates"]),
        ("source_pattern", config["source_patterns"]),
        ("target_pattern", config["target_patterns"]),
        ("imputer", config["imputers"]),
    ):
        if not set(results[column]).issubset(configured):
            raise ValueError(f"Unexpected {column} value in snapshot")
    expected_per_dataset = (
        len(config["seeds"]) * len(config["missing_rates"])
        * len(CONDITIONS) * len(config["imputers"])
    )
    counts = results.groupby("dataset").size()
    if not counts.eq(expected_per_dataset).all():
        raise ValueError(f"Incomplete dataset in snapshot: {counts.to_dict()}")
    keys = ["dataset", "seed", "missing_rate", "source_pattern", "target_pattern", "imputer"]
    if results.duplicated(keys).any():
        raise ValueError("Duplicate selection condition")
    if results[keys + ["regret"]].isna().any().any():
        raise ValueError("Missing condition or regret value")
    if (results.regret < -1e-12).any() or (results.regret > 1 + 1e-12).any():
        raise ValueError("Selection regret must be between 0 and 1")
    errors = results.selection_error.astype(str).str.lower().map({"true": 1.0, "false": 0.0})
    if errors.isna().any():
        raise ValueError("selection_error must contain only True or False")
    results["selection_error"] = errors
    return results


def summarize(selection: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average equally weighted, complete dataset-seed conditions."""
    groups = ["source_pattern", "target_pattern"]
    summary = selection.groupby(["missing_rate", *groups], as_index=False).agg(
        n_conditions=("regret", "size"),
        mean_regret=("regret", "mean"),
        selection_error_rate=("selection_error", "mean"),
    )
    overall = selection.groupby(groups, as_index=False).agg(
        n_conditions=("regret", "size"),
        mean_regret=("regret", "mean"),
        selection_error_rate=("selection_error", "mean"),
    )
    return summary, overall


def save_figures(selection: pd.DataFrame, output_dir: Path) -> None:
    """Write two Matplotlib figures and their underlying aggregate tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)
    summary, overall = summarize(selection)
    summary.to_csv(tables_dir / "by_rate_and_pattern.csv", index=False)
    overall.to_csv(tables_dir / "by_pattern_pair.csv", index=False)
    n_datasets = selection.dataset.nunique()
    n_seeds = selection.seed.nunique()
    colors = {"point": "#2366a8", "block": "#d37425"}
    markers = {"point": "o", "block": "s"}

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for source, target in CONDITIONS:
        rows = summary.loc[
            (summary.source_pattern == source) & (summary.target_pattern == target)
        ].sort_values("missing_rate")
        label = f"{source} → {target}"
        style = "-" if source == target else "--"
        common = dict(label=label, color=colors[source], marker=markers[target],
                      linestyle=style, linewidth=2, markersize=5)
        axes[0].plot(rows.missing_rate * 100, rows.mean_regret * 100, **common)
        axes[1].plot(rows.missing_rate * 100, rows.selection_error_rate * 100, **common)
    axes[0].set_ylabel("Mean selection regret (percentage points)")
    axes[1].set_ylabel("Selection error rate (%)")
    for axis in axes:
        axis.set_xlabel("Nominal missing rate (%)")
        axis.set_xticks(sorted(selection.missing_rate.unique() * 100))
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(bottom=0)
    axes[1].legend(loc="best", title="Validation → test", fontsize=8)
    figure.suptitle(f"INTERIM: {n_datasets}/64 UCR datasets | {n_seeds} seeds | linear imputation")
    figure.savefig(figures_dir / "outcomes_by_missing_rate.png", dpi=200)
    plt.close(figure)

    matrix = np.array([
        [overall.loc[(overall.source_pattern == source) & (overall.target_pattern == target),
                     "mean_regret"].iloc[0] * 100 for target in PATTERNS]
        for source in PATTERNS
    ])
    figure, axis = plt.subplots(figsize=(6.5, 5.2), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1.0, float(matrix.max()) * 1.15))
    axis.set_xticks(range(2), ["Point", "Block"])
    axis.set_yticks(range(2), ["Point", "Block"])
    axis.set_xlabel("Deployment/test mask")
    axis.set_ylabel("Validation mask")
    axis.set_title(f"INTERIM: mean selection regret | {n_datasets}/64 datasets")
    for i in range(2):
        for j in range(2):
            axis.text(j, i, f"{matrix[i, j]:.2f} pp", ha="center", va="center",
                      fontsize=15, color="white" if matrix[i, j] > matrix.max() * 0.55 else "black")
    figure.colorbar(image, ax=axis, label="Mean regret (percentage points)")
    figure.savefig(figures_dir / "regret_2x2.png", dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/main_protocol_balanced.yaml"))
    args = parser.parse_args()
    selection = load_and_check(args.results_dir, args.config)
    save_figures(selection, args.results_dir)
    print(f"Saved interim charts for {selection.dataset.nunique()}/64 datasets to {args.results_dir / 'figures'}")


if __name__ == "__main__":
    main()
