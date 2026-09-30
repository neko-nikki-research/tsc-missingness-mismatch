"""Publication figures for the completed validation/deployment mismatch study.

Every interval is a percentile bootstrap over datasets: seeds and missing rates
are averaged inside each dataset first, exactly as in ``src.paired_analysis``.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.paired_analysis import load_and_validate, pair_by_target, summarize_datasets

# Reference categorical order (light mode); identity never relies on color alone.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
WORSE, BETTER, NEUTRAL = "#eb6834", "#2a78d6", "#a3a29c"
BLOCK_LABELS = {"block": "Block", "linear_block": "Linear-block"}
CLASSIFIERS = [("minirocket", "MiniROCKET"), ("dtw", "1NN-DTW"), ("stat_rf", "Stat. features + RF")]


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 2,
    })


def _summaries(per_dataset: pd.DataFrame, groups: list[str], metric: str,
               n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows = []
    for key, group in per_dataset.groupby(groups):
        key = key if isinstance(key, tuple) else (key,)
        summary = summarize_datasets(group[metric], n_bootstrap, seed)
        rows.append(dict(zip(groups, key)) | {k: summary[k] for k in ("mean", "ci95_low", "ci95_high")})
    return pd.DataFrame(rows)


def _line_with_band(axis, frame, color, label, marker, linestyle="-"):
    x = frame.missing_rate * 100
    axis.fill_between(x, frame.ci95_low * 100, frame.ci95_high * 100, color=color, alpha=0.14, linewidth=0)
    axis.plot(x, frame["mean"] * 100, color=color, marker=marker, markersize=6, linestyle=linestyle,
              label=label, markeredgecolor=SURFACE, markeredgewidth=1.2)
    last = frame.iloc[-1]
    axis.annotate(label, (last.missing_rate * 100, last["mean"] * 100), xytext=(6, 0),
                  textcoords="offset points", va="center", fontsize=8, color=INK_2)


def _conditions(block: str) -> list[tuple[str, str, str]]:
    """PP, PB, BP and BB, where B is the block pattern of this analysis."""
    return [("point", "point", "PP"), ("point", block, "PB"), (block, "point", "BP"), (block, block, "BB")]


def _block_note(block: str) -> str:
    """Footnote naming the block construction when it is not the main study's."""
    if block == "block":
        return ""
    return " B = non-wrapping linear block (supplementary robustness analysis; PP reused from the main study)."


def _save(figure, figures_dir: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(figures_dir / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_mismatch_by_rate(pairs, figures_dir, n_bootstrap, seed, block="block"):
    """Figure 1: target-paired mismatch cost by missing rate."""
    metrics = [("delta_selected_test_ba", "Test BA lost by mismatched validation (pp)",
                "A  Deployment performance (= regret increase)"),
               ("delta_selection_error", "Increase in selection error (pp)",
                "B  Probability of choosing a sub-optimal classifier")]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharex=True)
    for axis, (metric, ylabel, title) in zip(axes, metrics):
        by_stratum = pairs.groupby(["dataset", "missing_rate", "target_pattern"])[metric].mean().reset_index()
        pooled = pairs.groupby(["dataset", "missing_rate"])[metric].mean().reset_index()
        axis.axhline(0, color=INK_2, linewidth=1)
        _line_with_band(axis, _summaries(by_stratum[by_stratum.target_pattern == "point"], ["missing_rate"],
                                         metric, n_bootstrap, seed), SERIES[0], "Point test: BP vs PP", "o")
        _line_with_band(axis, _summaries(by_stratum[by_stratum.target_pattern == block], ["missing_rate"],
                                         metric, n_bootstrap, seed), SERIES[1],
                        f"{BLOCK_LABELS.get(block, block)} test: PB vs BB", "s")
        _line_with_band(axis, _summaries(pooled, ["missing_rate"], metric, n_bootstrap, seed),
                        INK, "Both targets", "D", linestyle="--")
        axis.set(title=title, xlabel="Missing rate (%)", ylabel=ylabel, xticks=[5, 10, 15, 20, 25, 30])
        axis.set_xlim(3.5, 38)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    figure.text(0.5, -0.04, f"Positive = mismatch is worse. Mean over {pairs.dataset.nunique()} datasets; "
                "bands are 95% bootstrap CIs over datasets." + _block_note(block),
                ha="center", fontsize=8, color=INK_2, wrap=True)
    figure.tight_layout()
    _save(figure, figures_dir, "fig1_mismatch_cost_by_rate")


def plot_conditions_by_rate(selection, figures_dir, n_bootstrap, seed, block="block"):
    """Figure 2: regret and selection error for PP, PB, BP and BB."""
    per_dataset = selection.groupby(["dataset", "missing_rate", "source_pattern", "target_pattern"])[
        ["regret", "selection_error"]].mean().reset_index()
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharex=True)
    for axis, (metric, ylabel, title) in zip(axes, [
        ("regret", "Selection regret (pp of test BA)", "A  Regret against the candidate-set test oracle"),
        ("selection_error", "Selection error rate (%)", "B  Selection error rate"),
    ]):
        for color, marker, (source, target, name) in zip(SERIES, "osD^", _conditions(block)):
            group = per_dataset[(per_dataset.source_pattern == source) & (per_dataset.target_pattern == target)]
            _line_with_band(axis, _summaries(group, ["missing_rate"], metric, n_bootstrap, seed),
                            color, name, marker, linestyle="-" if source == target else "--")
        axis.set(title=title, xlabel="Missing rate (%)", ylabel=ylabel, xticks=[5, 10, 15, 20, 25, 30])
        axis.set_xlim(3.5, 34)
        axis.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, ["PP (matched)", "PB (mismatched)", "BP (mismatched)", "BB (matched)"],
                  loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04))
    figure.text(0.5, -0.04, "Validation pattern → test pattern. Solid = matched, dashed = mismatched; "
                "bands are 95% bootstrap CIs over datasets." + _block_note(block),
                ha="center", fontsize=8, color=INK_2, wrap=True)
    figure.tight_layout()
    _save(figure, figures_dir, "fig2_conditions_by_rate")


def plot_per_dataset(pairs, figures_dir):
    """Figure 3: one dataset-level mean effect per dataset."""
    effect = (pairs.groupby("dataset").delta_selected_test_ba.mean() * 100).sort_values()
    colors = [WORSE if v > 1e-7 else BETTER if v < -1e-7 else NEUTRAL for v in effect]
    figure, axis = plt.subplots(figsize=(6.2, 11))
    positions = np.arange(len(effect))
    axis.hlines(positions, 0, effect.to_numpy(), color=colors, linewidth=2)
    axis.scatter(effect.to_numpy(), positions, color=colors, s=28, zorder=3, edgecolor=SURFACE, linewidth=1)
    axis.axvline(0, color=INK_2, linewidth=1)
    axis.set_yticks(positions, effect.index, fontsize=7)
    axis.set_ylim(-1, len(effect))
    axis.grid(axis="y", visible=False)
    worse, better = int((effect > 1e-7).sum()), int((effect < -1e-7).sum())
    axis.set(xlabel="Test BA lost by mismatched validation (pp), mean over seeds, rates and targets",
             title=f"Per-dataset mismatch cost: {worse} worse, {better} better, "
                   f"{len(effect) - worse - better} no difference")
    _save(figure, figures_dir, "fig3_per_dataset_effect")


def plot_classifier_shares(selection, figures_dir, block="block"):
    """Figure 4: which classifier is selected, and which is the test oracle."""
    conditions = _conditions(block)
    rows = []
    for source, target, name in conditions:
        group = selection[(selection.source_pattern == source) & (selection.target_pattern == target)]
        for key, label in CLASSIFIERS:
            selected = (group.selected_classifier == key).astype(float).groupby(group.dataset).mean().mean()
            oracle = group.oracle_classifier.str.split("|").apply(
                lambda names: 1 / len(names) if key in names else 0.0).groupby(group.dataset).mean().mean()
            rows.append({"condition": name, "classifier": label, "Selected by validation": selected,
                         "Best on test (oracle)": oracle})
    shares = pd.DataFrame(rows)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for axis, measure in zip(axes, ["Selected by validation", "Best on test (oracle)"]):
        table = shares.pivot(index="condition", columns="classifier", values=measure).loc[
            [name for *_, name in conditions][::-1], [label for _, label in CLASSIFIERS]] * 100
        left = np.zeros(len(table))
        for color, label in zip(SERIES, table.columns):
            axis.barh(table.index, table[label], left=left, color=color, height=0.62,
                      edgecolor=SURFACE, linewidth=2, label=label)
            for y, (start, width) in enumerate(zip(left, table[label])):
                if width >= 8:
                    axis.text(start + width / 2, y, f"{width:.0f}%", ha="center", va="center",
                              fontsize=8, color=SURFACE if label != "Stat. features + RF" else INK)
            left += table[label].to_numpy()
        axis.set(title=measure, xlim=(0, 100), xlabel="Share of conditions (%), datasets weighted equally")
        axis.grid(axis="y", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    figure.text(0.5, -0.05, "Oracle ties between k classifiers give each 1/k." + _block_note(block),
                ha="center", fontsize=8, color=INK_2, wrap=True)
    figure.tight_layout()
    _save(figure, figures_dir, "fig4_classifier_selection")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--figures-dir", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    selection, datasets = load_and_validate(Path(args.results_dir), config, allow_incomplete=False)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _style()
    # The non-point pattern is B: circular "block" in the main study,
    # "linear_block" in the robustness analysis.
    block = next(pattern for pattern in config["target_patterns"] if pattern != "point")
    pairs = pair_by_target(selection)
    plot_mismatch_by_rate(pairs, figures_dir, args.n_bootstrap, args.seed, block)
    plot_conditions_by_rate(selection, figures_dir, args.n_bootstrap, args.seed, block)
    plot_per_dataset(pairs, figures_dir)
    plot_classifier_shares(selection, figures_dir, block)
    print(f"Saved 4 figures for {len(datasets)} datasets to {figures_dir}")


if __name__ == "__main__":
    main()
