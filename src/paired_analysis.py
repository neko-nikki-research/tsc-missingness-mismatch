"""Dataset-level paired analysis of validation/deployment missingness mismatch.

Each dataset is one independent unit. Seeds, missing rates and pattern strata
are repeated measurements inside a dataset, so they are averaged within the
dataset before any across-dataset interval or test is computed.

Two pairings are reported, with the sign convention "positive = mismatch is
worse":

* target-paired (primary): same test pattern, different validation pattern,
  e.g. PP vs BP and BB vs PB. Test data, candidate set and test oracle are
  identical, so the selected-model test BA difference equals the regret
  difference. This isolates the effect of the validation pattern on selection.
* source-paired (secondary): same validation pattern, different test pattern,
  e.g. PP vs PB. The selected model is identical, so the difference reflects
  how the deployment pattern changes test performance and the oracle.
"""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

from src.evaluation import selection_summary

CONDITION_KEYS = ["dataset", "seed", "missing_rate", "imputer"]
SELECTION_KEYS = CONDITION_KEYS + ["source_pattern", "target_pattern"]
TOLERANCE = 1e-9
# Dataset-level values and slopes are rounded to this many decimals before
# they are summarized. Values that are equal or zero in exact arithmetic
# (e.g. a dataset whose paired differences cancel) otherwise carry about
# 1e-17 of platform-dependent floating-point noise, which changes the
# Wilcoxon test's ties and zero handling.
DECIMALS = 12


def load_and_validate(results_dir: Path, config: dict, allow_incomplete: bool = False):
    """Load both CSVs and check them against the configured design.

    Returns the selection rows of fully completed datasets and the list of
    those datasets. Raises if the files are inconsistent, or if datasets are
    missing and ``allow_incomplete`` is False.
    """
    raw = pd.read_csv(results_dir / "raw_results.csv")
    selection = pd.read_csv(results_dir / "selection_results.csv")
    if raw.duplicated(SELECTION_KEYS + ["classifier"]).any():
        raise ValueError("Duplicate raw rows")
    if selection.duplicated(SELECTION_KEYS).any():
        raise ValueError("Duplicate selection rows")
    if raw.isna().any().any() or selection.isna().any().any():
        raise ValueError("Result files contain missing values")
    unknown = set(selection["dataset"]) - set(config["datasets"])
    if unknown:
        raise ValueError(f"Datasets outside the configuration: {sorted(unknown)}")

    classifiers = tuple(sorted(config["classifiers"]))
    candidate_sets = raw.groupby(SELECTION_KEYS)["classifier"].apply(lambda x: tuple(sorted(x)))
    if not (candidate_sets == classifiers).all():
        raise ValueError("Some conditions do not have exactly the configured classifiers")

    per_dataset = (
        len(config["seeds"]) * len(config["missing_rates"]) * len(config["imputers"])
        * len(config["source_patterns"]) * len(config["target_patterns"])
    )
    counts = selection.groupby("dataset").size()
    if not counts.eq(per_dataset).all():
        raise ValueError(f"Partially saved datasets: {counts[counts != per_dataset].to_dict()}")
    missing = [dataset for dataset in config["datasets"] if dataset not in counts.index]
    if missing and not allow_incomplete:
        raise ValueError(f"{len(missing)} configured datasets have no results yet: {missing}")
    # Counts alone would accept a wrong seed, rate or pattern in place of a
    # configured one, so the keys must be exactly the configured design.
    expected = set(product(
        counts.index, config["seeds"], config["missing_rates"], config["imputers"],
        config["source_patterns"], config["target_patterns"],
    ))
    if set(selection[SELECTION_KEYS].itertuples(index=False, name=None)) != expected:
        raise ValueError("Result keys do not match the configured seeds, rates, imputers and patterns")

    # Recompute selection, oracle, error and regret from the raw rows.
    recomputed = pd.DataFrame(
        [selection_summary(group.to_dict("records")) for _, group in raw.groupby(SELECTION_KEYS)]
    ).set_index(SELECTION_KEYS).sort_index()
    saved = selection.set_index(SELECTION_KEYS).sort_index()
    if not recomputed.index.equals(saved.index):
        raise ValueError("Raw and selection files cover different conditions")
    for column in ["selected_classifier", "oracle_classifier", "selection_error"]:
        if (recomputed[column].astype(str) != saved[column].astype(str)).any():
            raise ValueError(f"Saved {column} cannot be reproduced from raw results")
    for column in ["regret", "selected_test_balanced_accuracy", "oracle_test_balanced_accuracy"]:
        if (recomputed[column] - saved[column]).abs().max() > TOLERANCE:
            raise ValueError(f"Saved {column} cannot be reproduced from raw results")

    selection = selection.copy()
    selection["selection_error"] = selection["selection_error"].astype(str).str.lower().eq("true").astype(float)
    selection["validation_tie"] = (selection["n_validation_ties"] > 1).astype(float)
    return selection, sorted(counts.index)


def _split_matched(selection: pd.DataFrame, fixed: str):
    """Return matched and mismatched rows indexed by the pairing keys."""
    keys = CONDITION_KEYS + [fixed]
    matched = selection[selection.source_pattern == selection.target_pattern].set_index(keys)
    mismatched = selection[selection.source_pattern != selection.target_pattern].set_index(keys)
    if not matched.index.is_unique or not mismatched.index.is_unique:
        raise ValueError("Pairing needs exactly two patterns per side")
    if not matched.index.sort_values().equals(mismatched.index.sort_values()):
        raise ValueError("Every matched condition needs a mismatched partner")
    return matched, mismatched.loc[matched.index]


def pair_by_target(selection: pd.DataFrame) -> pd.DataFrame:
    """Primary pairing: same deployment pattern, different validation pattern."""
    matched, mismatched = _split_matched(selection, "target_pattern")
    oracle_gap = (matched.oracle_test_balanced_accuracy - mismatched.oracle_test_balanced_accuracy).abs()
    if oracle_gap.max() > TOLERANCE:
        raise ValueError("Target-paired conditions must share the same test oracle")
    pairs = pd.DataFrame({
        "matched_source": matched.source_pattern,
        "mismatched_source": mismatched.source_pattern,
        "delta_selected_test_ba": matched.selected_test_balanced_accuracy
        - mismatched.selected_test_balanced_accuracy,
        "delta_regret": mismatched.regret - matched.regret,
        "delta_selection_error": mismatched.selection_error - matched.selection_error,
        "selection_changed": (matched.selected_classifier != mismatched.selected_classifier).astype(float),
        "any_validation_tie": np.maximum(matched.validation_tie, mismatched.validation_tie),
    })
    # With a shared oracle, the regret difference is the BA difference.
    if (pairs.delta_selected_test_ba - pairs.delta_regret).abs().max() > TOLERANCE:
        raise ValueError("Target-paired regret and BA differences disagree")
    return pairs.reset_index()


def pair_by_source(selection: pd.DataFrame) -> pd.DataFrame:
    """Secondary pairing: same validation pattern, different deployment pattern."""
    matched, mismatched = _split_matched(selection, "source_pattern")
    pairs = pd.DataFrame({
        "matched_target": matched.target_pattern,
        "mismatched_target": mismatched.target_pattern,
        "delta_selected_test_ba": matched.selected_test_balanced_accuracy
        - mismatched.selected_test_balanced_accuracy,
        "delta_oracle_test_ba": matched.oracle_test_balanced_accuracy
        - mismatched.oracle_test_balanced_accuracy,
        "delta_regret": mismatched.regret - matched.regret,
        "delta_selection_error": mismatched.selection_error - matched.selection_error,
        "selection_changed": (matched.selected_classifier != mismatched.selected_classifier).astype(float),
        "any_validation_tie": np.maximum(matched.validation_tie, mismatched.validation_tie),
    })
    return pairs.reset_index()


def summarize_datasets(values: pd.Series, n_bootstrap: int, seed: int) -> dict:
    """Summarize one value per dataset: mean, bootstrap CI and Wilcoxon test."""
    values = np.round(values.to_numpy(dtype=float), DECIMALS)
    rng = np.random.default_rng(seed)
    resampled = values[rng.integers(0, len(values), size=(n_bootstrap, len(values)))].mean(axis=1)
    nonzero = np.abs(values) > TOLERANCE
    if nonzero.any():
        statistic, p_value = wilcoxon(values, alternative="two-sided", zero_method="wilcox")
    else:
        statistic, p_value = np.nan, 1.0
    return {
        "n_datasets": len(values),
        "mean": values.mean(),
        "median": np.median(values),
        "ci95_low": np.percentile(resampled, 2.5),
        "ci95_high": np.percentile(resampled, 97.5),
        "n_mismatch_worse": int((values > TOLERANCE).sum()),
        "n_mismatch_better": int((values < -TOLERANCE).sum()),
        "n_no_difference": int((~nonzero).sum()),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_two_sided": p_value,
    }


def rate_slope(missing_rates, values) -> float:
    """Least-squares slope of ``values`` per 1 percentage point of missingness.

    The closed form is rounded to ``DECIMALS`` places. Count metrics
    (selection error, selection change) have slopes on a coarse grid, so
    slopes that are equal or zero in exact arithmetic must stay equal or zero
    for the Wilcoxon test's ties and zero handling; see ``DECIMALS``.
    """
    x = np.asarray(missing_rates, dtype=float) * 100
    y = np.asarray(values, dtype=float)
    centred = x - x.mean()
    return round(float(centred @ y / (centred @ centred)), DECIMALS)


def holm(p_values: pd.Series) -> pd.Series:
    """Holm step-down adjustment for a family of p-values."""
    order = p_values.sort_values()
    m = len(order)
    adjusted = (order * (m - np.arange(m))).cummax().clip(upper=1.0)
    return adjusted.reindex(p_values.index)


def paired_tables(pairs: pd.DataFrame, metrics: list[str], stratum: str,
                  n_bootstrap: int, seed: int) -> dict[str, pd.DataFrame]:
    """Overall, per-stratum, per-rate and trend tables for one pairing."""
    def across(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
        rows = []
        per_dataset = frame.groupby(["dataset", *groups])[metrics].mean().reset_index()
        for key, group in (per_dataset.groupby(groups) if groups else [((), per_dataset)]):
            key = key if isinstance(key, tuple) else (key,)
            for metric in metrics:
                rows.append(dict(zip(groups, key)) | {"metric": metric}
                            | summarize_datasets(group[metric], n_bootstrap, seed))
        return pd.DataFrame(rows)

    overall = across(pairs, [])
    by_stratum = across(pairs, [stratum])
    by_rate = across(pairs, ["missing_rate"])
    by_rate["p_holm_across_rates"] = by_rate.groupby("metric")["wilcoxon_p_two_sided"].transform(holm)

    # Trend: one least-squares slope per dataset (per 1 percentage point of
    # missingness), then the slopes are summarized across datasets.
    per_rate = pairs.groupby(["dataset", "missing_rate"])[metrics].mean().reset_index()
    slopes = per_rate.groupby("dataset").apply(
        lambda group: pd.Series({
            metric: rate_slope(group.missing_rate, group[metric]) for metric in metrics
        }),
        include_groups=False,
    )
    trend = pd.DataFrame([
        {"metric": metric, "unit": "change per 1 percentage point"}
        | summarize_datasets(slopes[metric], n_bootstrap, seed)
        for metric in metrics
    ])
    no_ties = across(pairs[pairs.any_validation_tie == 0], [])
    return {"overall": overall, f"by_{stratum}": by_stratum, "by_rate": by_rate,
            "rate_trend": trend, "sensitivity_no_validation_ties": no_ties}


def condition_table(selection: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    """Dataset-level descriptive outcomes for each source/target condition."""
    metrics = ["regret", "selection_error", "selected_test_balanced_accuracy",
               "oracle_test_balanced_accuracy", "validation_tie"]
    per_dataset = selection.groupby(["dataset", "source_pattern", "target_pattern"])[metrics].mean()
    rows = []
    for (source, target), group in per_dataset.groupby(["source_pattern", "target_pattern"]):
        for metric in metrics:
            summary = summarize_datasets(group[metric], n_bootstrap, seed)
            rows.append({"source_pattern": source, "target_pattern": target, "metric": metric,
                         "n_datasets": summary["n_datasets"], "mean": summary["mean"],
                         "ci95_low": summary["ci95_low"], "ci95_high": summary["ci95_high"]})
    return pd.DataFrame(rows)


def classifier_table(selection: pd.DataFrame, classifiers: list[str]) -> pd.DataFrame:
    """Selection and oracle frequency per classifier, weighting datasets equally.

    A test-oracle tie between k classifiers gives each of them 1/k.
    ``mean_regret_when_selected`` is the gap to the candidate-set test oracle,
    averaged within dataset first and then over datasets that selected it.
    """
    rows = []
    for (source, target), group in selection.groupby(["source_pattern", "target_pattern"]):
        for name in classifiers:
            selected = (group.selected_classifier == name).astype(float)
            oracle_share = group.oracle_classifier.str.split("|").apply(
                lambda names: 1 / len(names) if name in names else 0.0
            )
            chosen = group[group.selected_classifier == name]
            rows.append({
                "source_pattern": source, "target_pattern": target, "classifier": name,
                "selected_share": selected.groupby(group.dataset).mean().mean(),
                "oracle_share": oracle_share.groupby(group.dataset).mean().mean(),
                "n_datasets_ever_selected": chosen.dataset.nunique(),
                "mean_regret_when_selected": chosen.groupby("dataset").regret.mean().mean(),
            })
    return pd.DataFrame(rows)


def run_analysis(results_dir: Path, config: dict, output_dir: Path, allow_incomplete: bool,
                 n_bootstrap: int = 10000, seed: int = 20260928) -> list[str]:
    """Write all tables and return the analysed datasets."""
    if output_dir.resolve() == results_dir.resolve():
        raise ValueError("Write analysis output to a separate directory from the checkpoint")
    selection, datasets = load_and_validate(results_dir, config, allow_incomplete)
    complete = len(datasets) == len(config["datasets"])
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    target_pairs = pair_by_target(selection)
    source_pairs = pair_by_source(selection)
    target_metrics = ["delta_selected_test_ba", "delta_selection_error", "selection_changed"]
    source_metrics = ["delta_regret", "delta_selection_error", "delta_selected_test_ba",
                      "delta_oracle_test_ba"]
    for name, table in paired_tables(target_pairs, target_metrics, "target_pattern", n_bootstrap, seed).items():
        table.to_csv(tables_dir / f"target_paired_{name}.csv", index=False)
    for name, table in paired_tables(source_pairs, source_metrics, "source_pattern", n_bootstrap, seed).items():
        table.to_csv(tables_dir / f"source_paired_{name}.csv", index=False)
    condition_table(selection, n_bootstrap, seed).to_csv(tables_dir / "conditions.csv", index=False)
    classifier_table(selection, config["classifiers"]).to_csv(tables_dir / "classifiers.csv", index=False)

    status = "COMPLETE" if complete else "INTERIM - NOT A FINAL RESULT"
    (output_dir / "ANALYSIS_STATUS.txt").write_text(
        f"{status}\n{len(datasets)}/{len(config['datasets'])} datasets analysed\n"
        f"source: {results_dir}\nbootstrap resamples: {n_bootstrap}, seed: {seed}\n",
        encoding="utf-8",
    )
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Directory with the two result CSVs")
    parser.add_argument("--config", required=True, help="YAML config that produced the results")
    parser.add_argument("--output-dir", required=True, help="Separate directory for analysis tables")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Analyse completed datasets only; output is marked INTERIM")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    datasets = run_analysis(Path(args.results_dir), config, Path(args.output_dir),
                            args.allow_incomplete, args.n_bootstrap)
    print(f"Analysed {len(datasets)}/{len(config['datasets'])} datasets into {args.output_dir}")


if __name__ == "__main__":
    main()
