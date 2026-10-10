# Final UCR64 results: protocol v1.4 main study

Complete results of the main experiment in `PROTOCOL.md`, copied unchanged from
the run directory `results/main_protocol_ucr64_v1_4/` after the benchmark
finished on 2026-09-29 (last checkpoint 02:32:48 +0900).

- Configuration: `configs/main_protocol_balanced.yaml`
  (fingerprint `195e4df2bf6545b7df2d65f21feba2b0e1ae3f9080bb2d0640b4eab70b3c4789`,
  matching `run_manifest.json`).
- `raw_results.csv`: 23,040 rows = 64 datasets × 5 seeds × 6 rates × 4 pattern pairs × 3 classifiers.
- `selection_results.csv`: 7,680 rows = 64 × 5 × 6 × 4.
- Training stays unmasked; validation has the source pattern; the official UCR
  test set has the target pattern. Linear interpolation is the only imputer.
- Selection uses validation balanced accuracy only; test balanced accuracy is
  used only for the post-hoc candidate-set oracle, selection error and regret.

## Verification

Before analysis the files were checked for: exact row counts; every configured
dataset/seed/rate/pattern condition present exactly once; exactly the three
configured classifiers per condition; no missing values; balanced accuracies in
[0, 1]; `k = floor(rT)`; and selected classifier, oracle, selection error and
regret recomputed from the raw rows with zero mismatches. Test results do not
vary with the source pattern and validation results do not vary with the target
pattern, as the design requires. 16.7% of conditions had a validation tie,
broken by the validation-only deterministic rule.

The first nine datasets (Adiac to Computers) were produced before prediction
caching was added; their raw rows satisfy the same invariances. This shows that
each version is internally consistent; it does not by itself establish that the
two code versions produce identical outputs.

### Checksums

`.gitattributes` stores these three files byte-for-byte (Windows CRLF line
endings, as the runner wrote them), so a clone on any platform reproduces the
original run output exactly:

| File | SHA-256 |
| --- | --- |
| `raw_results.csv` | `32b5b9eb429750836537e75496f8238905d05e23244634ca9de2b670c99ef03c` |
| `selection_results.csv` | `22dbbffc7fd6ddf6f47ef64fe17a3a30268141a7cb6a4eaff50f054efd67e382` |
| `run_manifest.json` | `3f7766f51d2f4b5b21e6db79cf20d231027ec262ba910596b7e845be9647300e` |

## Analysis (`analysis/`)

Produced by `src.paired_analysis` following `PROTOCOL.md` section 10. Each
dataset is one independent unit: seeds, rates and pattern strata are averaged
within a dataset before bootstrap CIs (10,000 resamples over datasets) and
two-sided Wilcoxon signed-rank tests across the 64 datasets.

Headline (target-paired, positive = mismatch is worse):

| Outcome | Mean | 95% CI | Worse / better / equal | Wilcoxon p |
| --- | --- | --- | --- | --- |
| Selected-model test BA lost (= regret increase) | 1.14 pp | 0.79 to 1.51 | 49 / 7 / 8 | 1.8e-9 |
| Selection error increase | 6.4 pp | 3.6 to 9.4 | 28 / 8 / 28 | 1.7e-5 |

The cost is asymmetric (point test, BP vs PP: 1.84 pp, p = 7.4e-7; block test,
PB vs BB: 0.44 pp, p = 0.20) and grows with the missing rate (−0.03 pp at 5% to
2.46 pp at 30%; slope 0.091 pp per percentage point, p = 6.4e-8).

### Revised analysis (`analysis_v2/`, 2026-10-06 and 2026-10-07)

`analysis/` is the original output and is kept unchanged. `analysis_v2/` was
produced from the same input files after two numerical changes to
`src.paired_analysis`, and the paper reports `analysis_v2/`:

1. The per-dataset trend slope is computed in closed form (`rate_slope`)
   instead of with `np.polyfit`.
2. Dataset-level values and slopes are rounded to 12 decimals (`DECIMALS`)
   before they are summarized.

Values that are equal or zero in exact arithmetic (slopes of count outcomes,
or a dataset whose paired differences cancel) otherwise carried about 1e-17
of floating-point noise. The direction counts already treated such values as
zero (tolerance 1e-9), but the Wilcoxon test ranked them, so its ties, zeros
and p-values depended on the numerical library. `scripts/compare_analysis_revisions.py`
shows that means, medians and CIs moved by less than 1e-12 and direction
counts did not change. These Wilcoxon results changed:

| Target-paired result | `analysis/` p | `analysis_v2/` p |
| --- | --- | --- |
| BA loss at 5%, Holm-adjusted | 0.793 | 0.800 |
| BA loss at 25%, Holm-adjusted | 9.54e-6 | 9.60e-6 |
| Selection-error trend | 5.65e-4 | 5.75e-4 |
| Selection-change trend | 3.25e-9 | 5.07e-9 |

The corresponding source-paired rows changed in the same way. All other
reported p-values, including the balanced-accuracy trend (p = 6.4e-8), are
unchanged, and no conclusion changes. `analysis_v2/tables/post_hoc_checks.csv`
(from `scripts/post_hoc_checks.py`) holds the post hoc checks reported in the
paper. `requirements-lock.txt` records the analysis environment for
`analysis_v2/`; the run environment was not recorded beyond aeon 1.6.0.

## Figures (`figures/`, PNG at 300 dpi and PDF)

Produced by `src.plot_final_results`; all intervals are bootstrap CIs over datasets.

1. `fig1_mismatch_cost_by_rate`: target-paired cost by missing rate.
2. `fig2_conditions_by_rate`: regret and selection error for PP, PB, BP, BB.
3. `fig3_per_dataset_effect`: dataset-level mean cost for all 64 datasets.
4. `fig4_classifier_selection`: which classifier is selected vs. best on test.

The supplementary linear-block analysis (`PROTOCOL.md` section 5.1) is not part
of these results; it is reported separately in `results/final_linear_block_v1_5/`.
