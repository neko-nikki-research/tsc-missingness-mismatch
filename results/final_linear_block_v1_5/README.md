# Linear-block robustness results — protocol v1.5 section 5.1

Supplementary sensitivity analysis of the main study. The **only** change is
that block missingness is a non-wrapping linear contiguous block instead of the
main study's circular block. It is reported separately and must not be pooled
with the circular-block main results in `results/final_ucr64_v1_4/`.

- Configuration: `configs/supplementary_linear_block.yaml` (64 datasets, seeds
  1–5, rates 5–30%, the three main-study classifiers and parameters;
  `n_jobs: 24`, which sets parallelism only).
- Run: commit `57e5ffa`, 2026-10-01 08:44–12:25 +0900, no uncommitted source
  changes and no resumes (`run/run_manifest.json`). An earlier 18-core start
  was stopped after 8 datasets and is not used.
- PP involves no block, so it is reused from the main study; the run produced
  PB, BP and BB only: 17,280 raw rows and 5,760 selection rows (`run/`).

## Verification

`run/` was checked for exact row counts; every configured condition present
exactly once and no PP rows; three classifiers per condition; no missing
values; balanced accuracies in [0, 1]; `k = floor(rT)`; and selection, oracle,
selection error and regret recomputed from the raw rows with zero mismatches.
As the protocol requires, PB and BB share the linear-block test mask (test BA
never varies with the source pattern) and BP and BB share the linear-block
validation mask (validation BA never varies with the target pattern). 15.3% of
conditions had a validation tie.

`src.assemble_linear_block` added the main-study PP rows only after verifying,
for all 64 datasets, that the run's point-side validation (PB) and test (BP)
balanced accuracies equal the main study's. This confirms that point masks and
fitted models are unchanged, and that `n_jobs` 24 versus 18 does not affect
results. The assembled `raw_results.csv` (23,040 rows) and
`selection_results.csv` (7,680 rows) are the analysis input;
`assembly_manifest.json` records the SHA-256 of both inputs.

| File in `run/` | SHA-256 |
| --- | --- |
| `raw_results.csv` | `bc9a491e2433d45f6ee6afecf6168cdb46842aba3b3aad33ea61222a9de76fd2` |
| `selection_results.csv` | `8b198f7ceb9ca9b1bf2602bdac88d3ac0efdb677ac051a43e85fe67821d03b17` |
| `run_manifest.json` | `6ed81c56f2473e5840a794e82717764545fa673f322cb06f5fc0d7fe5038e8cb` |

## Analysis (`analysis/`)

Same procedure as the main study (`PROTOCOL.md` section 10): datasets are the
independent unit; 10,000 bootstrap resamples over datasets; two-sided Wilcoxon
signed-rank tests; Holm correction across rates. Positive = mismatch is worse.

| Target-paired outcome | Linear block | Circular block (main study) |
| --- | --- | --- |
| Selected-model test BA lost (= regret increase) | **1.67 pp** (95% CI 1.23–2.14; 54 / 4 / 6 datasets worse / better / equal; p = 1.2e-10) | 1.14 pp (0.79–1.51; p = 1.8e-9) |
| Selection error increase | **11.2 pp** (7.3–15.4; p = 1.2e-7) | 6.4 pp (3.7–9.4; p = 1.7e-5) |
| Point test: BP vs PP | **2.24 pp** (1.62–2.90; p = 3.2e-9) | 1.84 pp (1.23–2.50; p = 7.4e-7) |
| Block test: PB vs BB | **1.10 pp** (0.45–1.81; p = 0.013) | 0.44 pp (−0.11–1.03; p = 0.20) |
| At 5% / 30% missingness | −0.03 pp / **3.65 pp** | −0.03 pp / 2.46 pp |
| Trend (pp per percentage point of missingness) | **0.145** (0.100–0.192; p = 8.6e-9) | 0.091 (0.061–0.124; p = 6.4e-8) |
| Excluding validation ties | 1.93 pp (62 datasets; 1.26–2.69) | 1.23 pp (62 datasets; 0.75–1.78) |

Per-rate effects are significant after Holm correction from 10% upward, as in
the main study. Mean regret by condition: PP 2.06, BB 2.43, PB 3.53, BP 4.30
pp. The main finding is therefore not an artefact of the circular block
construction; with linear blocks the mismatch cost is larger, and the
block-test direction (PB vs BB) also becomes significant.

## Figures (`figures/`, PNG at 300 dpi and PDF)

Produced by `src.plot_final_results`; all intervals are bootstrap CIs over
datasets.

1. `fig1_mismatch_cost_by_rate` — target-paired cost by missing rate.
2. `fig2_conditions_by_rate` — regret and selection error for PP, PB, BP, BB.
3. `fig3_per_dataset_effect` — dataset-level mean cost for all 64 datasets.
4. `fig4_classifier_selection` — which classifier is selected vs. best on test.
5. `fig5_circular_vs_linear_block` — target-paired cost under circular (main
   study) and linear (this analysis) blocks, by test pattern. Each analysis is
   summarised separately and shown side by side; they are not pooled.

In figures 1–4, B denotes the linear block and each figure carries that
footnote. Figure 5 shows the point-test cost is nearly the same under both
constructions, while the block-test cost is larger with linear blocks.
