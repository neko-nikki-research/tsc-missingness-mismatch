# Missingness Mismatch Benchmark for Time-Series Classification

This repository contains a protocol-compliant controlled empirical study of whether a mismatch between validation-time and deployment-time missingness patterns changes classifier selection and deployment performance in time-series classification.

The primary study uses random point missingness and circular contiguous block missingness. A supplementary robustness analysis replaces the circular block construction with a non-wrapping linear contiguous block while leaving the primary study unchanged.

## Primary study at a glance

| Component           | Fixed primary-study choice                                                       |
| ------------------- | -------------------------------------------------------------------------------- |
| Task                | Univariate time-series classification                                            |
| Data                | UCR univariate, equal-length, originally complete series                         |
| Datasets            | 64 fixed UCR2015 datasets                                                        |
| Training data       | Official UCR training set; kept complete and unchanged                           |
| Validation split    | Stratified split from official training data; 25% held out                       |
| Deployment data     | Official UCR test set                                                            |
| Validation patterns | Point, circular block                                                            |
| Test patterns       | Point, circular block                                                            |
| Pattern matrix      | PP, PB, BP, BB                                                                   |
| Missingness rates   | 5%, 10%, 15%, 20%, 25%, 30%                                                      |
| Imputation          | Linear interpolation only                                                        |
| Candidate methods   | 1NN-DTW; MiniROCKET + Ridge; statistical features + Random Forest                |
| Selection metric    | Validation balanced accuracy                                                     |
| Primary outcome     | Target-paired difference in selected-model test balanced accuracy                |
| Secondary outcomes  | Selection change, selection error, selection regret, classifier-selection shares |

The four primary conditions are:

```text
PP = Point validation → Point test
PB = Point validation → Circular-block test
BP = Circular-block validation → Point test
BB = Circular-block validation → Circular-block test
```

The primary analysis is target-paired. For a fixed target test condition, matched and mismatched validation patterns are compared on the same target test instances and masks.

The primary effect uses the sign convention

```text
positive = mismatch is worse
```

and is defined as

```text
ΔBA = BA_matched − BA_mismatched
```

Selection regret is

```text
selection regret
= oracle test balanced accuracy − selected test balanced accuracy
```

Because matched and mismatched target-paired conditions share the same candidate-set oracle, the difference in selected-model test balanced accuracy is equal to the difference in selection regret.

The training split is deliberately never masked. This keeps the training distribution fixed so that the primary comparison isolates the relationship between validation missingness pattern and the target deployment condition.

---

## Run the project

Create and activate a virtual environment, install the declared dependencies, and run the tests:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
python scripts\check_environment.py
```

Run the small protocol-faithful smoke test with:

```powershell
python -m src.run_benchmark --config configs\smoke.yaml
```

The smoke test is intended to verify the end-to-end execution path without producing the formal study results.

---

## Run the primary study

The primary experiment is specified by:

```text
configs/main_protocol_balanced.yaml
```

Run it with:

```powershell
python -m src.run_benchmark --config configs\main_protocol_balanced.yaml
```

The configuration defines:

* 64 fixed UCR2015 datasets;
* random seeds 1–5;
* nominal missingness rates 0.05–0.30;
* Point and circular-block source patterns;
* Point and circular-block target patterns;
* linear interpolation;
* the three prespecified classifier pipelines;
* a 25% stratified validation split.

For one complete primary run, the expected output is:

```text
64 × 5 × 6 × 4 × 3 = 23,040
```

classifier-condition result records and

```text
64 × 5 × 6 × 4 = 7,680
```

selection-condition records.

The configured runtime directory is:

```text
results/main_protocol_ucr64_v1_4/
```

The completed frozen primary-study archive is stored separately under:

```text
results/final_ucr64_v1_4/
```

The frozen directory contains the completed raw results, selection results, verification information, analysis tables, and four main-study figures.

---

## Confirmatory analysis

After the primary benchmark has completed, run the dataset-level paired analysis:

```powershell
python -m src.paired_analysis --results-dir results\main_protocol_ucr64_v1_4 --config configs\main_protocol_balanced.yaml --output-dir results\main_protocol_ucr64_v1_4\analysis
```

To analyse the frozen completed primary-study results instead:

```powershell
python -m src.paired_analysis --results-dir results\final_ucr64_v1_4 --config configs\main_protocol_balanced.yaml --output-dir results\final_ucr64_v1_4\analysis
```

The paired analysis:

* verifies the raw and selection result tables;
* checks the configured design;
* recomputes classifier selection from validation balanced accuracy;
* recomputes the candidate-set test oracle;
* recomputes selection error and regret;
* treats each dataset as one independent unit;
* averages repeated seeds and missingness rates within dataset before across-dataset inference;
* performs the prespecified paired analyses;
* computes percentile bootstrap intervals over datasets;
* performs two-sided Wilcoxon signed-rank tests;
* performs the prespecified missingness-rate analysis and trend analysis;
* reports the validation-tie sensitivity analysis.

An incomplete result directory is not analysed by default. Use `--allow-incomplete` only for an explicitly interim analysis; such output is marked as interim rather than final.

The bootstrap analysis uses 10,000 resamples and the fixed analysis seed specified by the protocol.

---

## Generate primary-study figures

Generate the four main-study figures with:

```powershell
python -m src.plot_final_results --results-dir results\final_ucr64_v1_4 --config configs\main_protocol_balanced.yaml --figures-dir results\final_ucr64_v1_4\figures
```

The primary plotting module produces four figures:

```text
fig1_mismatch_cost_by_rate
fig2_conditions_by_rate
fig3_per_dataset_effect
fig4_classifier_selection
```

All bootstrap bands are computed over datasets after repeated seeds and missingness rates have been aggregated within dataset.

---

## Descriptive summaries

The repository also provides descriptive summaries that are separate from the confirmatory paired analysis:

```powershell
python -m src.analyze_results --results-dir results\final_ucr64_v1_4
python -m src.report_results --results-dir results\final_ucr64_v1_4
```

These commands produce descriptive summaries and report figures. They do not replace the confirmatory paired analysis in `src.paired_analysis`.

---

## Checkpointing and reproducibility

The benchmark runner saves both result CSVs after each completed dataset.

If a run is interrupted, rerunning the same configuration resumes from the completed-dataset checkpoint rather than restarting completed datasets.

The benchmark writes:

```text
raw_results.csv
selection_results.csv
run_manifest.json
```

The run manifest records the configuration hash and source revision information. When a checkpoint is resumed, the later source revision is also recorded so that checkpoints created across multiple code revisions remain traceable.

The runner prevents results from different configurations from being mixed by checking the configuration hash stored in the existing manifest.

Because the training split remains complete and unchanged:

* each candidate classifier is fitted once per dataset and seed;
* fitted models are reused across missingness rates and pattern conditions;
* validation predictions are reused across target patterns;
* test predictions are reused across validation/source patterns.

This prediction reuse is an implementation optimization and does not change the experimental design.

For 1NN-DTW, query series are passed to aeon's distance kernel in batches. The parallel implementation produces the same prediction outcomes as the one-query-at-a-time reference implementation, including the deterministic smaller-training-index tie rule.

Timing fields in the raw output are not additive wall-clock measurements. `fit_time_seconds` records the classifier fit for the applicable dataset and seed, while `predict_time_seconds` records the corresponding validation-plus-test prediction timing. These values are repeated on rows to which the same fitted model or prediction operation applies.

The optimized runner was additionally checked on six datasets:

```text
Coffee
CBF
ECG200
ItalyPowerDemand
GunPoint
Car
```

The rerun reproduced every non-timing result value of the published primary results exactly.

---

# Linear-block robustness analysis

Protocol v1.5 adds a supplementary robustness analysis to assess whether the primary findings depend on the specific circular construction of contiguous block missingness.

The primary circular-block experiment remains unchanged.

The supplementary analysis replaces the circular block with a **non-wrapping linear contiguous block**. All scientific and statistical components remain the same:

* the same 64 datasets;
* the same five random seeds;
* the same six missingness rates;
* the same train/validation/test procedure;
* the same three candidate classifiers;
* the same classifier parameters;
* the same linear interpolation procedure;
* the same validation-based selection rule;
* the same evaluation metrics;
* the same dataset-level statistical analysis.

The linear-block analysis is reported separately and is never pooled with the circular-block primary study.

## Conditions

The supplementary run evaluates only:

```text
PL = Point validation → Linear-block test
LP = Linear-block validation → Point test
LL = Linear-block validation → Linear-block test
```

The primary-study `PP` condition is reused because it does not involve block missingness.

The newly executed supplementary benchmark therefore contains:

```text
64 × 5 × 6 × 3 × 3 = 17,280
```

classifier-condition result records.

The supplementary configuration is:

```text
configs/supplementary_linear_block.yaml
```

Run it with:

```powershell
python -m src.run_benchmark --config configs\supplementary_linear_block.yaml
```

The configured runtime directory is:

```text
results/supplementary_linear_block_v1_5/
```

---

## Assemble the linear-block result set

After the supplementary run has completed, assemble the complete PP/PL/LP/LL result set:

```powershell
python -m src.assemble_linear_block --main-results results\final_ucr64_v1_4 --supplementary-results results\supplementary_linear_block_v1_5 --config configs\supplementary_linear_block.yaml --output-dir results\final_linear_block_v1_5
```

The assembly step reuses the primary-study `PP` rows only after verifying the point-side results of the supplementary run against the primary study.

Specifically:

* the point-validation results in `PL` must agree with the primary `PP` validation results;
* the point-test results in `LP` must agree with the primary `PP` test results.

This verifies that the point-masking procedure and fitted models remain unchanged before the `PP` rows are reused.

The assembled robustness result set is stored under:

```text
results/final_linear_block_v1_5/
```

and contains:

```text
raw_results.csv
selection_results.csv
assembly_manifest.json
```

together with its analysis and figure outputs.

---

## Linear-block paired analysis

Run the dataset-level analysis on the assembled result set:

```powershell
python -m src.paired_analysis --results-dir results\final_linear_block_v1_5 --config configs\supplementary_linear_block.yaml --output-dir results\final_linear_block_v1_5\analysis
```

For point-test deployment, the supplementary target-paired comparison is:

```text
PP vs LP
```

For linear-block-test deployment, it is:

```text
LL vs PL
```

The linear-block analysis uses the same target-paired and source-paired framework as the primary study.

The circular-block and linear-block result sets are analysed separately because they correspond to different masking distributions.

---

## Compare circular and linear block constructions

To generate the robustness comparison figure:

```powershell
python -m src.plot_final_results --results-dir results\final_linear_block_v1_5 --config configs\supplementary_linear_block.yaml --figures-dir results\final_linear_block_v1_5\figures --compare-results results\final_ucr64_v1_4 --compare-config configs\main_protocol_balanced.yaml
```

This produces the four standard robustness-analysis figures plus:

```text
fig5_circular_vs_linear_block
```

The supplementary result set therefore contains five figures in total:

```text
fig1_mismatch_cost_by_rate
fig2_conditions_by_rate
fig3_per_dataset_effect
fig4_classifier_selection
fig5_circular_vs_linear_block
```

Figure 5 compares the primary circular-block results with the supplementary linear-block results. It does not redefine the primary estimand.

---

## Mask sharing in the linear-block analysis

The linear-block robustness analysis preserves the same paired-mask structure as the primary study.

For a fixed dataset, seed, and missingness rate:

* `LP` and `LL` share the same linear-block validation mask;
* `PL` and `LL` share the same linear-block test mask;
* `PP` and `PL` share the same point-validation mask generation rule;
* `PP` and `LP` share the same point-test mask generation rule.

The `condition_pairs` field in the supplementary configuration limits the runner to the three newly required conditions:

```yaml
condition_pairs:
  - [point, linear_block]
  - [linear_block, point]
  - [linear_block, linear_block]
```

Thus the supplementary benchmark does not rerun `PP`.

The different `n_jobs` value used by the supplementary configuration controls computational parallelism only and is not a scientific experimental factor.

---

## Formal result fields

The formal result files use explicit metric names:

```text
val_balanced_accuracy
test_balanced_accuracy
selected_test_balanced_accuracy
oracle_test_balanced_accuracy
```

Ordinary accuracy is not used for model selection or for the formal result tables.

The primary selection metric is balanced accuracy. Macro-F1 may be recorded as an additional descriptive metric but does not determine classifier selection or the candidate-set oracle.

---

# Repository map

```text
configs/
  smoke.yaml
      # Small protocol-faithful end-to-end check

  main_protocol_balanced.yaml
      # Primary PP/PB/BP/BB circular-block study

  supplementary_linear_block.yaml
      # Protocol v1.5 supplementary PL/LP/LL linear-block robustness run

  exploratory/
      # Separate exploratory analyses; not part of the primary study

  legacy/
      # Superseded configurations retained for historical reference

src/
  run_benchmark.py
      # Benchmark runner, checkpointing, condition selection, and prediction reuse

  datasets.py
      # UCR loading and stratified train/validation splitting

  masking.py
      # Point, circular-block, and linear-block masking

  imputation.py
      # Imputation implementations

  classifiers.py
      # Three prespecified candidate pipelines

  statistical_features.py
      # Statistical feature extraction for the feature-based pipeline

  evaluation.py
      # Classification metrics, selection, oracle, and regret logic

  paired_analysis.py
      # Dataset-level paired statistical analysis

  assemble_linear_block.py
      # PP reuse verification and assembly of the linear-block robustness result set

  analyze_results.py
      # Descriptive result summaries

  report_results.py
      # Additional descriptive tables and figures

  plot_final_results.py
      # Main-study and supplementary publication figures

  plot_interim_results.py
      # Descriptive plotting for partial checkpoint runs

tests/
      # Unit tests and protocol-invariant tests

results/
      # Generated and archived experiment outputs

figures/
      # Experimental-design figure assets

scripts/
  check_environment.py
      # Environment checks

  preflight_datasets.py
      # Dataset loading and split preflight

  switch_after_checkpoint.ps1
      # Checkpoint-related workflow helper

PROTOCOL.md
      # Full study protocol

CONTRIBUTING.md
      # Contribution and project workflow guidance
```

`configs/exploratory/` is reserved for analyses that are outside the primary study, such as alternative imputation methods, alternative block constructions, prefix/suffix masking, or additional missingness rates.

Exploratory configurations must remain separate from the primary PP/PB/BP/BB conclusions and must not be pooled with the formal primary results.

---

# Reproducibility guarantees

The implementation enforces the following experimental invariants:

* A fixed experimental seed produces deterministic masks.
* Masking does not mutate the original input data.
* All candidate classifiers within an experimental condition receive identical validation and test masks.
* Validation and test masks are generated independently through distinct seed offsets.
* Validation balanced accuracy is the only external criterion used for classifier selection.
* Exact validation ties are resolved using a reproducible random rule based only on validation-side metadata.
* Test-oracle ties are reported as ties and are not treated as arbitrary classifier identities.
* A classifier tied with the candidate-set test oracle is not counted as a selection error.
* Imputation uses only observed values from the same series.
* No test-set parameter is learned for imputation.
* The official UCR test labels are never used for model selection or tuning.
* Target-paired comparisons share the same target test condition and therefore the same candidate-set oracle.
* Source-paired comparisons keep the validation condition fixed and therefore isolate the effect of changing the deployment pattern.
* Raw classifier-level results are sufficient to reconstruct classifier selection, oracle membership, selection error, and regret.

---

# Dataset and result integrity

The formal dataset list is fixed before formal result analysis.

The main-study result set is considered complete only when all 64 configured datasets have completed the full set of seeds, missingness rates, pattern conditions, imputers, and candidate classifiers.

Partial checkpoint results may be inspected descriptively but are not treated as final inferential results.

The repository retains two historical interim checkpoints:

```text
results/interim_ucr64_2026-09-25_27datasets/
results/interim_ucr64_2026-09-27_46datasets/
```

These are ordered prefixes of the fixed dataset list rather than random or representative samples. They were inspected during the primary run but are not used as formal final analyses.

The completed v1.4 primary study is archived under:

```text
results/final_ucr64_v1_4/
```

The completed v1.5 linear-block robustness analysis is archived under:

```text
results/final_linear_block_v1_5/
```

---

# Protocol and contribution documents

The complete study protocol is:

[PROTOCOL.md](PROTOCOL.md)

Contribution and project workflow guidance:

[CONTRIBUTING.md](CONTRIBUTING.md)
