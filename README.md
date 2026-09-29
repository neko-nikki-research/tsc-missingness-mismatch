# Missingness Mismatch Benchmark for Time-Series Classification

This branch contains the protocol-compliant benchmark for testing whether a
mismatch between validation-time and deployment-time missingness changes
classifier selection and deployment performance.

## Main experiment at a glance

| Component | Fixed main-experiment choice |
| --- | --- |
| Data | UCR univariate, equal-length, originally complete series |
| Datasets | 64 fixed UCR2015 univariate, equal-length datasets (listed in `PROTOCOL.md`) |
| Train split | Official UCR train; kept complete and unchanged |
| Validation split | Stratified split from official train; receives the source mask |
| Deployment split | Official UCR test; receives the target mask |
| Missingness matrix | Point → Point, Point → circular Block, circular Block → Point, circular Block → circular Block |
| Missingness rates | 5%, 10%, 15%, 20%, 25%, 30% |
| Imputation | Linear interpolation only |
| Candidate methods | 1NN-DTW; MiniROCKET + Ridge; statistical features + Random Forest |
| Selection metric | Validation balanced accuracy |
| Deployment metrics | Test balanced accuracy, post-hoc oracle, selection error, regret |

`selection regret = oracle test balanced accuracy − selected test balanced accuracy`

The train split is deliberately never masked. This holds training conditions
constant, so a difference between PP/BB and PB/BP can be attributed to the
validation-to-deployment mismatch rather than a changed training distribution.

## Run it

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest
python scripts\check_environment.py
python -m src.run_benchmark --config configs\smoke.yaml
```

Run the full protocol study with:

```powershell
python -m src.run_benchmark --config configs\main_protocol_balanced.yaml
python -m src.paired_analysis --results-dir results\final_ucr64_v1_4 --config configs\main_protocol_balanced.yaml --output-dir results\final_ucr64_v1_4\analysis
python -m src.plot_final_results --results-dir results\final_ucr64_v1_4 --config configs\main_protocol_balanced.yaml --figures-dir results\final_ucr64_v1_4\figures
python -m src.analyze_results --results-dir results\main_protocol_ucr64_v1_4
python -m src.report_results --results-dir results\main_protocol_ucr64_v1_4
```

`src.paired_analysis` is the confirmatory analysis described in `PROTOCOL.md`
section 10. It verifies both CSVs, recomputes selection and regret from the raw
rows, and treats each dataset as one independent unit. It refuses to analyse an
unfinished run unless `--allow-incomplete` is given, in which case the output is
marked `INTERIM`. `src.plot_final_results` draws the four publication figures
with bootstrap intervals over datasets. The last two commands produce
descriptive summaries only.

The completed main-study results, verification notes, analysis tables and
figures are committed under `results/final_ucr64_v1_4/`.

The runner saves both result CSVs after each complete dataset. Rerunning the
same command resumes from that checkpoint, and `run_manifest.json` records the
git commit that created the checkpoint and every later resume.

Because the training split is never masked, each candidate is fitted once per
seed and reused for every rate and pattern. Within one rate, validation
predictions are reused across target patterns and test predictions across source
patterns. 1NN-DTW sends all query series to aeon's distance kernel in one call,
so `n_jobs` parallelises it; predictions, including the smaller-training-index
tie rule, are identical to aeon's one-query-at-a-time loop. `fit_time_seconds`
is the single fit of that classifier and seed; `predict_time_seconds` is one
validation plus one test prediction. Both are repeated on every row they apply
to, so they are not additive wall time.

Re-running six datasets (Coffee, CBF, ECG200, ItalyPowerDemand, GunPoint, Car)
with the optimized runner reproduced every non-timing value of the published
results exactly.

The raw output uses only explicit metric names: `val_balanced_accuracy`,
`test_balanced_accuracy`, `selected_test_balanced_accuracy`, and
`oracle_test_balanced_accuracy`. Ordinary accuracy is neither used for model
selection nor written to the formal result files.

## Repository map

```text
configs/
  smoke.yaml                    # Small protocol-faithful end-to-end check
  main_protocol_balanced.yaml   # Formal PP/PB/BP/BB study
  exploratory/                  # Separate robustness analyses; not main results
  legacy/                       # Superseded configs retained for history only
src/
  run_benchmark.py              # Main experiment runner
  datasets.py                   # UCR loading and stratified split
  masking.py                    # Point/block masking (prefix/suffix available only for exploration)
  imputation.py                 # Linear/zero/mean implementations
  classifiers.py                # Three pre-specified candidate methods
  evaluation.py                 # Selection, oracle, and regret logic
tests/                          # Unit and protocol-invariant tests
results/                        # Local generated outputs; CSVs and figures are gitignored
```

`configs/exploratory/` may compare zero/mean imputation, prefix/suffix
missingness, linear blocks, or extra missingness rates in future robustness
work. These configurations must be reported separately and must not be pooled
with the main PP/PB/BP/BB conclusions.


## Reproducibility guarantees

- A fixed seed gives the same mask, and masking never mutates its input.
- All candidate classifiers in an experimental cell see identical validation
  and test masks.
- Exact validation balanced-accuracy ties are resolved by a reproducible,
  uniform random draw that never uses test outcomes; test-oracle ties are
  reported rather than treated as arbitrary classifier identities.
- Imputation only uses observed values from the same series; no test-set
  parameters are learned.
- The official UCR test set is never used for selection or tuning.

The repository protocol and contribution guide are in [PROTOCOL.md](PROTOCOL.md)
and [CONTRIBUTING.md](CONTRIBUTING.md).
