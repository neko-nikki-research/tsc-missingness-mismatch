# Missingness Mismatch Benchmark for Time-Series Classification

This branch contains the protocol-compliant benchmark for testing whether a
mismatch between validation-time and deployment-time missingness changes
classifier selection and deployment performance.

## Main experiment at a glance

| Component | Fixed main-experiment choice |
| --- | --- |
| Data | UCR univariate, equal-length, originally complete series |
| Datasets | GunPoint, ECG200, ItalyPowerDemand |
| Train split | Official UCR train; kept complete and unchanged |
| Validation split | Stratified split from official train; receives the source mask |
| Deployment split | Official UCR test; receives the target mask |
| Missingness matrix | Point → Point, Point → Block, Block → Point, Block → Block |
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
python -m src.analyze_results --results-dir results\main_protocol_balanced
python -m src.report_results --results-dir results\main_protocol_balanced
```

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

`configs/exploratory/` may compare zero/mean imputation or prefix/suffix
missingness in future robustness work. These configurations must be reported
separately and must not be pooled with the main PP/PB/BP/BB conclusions.

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
