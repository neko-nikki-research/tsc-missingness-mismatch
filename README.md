# The Effect of Missingness Pattern Mismatch on Method Selection for Time Series Classification

This repository contains a reproducible benchmark for testing whether a mismatch between validation-time and deployment-time temporal missingness affects classifier selection and deployment performance.

## Team

* Ruiqi Zhao - The University of Tokyo - First Author and Project Lead
* Zishun Yuan - State University of New York Korea - Experimental Lead
* Jianfan Deng - Anhui University of Science and Technology - Reproducibility Lead

## Implemented benchmark

- UCR datasets are loaded through aeon. Official train is split into stratified train and validation data; official train remains complete, source missingness is applied only to validation, and official test receives target missingness for simulated deployment.
- **Main experiment only:** point and block missingness, forming the pre-specified 2 x 2 source/target matrix (PP, PB, BP, BB).
- **Main experiment only:** linear interpolation.
- **Main classifiers:** 1NN-DTW, MiniROCKET + Ridge linear classifier, and fixed statistical features + Random Forest.
- Prefix/suffix missingness and zero/mean imputation are supported only for a future, separately labelled exploratory robustness analysis.
- Selection uses validation balanced accuracy only. The test set is used only to measure deployment balanced accuracy, identify the post-hoc oracle, and calculate selection regret.

`selection regret = oracle test balanced accuracy - selected model test balanced accuracy`

The raw result file uses the explicit fields `val_balanced_accuracy` and
`test_balanced_accuracy`. The selection result file uses
`selected_test_balanced_accuracy` and `oracle_test_balanced_accuracy`.
Ordinary accuracy is not calculated or written by the benchmark.

## Quick start

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest
python scripts\check_environment.py
python -m src.run_benchmark --config configs\smoke.yaml
```

The smoke outputs are written to `results/raw_results.csv` and `results/selection_results.csv`.

## Experiment configurations

- `configs/smoke.yaml`: one small GunPoint check.
- `configs/pilot.yaml`: single GunPoint pilot with complete classifier settings.
- `configs/rate_expanded.yaml`: 3 datasets, 5 seeds, point/block patterns, and 10%, 20%, and 30% missingness.
- `configs/multidataset_imputers.yaml`: exploratory imputation robustness comparison; not part of the main result.
- `configs/pattern_rate_smoke.yaml` and `configs/pattern_expanded.yaml`: exploratory prefix/suffix robustness checks; not part of the main result.
- `configs/main_protocol_balanced.yaml`: main 2 x 2, linear-imputation study across six missingness rates, using balanced-accuracy selection.
- `configs/rate_expanded_six_rates_balanced.yaml`: superseded historical configuration; its old TSF results are not formal protocol results.

Generate summaries or the rate report with:

```powershell
python -m src.analyze_results --results-dir results\multidataset_expanded
python -m src.report_results --results-dir results\rate_expanded
```

## Reproducibility

Mask generation is deterministic for a fixed seed and never mutates its input. Every classifier in the same experimental cell receives the same masked and imputed train, validation, and test arrays. Unit tests cover masking and imputation behavior.

The repository protocol and contribution guidance are in [PROTOCOL.md](PROTOCOL.md) and [CONTRIBUTING.md](CONTRIBUTING.md).
