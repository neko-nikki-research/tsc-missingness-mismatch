# The Effect of Missingness Pattern Mismatch on Method Selection for Time Series Classification

This repository contains a reproducible benchmark for testing whether a mismatch between validation-time and deployment-time temporal missingness affects classifier selection and deployment performance.

## Team

* Ruiqi Zhao - The University of Tokyo - First Author and Project Lead
* Zishun Yuan - State University of New York Korea - Experimental Lead
* Jianfan Deng - Anhui University of Science and Technology - Reproducibility Lead

## Implemented benchmark

- UCR datasets are loaded through aeon. Official train is split into stratified train and validation data; official test is reserved for simulated deployment.
- Missingness mechanisms: point, block, prefix, and suffix.
- Imputers: zero, mean, and linear interpolation.
- Classifiers: 1NN-DTW, Time Series Forest, and MiniROCKET.
- Selection uses validation accuracy only. The test set is used only to measure deployment accuracy, identify the post-hoc oracle, and calculate selection regret.

`selection regret = oracle test accuracy - selected model test accuracy`

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
- `configs/multidataset_imputers.yaml`: compares zero, mean, and linear imputation.
- `configs/pattern_rate_smoke.yaml`: validates prefix/suffix and multi-rate support before a larger run.

Generate summaries or the rate report with:

```powershell
python -m src.analyze_results --results-dir results\multidataset_expanded
python -m src.report_results --results-dir results\rate_expanded
```

## Reproducibility

Mask generation is deterministic for a fixed seed and never mutates its input. Every classifier in the same experimental cell receives the same masked and imputed train, validation, and test arrays. Unit tests cover masking and imputation behavior.

The repository protocol and contribution guidance are in [PROTOCOL.md](PROTOCOL.md) and [CONTRIBUTING.md](CONTRIBUTING.md).
