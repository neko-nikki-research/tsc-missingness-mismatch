# The Effect of Missingness-Pattern Mismatch on Method Selection for Time-Series Classification

A controlled empirical study of how mismatched missingness patterns affect method selection in time-series classification.

## Status

Development stage

## Team

* Ruiqi Zhao — The University of Tokyo (UTokyo) — First Author / Project Lead
* Zishun Yuan — State University of New York - Korea (SUNY-Korea) — Second Author / Experimental Lead
* Jianfan Deng — Anhui University of Science and Technology (AHUST) — Third Author / Reproducibility Lead

## Research Question

Does a mismatch between the missingness pattern used for validation and the missingness pattern encountered during testing affect classifier selection and final classification performance?

## Scope

This study focuses on:

* Univariate time-series classification
* Equal-length time series
* Point missingness
* Contiguous block missingness
* Missing rates of 10%, 20%, and 30%
* Linear interpolation
* Three candidate classification methods

## Experimental Design

The core experiment uses a 2 × 2 design:

| Validation | Test  | Condition  |
| ---------- | ----- | ---------- |
| Point      | Point | Matched    |
| Point      | Block | Mismatched |
| Block      | Point | Mismatched |
| Block      | Block | Matched    |

The primary evaluation metric is balanced accuracy.

## Candidate Methods

1. 1-NN DTW
2. MiniRocket + Linear Classifier
3. Statistical Features + Random Forest

## Repository Structure

```text
tsc-missingness-mismatch/
├── README.md
├── PROTOCOL.md
├── AUTHORS.md
├── CONTRIBUTING.md
├── configs/
├── docs/
├── src/
├── experiments/
├── notebooks/
├── results/
├── figures/
└── paper/
```

## Protocol

The complete experimental design, evaluation rules, reproducibility requirements, and protocol-change policy are documented in [`PROTOCOL.md`](PROTOCOL.md).

## Reproducibility

All formal experiments will record:

* Dataset sources and versions
* Masking procedures
* Random seeds
* Imputation settings
* Classifier configurations
* Software and dependency versions
* Raw experimental outputs
* Analysis scripts
* Figure-generation scripts

Test data will not be used for model selection or protocol tuning.

## Authorship

The current authorship order is provisional and will be confirmed according to actual contributions and the requirements of the eventual publication venue.
