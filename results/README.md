# Generated experiment outputs

Benchmark CSVs, tables and figures are generated locally and ignored by Git.
The directories below are intentional, versioned exceptions: their CSVs,
manifests, analysis tables and Matplotlib figures are committed. Each has its
own README with configuration, verification notes, checksums and headline
numbers.

## Final results

- `final_ucr64_v1_4/`: **main study** (`PROTOCOL.md`, circular block). All 64
  datasets, PP/PB/BP/BB; 23,040 raw and 7,680 selection rows, copied unchanged
  from the run directory `main_protocol_ucr64_v1_4/`.
- `final_linear_block_v1_5/`: **linear-block robustness analysis**
  (`PROTOCOL.md` v1.5 section 5.1). The only change is a non-wrapping linear
  block instead of the circular block. `run/` holds the byte-exact PB/BP/BB run
  output (17,280 raw and 5,760 selection rows, copied from
  `supplementary_linear_block_v1_5/`); the top-level CSVs add the main-study PP
  rows, after a reuse check recorded in `assembly_manifest.json`. Figure 5
  compares circular and linear blocks side by side.

The linear-block results are a supplementary sensitivity analysis. They are
reported separately and must never be pooled with the circular-block main
study.

## Dataset inventory

`dataset_inventory.csv` (from `scripts/dataset_inventory.py`, 2026-10-07)
records, for each of the 64 datasets, the file aeon 1.6.0 reads (4 bundled, 60
from the download cache), its SHA-256 and modification time, train/test sizes,
series length, class counts, and for each seed a SHA-256 of the validation
indices. It was made after the runs, not during them: the data files were last
modified on 2026-09-21 to 2026-09-23, before the main run's first checkpoint, but
the run manifests do not record file hashes. Validation subsets lack one class
in 13 of the 320 dataset-seed splits (ECG5000 seeds 2-4; Mallat and
WordSynonyms, all seeds); every fitting subset contains all classes.

## Interim checkpoints

Frozen during the main-study run so the PR could be reviewed before all 64
datasets finished. They are **not** final study results and are superseded by
`final_ucr64_v1_4/`.

- `interim_ucr64_2026-09-25_27datasets/`: checkpoint after 27/64 datasets.
- `interim_ucr64_2026-09-27_46datasets/`: checkpoint after 46/64 datasets.

## Local run directories (not committed)

- `main_protocol_ucr64_v1_4/`: live output of the main-study run.
- `supplementary_linear_block_v1_5/`: live output of the 24-core linear-block
  run used for the final results.
- `supplementary_linear_block_v1_5_aborted_18cores/`: an earlier 18-core start
  that was stopped after 8 datasets; not used.
- Other directories hold superseded or exploratory runs and are not study
  results.

## Reproduce

Main study:

```powershell
python -m src.run_benchmark --config configs\main_protocol_balanced.yaml
python -m src.paired_analysis --results-dir results\final_ucr64_v1_4 --config configs\main_protocol_balanced.yaml --output-dir results\final_ucr64_v1_4\analysis
python -m src.plot_final_results --results-dir results\final_ucr64_v1_4 --config configs\main_protocol_balanced.yaml --figures-dir results\final_ucr64_v1_4\figures
```

Linear-block robustness analysis (needs the main study's PP rows):

```powershell
python -m src.run_benchmark --config configs\supplementary_linear_block.yaml
python -m src.assemble_linear_block --main-results results\final_ucr64_v1_4 --supplementary-results results\supplementary_linear_block_v1_5 --config configs\supplementary_linear_block.yaml --output-dir results\final_linear_block_v1_5
python -m src.paired_analysis --results-dir results\final_linear_block_v1_5 --config configs\supplementary_linear_block.yaml --output-dir results\final_linear_block_v1_5\analysis
python -m src.plot_final_results --results-dir results\final_linear_block_v1_5 --config configs\supplementary_linear_block.yaml --figures-dir results\final_linear_block_v1_5\figures --compare-results results\final_ucr64_v1_4 --compare-config configs\main_protocol_balanced.yaml
```

Interim figures:

```powershell
python -m src.plot_interim_results --results-dir results\interim_ucr64_2026-09-25_27datasets
```

The tracked configurations and fixed seeds make every experiment reproducible.
