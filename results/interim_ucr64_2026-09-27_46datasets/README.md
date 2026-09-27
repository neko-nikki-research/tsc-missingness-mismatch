# Interim UCR64 checkpoint — 46/64 datasets

This is a frozen, partial checkpoint of the main experiment, copied from
`results/main_protocol_ucr64_v1_4/` on 2026-09-27. The benchmark process was
still running when the snapshot was made; the live experiment files were not
modified. The 46 datasets are the completed prefix in the configured order,
not a random or representative sample of the 64 datasets.

- Protocol/configuration: `PROTOCOL.md` and `configs/main_protocol_balanced.yaml`.
- Configuration fingerprint: `195e4df2bf6545b7df2d65f21feba2b0e1ae3f9080bb2d0640b4eab70b3c4789`.
- `raw_results.csv`: 16,560 rows = 46 datasets × 5 seeds × 6 rates × 4 pattern pairs × 3 classifiers.
- `selection_results.csv`: 5,520 rows = 46 × 5 × 6 × 4.
- Rates: 5%, 10%, 15%, 20%, 25%, 30%; validation/test masks: point or block.
- Training stays unmasked; validation has the source pattern; official UCR test has the target pattern.
- Linear interpolation is the only main-study imputer.
- Classifier selection uses validation balanced accuracy; test balanced accuracy is used only for post-selection test metrics and candidate-set oracle regret.

The tables and figures are descriptive summaries of this partial, ordered
subset. They are not final confirmatory results, and the unfinished datasets
may change the aggregate findings. The run manifest accompanies the CSVs.
