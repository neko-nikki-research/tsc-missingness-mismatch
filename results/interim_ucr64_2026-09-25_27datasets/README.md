# Interim UCR64 checkpoint — 27/64 datasets

This is a frozen, **partial** checkpoint of the main experiment, captured from
`results/main_protocol_ucr64_v1_4/` after `CinCECGTorso` completed on
2026-09-25 at 20:58 KST. The continuing run and its live checkpoints were not
modified or stopped. The 27 datasets are the first completed datasets in the
configured order, **not a random or representative sample** of all 64.

- Protocol/configuration: `PROTOCOL.md` and `configs/main_protocol_balanced.yaml`.
- Run configuration fingerprint: `195e4df2bf6545b7df2d65f21feba2b0e1ae3f9080bb2d0640b4eab70b3c4789`.
- `raw_results.csv`: 9,720 rows = 27 datasets × 5 seeds × 6 rates × 4 pattern pairs × 3 classifiers.
- `selection_results.csv`: 3,240 rows = 27 × 5 × 6 × 4.
- Rates: 5%, 10%, 15%, 20%, 25%, 30%; masks: point/block for validation and test.
- Training stays unmasked; validation has source masking; official UCR test has target masking.
- Linear interpolation is the only main-study imputer.
- Classifier selection uses validation **balanced accuracy**; the test oracle
  and selection regret are calculated afterward from test balanced accuracy.

The CSVs were checked for full row counts, duplicate experiment keys, missing
metric values, nonnegative regret, source/target invariance, and exact
selection/oracle/regret reproduction from every three-classifier raw group.
The snapshot's `run_manifest.json` matches the tracked configuration hash.

SHA-256:

- `raw_results.csv`: `315b153c9b120778fec2d42dfccedf8b4a0250301a2ffa0263f332805d9736e8`
- `selection_results.csv`: `b8539d06d39efb9a882d419c0cbea3e20812fb7d78ecaf62031e526436174025`

Figures and aggregate tables are generated from this frozen selection CSV with
Matplotlib, using:

```powershell
python -m src.plot_interim_results --results-dir results/interim_ucr64_2026-09-25_27datasets
```

The plotted values are descriptive. Do not treat them or any comparison between
matched and mismatched patterns as final evidence until the 64-dataset run is
complete and the planned paired analysis has been performed.
