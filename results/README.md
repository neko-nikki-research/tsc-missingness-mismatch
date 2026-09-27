# Generated experiment outputs

The live benchmark CSVs, tables, and figures are generated locally and ignored
by Git. Frozen interim checkpoints are intentional, versioned exceptions:
their CSVs, run manifest, aggregate tables, and Matplotlib figures are
committed so the PR can be reviewed before all 64 datasets finish. They are
**not** final study results.

- `interim_ucr64_2026-09-25_27datasets/`: checkpoint after 27/64 datasets.
- `interim_ucr64_2026-09-27_46datasets/`: checkpoint after 46/64 datasets.

For the formal study, run:

```powershell
python -m src.run_benchmark --config configs\main_protocol_balanced.yaml
python -m src.analyze_results --results-dir results\main_protocol_ucr64_v1_4
python -m src.report_results --results-dir results\main_protocol_ucr64_v1_4
```

For the committed interim figures, run:

```powershell
python -m src.plot_interim_results --results-dir results\interim_ucr64_2026-09-25_27datasets
```

The tracked configuration and fixed seeds make the experiment reproducible.
