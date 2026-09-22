# Generated experiment outputs

Benchmark CSVs, tables, and figures are generated locally and ignored by Git.

For the formal study, run:

```powershell
python -m src.run_benchmark --config configs\main_protocol_balanced.yaml
python -m src.analyze_results --results-dir results\main_protocol_balanced
python -m src.report_results --results-dir results\main_protocol_balanced
```

This keeps the repository source-focused while allowing a complete experiment
to be reproduced from the tracked configuration and fixed seeds.
