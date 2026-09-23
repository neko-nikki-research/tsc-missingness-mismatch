"""Load and validate every dataset named by a benchmark configuration."""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import load_ucr_with_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML benchmark config.")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    datasets = config["datasets"]
    print(f"Preflight: {len(datasets)} datasets")
    for index, dataset in enumerate(datasets, start=1):
        load_ucr_with_validation(dataset, config["validation_size"], config["seeds"][0])
        print(f"[{index}/{len(datasets)}] OK {dataset}", flush=True)
    print("PREFLIGHT PASSED", flush=True)


if __name__ == "__main__":
    main()
