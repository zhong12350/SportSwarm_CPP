#!/usr/bin/env python3
"""Run batch experiments across multiple seeds."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.batch_experiments import run_batch, run_multi_court_batch
from src.config_loader import load_config


def main() -> int:
    config_path = ROOT / "configs" / "football_full.yaml"
    if len(sys.argv) > 1:
        config_path = ROOT / sys.argv[1]

    if sys.argv[-1] == "--multi-court":
        print("Running multi-court batch...")
        metrics = run_multi_court_batch(ROOT)
    else:
        config = load_config(config_path)
        print(f"Batch run | config: {config_path} | seeds: {config.experiment.batch_seeds}")
        metrics = run_batch(config, ROOT)

    print(f"Completed {len(metrics)} runs → outputs/results/batch_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
