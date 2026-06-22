#!/usr/bin/env python3
"""Run proposal §11.2 ablation experiments."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import run_pipeline
from src.ablation import ABLATION_SPECS, apply_ablation, default_ablation_groups
from src.config_loader import load_config


def main() -> int:
    config_path = ROOT / "configs" / "football_full.yaml"
    groups = default_ablation_groups()
    if len(sys.argv) > 1:
        groups = sys.argv[1:]

    base = load_config(config_path)
    all_rows = []

    for name in groups:
        if name not in ABLATION_SPECS:
            print(f"Unknown ablation: {name}")
            continue
        spec = ABLATION_SPECS[name]
        print(f"\n=== Ablation: {name} — {spec.description} ===")
        cfg = apply_ablation(base, name)
        _, metrics = run_pipeline(cfg, ROOT)
        for m in metrics:
            row = m.to_dict()
            row["ablation"] = name
            all_rows.append(row)

    out = ROOT / "outputs" / "results" / "ablation_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if all_rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
