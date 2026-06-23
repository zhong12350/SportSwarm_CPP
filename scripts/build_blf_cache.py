#!/usr/bin/env python3
"""One-time BLF cache build from StatsBomb events (run before main.py)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config
from src.geometry import build_court
from src.statsbomb_blf import build_blf_from_statsbomb


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and cache StatsBomb BLF")
    parser.add_argument("--config", default="configs/football_full.yaml")
    parser.add_argument("--data-dir", default="data/statsbomb/data/events")
    parser.add_argument("--style", default="all", choices=["all", "offensive", "defensive"])
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    config.blf.statsbomb_dir = Path(args.data_dir)
    config.blf.tactical_style = args.style
    config.blf.source = "statsbomb"

    court = build_court(config.court)
    print("Keep Mac awake; first build scans 4235 files (~5-15 min).", flush=True)
    build_blf_from_statsbomb(
        court,
        config.blf,
        cell_size=config.court.cell_size_m,
        verbose=True,
        use_cache=True,
    )
    print("Cache ready. Now run: python3 main.py configs/football_full.yaml", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
