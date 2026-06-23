#!/usr/bin/env python3
"""Build BLF heatmap from StatsBomb event JSON files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import BLFConfig, load_config
from src.geometry import build_court
from src.statsbomb_blf import build_blf_field


def main() -> int:
    parser = argparse.ArgumentParser(description="StatsBomb → BLF heatmap")
    parser.add_argument("--config", default="configs/football_full.yaml")
    parser.add_argument("--data-dir", default=None, help="StatsBomb events directory")
    parser.add_argument("--style", default="all", choices=["all", "offensive", "defensive"])
    parser.add_argument("--output", default="outputs/figures/blf_statsbomb.png")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    if args.data_dir:
        config.blf.statsbomb_dir = Path(args.data_dir)
    config.blf.source = "statsbomb"
    config.blf.tactical_style = args.style

    court = build_court(config.court)
    print("Reading StatsBomb events (uses cache if available)...", flush=True)
    blf = build_blf_field(court, config.blf, cell_size=config.court.cell_size_m, use_cache=True)
    print("Rendering figure...", flush=True)

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(
        blf.values,
        origin="lower",
        extent=[0, court.width_m, 0, court.height_m],
        cmap="YlOrRd",
    )
    ax.set_title(f"BLF from StatsBomb ({args.style})")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
