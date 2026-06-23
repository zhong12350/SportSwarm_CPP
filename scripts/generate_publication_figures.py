#!/usr/bin/env python3
"""Generate publication-quality figures (300 DPI PNG + vector PDF).

Usage:
    python3 scripts/generate_publication_figures.py
    python3 scripts/generate_publication_figures.py configs/football_full.yaml
    python3 scripts/generate_publication_figures.py configs/default.yaml --dpi 600
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ball_events import BallEventStream
from src.config_loader import AppConfig, load_config
from src.geometry import build_court
from src.metrics import evaluate_all
from src.pub_visualization import (
    plot_comparison_pub,
    plot_method_comparison_panel,
    plot_pareto_pub,
    plot_run_pub,
    plot_scaling_pub,
)
from src.simulator import run_all_experiments
from src.statsbomb_blf import build_blf_field


def _setup_matplotlib_cache(root: Path) -> None:
    mpl_dir = root / ".matplotlib_cache"
    mpl_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


def _pick_comparison_pair(results, metrics):
    """Baseline vs BLF-informed at largest N available for both."""
    by_method: dict[str, list] = {}
    for r, m in zip(results, metrics):
        by_method.setdefault(r.method, []).append((r, m))

    baseline = "multi_greedy" if "multi_greedy" in by_method else "single_greedy"
    proposed = "blf_informed_deployment"
    if proposed not in by_method:
        proposed = "sportswarm_full" if "sportswarm_full" in by_method else baseline

    if baseline not in by_method or proposed not in by_method:
        return list(zip(results[:2], metrics[:2]))

    n_common = max(r.num_robots for r, _ in by_method[baseline])
    if any(r.num_robots == n_common for r, _ in by_method[proposed]):
        pass
    else:
        n_common = min(
            {r.num_robots for r, _ in by_method[baseline]}
            & {r.num_robots for r, _ in by_method[proposed]}
        )

    base = next((p for p in by_method[baseline] if p[0].num_robots == n_common), by_method[baseline][-1])
    prop = next((p for p in by_method[proposed] if p[0].num_robots == n_common), by_method[proposed][-1])
    return [base, prop]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate publication-quality SportSwarm figures.")
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/default.yaml",
        help="YAML config path (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: outputs/figures/pub)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster DPI (default: 300)")
    parser.add_argument("--show-gmm", action="store_true", help="Overlay GMM ellipses on run figures")
    args = parser.parse_args(argv)

    _setup_matplotlib_cache(ROOT)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    config: AppConfig = load_config(config_path)
    out_dir = args.out_dir or (ROOT / "outputs" / "figures" / "pub")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.seed)
    court = build_court(config.court)
    blf = build_blf_field(court, config.blf, cell_size=config.court.cell_size_m)
    ball_stream = BallEventStream.create(court, blf, config.balls, rng)

    print(f"Running simulation: {config_path.name} (seed={config.seed})")
    results = run_all_experiments(config, court, blf, ball_stream, rng)
    metrics = evaluate_all(results, seed=config.seed, court_preset=config.court.preset)

    for result, m in zip(results, metrics):
        tag = f"{result.method}_N{result.num_robots}"
        path = out_dir / f"pub_{tag}.png"
        plot_run_pub(
            court,
            blf,
            result,
            m,
            path,
            dpi=args.dpi,
            show_gmm=args.show_gmm,
            gmm_config=config.gmm,
        )
        print(f"  saved {path.relative_to(ROOT)} (+ PDF)")

    pair = _pick_comparison_pair(results, metrics)
    panel_path = out_dir / "pub_fig1_method_comparison.png"
    plot_method_comparison_panel(court, blf, pair, panel_path, dpi=args.dpi)
    print(f"  saved {panel_path.relative_to(ROOT)} (+ PDF)")

    cmp_path = out_dir / "pub_comparison_bars.png"
    plot_comparison_pub(metrics, cmp_path, dpi=args.dpi)
    print(f"  saved {cmp_path.relative_to(ROOT)} (+ PDF)")

    for method in config.methods:
        if method in ("single_greedy", "gb_greedy"):
            continue
        scaling_path = out_dir / f"pub_scaling_{method}.png"
        plot_scaling_pub(metrics, method, scaling_path, dpi=args.dpi)
        print(f"  saved {scaling_path.relative_to(ROOT)} (+ PDF)")

        pareto_path = out_dir / f"pub_pareto_{method}.png"
        plot_pareto_pub(metrics, method, pareto_path, dpi=args.dpi)
        print(f"  saved {pareto_path.relative_to(ROOT)} (+ PDF)")

    print(f"\nDone. Publication figures in {out_dir.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
