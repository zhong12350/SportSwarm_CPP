"""SportSwarm-CPP demo pipeline."""

from __future__ import annotations

import csv
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

from src.ball_events import BallEventStream
from src.config_loader import AppConfig, load_config
from src.geometry import build_court
from src.metrics import RunMetrics, evaluate_all
from src.simulator import SimResult, run_all_experiments
from src.statsbomb_blf import build_blf_field
from src.visualization import (
    plot_blf_gain,
    plot_comparison,
    plot_n_scaling,
    plot_pareto,
    plot_run,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _setup_matplotlib_cache(root: Path) -> None:
    mpl_dir = root / ".matplotlib_cache"
    mpl_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


def run_pipeline(config: AppConfig, project_root: Path | None = None) -> tuple[list[SimResult], list[RunMetrics]]:
    root = project_root or _project_root()
    _setup_matplotlib_cache(root)
    rng = np.random.default_rng(config.seed)

    court = build_court(config.court)
    blf = build_blf_field(court, config.blf, cell_size=config.court.cell_size_m)
    ball_stream = BallEventStream.create(court, blf, config.balls, rng)

    results = run_all_experiments(config, court, blf, ball_stream, rng)
    metrics = evaluate_all(results, seed=config.seed, court_preset=config.court.preset)

    figures_dir = root / config.output.figures_dir
    results_dir = root / config.output.results_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    for result, m in zip(results, metrics):
        tag = f"{result.method}_N{result.num_robots}"
        fig_path = figures_dir / f"{tag}.png"
        plot_run(court, blf, result, m, fig_path, dpi=config.output.dpi)
        print(f"  saved {fig_path.relative_to(root)}")

    plot_comparison(metrics, figures_dir / "comparison.png", dpi=config.output.dpi)
    print(f"  saved {(figures_dir / 'comparison.png').relative_to(root)}")

    plot_blf_gain(metrics, figures_dir / "blf_gain.png", dpi=config.output.dpi)
    print(f"  saved {(figures_dir / 'blf_gain.png').relative_to(root)}")

    for method in config.methods:
        if method in ("single_greedy", "gb_greedy"):
            continue
        scaling_path = figures_dir / f"scaling_{method}.png"
        plot_n_scaling(metrics, method, scaling_path, dpi=config.output.dpi)
        print(f"  saved {scaling_path.relative_to(root)}")

        pareto_path = figures_dir / f"pareto_{method}.png"
        plot_pareto(metrics, method, pareto_path, dpi=config.output.dpi)
        print(f"  saved {pareto_path.relative_to(root)}")

    csv_path = results_dir / "metrics.csv"
    _write_csv(metrics, csv_path)
    print(f"  saved {csv_path.relative_to(root)}")

    _print_summary(metrics)
    return results, metrics


def _write_csv(metrics: list[RunMetrics], path: Path) -> None:
    if not metrics:
        return
    rows = [m.to_dict() for m in metrics]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(metrics: list[RunMetrics]) -> None:
    print("\n--- Metrics summary ---")
    for m in metrics:
        print(
            f"  {m.method} N={m.num_robots}: "
            f"T_clear={m.time_to_clear_s:.1f}s, "
            f"dist={m.total_distance_m:.1f}m, "
            f"collected={m.balls_collected}/{m.balls_total}, "
            f"cost={m.total_cost:.0f}"
        )


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    root = _project_root()
    config_path = root / "configs" / "football_full.yaml"
    if argv:
        config_path = Path(argv[0])
        if not config_path.is_absolute():
            config_path = root / config_path

    print(f"SportSwarm-CPP | config: {config_path}")
    config = load_config(config_path)
    run_pipeline(config, root)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
