"""Batch experiment runner across seeds and court presets."""

from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

import numpy as np

from src.ball_events import BallEventStream
from src.config_loader import AppConfig, load_config
from src.fields import BallLandingField
from src.geometry import build_court
from src.metrics import RunMetrics, evaluate_all
from src.simulator import SimResult, run_all_experiments
from src.statsbomb_blf import build_blf_field


def run_batch(
    config: AppConfig,
    project_root: Path,
    seeds: list[int] | None = None,
) -> list[RunMetrics]:
    seeds = seeds or config.experiment.batch_seeds
    all_metrics: list[RunMetrics] = []

    court = build_court(config.court)
    blf = build_blf_field(court, config.blf, cell_size=config.court.cell_size_m)

    for seed in seeds:
        rng = np.random.default_rng(seed)
        stream = BallEventStream.create(court, blf, config.balls, rng)
        cfg_seed = deepcopy(config)
        cfg_seed.seed = seed
        results = run_all_experiments(cfg_seed, court, blf, stream, rng)
        metrics = evaluate_all(results, seed=seed, court_preset=config.court.preset)
        all_metrics.extend(metrics)

    results_dir = project_root / config.output.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "batch_results.csv"
    _write_csv(all_metrics, csv_path)
    return all_metrics


def _write_csv(metrics: list[RunMetrics], path: Path) -> None:
    if not metrics:
        return
    rows = [m.to_dict() for m in metrics]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_multi_court_batch(project_root: Path) -> list[RunMetrics]:
    configs = [
        project_root / "configs" / "football_full.yaml",
        project_root / "configs" / "tennis_court.yaml",
    ]
    combined: list[RunMetrics] = []
    for cfg_path in configs:
        if not cfg_path.exists():
            continue
        config = load_config(cfg_path)
        combined.extend(run_batch(config, project_root))
    return combined
