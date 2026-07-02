"""Batch experiment runner across seeds and court presets."""

from __future__ import annotations

import csv
from copy import deepcopy
from collections import defaultdict
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
    _write_summary_csv(all_metrics, results_dir / "batch_summary.csv")
    return all_metrics


def _write_csv(metrics: list[RunMetrics], path: Path) -> None:
    if not metrics:
        return
    rows = [m.to_dict() for m in metrics]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(metrics: list[RunMetrics], path: Path) -> None:
    if not metrics:
        return

    grouped: dict[tuple[str, str, int], list[RunMetrics]] = defaultdict(list)
    for metric in metrics:
        grouped[(metric.court_preset, metric.method, metric.num_robots)].append(metric)

    baseline_time: dict[tuple[str, int], float] = {}
    for (court, method, n), rows in grouped.items():
        if method == "multi_greedy":
            baseline_time[(court, n)] = float(np.mean([m.time_to_clear_s for m in rows]))

    summary_rows = []
    for (court, method, n), rows in sorted(grouped.items()):
        times = np.array([m.time_to_clear_s for m in rows], dtype=float)
        dists = np.array([m.total_distance_m for m in rows], dtype=float)
        clearances = np.array([m.clearance_rate for m in rows], dtype=float)
        collisions = np.array([m.collision_count for m in rows], dtype=float)
        violations = np.array([m.safety_violation_count for m in rows], dtype=float)
        finite_clearances = np.array(
            [
                m.min_player_clearance_m
                for m in rows
                if m.min_player_clearance_m is not None
                and np.isfinite(m.min_player_clearance_m)
            ],
            dtype=float,
        )
        base = baseline_time.get((court, n))
        gain = ""
        if base and base > 0:
            gain = 100.0 * (base - float(times.mean())) / base
        summary_rows.append(
            {
                "court_preset": court,
                "method": method,
                "num_robots": n,
                "n_seeds": len(rows),
                "time_mean_s": float(times.mean()),
                "time_std_s": float(times.std(ddof=1)) if len(times) > 1 else 0.0,
                "distance_mean_m": float(dists.mean()),
                "distance_std_m": float(dists.std(ddof=1)) if len(dists) > 1 else 0.0,
                "clearance_rate_mean": float(clearances.mean()),
                "time_gain_vs_multi_greedy_pct": gain,
                "collision_count_mean": float(collisions.mean()),
                "safety_violation_count_mean": float(violations.mean()),
                "min_player_clearance_mean_m": (
                    float(finite_clearances.mean()) if len(finite_clearances) else ""
                ),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


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
