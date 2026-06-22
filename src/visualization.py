"""Matplotlib visualization for ball retrieval demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.fields import BallLandingField
from src.geometry import Court
from src.metrics import RunMetrics, pareto_front
from src.simulator import SimResult

METHOD_LABELS = {
    "single_greedy": "Single Greedy (Tennibot)",
    "gb_greedy": "GB-Greedy",
    "multi_greedy": "Multi Greedy",
    "voronoi_assignment": "Voronoi (MDCPP-style)",
    "mdcpp_voronoi": "MDCPP Voronoi",
    "blf_informed_deployment": "BLF-Informed Deployment",
    "uniform_cpp": "Uniform CPP (Boustrophedon)",
    "blf_weighted_cpp": "BLF-Weighted CPP",
    "gmm_swarm": "GMM Swarm (SwarmPRM-lite)",
    "sportswarm_full": "SportSwarm-Full (BLF+GMM+CVaR)",
}

ROBOT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def plot_run(
    court: Court,
    blf: BallLandingField,
    result: SimResult,
    metrics: RunMetrics,
    out_path: Path,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.5))
    extent = [0, court.width_m, 0, court.height_m]
    im = ax.imshow(
        blf.values,
        origin="lower",
        extent=extent,
        cmap="YlOrRd",
        alpha=0.75,
        vmin=0,
        vmax=1,
    )
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Ball-Landing Field")

    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            court.width_m,
            court.height_m,
            fill=False,
            edgecolor="black",
            linewidth=2,
        )
    )

    for ball in result.balls:
        if ball.collected:
            ax.scatter(ball.x, ball.y, c="lightgray", s=25, marker="o", alpha=0.5)
        else:
            ax.scatter(ball.x, ball.y, c="red", s=40, marker="o", edgecolors="darkred")

    for robot in result.robots:
        color = ROBOT_COLORS[robot.robot_id % len(ROBOT_COLORS)]
        if len(robot.path) > 1:
            xs = [p[0] for p in robot.path]
            ys = [p[1] for p in robot.path]
            ax.plot(xs, ys, color=color, linewidth=1.5, label=f"Robot {robot.robot_id}")
        ax.scatter(robot.path[0][0], robot.path[0][1], c=color, s=80, marker="s", zorder=5)
        ax.scatter(robot.x, robot.y, c=color, s=60, marker="^", zorder=5)

    for player in result.players:
        circle = plt.Circle(
            (player.x, player.y),
            player.radius_m,
            color="purple",
            alpha=0.3,
            label="Player" if player.player_id == 0 else None,
        )
        ax.add_patch(circle)

    title = (
        f"{_label(result.method)} | N={result.num_robots}\n"
        f"T_clear={metrics.time_to_clear_s:.1f}s, "
        f"dist={metrics.total_distance_m:.1f}m, "
        f"collected={metrics.balls_collected}/{metrics.balls_total}"
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0, court.width_m)
    ax.set_ylim(0, court.height_m)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(
    metrics_list: list[RunMetrics],
    out_path: Path,
    dpi: int = 150,
) -> None:
    robot_counts = sorted({m.num_robots for m in metrics_list})
    methods = list(dict.fromkeys(m.method for m in metrics_list))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    width = 0.8 / max(len(robot_counts), 1)

    for i, n in enumerate(robot_counts):
        subset = [m for m in metrics_list if m.num_robots == n]
        x = np.arange(len(methods))
        times = []
        dists = []
        for method in methods:
            row = next((m for m in subset if m.method == method), None)
            times.append(row.time_to_clear_s if row else 0)
            dists.append(row.total_distance_m if row else 0)
        offset = (i - len(robot_counts) / 2 + 0.5) * width
        axes[0].bar(x + offset, times, width=width, label=f"N={n}")
        axes[1].bar(x + offset, dists, width=width, label=f"N={n}")

    labels = [_label(m) for m in methods]
    for ax in axes:
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Time to clear (s)")
    axes[0].set_title("Clearance time")
    axes[1].set_ylabel("Total distance (m)")
    axes[1].set_title("Total robot distance")

    fig.suptitle("SportSwarm-CPP: Method Comparison", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_n_scaling(
    metrics_list: list[RunMetrics],
    method: str,
    out_path: Path,
    dpi: int = 150,
) -> None:
    subset = sorted(
        [m for m in metrics_list if m.method == method],
        key=lambda m: m.num_robots,
    )
    if not subset:
        return

    ns = [m.num_robots for m in subset]
    times = [m.time_to_clear_s for m in subset]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ns, times, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.set_xlabel("Number of robots")
    ax.set_ylabel("Time to clear (s)")
    ax.set_title(f"N scaling: {_label(method)}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(
    metrics_list: list[RunMetrics],
    method: str,
    out_path: Path,
    dpi: int = 150,
) -> None:
    subset = [m for m in metrics_list if m.method == method]
    if not subset:
        return

    front = pareto_front(subset)
    costs = [m.total_cost for m in subset]
    times = [m.time_to_clear_s for m in subset]
    f_costs = [m.total_cost for m in front]
    f_times = [m.time_to_clear_s for m in front]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(costs, times, c="lightgray", s=60, label="All runs")
    ax.plot(f_costs, f_times, "ro-", linewidth=2, markersize=8, label="Pareto front")
    ax.set_xlabel("Total cost (N × unit cost)")
    ax.set_ylabel("Time to clear (s)")
    ax.set_title(f"Cost–Performance Pareto: {_label(method)}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_blf_gain(
    metrics_list: list[RunMetrics],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """BLF vs uniform gain at equal N."""
    pairs = [
        ("multi_greedy", "blf_informed_deployment"),
        ("uniform_cpp", "blf_weighted_cpp"),
    ]
    robot_counts = sorted({m.num_robots for m in metrics_list})

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(robot_counts))
    width = 0.35

    for idx, (base, improved) in enumerate(pairs):
        gains = []
        for n in robot_counts:
            b = next((m for m in metrics_list if m.method == base and m.num_robots == n), None)
            i = next((m for m in metrics_list if m.method == improved and m.num_robots == n), None)
            if b and i and b.time_to_clear_s > 0:
                gains.append(100 * (b.time_to_clear_s - i.time_to_clear_s) / b.time_to_clear_s)
            else:
                gains.append(0)
        ax.bar(x + idx * width, gains, width=width, label=f"{_label(improved)} vs {_label(base)}")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f"N={n}" for n in robot_counts])
    ax.set_ylabel("Time reduction (%)")
    ax.set_title("BLF-informed gain vs baseline")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
