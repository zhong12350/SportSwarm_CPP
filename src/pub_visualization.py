"""High-fidelity publication figures for SportSwarm-CPP simulation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Rectangle
from scipy.ndimage import gaussian_filter, zoom

from src.fields import BallLandingField
from src.geometry import Court
from src.gmm import fit_gmm_to_balls
from src.metrics import RunMetrics, pareto_front
from src.pub_style import (
    BLF_CMAP,
    COURT_GREEN,
    COURT_LINE,
    COURT_LINE_ALPHA,
    OKABE_ITO,
    apply_pub_style,
    pub_figure_context,
    save_figure,
)
from src.simulator import SimResult
from src.visualization import METHOD_LABELS, _label

ROBOT_COLORS = OKABE_ITO


def _upsample_field(values: np.ndarray, factor: int = 4, smooth_sigma: float = 1.2) -> np.ndarray:
    if factor > 1:
        field = zoom(values, factor, order=3)
    else:
        field = values.copy()
    if smooth_sigma > 0:
        field = gaussian_filter(field, sigma=smooth_sigma)
    peak = field.max()
    if peak > 0:
        field /= peak
    return field


def _draw_football_markings(ax: plt.Axes, court: Court) -> None:
    x0, y0, x1, y1 = court.playable_bounds
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    lw = 1.2

    ax.add_patch(Rectangle((x0, y0), w, h, facecolor=COURT_GREEN, edgecolor="none", zorder=0))

    for x_line in (cx,):
        ax.plot([x_line, x_line], [y0, y1], color=COURT_LINE, lw=lw, alpha=COURT_LINE_ALPHA, zorder=1)

    circle = Circle((cx, cy), min(w, h) * 0.12, fill=False, edgecolor=COURT_LINE, lw=lw, alpha=COURT_LINE_ALPHA, zorder=1)
    ax.add_patch(circle)

    pa_depth = w * 0.16
    pa_width = h * 0.55
    for side_x in (x0, x1 - pa_depth):
        ax.add_patch(
            Rectangle(
                (side_x, cy - pa_width / 2),
                pa_depth,
                pa_width,
                fill=False,
                edgecolor=COURT_LINE,
                lw=lw * 0.9,
                alpha=COURT_LINE_ALPHA * 0.85,
                zorder=1,
            )
        )


def _draw_tennis_markings(ax: plt.Axes, court: Court) -> None:
    x0, y0, x1, y1 = court.playable_bounds
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    singles_half = min(h * 0.375, 4.115)
    service_y = cy - singles_half / 2
    lw = 1.0

    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="#3A7D44", edgecolor="none", zorder=0))

    ax.plot([x0, x1], [cy, cy], color=COURT_LINE, lw=lw, alpha=COURT_LINE_ALPHA, zorder=1)
    ax.plot([x0, x1], [cy - singles_half, cy - singles_half], color=COURT_LINE, lw=lw, alpha=0.75, zorder=1)
    ax.plot([x0, x1], [cy + singles_half, cy + singles_half], color=COURT_LINE, lw=lw, alpha=0.75, zorder=1)
    ax.plot([cx, cx], [y0, y1], color=COURT_LINE, lw=lw, alpha=COURT_LINE_ALPHA, zorder=1)
    ax.plot([cx, cx], [service_y, cy + singles_half], color=COURT_LINE, lw=lw * 0.8, alpha=0.7, zorder=1)
    ax.plot([cx, cx], [cy - singles_half, service_y], color=COURT_LINE, lw=lw * 0.8, alpha=0.7, zorder=1)


def draw_court_surface(ax: plt.Axes, court: Court) -> None:
    if court.preset == "tennis_court":
        _draw_tennis_markings(ax, court)
    else:
        _draw_football_markings(ax, court)


def _draw_blf_heatmap(
    ax: plt.Axes,
    court: Court,
    blf: BallLandingField,
    upsample: int = 4,
    alpha: float = 0.72,
) -> plt.AxesImage:
    field = _upsample_field(blf.values, factor=upsample)
    extent = [0, court.width_m, 0, court.height_m]
    return ax.imshow(
        field,
        origin="lower",
        extent=extent,
        cmap=BLF_CMAP,
        alpha=alpha,
        vmin=0,
        vmax=1,
        interpolation="bilinear",
        zorder=2,
    )


def _plot_time_colored_path(
    ax: plt.Axes,
    path: list[tuple[float, float]],
    base_color: str,
    zorder: int = 4,
) -> None:
    if len(path) < 2:
        return
    pts = np.asarray(path, dtype=float)
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    n = len(segments)
    colors = np.tile(mpl_color_rgba(base_color), (n, 1))
    colors[:, 3] = np.linspace(0.25, 1.0, n)
    lc = LineCollection(segments, colors=colors, linewidths=2.0, capstyle="round", joinstyle="round", zorder=zorder)
    ax.add_collection(lc)

    step = max(len(path) // 6, 1)
    for i in range(step, len(path) - 1, step):
        x0, y0 = path[i - 1]
        x1, y1 = path[i]
        dx, dy = x1 - x0, y1 - y0
        if dx * dx + dy * dy < 1e-6:
            continue
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0,
                color=base_color,
                alpha=0.55,
                zorder=zorder + 1,
            )
        )


def mpl_color_rgba(hex_color: str) -> tuple[float, float, float, float]:
    from matplotlib.colors import to_rgba

    return to_rgba(hex_color)


def _draw_gmm_ellipses(ax: plt.Axes, balls, num_robots: int, config_gmm) -> None:
    gmm = fit_gmm_to_balls(balls, num_robots, config_gmm)
    if gmm is None:
        return
    for k in range(gmm.n_components):
        mean = gmm.means_[k]
        cov = gmm.covariances_[k]
        if cov.ndim == 1:
            cov = np.diag(cov)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2 * 2.0 * np.sqrt(np.maximum(vals, 1e-6))
        ell = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            fill=False,
            edgecolor="#444444",
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
            zorder=3,
        )
        ax.add_patch(ell)


def plot_run_pub(
    court: Court,
    blf: BallLandingField,
    result: SimResult,
    metrics: RunMetrics,
    out_path: Path,
    dpi: int = 300,
    show_gmm: bool = False,
    gmm_config=None,
    panel_label: str | None = None,
) -> None:
    """Single high-fidelity simulation snapshot (trajectories + BLF + court)."""
    apply_pub_style()

    fig_w = 7.2 if court.preset == "football_7v7" else 5.5
    fig_h = fig_w * court.height_m / court.width_m * 1.15
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    draw_court_surface(ax, court)
    im = _draw_blf_heatmap(ax, court, blf)

    if show_gmm and gmm_config is not None:
        _draw_gmm_ellipses(ax, result.balls, result.num_robots, gmm_config)

    for ball in result.balls:
        if ball.collected:
            ax.scatter(
                ball.x,
                ball.y,
                s=18,
                c="#BDBDBD",
                edgecolors="#757575",
                linewidths=0.3,
                alpha=0.55,
                zorder=5,
            )
        else:
            ax.scatter(
                ball.x,
                ball.y,
                s=28,
                c="#F4D03F",
                edgecolors="#1A1A1A",
                linewidths=0.45,
                zorder=6,
            )

    for player in result.players:
        ax.add_patch(
            Circle(
                (player.x, player.y),
                player.radius_m,
                facecolor="#9B59B6",
                edgecolor="#4A235A",
                alpha=0.35,
                linewidth=0.8,
                zorder=4,
            )
        )

    for robot in result.robots:
        color = ROBOT_COLORS[robot.robot_id % len(ROBOT_COLORS)]
        _plot_time_colored_path(ax, robot.path, color)
        sx, sy = robot.path[0]
        ax.scatter(sx, sy, s=55, c=color, marker="s", edgecolors="white", linewidths=0.8, zorder=7, label="_nolegend_")
        ax.scatter(robot.x, robot.y, s=48, c=color, marker="^", edgecolors="white", linewidths=0.8, zorder=7)

    x0, y0, x1, y1 = court.playable_bounds
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor="#1A1A1A",
            linewidth=1.4,
            zorder=8,
        )
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.82)
    cbar.set_label("Ball-Landing Field (normalized)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    subtitle = (
        rf"$T_{{\mathrm{{clear}}}}$ = {metrics.time_to_clear_s:.1f}\,s \quad "
        rf"$D_{{\mathrm{{tot}}}}$ = {metrics.total_distance_m:.0f}\,m \quad "
        rf"Collected = {metrics.balls_collected}/{metrics.balls_total}"
    )
    ax.set_title(f"{_label(result.method)}  ($N={result.num_robots}$)\n{subtitle}", pad=8)
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.set_xlim(0, court.width_m)
    ax.set_ylim(0, court.height_m)
    ax.set_aspect("equal")
    ax.grid(False)

    if panel_label:
        ax.text(
            0.02,
            0.98,
            panel_label,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85),
        )

    fig.tight_layout()
    save_figure(fig, out_path, dpi=dpi)
    plt.close(fig)


def plot_method_comparison_panel(
    court: Court,
    blf: BallLandingField,
    results: list[tuple[SimResult, RunMetrics]],
    out_path: Path,
    dpi: int = 300,
    panel_labels: list[str] | None = None,
) -> None:
    """Side-by-side baseline vs proposed (typical paper Figure 1 layout)."""
    apply_pub_style()
    n = len(results)
    fig_w = 7.16 * n
    fig_h = 3.8 if court.preset == "football_7v7" else 3.2
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), squeeze=False)

    for col, (result, metrics) in enumerate(results):
        ax = axes[0, col]
        draw_court_surface(ax, court)
        im = _draw_blf_heatmap(ax, court, blf, alpha=0.68)

        for ball in result.balls:
            marker_kw = dict(s=16 if ball.collected else 24, zorder=5)
            if ball.collected:
                ax.scatter(ball.x, ball.y, c="#CCCCCC", edgecolors="none", alpha=0.5, **marker_kw)
            else:
                ax.scatter(ball.x, ball.y, c="#F4D03F", edgecolors="#222", linewidths=0.35, **marker_kw)

        for robot in result.robots:
            color = ROBOT_COLORS[robot.robot_id % len(ROBOT_COLORS)]
            _plot_time_colored_path(ax, robot.path, color)

        x0, y0, x1, y1 = court.playable_bounds
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#111", linewidth=1.2, zorder=8))

        label = panel_labels[col] if panel_labels and col < len(panel_labels) else chr(ord("a") + col)
        ax.set_title(
            f"({label}) {_label(result.method)}, $N={result.num_robots}$\n"
            rf"$T_{{\mathrm{{clear}}}}$={metrics.time_to_clear_s:.0f}\,s, "
            rf"$D={metrics.total_distance_m:.0f}\,m",
            fontsize=9,
        )
        ax.set_xlabel("$x$ (m)")
        if col == 0:
            ax.set_ylabel("$y$ (m)")
        ax.set_xlim(0, court.width_m)
        ax.set_ylim(0, court.height_m)
        ax.set_aspect("equal")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03, shrink=0.9)
    cbar.set_label("BLF intensity", fontsize=8)
    fig.tight_layout(w_pad=1.5)
    save_figure(fig, out_path, dpi=dpi)
    plt.close(fig)


def plot_comparison_pub(
    metrics_list: list[RunMetrics],
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Grouped bar chart with journal styling."""
    apply_pub_style()
    robot_counts = sorted({m.num_robots for m in metrics_list})
    methods = list(dict.fromkeys(m.method for m in metrics_list))

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))
    width = 0.75 / max(len(robot_counts), 1)
    x = np.arange(len(methods))

    palette = OKABE_ITO[: len(robot_counts)]
    for i, n in enumerate(robot_counts):
        subset = [m for m in metrics_list if m.num_robots == n]
        times, dists = [], []
        for method in methods:
            row = next((m for m in subset if m.method == method), None)
            times.append(row.time_to_clear_s if row else 0)
            dists.append(row.total_distance_m if row else 0)
        offset = (i - len(robot_counts) / 2 + 0.5) * width
        axes[0].bar(x + offset, times, width=width, color=palette[i], label=f"$N={n}$", edgecolor="white", linewidth=0.4)
        axes[1].bar(x + offset, dists, width=width, color=palette[i], label=f"$N={n}$", edgecolor="white", linewidth=0.4)

    short_labels = [_short_method_label(m) for m in methods]
    for ax, ylabel, title in zip(
        axes,
        [r"Clearance time $T_{\mathrm{clear}}$ (s)", r"Total distance $D_{\mathrm{tot}}$ (m)"],
        ["Clearance time", "Total robot distance"],
    ):
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=22, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    save_figure(fig, out_path, dpi=dpi)
    plt.close(fig)


def _short_method_label(method: str) -> str:
    short = {
        "single_greedy": "Single\nGreedy",
        "multi_greedy": "Multi\nGreedy",
        "voronoi_assignment": "Voronoi",
        "blf_informed_deployment": "BLF\nDeploy",
        "sportswarm_full": "SportSwarm\nFull",
        "gmm_swarm": "GMM\nSwarm",
        "uniform_cpp": "Uniform\nCPP",
        "blf_weighted_cpp": "BLF\nCPP",
    }
    return short.get(method, method.replace("_", "\n"))


def plot_scaling_pub(
    metrics_list: list[RunMetrics],
    method: str,
    out_path: Path,
    dpi: int = 300,
) -> None:
    subset = sorted([m for m in metrics_list if m.method == method], key=lambda m: m.num_robots)
    if not subset:
        return
    apply_pub_style()
    ns = [m.num_robots for m in subset]
    times = [m.time_to_clear_s for m in subset]

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(ns, times, "o-", color=OKABE_ITO[0], linewidth=1.8, markersize=7, markerfacecolor="white", markeredgewidth=1.4)
    ax.set_xlabel("Number of robots $N$")
    ax.set_ylabel(r"$T_{\mathrm{clear}}$ (s)")
    ax.set_title(f"Scaling: {_label(method)}")
    ax.set_xticks(ns)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, out_path, dpi=dpi)
    plt.close(fig)


def plot_pareto_pub(
    metrics_list: list[RunMetrics],
    method: str,
    out_path: Path,
    dpi: int = 300,
) -> None:
    subset = [m for m in metrics_list if m.method == method]
    if not subset:
        return
    apply_pub_style()
    front = pareto_front(subset)
    costs = [m.total_cost for m in subset]
    times = [m.time_to_clear_s for m in subset]
    f_costs = [m.total_cost for m in front]
    f_times = [m.time_to_clear_s for m in front]

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.scatter(costs, times, c="#CCCCCC", s=45, edgecolors="#888888", linewidths=0.4, label="All runs", zorder=2)
    ax.plot(f_costs, f_times, "o-", color=OKABE_ITO[3], linewidth=1.8, markersize=7, label="Pareto front", zorder=3)
    ax.set_xlabel(r"Total cost ($N \times c_{\mathrm{unit}}$)")
    ax.set_ylabel(r"$T_{\mathrm{clear}}$ (s)")
    ax.set_title(f"Pareto: {_label(method)}")
    ax.legend(frameon=False, fontsize=7)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, out_path, dpi=dpi)
    plt.close(fig)
