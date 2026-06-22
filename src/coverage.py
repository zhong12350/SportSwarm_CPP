"""Boustrophedon coverage path generation for CPP baselines."""

from __future__ import annotations

import numpy as np

from src.fields import BallLandingField
from src.geometry import Court


def generate_boustrophedon_waypoints(
    court: Court,
    robot_id: int,
    num_robots: int,
    cell_size_m: float = 4.0,
    blf: BallLandingField | None = None,
    blf_weighted: bool = False,
) -> list[tuple[float, float]]:
    """Visit grid cell centers in boustrophedon order (JTEC-style CPP baseline)."""
    x_min, y_min, x_max, y_max = court.playable_bounds
    xs = np.arange(x_min + cell_size_m / 2, x_max, cell_size_m)
    ys = np.arange(y_min + cell_size_m / 2, y_max, cell_size_m)
    if len(xs) == 0:
        xs = np.array([(x_min + x_max) / 2])
    if len(ys) == 0:
        ys = np.array([(y_min + y_max) / 2])

    cells: list[tuple[float, float, float]] = []
    for j, y in enumerate(ys):
        row = [(float(x), float(y), float(j)) for x in xs]
        if j % 2 == 1:
            row = list(reversed(row))
        cells.extend(row)

    if blf_weighted and blf is not None:
        cells.sort(
            key=lambda c: blf.intensity(c[0], c[1]),
            reverse=True,
        )

    my_cells = [c for i, c in enumerate(cells) if i % num_robots == robot_id]
    return [(c[0], c[1]) for c in my_cells]


def init_coverage_paths(
    court: Court,
    num_robots: int,
    blf: BallLandingField | None = None,
    blf_weighted: bool = False,
    cell_size_m: float = 4.0,
) -> list[list[tuple[float, float]]]:
    return [
        generate_boustrophedon_waypoints(
            court, i, num_robots, cell_size_m, blf, blf_weighted
        )
        for i in range(num_robots)
    ]
