"""GMM-based macroscopic task allocation (SwarmPRM simplified)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture

from src.config_loader import GMMConfig
from src.geometry import Ball, Robot


def fit_gmm_to_balls(
    balls: list[Ball],
    num_components: int,
    cfg: GMMConfig,
) -> GaussianMixture | None:
    active = [b for b in balls if not b.collected]
    if len(active) < cfg.min_samples_per_component:
        return None
    k = min(num_components, len(active), cfg.max_components)
    k = max(1, k)
    pts = np.array([[b.x, b.y] for b in active])
    gmm = GaussianMixture(
        n_components=k,
        covariance_type=cfg.covariance_type,
        random_state=42,
        reg_covar=1e-3,
    )
    gmm.fit(pts)
    return gmm


def assign_robot_components(
    robots: list[Robot],
    gmm: GaussianMixture,
) -> None:
    """Assign each robot to a GMM mode using robot-to-centroid distance.

    This is the macroscopic allocation step.  When there are more robots than
    modes, extra robots are placed on the heaviest modes to increase capacity in
    dense ball clusters.
    """
    if not robots:
        return

    centers = np.asarray(gmm.means_, dtype=float)
    weights = np.asarray(gmm.weights_, dtype=float)
    robot_xy = np.array([[r.x, r.y] for r in robots], dtype=float)
    dists = np.linalg.norm(robot_xy[:, None, :] - centers[None, :, :], axis=2)

    for robot in robots:
        robot.gmm_component = None

    if len(robots) <= len(centers):
        row_idx, col_idx = linear_sum_assignment(dists)
        for r_i, c_i in zip(row_idx, col_idx):
            robots[int(r_i)].gmm_component = int(c_i)
        return

    # Cover every component at least once, then duplicate high-mass modes.
    row_idx, col_idx = linear_sum_assignment(dists)
    assigned_robot_ids = set()
    for r_i, c_i in zip(row_idx, col_idx):
        robots[int(r_i)].gmm_component = int(c_i)
        assigned_robot_ids.add(int(r_i))

    ranked_components = list(np.argsort(weights)[::-1])
    for robot in robots:
        if robot.robot_id in assigned_robot_ids:
            continue
        best_component = min(
            ranked_components,
            key=lambda c: dists[robot.robot_id, int(c)] / max(weights[int(c)], 1e-6),
        )
        robot.gmm_component = int(best_component)


def assign_robots_to_components(
    robots: list[Robot],
    balls: list[Ball],
    gmm: GaussianMixture,
) -> dict[int, list[int]]:
    """Map robot_id → list of ball_ids in assigned GMM component."""
    active = [b for b in balls if not b.collected]
    if not active:
        return {r.robot_id: [] for r in robots}

    pts = np.array([[b.x, b.y] for b in active])
    labels = gmm.predict(pts)
    n_comp = gmm.n_components

    # Assign each robot to a component (round-robin if more robots than components)
    comp_to_robots: dict[int, list[int]] = {c: [] for c in range(n_comp)}
    for i, robot in enumerate(robots):
        comp = i % n_comp
        comp_to_robots[comp].append(robot.robot_id)
        robot.gmm_component = comp

    assignment: dict[int, list[int]] = {r.robot_id: [] for r in robots}
    for ball, label in zip(active, labels):
        robot_ids = comp_to_robots[int(label)]
        if not robot_ids:
            continue
        # Assign ball to nearest robot in that component
        best_rid = min(
            robot_ids,
            key=lambda rid: _robot_ball_dist(robots[rid], ball),
        )
        assignment[best_rid].append(ball.ball_id)

    return assignment


def pick_ball_in_component(
    robot: Robot,
    available: list[Ball],
    gmm: GaussianMixture | None,
) -> Ball | None:
    if not available:
        return None
    if robot.gmm_component is not None and gmm is not None:
        pts = np.array([[b.x, b.y] for b in available], dtype=float)
        labels = gmm.predict(pts)
        owned = [
            ball
            for ball, label in zip(available, labels)
            if int(label) == robot.gmm_component
        ]
        if owned:
            return min(
                owned,
                key=lambda b: abs(b.x - robot.x) + abs(b.y - robot.y),
            )
    return min(available, key=lambda b: abs(b.x - robot.x) + abs(b.y - robot.y))


def _robot_ball_dist(robot: Robot, ball: Ball) -> float:
    return abs(robot.x - ball.x) + abs(robot.y - ball.y)
