"""GMM-based macroscopic task allocation (SwarmPRM simplified)."""

from __future__ import annotations

import numpy as np
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
    component_centers: np.ndarray | None,
) -> Ball | None:
    if not available:
        return None
    if robot.gmm_component is not None and component_centers is not None:
        if robot.gmm_component < len(component_centers):
            cx, cy = component_centers[robot.gmm_component]
            owned = sorted(
                available,
                key=lambda b: abs(b.x - cx) + abs(b.y - cy),
            )
            # Prefer balls near component center
            pool = owned[: max(1, len(owned) // 2)]
            return min(
                pool,
                key=lambda b: abs(b.x - robot.x) + abs(b.y - robot.y),
            )
    return min(available, key=lambda b: abs(b.x - robot.x) + abs(b.y - robot.y))


def _robot_ball_dist(robot: Robot, ball: Ball) -> float:
    return abs(robot.x - ball.x) + abs(robot.y - ball.y)
