"""Task assignment strategies for multi-robot ball retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture

from src.coverage import init_coverage_paths
from src.fields import BallLandingField
from src.geometry import Ball, Court, Robot, manhattan
from src.gmm import assign_robot_components, pick_ball_in_component


class BaseAssigner(ABC):
    @property
    @abstractmethod
    def method_name(self) -> str:
        ...

    def on_init(
        self,
        robots: list[Robot],
        court: Court,
        blf: BallLandingField | None = None,
        num_robots: int = 1,
    ) -> None:
        """Optional setup hook (coverage paths, etc.)."""

    def on_realloc(
        self,
        robots: list[Robot],
        balls: list[Ball],
        t: float,
        gmm: GaussianMixture | None = None,
    ) -> None:
        """Optional periodic reassignment hook."""

    def assign_targets(
        self,
        robots: list[Robot],
        available: list[Ball],
        claimed: set[int],
        anchors: list[tuple[float, float]],
    ) -> bool:
        """Optional batch assignment hook for centralized baselines."""
        return False

    @abstractmethod
    def pick_ball(
        self,
        robot: Robot,
        available: list[Ball],
        anchors: list[tuple[float, float]],
    ) -> Ball | None:
        ...


class SingleGreedyAssigner(BaseAssigner):
    @property
    def method_name(self) -> str:
        return "single_greedy"

    def pick_ball(
        self,
        robot: Robot,
        available: list[Ball],
        anchors: list[tuple[float, float]],
    ) -> Ball | None:
        if not available:
            return None
        return min(available, key=lambda b: manhattan((robot.x, robot.y), (b.x, b.y)))


class MultiGreedyAssigner(BaseAssigner):
    @property
    def method_name(self) -> str:
        return "multi_greedy"

    def pick_ball(
        self,
        robot: Robot,
        available: list[Ball],
        anchors: list[tuple[float, float]],
    ) -> Ball | None:
        if not available:
            return None
        return min(available, key=lambda b: manhattan((robot.x, robot.y), (b.x, b.y)))


class VoronoiAssigner(BaseAssigner):
    @property
    def method_name(self) -> str:
        return "voronoi_assignment"

    def pick_ball(
        self,
        robot: Robot,
        available: list[Ball],
        anchors: list[tuple[float, float]],
    ) -> Ball | None:
        if not available:
            return None
        owned = [
            b
            for b in available
            if _nearest_anchor_id(b, anchors) == robot.robot_id
        ]
        pool = owned if owned else available
        return min(pool, key=lambda b: manhattan((robot.x, robot.y), (b.x, b.y)))


class HungarianAssigner(BaseAssigner):
    @property
    def method_name(self) -> str:
        return "hungarian_assignment"

    def assign_targets(
        self,
        robots: list[Robot],
        available: list[Ball],
        claimed: set[int],
        anchors: list[tuple[float, float]],
    ) -> bool:
        idle = [
            robot
            for robot in robots
            if robot.target_ball_id is None
            and not (robot.waypoint_queue and not robot.coverage_complete)
        ]
        if not idle or not available:
            return False

        cost = np.array(
            [
                [manhattan((robot.x, robot.y), (ball.x, ball.y)) for ball in available]
                for robot in idle
            ],
            dtype=float,
        )
        row_idx, col_idx = linear_sum_assignment(cost)
        for r_i, b_i in zip(row_idx, col_idx):
            robot = idle[int(r_i)]
            ball = available[int(b_i)]
            robot.target_ball_id = ball.ball_id
            claimed.add(ball.ball_id)
        return True

    def pick_ball(
        self,
        robot: Robot,
        available: list[Ball],
        anchors: list[tuple[float, float]],
    ) -> Ball | None:
        if not available:
            return None
        return min(available, key=lambda b: manhattan((robot.x, robot.y), (b.x, b.y)))


class BLFInformedAssigner(MultiGreedyAssigner):
    @property
    def method_name(self) -> str:
        return "blf_informed_deployment"


class UniformCPPAssigner(BaseAssigner):
    def __init__(self) -> None:
        self._blf: BallLandingField | None = None

    @property
    def method_name(self) -> str:
        return "uniform_cpp"

    def on_init(
        self,
        robots: list[Robot],
        court: Court,
        blf: BallLandingField | None = None,
        num_robots: int = 1,
    ) -> None:
        self._blf = blf
        paths = init_coverage_paths(court, len(robots), blf_weighted=False)
        for robot, wps in zip(robots, paths):
            robot.waypoint_queue = list(wps)
            robot.coverage_complete = False

    def pick_ball(
        self,
        robot: Robot,
        available: list[Ball],
        anchors: list[tuple[float, float]],
    ) -> Ball | None:
        if not available:
            return None
        if robot.waypoint_queue:
            wx, wy = robot.waypoint_queue[0]
            return min(
                available,
                key=lambda b: (
                    0.65 * manhattan((robot.x, robot.y), (b.x, b.y))
                    + 0.35 * manhattan((wx, wy), (b.x, b.y))
                ),
            )
        return min(available, key=lambda b: manhattan((robot.x, robot.y), (b.x, b.y)))


class BLFWeightedCPPAssigner(BaseAssigner):
    def __init__(self) -> None:
        self._blf: BallLandingField | None = None

    @property
    def method_name(self) -> str:
        return "blf_weighted_cpp"

    def on_init(
        self,
        robots: list[Robot],
        court: Court,
        blf: BallLandingField | None = None,
        num_robots: int = 1,
    ) -> None:
        self._blf = blf
        paths = init_coverage_paths(
            court, len(robots), blf=blf, blf_weighted=True
        )
        for robot, wps in zip(robots, paths):
            robot.waypoint_queue = list(wps)
            robot.coverage_complete = False

    def pick_ball(
        self,
        robot: Robot,
        available: list[Ball],
        anchors: list[tuple[float, float]],
    ) -> Ball | None:
        if not available:
            return None
        if self._blf is None:
            return min(available, key=lambda b: manhattan((robot.x, robot.y), (b.x, b.y)))

        def score(ball: Ball) -> float:
            travel = manhattan((robot.x, robot.y), (ball.x, ball.y))
            waypoint = 0.0
            if robot.waypoint_queue:
                waypoint = manhattan(robot.waypoint_queue[0], (ball.x, ball.y))
            intensity = self._blf.intensity(ball.x, ball.y)
            return 0.9 * travel + 0.1 * waypoint - 3.0 * intensity

        return min(available, key=score)


class GMMSwarmAssigner(BaseAssigner):
    def __init__(self) -> None:
        self._gmm: GaussianMixture | None = None

    @property
    def method_name(self) -> str:
        return "gmm_swarm"

    def on_realloc(
        self,
        robots: list[Robot],
        balls: list[Ball],
        t: float,
        gmm: GaussianMixture | None = None,
    ) -> None:
        self._gmm = gmm
        if gmm is not None:
            assign_robot_components(robots, gmm)

    def pick_ball(
        self,
        robot: Robot,
        available: list[Ball],
        anchors: list[tuple[float, float]],
    ) -> Ball | None:
        return pick_ball_in_component(robot, available, self._gmm)


class SportSwarmFullAssigner(GMMSwarmAssigner):
    @property
    def method_name(self) -> str:
        return "sportswarm_full"


def _nearest_anchor_id(ball: Ball, anchors: list[tuple[float, float]]) -> int:
    def dist(a: tuple[float, float]) -> float:
        return abs(a[0] - ball.x) + abs(a[1] - ball.y)

    return int(min(range(len(anchors)), key=lambda i: dist(anchors[i])))


ASSIGNER_REGISTRY: dict[str, type[BaseAssigner]] = {
    "single_greedy": SingleGreedyAssigner,
    "gb_greedy": SingleGreedyAssigner,
    "multi_greedy": MultiGreedyAssigner,
    "hungarian_assignment": HungarianAssigner,
    "voronoi_assignment": VoronoiAssigner,
    "mdcpp_voronoi": VoronoiAssigner,
    "blf_informed_deployment": BLFInformedAssigner,
    "uniform_cpp": UniformCPPAssigner,
    "blf_weighted_cpp": BLFWeightedCPPAssigner,
    "gmm_swarm": GMMSwarmAssigner,
    "sportswarm_full": SportSwarmFullAssigner,
}


def create_assigner(method: str) -> BaseAssigner:
    if method not in ASSIGNER_REGISTRY:
        raise ValueError(f"Unknown method '{method}'. Available: {list(ASSIGNER_REGISTRY)}")
    return ASSIGNER_REGISTRY[method]()
