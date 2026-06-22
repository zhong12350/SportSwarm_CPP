"""Task assignment strategies for multi-robot ball retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod

from sklearn.mixture import GaussianMixture

from src.coverage import init_coverage_paths
from src.fields import BallLandingField
from src.geometry import Ball, Court, Robot, manhattan
from src.gmm import fit_gmm_to_balls, pick_ball_in_component


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


class BLFInformedAssigner(MultiGreedyAssigner):
    @property
    def method_name(self) -> str:
        return "blf_informed_deployment"


class UniformCPPAssigner(BaseAssigner):
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
        if not robot.coverage_complete and not robot.waypoint_queue:
            return None
        return min(available, key=lambda b: manhattan((robot.x, robot.y), (b.x, b.y)))


class BLFWeightedCPPAssigner(BaseAssigner):
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
        if not robot.coverage_complete and not robot.waypoint_queue:
            return None
        return min(available, key=lambda b: manhattan((robot.x, robot.y), (b.x, b.y)))


class GMMSwarmAssigner(BaseAssigner):
    def __init__(self) -> None:
        self._gmm: GaussianMixture | None = None
        self._centers: list[tuple[float, float]] | None = None

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
            self._centers = [
                (float(c[0]), float(c[1])) for c in gmm.means_
            ]

    def pick_ball(
        self,
        robot: Robot,
        available: list[Ball],
        anchors: list[tuple[float, float]],
    ) -> Ball | None:
        centers = (
            [[c[0], c[1]] for c in self._centers]
            if self._centers
            else None
        )
        import numpy as np

        arr = np.array(centers) if centers else None
        return pick_ball_in_component(robot, available, arr)


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
