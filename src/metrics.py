"""Simulation metrics, Pareto analysis, and CSV export."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.simulator import SimResult


@dataclass
class RunMetrics:
    method: str
    num_robots: int
    time_to_clear_s: float
    total_distance_m: float
    max_robot_distance_m: float
    balls_collected: int
    balls_total: int
    clearance_rate: float
    distance_per_ball: float
    total_cost: float
    cost_efficiency: float  # clearance / cost
    min_player_clearance_m: float | None
    safety_violation_count: int
    collision_count: int
    seed: int = 0
    court_preset: str = "football_7v7"

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_result(result: SimResult, seed: int = 0, court_preset: str = "football_7v7") -> RunMetrics:
    collected = result.balls_collected
    clearance = result.clearance_rate
    dist_per_ball = (
        result.total_distance_m / collected if collected > 0 else float("inf")
    )
    cost_eff = clearance / result.total_cost if result.total_cost > 0 else 0.0
    return RunMetrics(
        method=result.method,
        num_robots=result.num_robots,
        time_to_clear_s=result.time_to_clear_s,
        total_distance_m=result.total_distance_m,
        max_robot_distance_m=result.max_robot_distance_m,
        balls_collected=collected,
        balls_total=result.balls_total,
        clearance_rate=clearance,
        distance_per_ball=dist_per_ball,
        total_cost=result.total_cost,
        cost_efficiency=cost_eff,
        min_player_clearance_m=(
            result.min_player_clearance_m
            if result.min_player_clearance_m != float("inf")
            else None
        ),
        safety_violation_count=result.safety_violation_count,
        collision_count=result.collision_count,
        seed=seed,
        court_preset=court_preset,
    )


def evaluate_all(
    results: list[SimResult],
    seed: int = 0,
    court_preset: str = "football_7v7",
) -> list[RunMetrics]:
    return [evaluate_result(r, seed, court_preset) for r in results]


def pareto_front(metrics: list[RunMetrics]) -> list[RunMetrics]:
    """Non-dominated points minimizing (time, cost) for fixed method."""
    sorted_m = sorted(metrics, key=lambda m: (m.time_to_clear_s, m.total_cost))
    front: list[RunMetrics] = []
    best_cost = float("inf")
    for m in sorted_m:
        if m.total_cost < best_cost:
            front.append(m)
            best_cost = m.total_cost
    return front
