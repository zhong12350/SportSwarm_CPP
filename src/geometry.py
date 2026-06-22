"""Court geometry and entity definitions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.config_loader import CourtConfig


@dataclass
class Court:
    width_m: float
    height_m: float
    margin_m: float
    preset: str = "football_7v7"

    @property
    def playable_bounds(self) -> tuple[float, float, float, float]:
        m = self.margin_m
        return m, m, self.width_m - m, self.height_m - m

    def contains(self, x: float, y: float) -> bool:
        x_min, y_min, x_max, y_max = self.playable_bounds
        return x_min <= x <= x_max and y_min <= y <= y_max

    def clip(self, x: float, y: float) -> tuple[float, float]:
        x_min, y_min, x_max, y_max = self.playable_bounds
        return (
            float(np.clip(x, x_min, x_max)),
            float(np.clip(y, y_min, y_max)),
        )


@dataclass
class Ball:
    ball_id: int
    x: float
    y: float
    collected: bool = False
    spawn_time_s: float = 0.0


@dataclass
class Robot:
    robot_id: int
    x: float
    y: float
    target_ball_id: int | None = None
    target_waypoint: tuple[float, float] | None = None
    waypoint_queue: list[tuple[float, float]] = field(default_factory=list)
    gmm_component: int | None = None
    distance_traveled_m: float = 0.0
    path: list[tuple[float, float]] = field(default_factory=list)
    coverage_complete: bool = False

    def record_position(self) -> None:
        self.path.append((self.x, self.y))


@dataclass
class Player:
    player_id: int
    x: float
    y: float
    vx: float
    vy: float
    radius_m: float
    path: list[tuple[float, float]] = field(default_factory=list)

    def step(self, dt: float, court: Court) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt
        x_min, y_min, x_max, y_max = court.playable_bounds
        if self.x <= x_min or self.x >= x_max:
            self.vx *= -1
            self.x = float(np.clip(self.x, x_min, x_max))
        if self.y <= y_min or self.y >= y_max:
            self.vy *= -1
            self.y = float(np.clip(self.y, y_min, y_max))
        self.path.append((self.x, self.y))


def build_court(cfg: CourtConfig) -> Court:
    return Court(
        width_m=cfg.width_m,
        height_m=cfg.height_m,
        margin_m=cfg.margin_m,
        preset=cfg.preset,
    )


def default_robot_positions(court: Court, num_robots: int) -> list[tuple[float, float]]:
    x_min, y_min, x_max, _ = court.playable_bounds
    if num_robots == 1:
        xs = [(x_min + x_max) / 2]
    else:
        xs = np.linspace(x_min + 2, x_max - 2, num_robots)
    y = y_min
    return [(float(x), float(y)) for x in xs]


def manhattan(a: tuple[float, float], b: tuple[float, float]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))
