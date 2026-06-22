"""Dynamic player obstacles and CVaR-aware local planning."""

from __future__ import annotations

import numpy as np

from src.config_loader import CVaRConfig, PlayerConfig
from src.geometry import Court, Player, Robot, euclidean


def spawn_players(court: Court, cfg: PlayerConfig, rng: np.random.Generator) -> list[Player]:
    if not cfg.enabled:
        return []
    x_min, y_min, x_max, y_max = court.playable_bounds
    players: list[Player] = []
    for i in range(cfg.count):
        x = rng.uniform(x_min + 5, x_max - 5)
        if cfg.sideline == "bottom":
            y = y_min + 0.5
            vy = 0.0
        elif cfg.sideline == "top":
            y = y_max - 0.5
            vy = 0.0
        else:
            y = y_min + 0.5 if i % 2 == 0 else y_max - 0.5
            vy = 0.0
        vx = cfg.speed_mps * (1 if rng.random() > 0.5 else -1)
        players.append(
            Player(
                player_id=i,
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                radius_m=cfg.radius_m,
            )
        )
    return players


def step_players(players: list[Player], court: Court, dt: float) -> None:
    for p in players:
        p.step(dt, court)


def collision_cost(
    pos: tuple[float, float],
    players: list[Player],
    robot_radius: float = 0.3,
) -> float:
    cost = 0.0
    for p in players:
        d = euclidean(pos, (p.x, p.y)) - p.radius_m - robot_radius
        if d < 0:
            cost += (-d) ** 2
    return cost


def cvar_best_direction(
    robot: Robot,
    target: tuple[float, float],
    players: list[Player],
    speed: float,
    dt: float,
    cfg: CVaRConfig,
    rng: np.random.Generator,
    robot_radius: float = 0.3,
) -> tuple[float, float]:
    """Choose velocity minimizing CVaR collision cost while progressing toward target."""
    dx = target[0] - robot.x
    dy = target[1] - robot.y
    dist = np.hypot(dx, dy)
    if dist < 1e-9:
        return 0.0, 0.0

    desired = np.array([dx / dist, dy / dist])
    step_len = min(speed * dt, dist)

    candidates = [desired]
    for angle in np.linspace(-np.pi / 3, np.pi / 3, 7):
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        candidates.append(rot @ desired)

    best_v = desired * step_len
    best_score = float("inf")

    for cand in candidates:
        cand = cand / (np.linalg.norm(cand) + 1e-9)
        end = (robot.x + cand[0] * step_len, robot.y + cand[1] * step_len)
        costs = []
        for _ in range(cfg.num_scenarios):
            scenario_players = _sample_player_scenarios(players, cfg, rng)
            costs.append(
                collision_cost(end, scenario_players, robot_radius)
                + 0.1 * euclidean(end, target)
            )
        costs_arr = np.array(costs)
        threshold = np.quantile(costs_arr, cfg.alpha)
        tail = costs_arr[costs_arr >= threshold]
        cvar = float(tail.mean()) if len(tail) > 0 else float(costs_arr.mean())
        if cvar < best_score:
            best_score = cvar
            best_v = cand * step_len

    return float(best_v[0]), float(best_v[1])


def _sample_player_scenarios(
    players: list[Player],
    cfg: CVaRConfig,
    rng: np.random.Generator,
) -> list[Player]:
    """Perturb player positions along their velocity for scenario sampling."""
    scenarios: list[Player] = []
    for p in players:
        noise_x = rng.normal(0, p.radius_m * 0.5)
        noise_y = rng.normal(0, p.radius_m * 0.3)
        dt_h = cfg.horizon_s * rng.uniform(0.3, 1.0)
        scenarios.append(
            Player(
                player_id=p.player_id,
                x=p.x + p.vx * dt_h + noise_x,
                y=p.y + p.vy * dt_h + noise_y,
                vx=p.vx,
                vy=p.vy,
                radius_m=p.radius_m,
            )
        )
    return scenarios
