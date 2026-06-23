"""Discrete-time ball retrieval simulation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from src.assigners import BaseAssigner, create_assigner
from src.ball_events import BallEventStream
from src.config_loader import AppConfig, CVaRConfig, RobotsConfig, SimulationConfig
from src.cvar_planner import cvar_best_direction, spawn_players, step_players
from src.deployment import optimize_deployment
from src.fields import BallLandingField
from src.geometry import Ball, Court, Player, Robot, euclidean
from src.gmm import fit_gmm_to_balls


@dataclass(frozen=True)
class SimResult:
    method: str
    num_robots: int
    time_to_clear_s: float
    total_distance_m: float
    max_robot_distance_m: float
    balls_collected: int
    balls_total: int
    clearance_rate: float
    total_cost: float
    robots: list[Robot]
    balls: list[Ball]
    players: list[Player] = field(default_factory=list)


def deploy_robots(
    method: str,
    court: Court,
    num_robots: int,
    blf: BallLandingField,
    config: AppConfig,
) -> tuple[list[Robot], list[tuple[float, float]]]:
    positions = optimize_deployment(
        method, court, blf, num_robots, config.deployment
    )
    robots = [
        Robot(robot_id=i, x=pos[0], y=pos[1], path=[pos])
        for i, pos in enumerate(positions)
    ]
    return robots, positions


def run_simulation(
    method: str,
    court: Court,
    blf: BallLandingField,
    ball_stream: BallEventStream,
    robots_cfg: RobotsConfig,
    sim_cfg: SimulationConfig,
    cvar_cfg: CVaRConfig,
    config: AppConfig,
    num_robots: int,
    rng: np.random.Generator | None = None,
) -> SimResult:
    rng = rng or np.random.default_rng(0)
    robots, anchors = deploy_robots(method, court, num_robots, blf, config)
    if method in ("single_greedy", "gb_greedy"):
        robots = robots[:1]
        anchors = anchors[:1]

    balls_sim = deepcopy(ball_stream.balls)
    assigner = create_assigner(method)
    assigner.on_init(robots, court, blf, len(robots))

    players = spawn_players(court, config.players, rng) if (
        config.players.enabled and method == "sportswarm_full"
    ) else []
    use_cvar = cvar_cfg.enabled and method == "sportswarm_full" and bool(players)
    use_gmm = method in ("gmm_swarm", "sportswarm_full")

    claimed: set[int] = set()
    t = 0.0
    dt = sim_cfg.dt_s
    speed = robots_cfg.speed_mps
    pickup_r = robots_cfg.pickup_radius_m
    last_realloc = -sim_cfg.gmm_realloc_interval_s
    gmm = None

    def _is_done() -> bool:
        if ball_stream.is_streaming:
            if t >= ball_stream.cfg.semi_markov.session_duration_s:
                return all(b.collected for b in balls_sim)
            return False
        return all(b.collected for b in balls_sim)

    while t <= sim_cfg.max_time_s:
        if use_gmm and t - last_realloc >= sim_cfg.gmm_realloc_interval_s:
            gmm = fit_gmm_to_balls(balls_sim, len(robots), config.gmm)
            assigner.on_realloc(robots, balls_sim, t, gmm)
            if gmm is not None:
                _drop_cross_component_targets(robots, balls_sim, gmm, claimed)
                claimed.clear()
            last_realloc = t

        spawned = ball_stream.step(dt)
        balls_sim.extend(spawned)

        _assign_with_claims(assigner, robots, balls_sim, anchors, claimed)
        if _is_done():
            break

        step_players(players, court, dt)
        any_moved = False

        for robot in robots:
            moved = _move_robot(
                robot,
                balls_sim,
                speed,
                dt,
                pickup_r,
                use_cvar,
                players,
                cvar_cfg,
                rng,
            )
            _collect_nearby_balls(robot, balls_sim, pickup_r, claimed)
            any_moved = any_moved or moved

        if not any_moved and not _is_done():
            for robot in robots:
                robot.target_ball_id = None
                robot.target_waypoint = None
            claimed.clear()

        t += dt

    collected = sum(1 for b in balls_sim if b.collected)
    total = len(balls_sim)
    clearance = collected / total if total else 1.0
    total_dist = sum(r.distance_traveled_m for r in robots)
    max_dist = max((r.distance_traveled_m for r in robots), default=0.0)
    cost = len(robots) * robots_cfg.cost_per_robot

    cleared = (
        clearance >= sim_cfg.coverage_threshold
        if not ball_stream.is_streaming
        else _is_done()
    )

    return SimResult(
        method=method,
        num_robots=len(robots),
        time_to_clear_s=t if cleared else sim_cfg.max_time_s,
        total_distance_m=total_dist,
        max_robot_distance_m=max_dist,
        balls_collected=collected,
        balls_total=total,
        clearance_rate=clearance,
        total_cost=cost,
        robots=robots,
        balls=balls_sim,
        players=players,
    )


def _collect_nearby_balls(
    robot: Robot,
    balls: list[Ball],
    pickup_r: float,
    claimed: set[int],
) -> None:
    for ball in balls:
        if ball.collected:
            continue
        if euclidean((robot.x, robot.y), (ball.x, ball.y)) <= pickup_r + 1e-6:
            ball.collected = True
            claimed.discard(ball.ball_id)
            if robot.target_ball_id == ball.ball_id:
                robot.target_ball_id = None


def _drop_cross_component_targets(
    robots: list[Robot],
    balls: list[Ball],
    gmm,
    claimed: set[int],
) -> None:
    """Keep useful targets, but release targets that moved outside robot's macro mode."""
    for robot in robots:
        if robot.target_ball_id is None or robot.gmm_component is None:
            continue
        ball = balls[robot.target_ball_id]
        if ball.collected:
            robot.target_ball_id = None
            claimed.discard(ball.ball_id)
            continue
        label = int(gmm.predict(np.array([[ball.x, ball.y]], dtype=float))[0])
        if label != robot.gmm_component:
            claimed.discard(ball.ball_id)
            robot.target_ball_id = None


def _move_robot(
    robot: Robot,
    balls: list[Ball],
    speed: float,
    dt: float,
    pickup_r: float,
    use_cvar: bool,
    players: list[Player],
    cvar_cfg: CVaRConfig,
    rng: np.random.Generator,
) -> bool:
    # Ball target following (greedy / opportunistic during CPP)
    if robot.target_ball_id is not None:
        ball = balls[robot.target_ball_id]
        if ball.collected:
            robot.target_ball_id = None
            return False

        dist = euclidean((robot.x, robot.y), (ball.x, ball.y))
        if dist <= pickup_r + 1e-6:
            ball.collected = True
            robot.target_ball_id = None
            return False

        return _step_toward(
            robot, (ball.x, ball.y), speed, dt, use_cvar, players, cvar_cfg, rng
        )

    # Coverage waypoint following
    if robot.waypoint_queue and not robot.coverage_complete:
        wp = robot.waypoint_queue[0]
        dist_wp = euclidean((robot.x, robot.y), wp)
        if dist_wp <= pickup_r:
            robot.waypoint_queue.pop(0)
            if not robot.waypoint_queue:
                robot.coverage_complete = True
        else:
            target = wp
            return _step_toward(
                robot, target, speed, dt, use_cvar, players, cvar_cfg, rng
            )

    return False


def _step_toward(
    robot: Robot,
    target: tuple[float, float],
    speed: float,
    dt: float,
    use_cvar: bool,
    players: list[Player],
    cvar_cfg: CVaRConfig,
    rng: np.random.Generator,
) -> bool:
    if use_cvar and players:
        dx, dy = cvar_best_direction(
            robot, target, players, speed, dt, cvar_cfg, rng
        )
    else:
        tx, ty = target
        dx_raw = tx - robot.x
        dy_raw = ty - robot.y
        dist = np.hypot(dx_raw, dy_raw)
        if dist < 1e-9:
            return False
        step = min(speed * dt, dist)
        dx = step * dx_raw / dist
        dy = step * dy_raw / dist

    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return False

    robot.x += dx
    robot.y += dy
    step_len = np.hypot(dx, dy)
    robot.distance_traveled_m += step_len
    robot.record_position()
    return True


def _assign_with_claims(
    assigner: BaseAssigner,
    robots: list[Robot],
    balls: list[Ball],
    anchors: list[tuple[float, float]],
    claimed: set[int],
) -> None:
    for robot in robots:
        if robot.target_ball_id is not None:
            claimed.add(robot.target_ball_id)

    available = [b for b in balls if not b.collected and b.ball_id not in claimed]

    for robot in robots:
        if robot.target_ball_id is not None:
            continue
        # During CPP: opportunistically grab nearby balls before next waypoint
        if robot.waypoint_queue and not robot.coverage_complete:
            ball = assigner.pick_ball(robot, available, anchors)
            if ball is not None:
                robot.target_ball_id = ball.ball_id
                claimed.add(ball.ball_id)
                available.remove(ball)
            continue
        ball = assigner.pick_ball(robot, available, anchors)
        if ball is None:
            continue
        robot.target_ball_id = ball.ball_id
        claimed.add(ball.ball_id)
        available.remove(ball)


def run_all_experiments(
    config: AppConfig,
    court: Court,
    blf: BallLandingField,
    ball_stream: BallEventStream,
    rng: np.random.Generator,
) -> list[SimResult]:
    results: list[SimResult] = []
    for method in config.methods:
        counts = [1] if method in ("single_greedy", "gb_greedy") else config.robots.counts
        for n in counts:
            stream_copy = BallEventStream(
                court=court,
                blf=blf,
                cfg=ball_stream.cfg,
                rng=np.random.default_rng(config.seed),
                balls=deepcopy(ball_stream.initial_balls()),
                next_ball_id=ball_stream.next_ball_id,
            )
            results.append(
                run_simulation(
                    method=method,
                    court=court,
                    blf=blf,
                    ball_stream=stream_copy,
                    robots_cfg=config.robots,
                    sim_cfg=config.simulation,
                    cvar_cfg=config.cvar,
                    config=config,
                    num_robots=n,
                    rng=np.random.default_rng(config.seed + n),
                )
            )
    return results
