"""Ball event generation: static batches and semi-Markov streaming."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config_loader import BallsConfig
from src.fields import BallLandingField, generate_balls
from src.geometry import Ball, Court


@dataclass
class BallEventStream:
    """Time-varying ball arrival for semi-Markov session simulation."""

    court: Court
    blf: BallLandingField
    cfg: BallsConfig
    rng: np.random.Generator
    balls: list[Ball]
    next_ball_id: int
    t: float = 0.0

    @classmethod
    def create(
        cls,
        court: Court,
        blf: BallLandingField,
        cfg: BallsConfig,
        rng: np.random.Generator,
    ) -> BallEventStream:
        if cfg.mode == "semi_markov":
            return cls(
                court=court,
                blf=blf,
                cfg=cfg,
                rng=rng,
                balls=[],
                next_ball_id=0,
            )
        static = generate_balls(court, blf, cfg, rng)
        return cls(
            court=court,
            blf=blf,
            cfg=cfg,
            rng=rng,
            balls=static,
            next_ball_id=len(static),
        )

    @property
    def is_streaming(self) -> bool:
        return self.cfg.mode == "semi_markov"

    def _rate_per_s(self, t: float) -> float:
        sm = self.cfg.semi_markov
        if t <= sm.high_rate_period_s:
            return sm.high_rate_per_min / 60.0
        return sm.low_rate_per_min / 60.0

    def _spawn_ball(self, t: float) -> Ball | None:
        x_min, y_min, x_max, y_max = self.court.playable_bounds
        for _ in range(50):
            if self.cfg.mode == "uniform" or self.rng.random() < 0.2:
                x = self.rng.uniform(x_min, x_max)
                y = self.rng.uniform(y_min, y_max)
            else:
                x, y = self.blf.sample_point(self.rng, self.court)
            too_close = any(
                np.hypot(x - b.x, y - b.y) < self.cfg.min_separation_m
                for b in self.balls
                if not b.collected
            )
            if too_close:
                continue
            ball = Ball(
                ball_id=self.next_ball_id,
                x=x,
                y=y,
                spawn_time_s=t,
            )
            self.next_ball_id += 1
            return ball
        return None

    def step(self, dt: float) -> list[Ball]:
        """Advance time and possibly spawn new balls."""
        if not self.is_streaming:
            return []
        self.t += dt
        sm = self.cfg.semi_markov
        if self.t > sm.session_duration_s:
            return []
        active = sum(1 for b in self.balls if not b.collected)
        if active + len(self.balls) >= sm.max_balls:
            return []

        rate = self._rate_per_s(self.t)
        expected = rate * dt
        n_spawn = int(self.rng.poisson(expected))
        spawned: list[Ball] = []
        for _ in range(n_spawn):
            ball = self._spawn_ball(self.t)
            if ball is None:
                break
            self.balls.append(ball)
            spawned.append(ball)
        return spawned

    def initial_balls(self) -> list[Ball]:
        return list(self.balls)

    def all_collected(self) -> bool:
        return bool(self.balls) and all(b.collected for b in self.balls)

    def clearance_rate(self) -> float:
        if not self.balls:
            return 1.0
        return sum(1 for b in self.balls if b.collected) / len(self.balls)
