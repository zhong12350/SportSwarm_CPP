"""Ball-Landing Field (BLF) generation and sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config_loader import BLFConfig, BallsConfig
from src.geometry import Ball, Court


@dataclass
class BallLandingField:
    values: np.ndarray
    x_coords: np.ndarray
    y_coords: np.ndarray

    def intensity(self, x: float, y: float) -> float:
        ix = int(np.clip(np.searchsorted(self.x_coords, x), 0, len(self.x_coords) - 1))
        iy = int(np.clip(np.searchsorted(self.y_coords, y), 0, len(self.y_coords) - 1))
        return float(self.values[iy, ix])

    def sample_point(self, rng: np.random.Generator, court: Court) -> tuple[float, float]:
        flat = self.values.ravel()
        probs = flat / flat.sum() if flat.sum() > 0 else np.ones_like(flat) / flat.size
        idx = rng.choice(flat.size, p=probs)
        iy, ix = divmod(int(idx), self.values.shape[1])
        x = float(self.x_coords[ix])
        y = float(self.y_coords[iy])
        return court.clip(x, y)


def build_blf(court: Court, cfg: BLFConfig, cell_size: float = 1.0) -> BallLandingField:
    nx = max(1, int(np.ceil(court.width_m / cell_size)))
    ny = max(1, int(np.ceil(court.height_m / cell_size)))
    x_coords = np.arange(nx) * cell_size + cell_size / 2
    y_coords = np.arange(ny) * cell_size + cell_size / 2
    xx, yy = np.meshgrid(x_coords, y_coords)
    blf = np.zeros((ny, nx), dtype=float)

    for spec in cfg.gaussians:
        cx, cy = spec.center
        sx, sy = spec.sigma
        blf += spec.amplitude * np.exp(
            -0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2)
        )

    if cfg.normalize and blf.max() > 0:
        blf /= blf.max()

    return BallLandingField(values=blf, x_coords=x_coords, y_coords=y_coords)


def generate_balls(
    court: Court,
    blf: BallLandingField,
    balls_cfg: BallsConfig,
    rng: np.random.Generator,
) -> list[Ball]:
    balls: list[Ball] = []
    x_min, y_min, x_max, y_max = court.playable_bounds
    attempts = 0
    max_attempts = balls_cfg.count * 200

    while len(balls) < balls_cfg.count and attempts < max_attempts:
        attempts += 1
        if balls_cfg.mode == "uniform":
            x = rng.uniform(x_min, x_max)
            y = rng.uniform(y_min, y_max)
        else:
            x, y = blf.sample_point(rng, court)

        candidate = (x, y)
        too_close = any(
            np.hypot(x - b.x, y - b.y) < balls_cfg.min_separation_m for b in balls
        )
        if too_close:
            continue
        balls.append(Ball(ball_id=len(balls), x=x, y=y))

    return balls


def blf_peak_positions(blf: BallLandingField, court: Court, k: int) -> list[tuple[float, float]]:
    """Return top-k BLF peak locations for informed robot deployment."""
    flat = blf.values.ravel()
    top_idx = np.argsort(flat)[::-1][: max(k * 3, k)]
    peaks: list[tuple[float, float]] = []
    used: list[tuple[float, float]] = []

    for idx in top_idx:
        iy, ix = divmod(int(idx), blf.values.shape[1])
        x, y = float(blf.x_coords[ix]), float(blf.y_coords[iy])
        if not court.contains(x, y):
            continue
        if any(np.hypot(x - px, y - py) < 4.0 for px, py in used):
            continue
        peaks.append((x, y))
        used.append((x, y))
        if len(peaks) >= k:
            break

    while len(peaks) < k:
        x_min, y_min, x_max, y_max = court.playable_bounds
        peaks.append((x_min + len(peaks) * 3, y_min))
    return peaks[:k]
