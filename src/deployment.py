"""Robot deployment optimization on BLF-weighted fields."""

from __future__ import annotations

import numpy as np

from src.config_loader import DeploymentConfig
from src.fields import BallLandingField, blf_peak_positions
from src.geometry import Court, default_robot_positions


def deploy_uniform(court: Court, num_robots: int) -> list[tuple[float, float]]:
    return default_robot_positions(court, num_robots)


def deploy_blf_peaks(
    blf: BallLandingField,
    court: Court,
    num_robots: int,
) -> list[tuple[float, float]]:
    return blf_peak_positions(blf, court, num_robots)


def deploy_lloyd(
    blf: BallLandingField,
    court: Court,
    num_robots: int,
    cfg: DeploymentConfig,
    max_iter: int = 30,
) -> list[tuple[float, float]]:
    """Lloyd relaxation on BLF-weighted samples for informed deployment."""
    if num_robots <= 0:
        return []
    if num_robots == 1:
        peaks = deploy_blf_peaks(blf, court, 1)
        return peaks

    rng = np.random.default_rng(42)
    n_samples = max(500, num_robots * 100)
    samples = _sample_blf_points(blf, court, rng, n_samples)

    x_min, y_min, x_max, y_max = court.playable_bounds
    init = deploy_blf_peaks(blf, court, num_robots)
    centroids = np.array(init, dtype=float)

    for _ in range(max_iter):
        dists = np.linalg.norm(samples[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = centroids.copy()
        for k in range(num_robots):
            mask = labels == k
            if mask.sum() == 0:
                continue
            new_centroids[k] = samples[mask].mean(axis=0)
            new_centroids[k, 0] = np.clip(new_centroids[k, 0], x_min, x_max)
            new_centroids[k, 1] = np.clip(new_centroids[k, 1], y_min, y_max)
        if np.allclose(new_centroids, centroids, atol=0.05):
            break
        centroids = new_centroids

    # Enforce minimum separation
    sep = cfg.min_separation_m
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            d = np.linalg.norm(centroids[i] - centroids[j])
            if d < sep and d > 1e-6:
                mid = (centroids[i] + centroids[j]) / 2
                offset = (centroids[i] - centroids[j]) / d * (sep / 2)
                centroids[i] = mid + offset
                centroids[j] = mid - offset

    return [(float(c[0]), float(c[1])) for c in centroids]


def _sample_blf_points(
    blf: BallLandingField,
    court: Court,
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    pts = np.array([blf.sample_point(rng, court) for _ in range(n)])
    return pts


def optimize_deployment(
    method: str,
    court: Court,
    blf: BallLandingField,
    num_robots: int,
    deploy_cfg: DeploymentConfig,
) -> list[tuple[float, float]]:
    """Return robot initial positions based on method and deployment config."""
    if method in ("blf_informed_deployment", "blf_weighted_cpp", "sportswarm_full"):
        if deploy_cfg.mode == "lloyd":
            return deploy_lloyd(blf, court, num_robots, deploy_cfg)
        return deploy_blf_peaks(blf, court, num_robots)

    if method == "gmm_swarm" and deploy_cfg.mode in ("lloyd", "blf_peaks"):
        if deploy_cfg.mode == "lloyd":
            return deploy_lloyd(blf, court, num_robots, deploy_cfg)
        return deploy_blf_peaks(blf, court, num_robots)

    return deploy_uniform(court, num_robots)
