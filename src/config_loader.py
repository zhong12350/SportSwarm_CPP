"""Load and validate YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

import yaml

COURT_PRESETS: dict[str, dict[str, float]] = {
    "football_7v7": {"width_m": 60.0, "height_m": 40.0, "margin_m": 2.0},
    "tennis_court": {"width_m": 23.77, "height_m": 10.97, "margin_m": 0.5},
}


@dataclass
class CourtConfig:
    preset: str = "football_7v7"
    width_m: float = 60.0
    height_m: float = 40.0
    cell_size_m: float = 1.0
    margin_m: float = 2.0


@dataclass
class GaussianSpec:
    center: tuple[float, float]
    sigma: tuple[float, float]
    amplitude: float


@dataclass
class BLFConfig:
    source: str = "gaussian"  # gaussian | statsbomb | hybrid
    gaussians: list[GaussianSpec] = dc_field(default_factory=list)
    normalize: bool = True
    statsbomb_dir: Path = Path("data/statsbomb_sample")
    tactical_style: str = "all"  # all | offensive | defensive
    grid_cell_m: float = 1.0


@dataclass
class BallsConfig:
    count: int = 40
    mode: str = "blf"  # uniform | blf | semi_markov
    min_separation_m: float = 1.5
    semi_markov: "SemiMarkovConfig" = dc_field(default_factory=lambda: SemiMarkovConfig())


@dataclass
class SemiMarkovConfig:
    session_duration_s: float = 1800.0
    high_rate_period_s: float = 1200.0
    high_rate_per_min: float = 2.5
    low_rate_per_min: float = 0.8
    max_balls: int = 80


@dataclass
class RobotsConfig:
    speed_mps: float = 0.8
    pickup_radius_m: float = 0.4
    counts: list[int] = dc_field(default_factory=lambda: [1, 2, 4, 8])
    cost_per_robot: float = 1.0


@dataclass
class SimulationConfig:
    dt_s: float = 0.1
    max_time_s: float = 600.0
    coverage_threshold: float = 0.95
    gmm_realloc_interval_s: float = 5.0


@dataclass
class DeploymentConfig:
    mode: str = "uniform"  # uniform | blf_peaks | lloyd
    min_separation_m: float = 4.0


@dataclass
class GMMConfig:
    max_components: int = 8
    min_samples_per_component: int = 2
    covariance_type: str = "full"


@dataclass
class PlayerConfig:
    enabled: bool = False
    count: int = 3
    speed_mps: float = 1.2
    radius_m: float = 0.6
    sideline: str = "bottom"  # bottom | top | both


@dataclass
class CVaRConfig:
    enabled: bool = False
    alpha: float = 0.9
    num_scenarios: int = 16
    horizon_s: float = 2.0
    collision_penalty: float = 100.0


@dataclass
class OutputConfig:
    figures_dir: Path = Path("outputs/figures")
    results_dir: Path = Path("outputs/results")
    dpi: int = 150


@dataclass
class ExperimentConfig:
    batch_seeds: list[int] = dc_field(default_factory=lambda: list(range(42, 52)))
    ablation_groups: list[str] = dc_field(default_factory=list)


@dataclass
class AppConfig:
    seed: int = 42
    court: CourtConfig = dc_field(default_factory=CourtConfig)
    blf: BLFConfig = dc_field(default_factory=BLFConfig)
    balls: BallsConfig = dc_field(default_factory=BallsConfig)
    robots: RobotsConfig = dc_field(default_factory=RobotsConfig)
    simulation: SimulationConfig = dc_field(default_factory=SimulationConfig)
    deployment: DeploymentConfig = dc_field(default_factory=DeploymentConfig)
    gmm: GMMConfig = dc_field(default_factory=GMMConfig)
    players: PlayerConfig = dc_field(default_factory=PlayerConfig)
    cvar: CVaRConfig = dc_field(default_factory=CVaRConfig)
    methods: list[str] = dc_field(default_factory=list)
    output: OutputConfig = dc_field(default_factory=OutputConfig)
    experiment: ExperimentConfig = dc_field(default_factory=ExperimentConfig)


def _parse_gaussian(raw: dict[str, Any]) -> GaussianSpec:
    return GaussianSpec(
        center=(float(raw["center"][0]), float(raw["center"][1])),
        sigma=(float(raw["sigma"][0]), float(raw["sigma"][1])),
        amplitude=float(raw["amplitude"]),
    )


def _apply_court_preset(court_raw: dict[str, Any]) -> CourtConfig:
    preset = str(court_raw.get("preset", "football_7v7"))
    base = dict(COURT_PRESETS.get(preset, COURT_PRESETS["football_7v7"]))
    for key in ("width_m", "height_m", "margin_m", "cell_size_m"):
        if key in court_raw:
            base[key] = float(court_raw[key])
    return CourtConfig(
        preset=preset,
        width_m=float(base["width_m"]),
        height_m=float(base["height_m"]),
        cell_size_m=float(base.get("cell_size_m", 1.0)),
        margin_m=float(base["margin_m"]),
    )


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    court_raw = raw.get("court", {})
    blf_raw = raw.get("blf", {})
    balls_raw = raw.get("balls", {})
    robots_raw = raw.get("robots", {})
    sim_raw = raw.get("simulation", {})
    output_raw = raw.get("output", {})
    deploy_raw = raw.get("deployment", {})
    gmm_raw = raw.get("gmm", {})
    players_raw = raw.get("players", {})
    cvar_raw = raw.get("cvar", {})
    exp_raw = raw.get("experiment", {})
    sm_raw = balls_raw.get("semi_markov", {})

    return AppConfig(
        seed=int(raw.get("seed", 42)),
        court=_apply_court_preset(court_raw),
        blf=BLFConfig(
            source=str(blf_raw.get("source", "gaussian")),
            gaussians=[_parse_gaussian(g) for g in blf_raw.get("gaussians", [])],
            normalize=bool(blf_raw.get("normalize", True)),
            statsbomb_dir=Path(blf_raw.get("statsbomb_dir", "data/statsbomb_sample")),
            tactical_style=str(blf_raw.get("tactical_style", "all")),
            grid_cell_m=float(blf_raw.get("grid_cell_m", 1.0)),
        ),
        balls=BallsConfig(
            count=int(balls_raw.get("count", 40)),
            mode=str(balls_raw.get("mode", "blf")),
            min_separation_m=float(balls_raw.get("min_separation_m", 1.5)),
            semi_markov=SemiMarkovConfig(
                session_duration_s=float(sm_raw.get("session_duration_s", 1800.0)),
                high_rate_period_s=float(sm_raw.get("high_rate_period_s", 1200.0)),
                high_rate_per_min=float(sm_raw.get("high_rate_per_min", 2.5)),
                low_rate_per_min=float(sm_raw.get("low_rate_per_min", 0.8)),
                max_balls=int(sm_raw.get("max_balls", 80)),
            ),
        ),
        robots=RobotsConfig(
            speed_mps=float(robots_raw.get("speed_mps", 0.8)),
            pickup_radius_m=float(robots_raw.get("pickup_radius_m", 0.4)),
            counts=[int(n) for n in robots_raw.get("counts", [1, 2, 4, 8])],
            cost_per_robot=float(robots_raw.get("cost_per_robot", 1.0)),
        ),
        simulation=SimulationConfig(
            dt_s=float(sim_raw.get("dt_s", 0.1)),
            max_time_s=float(sim_raw.get("max_time_s", 600.0)),
            coverage_threshold=float(sim_raw.get("coverage_threshold", 0.95)),
            gmm_realloc_interval_s=float(sim_raw.get("gmm_realloc_interval_s", 5.0)),
        ),
        deployment=DeploymentConfig(
            mode=str(deploy_raw.get("mode", "uniform")),
            min_separation_m=float(deploy_raw.get("min_separation_m", 4.0)),
        ),
        gmm=GMMConfig(
            max_components=int(gmm_raw.get("max_components", 8)),
            min_samples_per_component=int(gmm_raw.get("min_samples_per_component", 2)),
            covariance_type=str(gmm_raw.get("covariance_type", "full")),
        ),
        players=PlayerConfig(
            enabled=bool(players_raw.get("enabled", False)),
            count=int(players_raw.get("count", 3)),
            speed_mps=float(players_raw.get("speed_mps", 1.2)),
            radius_m=float(players_raw.get("radius_m", 0.6)),
            sideline=str(players_raw.get("sideline", "bottom")),
        ),
        cvar=CVaRConfig(
            enabled=bool(cvar_raw.get("enabled", False)),
            alpha=float(cvar_raw.get("alpha", 0.9)),
            num_scenarios=int(cvar_raw.get("num_scenarios", 16)),
            horizon_s=float(cvar_raw.get("horizon_s", 2.0)),
            collision_penalty=float(cvar_raw.get("collision_penalty", 100.0)),
        ),
        methods=list(raw.get("methods", [])),
        output=OutputConfig(
            figures_dir=Path(output_raw.get("figures_dir", "outputs/figures")),
            results_dir=Path(output_raw.get("results_dir", "outputs/results")),
            dpi=int(output_raw.get("dpi", 150)),
        ),
        experiment=ExperimentConfig(
            batch_seeds=[int(s) for s in exp_raw.get("batch_seeds", list(range(42, 52)))],
            ablation_groups=list(exp_raw.get("ablation_groups", [])),
        ),
    )
