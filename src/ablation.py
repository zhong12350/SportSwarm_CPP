"""Ablation study configurations per proposal §11.2."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from src.config_loader import AppConfig


@dataclass(frozen=True)
class AblationSpec:
    name: str
    description: str


ABLATION_SPECS: dict[str, AblationSpec] = {
    "no_blf": AblationSpec(
        "no_blf",
        "BLF disabled → uniform ball sampling + uniform deployment",
    ),
    "no_multi_robot": AblationSpec(
        "no_multi_robot",
        "Only N=1 to test multi-robot necessity",
    ),
    "fixed_deployment": AblationSpec(
        "fixed_deployment",
        "Fixed uniform deployment vs optimized BLF/Lloyd deployment",
    ),
    "no_dynamic_realloc": AblationSpec(
        "no_dynamic_realloc",
        "Static Voronoi vs GMM online update",
    ),
    "no_cvar": AblationSpec(
        "no_cvar",
        "Ignore player obstacles vs CVaR-aware planning",
    ),
}


def apply_ablation(config: AppConfig, ablation: str) -> AppConfig:
    cfg = deepcopy(config)

    if ablation == "no_blf":
        cfg.balls.mode = "uniform"
        cfg.blf.source = "gaussian"
        cfg.blf.gaussians = []
        cfg.deployment.mode = "uniform"
        cfg.methods = [m for m in cfg.methods if "blf" not in m and m != "sportswarm_full"]
        if not cfg.methods:
            cfg.methods = ["single_greedy", "multi_greedy", "uniform_cpp"]

    elif ablation == "no_multi_robot":
        cfg.robots.counts = [1]
        cfg.methods = ["single_greedy", "uniform_cpp", "blf_weighted_cpp"]

    elif ablation == "fixed_deployment":
        cfg.deployment.mode = "uniform"
        cfg.methods = ["blf_informed_deployment", "gmm_swarm", "sportswarm_full"]

    elif ablation == "no_dynamic_realloc":
        cfg.simulation.gmm_realloc_interval_s = 1e9
        cfg.methods = ["voronoi_assignment", "gmm_swarm", "sportswarm_full"]

    elif ablation == "no_cvar":
        cfg.cvar.enabled = False
        cfg.players.enabled = True
        cfg.methods = ["sportswarm_full", "multi_greedy"]

    elif ablation == "full_cvar":
        cfg.cvar.enabled = True
        cfg.players.enabled = True
        cfg.methods = ["sportswarm_full"]

    return cfg


def default_ablation_groups() -> list[str]:
    return list(ABLATION_SPECS.keys())
