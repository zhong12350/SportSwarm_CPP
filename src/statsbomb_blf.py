"""StatsBomb event data → Ball-Landing Field construction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config_loader import BLFConfig
from src.fields import BallLandingField, build_blf
from src.geometry import Court

# StatsBomb pitch coordinates (120 x 80)
SB_WIDTH = 120.0
SB_HEIGHT = 80.0

OFFENSIVE_TYPES = {
    "Shot", "Pass", "Carry", "Dribble", "Cross", "Through Ball", "Goal Keeper",
}
DEFENSIVE_TYPES = {
    "Block", "Clearance", "Interception", "Duel", "Pressure", "Ball Recovery",
}
OUT_TYPES = {"Out", "Miscontrol"}


def _map_to_court(x_sb: float, y_sb: float, court: Court) -> tuple[float, float]:
    x_min, y_min, x_max, y_max = court.playable_bounds
    x = x_min + (x_sb / SB_WIDTH) * (x_max - x_min)
    y = y_min + (y_sb / SB_HEIGHT) * (y_max - y_min)
    return court.clip(x, y)


def _load_events_from_dir(data_dir: Path) -> list[dict]:
    events: list[dict] = []
    if not data_dir.exists():
        return events
    for path in sorted(data_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            events.extend(payload)
        elif isinstance(payload, dict) and "events" in payload:
            events.extend(payload["events"])
    return events


def _event_style(event: dict) -> str:
    etype = event.get("type", {})
    name = etype.get("name", "") if isinstance(etype, dict) else str(etype)
    if name in DEFENSIVE_TYPES:
        return "defensive"
    if name in OFFENSIVE_TYPES:
        return "offensive"
    return "neutral"


def _is_out_event(event: dict) -> bool:
    etype = event.get("type", {})
    name = etype.get("name", "") if isinstance(etype, dict) else str(etype)
    if name in OUT_TYPES:
        return True
    if event.get("out", False):
        return True
    return False


def build_blf_from_statsbomb(
    court: Court,
    cfg: BLFConfig,
    cell_size: float | None = None,
) -> BallLandingField:
    """Build BLF from StatsBomb JSON events; fall back to Gaussian if no data."""
    cell = cell_size or cfg.grid_cell_m or 1.0
    events = _load_events_from_dir(cfg.statsbomb_dir)
    if not events:
        return build_blf(court, cfg, cell_size=cell)

    nx = max(1, int(np.ceil(court.width_m / cell)))
    ny = max(1, int(np.ceil(court.height_m / cell)))
    x_coords = np.arange(nx) * cell + cell / 2
    y_coords = np.arange(ny) * cell + cell / 2
    blf = np.zeros((ny, nx), dtype=float)

    style_filter = cfg.tactical_style
    for event in events:
        if not _is_out_event(event):
            continue
        loc = event.get("location")
        if not loc or len(loc) < 2:
            continue
        if style_filter in ("offensive", "defensive"):
            if _event_style(event) != style_filter:
                continue
        x, y = _map_to_court(float(loc[0]), float(loc[1]), court)
        ix = int(np.clip(np.searchsorted(x_coords, x), 0, nx - 1))
        iy = int(np.clip(np.searchsorted(y_coords, y), 0, ny - 1))
        blf[iy, ix] += 1.0

    if blf.sum() == 0:
        return build_blf(court, cfg, cell_size=cell)

    # Smooth with Gaussian kernel for sparse events
    from scipy.ndimage import gaussian_filter

    blf = gaussian_filter(blf, sigma=1.5)

    if cfg.normalize and blf.max() > 0:
        blf /= blf.max()

    return BallLandingField(values=blf, x_coords=x_coords, y_coords=y_coords)


def build_blf_field(
    court: Court,
    cfg: BLFConfig,
    cell_size: float = 1.0,
) -> BallLandingField:
    """Unified BLF builder: StatsBomb, Gaussian, or hybrid."""
    if cfg.source == "statsbomb":
        return build_blf_from_statsbomb(court, cfg, cell_size=cell_size)
    if cfg.source == "hybrid":
        sb = build_blf_from_statsbomb(court, cfg, cell_size=cell_size)
        gauss = build_blf(court, cfg, cell_size=cell_size)
        combined = 0.6 * sb.values + 0.4 * gauss.values
        if cfg.normalize and combined.max() > 0:
            combined /= combined.max()
        return BallLandingField(values=combined, x_coords=sb.x_coords, y_coords=sb.y_coords)
    return build_blf(court, cfg, cell_size=cell_size)
