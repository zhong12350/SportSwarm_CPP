"""StatsBomb event data → Ball-Landing Field construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config_loader import BLFConfig
from src.fields import BallLandingField, build_blf
from src.geometry import Court

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
    return bool(event.get("out", False))


def _cache_path(
    cfg: BLFConfig,
    court: Court,
    cell_size: float,
    cache_dir: Path | None = None,
) -> Path:
    base = cache_dir or Path("data/statsbomb/cache")
    tag = (
        f"blf_{cfg.tactical_style}_"
        f"{court.width_m:.0f}x{court.height_m:.0f}_"
        f"cell{cell_size:.1f}.npz"
    )
    return base / tag


def _iter_event_lists(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "events" in payload:
        return payload["events"]
    return []


def _accumulate_out_events(
    data_dir: Path,
    court: Court,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    style_filter: str,
    verbose: bool,
) -> tuple[np.ndarray, int]:
    """Stream JSON files; only count out-events into grid (low memory)."""
    blf = np.zeros((len(y_coords), len(x_coords)), dtype=np.float64)
    paths = sorted(data_dir.glob("*.json"))
    total = len(paths)
    out_count = 0

    for i, path in enumerate(paths, start=1):
        for event in _iter_event_lists(path):
            if not _is_out_event(event):
                continue
            loc = event.get("location")
            if not loc or len(loc) < 2:
                continue
            if style_filter in ("offensive", "defensive"):
                if _event_style(event) != style_filter:
                    continue
            x, y = _map_to_court(float(loc[0]), float(loc[1]), court)
            ix = int(np.clip(np.searchsorted(x_coords, x), 0, len(x_coords) - 1))
            iy = int(np.clip(np.searchsorted(y_coords, y), 0, len(y_coords) - 1))
            blf[iy, ix] += 1.0
            out_count += 1

        if verbose and (i == 1 or i % 200 == 0 or i == total):
            print(f"  Scanning events: {i}/{total} files...", flush=True)

    if verbose and total:
        print(f"  Out events counted: {out_count:,} from {total} files.", flush=True)
    return blf, total


def _load_cached_blf(path: Path, expected_files: int) -> BallLandingField | None:
    if not path.exists():
        return None
    try:
        data = np.load(path)
        if int(data["n_files"]) != expected_files:
            return None
        return BallLandingField(
            values=data["values"],
            x_coords=data["x_coords"],
            y_coords=data["y_coords"],
        )
    except (OSError, KeyError, ValueError):
        return None


def _save_cached_blf(
    path: Path,
    field: BallLandingField,
    n_files: int,
    out_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        values=field.values,
        x_coords=field.x_coords,
        y_coords=field.y_coords,
        n_files=np.array(n_files),
        out_count=np.array(out_count),
    )


def build_blf_from_statsbomb(
    court: Court,
    cfg: BLFConfig,
    cell_size: float | None = None,
    verbose: bool = True,
    use_cache: bool = True,
    cache_dir: Path | None = None,
) -> BallLandingField:
    """Build BLF from StatsBomb JSON; stream files + optional disk cache."""
    cell = cell_size or cfg.grid_cell_m or 1.0
    data_dir = cfg.statsbomb_dir
    if not data_dir.exists():
        if verbose:
            print(f"  Missing {data_dir}; falling back to Gaussian BLF.", flush=True)
        return build_blf(court, cfg, cell_size=cell)

    n_files = len(list(data_dir.glob("*.json")))
    cache_path = _cache_path(cfg, court, cell, cache_dir)

    if use_cache:
        cached = _load_cached_blf(cache_path, n_files)
        if cached is not None:
            if verbose:
                print(f"  Loaded BLF cache: {cache_path}", flush=True)
            return cached

    if verbose:
        print(f"Building BLF from {data_dir} ({n_files} files, streaming)...", flush=True)

    nx = max(1, int(np.ceil(court.width_m / cell)))
    ny = max(1, int(np.ceil(court.height_m / cell)))
    x_coords = np.arange(nx) * cell + cell / 2
    y_coords = np.arange(ny) * cell + cell / 2

    raw, scanned = _accumulate_out_events(
        data_dir, court, x_coords, y_coords, cfg.tactical_style, verbose
    )

    if raw.sum() == 0:
        if verbose:
            print("  No out events; falling back to Gaussian BLF.", flush=True)
        return build_blf(court, cfg, cell_size=cell)

    from scipy.ndimage import gaussian_filter

    blf = gaussian_filter(raw, sigma=1.5)
    if cfg.normalize and blf.max() > 0:
        blf = blf / blf.max()

    field = BallLandingField(values=blf, x_coords=x_coords, y_coords=y_coords)
    out_count = int(raw.sum())

    if use_cache:
        _save_cached_blf(cache_path, field, scanned, out_count)
        if verbose:
            print(f"  Saved BLF cache: {cache_path}", flush=True)

    if verbose:
        print(f"  BLF grid: {blf.shape}, out events accumulated.", flush=True)
    return field


def build_blf_field(
    court: Court,
    cfg: BLFConfig,
    cell_size: float = 1.0,
    use_cache: bool = True,
) -> BallLandingField:
    """Unified BLF builder: StatsBomb, Gaussian, or hybrid."""
    if cfg.source == "statsbomb":
        return build_blf_from_statsbomb(
            court, cfg, cell_size=cell_size, use_cache=use_cache
        )
    if cfg.source == "hybrid":
        sb = build_blf_from_statsbomb(
            court, cfg, cell_size=cell_size, use_cache=use_cache
        )
        gauss = build_blf(court, cfg, cell_size=cell_size)
        combined = 0.6 * sb.values + 0.4 * gauss.values
        if cfg.normalize and combined.max() > 0:
            combined /= combined.max()
        return BallLandingField(values=combined, x_coords=sb.x_coords, y_coords=sb.y_coords)
    return build_blf(court, cfg, cell_size=cell_size)


def summarize_blf_source(cfg: BLFConfig) -> dict[str, Any]:
    """Report whether a run is backed by real StatsBomb files or a synthetic BLF."""
    report: dict[str, Any] = {
        "configured_source": cfg.source,
        "effective_source": cfg.source,
        "statsbomb_dir": str(cfg.statsbomb_dir),
        "statsbomb_dir_exists": cfg.statsbomb_dir.exists(),
        "json_files": 0,
        "out_events": 0,
        "tactical_style": cfg.tactical_style,
        "fallback_reason": None,
    }
    if cfg.source not in ("statsbomb", "hybrid"):
        return report

    if not cfg.statsbomb_dir.exists():
        report["effective_source"] = "gaussian_fallback"
        report["fallback_reason"] = "statsbomb_dir_missing"
        return report

    paths = sorted(cfg.statsbomb_dir.glob("*.json"))
    report["json_files"] = len(paths)
    if not paths:
        report["effective_source"] = "gaussian_fallback"
        report["fallback_reason"] = "no_json_files"
        return report

    out_events = 0
    for path in paths:
        for event in _iter_event_lists(path):
            if not _is_out_event(event):
                continue
            if cfg.tactical_style in ("offensive", "defensive"):
                if _event_style(event) != cfg.tactical_style:
                    continue
            out_events += 1
    report["out_events"] = out_events
    if out_events == 0:
        report["effective_source"] = "gaussian_fallback"
        report["fallback_reason"] = "no_matching_out_events"
    return report
