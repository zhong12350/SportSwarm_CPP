"""Publication-quality Matplotlib styling (IEEE / RAL / ICRA)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Okabe–Ito colorblind-safe palette
OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# Warm BLF colormap: low → high landing probability
BLF_CMAP = LinearSegmentedColormap.from_list(
    "blf_warm",
    ["#F7F9F2", "#FFF3C4", "#FDB462", "#E6550D", "#A63603", "#67000D"],
    N=256,
)

# Court surface tint (under BLF overlay)
COURT_GREEN = "#2D6A4F"
COURT_LINE = "#FFFFFF"
COURT_LINE_ALPHA = 0.85

_DEFAULT_RC: dict[str, object] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.4,
    "lines.markersize": 5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 150,
    "savefig.dpi": 300,
}


def apply_pub_style(font_size: float = 9) -> None:
    """Apply global rcParams for journal figures."""
    rc = dict(_DEFAULT_RC)
    rc["font.size"] = font_size
    rc["axes.labelsize"] = font_size
    rc["axes.titlesize"] = font_size + 1
    rc["legend.fontsize"] = max(font_size - 1, 7)
    rc["xtick.labelsize"] = max(font_size - 1, 7)
    rc["ytick.labelsize"] = max(font_size - 1, 7)
    mpl.rcParams.update(rc)


@contextmanager
def pub_figure_context(font_size: float = 9) -> Iterator[None]:
    """Temporarily switch to publication style and restore on exit."""
    backup = mpl.rcParams.copy()
    apply_pub_style(font_size)
    try:
        yield
    finally:
        mpl.rcParams.update(backup)


def save_figure(fig: plt.Figure, path, dpi: int = 300) -> None:
    """Save PNG + vector PDF alongside each other."""
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    if out.suffix.lower() != ".pdf":
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
