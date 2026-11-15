"""Shared figure style for the Main HER2 figures (07).

Journal conventions rather than plotting-library defaults: figure widths in
mm (89 single column, 183 double), Arial at 5-7 pt, hairline axes with the top
and right spines removed, no titles on panels (the legend carries them), bold
lowercase panel letters, scale bars instead of axes on tissue panels, and the
Okabe-Ito colour-blind-safe palette with grey for context.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MM = 1 / 25.4  # inches per mm
SINGLE, ONE_HALF, DOUBLE = 89 * MM, 120 * MM, 183 * MM

# Okabe & Ito colour-universal-design palette.
BLUE, VERMILLION, GREEN = "#0072B2", "#D55E00", "#009E73"
ORANGE, SKY, PURPLE, YELLOW = "#E69F00", "#56B4E9", "#CC79A7", "#F0E442"
INK, MID, CONTEXT, HAIR = "#222222", "#6b6b6b", "#c9c9c9", "#e3e3e3"

# One meaning per colour, used in every figure.
BORDER_CLASS = {"3D only": VERMILLION, "2D and 3D": BLUE, "2D only": GREEN}
KIND = {"2D vs 3D": VERMILLION, "k only": BLUE, "seed only": MID}
DIM = {"3D": VERMILLION, "2D": BLUE}
# Yellow is too faint for small points on white, so the seventh slot is near-black.
NEIGHBORHOOD_COLORS = (BLUE, VERMILLION, GREEN, ORANGE, SKY, PURPLE, "#3a3a3a")


def use_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.5,
        "axes.labelsize": 6.5,
        "axes.titlesize": 6.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelpad": 2.5,
        "lines.linewidth": 0.9,
        "lines.markersize": 3,
        "patch.linewidth": 0.4,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.2,
        "legend.labelspacing": 0.3,
        "figure.dpi": 150,
        "savefig.dpi": 450,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def panel_letter(ax, letter: str, dx: float = -0.02, dy: float = 1.02) -> None:
    """Bold lowercase panel letter at the top-left of ``ax`` (axes fraction offsets)."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")


def panel_letter_fig(fig, x: float, y: float, letter: str) -> None:
    fig.text(x, y, letter, fontsize=8, fontweight="bold", va="top", ha="left")


def bare(ax) -> None:
    """Tissue panel: no ticks, no spines, equal aspect."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")


def scale_bar(ax, length: float, label: str | None = None, loc=(0.04, 0.05), color=INK) -> None:
    """Horizontal scale bar of ``length`` data units at axes-fraction ``loc``."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + loc[0] * (x1 - x0)
    y = y0 + loc[1] * (y1 - y0)
    sign = 1 if x1 > x0 else -1
    ax.plot([x, x + sign * length], [y, y], color=color, lw=1.2, solid_capstyle="butt", clip_on=False)
    ax.text(x + sign * length / 2, y + 0.025 * (y1 - y0), label or f"{length:g} µm",
            ha="center", va="bottom", fontsize=6, color=color)


def box_edges(ax3d, lo, hi, color=MID, lw=0.4) -> None:
    """Wireframe of the imaged block on a 3D axis."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    for a, b in (((x0, y0, z0), (x1, y0, z0)), ((x0, y1, z0), (x1, y1, z0)), ((x0, y0, z1), (x1, y0, z1)),
                 ((x0, y1, z1), (x1, y1, z1)), ((x0, y0, z0), (x0, y1, z0)), ((x1, y0, z0), (x1, y1, z0)),
                 ((x0, y0, z1), (x0, y1, z1)), ((x1, y0, z1), (x1, y1, z1)), ((x0, y0, z0), (x0, y0, z1)),
                 ((x1, y0, z0), (x1, y0, z1)), ((x0, y1, z0), (x0, y1, z1)), ((x1, y1, z0), (x1, y1, z1))):
        ax3d.plot(*np.array([a, b]).T, color=color, lw=lw)


def bare3d(ax3d, lo, hi, elev=22, azim=-58) -> None:
    """Orthographic 3D tissue panel: true proportions, no panes, grid or ticks."""
    ax3d.set_proj_type("ortho")
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_xlim(lo[0], hi[0])
    ax3d.set_ylim(lo[1], hi[1])
    ax3d.set_zlim(lo[2], hi[2])
    ax3d.set_box_aspect(tuple(np.asarray(hi) - np.asarray(lo)))
    ax3d.set_axis_off()


def save(fig, path_stem) -> None:
    """Write PDF (vector text, rasterised point clouds) and PNG."""
    fig.savefig(f"{path_stem}.pdf")
    fig.savefig(f"{path_stem}.png")
    plt.close(fig)
