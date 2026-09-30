"""Shared figure style, palettes and small drawing helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "set_style",
    "cell_type_palette",
    "save_figure",
    "save_panels",
    "panel_label",
    "stacked_composition_bars",
    "annotate_significance",
    "compact_legend",
    "legend_from",
    "top_cell_types",
    "collapse_to_top",
    "BORDER_MAP_COLORS",
    "TOP_N_CELL_TYPES",
]

#: Cell types shown individually in composition panels. ``None`` shows all of
#: them (MINGL's own behaviour); an int truncates and pools the rest as "Other".
TOP_N_CELL_TYPES = None  # None == show every cell type, as MINGL does

_COLOR_MAP_PATH = Path(__file__).resolve().parents[1] / "src" / "mingl" / "pl" / "cell_type_color_map.json"


def set_style() -> None:
    """Publication-ish matplotlib defaults (vector text, no chartjunk)."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def cell_type_palette(cell_types: Sequence[str]) -> dict[str, str]:
    """Colors for cell types, reusing the package's map where the names match.

    Falls back to a concatenation of qualitative colormaps for names the shipped
    map does not cover (the melanoma and esophagus panels use different
    vocabularies), so colors stay stable for a given ordered input.
    """
    shipped: dict[str, str] = {}
    try:
        with open(_COLOR_MAP_PATH) as fh:
            shipped = {str(k): str(v) for k, v in json.load(fh).items()}
    except (OSError, ValueError):
        pass

    fallback = []
    for name in ("tab20", "tab20b", "tab20c", "Set3", "Dark2"):
        cmap = matplotlib.colormaps[name]
        fallback.extend(matplotlib.colors.to_hex(cmap(i)) for i in range(cmap.N))

    palette, spare = {}, 0
    for ct in cell_types:
        key = str(ct)
        if key == "Other":
            palette[key] = "#bdbdbd"
        elif key in shipped:
            palette[key] = shipped[key]
        else:
            palette[key] = fallback[spare % len(fallback)]
            spare += 1
    return palette


def save_figure(fig: plt.Figure, out_dir: str | Path, name: str, *, pdf: bool = True) -> list[Path]:
    """Save a figure as PNG (+ PDF) and return the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png"]
    fig.savefig(paths[0])
    if pdf:
        paths.append(out_dir / f"{name}.pdf")
        fig.savefig(paths[1])
    plt.close(fig)
    return paths


def save_panels(fig: plt.Figure, out_dir: str | Path, prefix: str, panels: dict) -> list[Path]:
    """Save each panel of a composite figure as its own file.

    ``panels`` maps a name to either a single Axes or a list of Axes; the saved
    crop is the tight bounding box around that group (including its labels and
    any colorbar passed in the group), so a panel is usable standalone.
    """
    out_dir = Path(out_dir)
    (out_dir / "panels").mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()  # renderer must exist before tightbbox is meaningful
    written: list[Path] = []
    for name, group in panels.items():
        axes = group if isinstance(group, (list, tuple)) else [group]
        axes = [a for a in axes if a is not None and a.get_visible()]
        if not axes:
            continue
        boxes = [a.get_tightbbox(fig.canvas.get_renderer()) for a in axes]
        bbox = boxes[0].union(boxes).transformed(fig.dpi_scale_trans.inverted())
        bbox = bbox.expanded(1.03, 1.05)
        path = out_dir / "panels" / f"{prefix}_{name}.png"
        fig.savefig(path, bbox_inches=bbox, dpi=300)
        written.append(path)
    return written


#: Style of the composite-figure panel letters, named so that anything needing to
#: reproduce or measure a letter reads it from here rather than restating 11 pt bold.
PANEL_LABEL_SIZE = 11
PANEL_LABEL_WEIGHT = "bold"


def panel_label(ax: plt.Axes, letter: str, *, dx: float = -0.08, dy: float = 1.06) -> None:
    """Bold panel letter in axes coordinates."""
    ax.text(
        dx, dy, letter, transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE,
        fontweight=PANEL_LABEL_WEIGHT, va="bottom", ha="right",
    )


def top_cell_types(
    proportions: pd.DataFrame, *, top_n: int | None = TOP_N_CELL_TYPES
) -> list[str]:
    """Cell types ordered by mean proportion, optionally truncated.

    ``proportions`` is (conditions x cell types). ``top_n=None`` returns every
    cell type with no ``"Other"`` bucket -- what MINGL's own composition plots
    do, and the default here.
    """
    if proportions.empty:
        return []
    means = proportions.mean(axis=0).sort_values(ascending=False)
    if top_n is None or len(means) <= top_n:
        return list(means.index)
    return list(means.index[:top_n]) + ["Other"]


def collapse_to_top(proportions: pd.DataFrame, order: Sequence[str]) -> pd.DataFrame:
    """Reduce a (conditions x cell types) frame to ``order``, pooling the rest."""
    order = list(order)
    named = [c for c in order if c != "Other"]
    out = proportions.reindex(columns=named).fillna(0.0)
    if "Other" in order:
        out["Other"] = proportions.drop(columns=[c for c in named if c in proportions.columns]).sum(axis=1)
    return out.loc[:, order]


def stacked_composition_bars(
    ax: plt.Axes,
    proportions: pd.DataFrame,
    *,
    palette: dict[str, str],
    x_labels: Sequence[str] | None = None,
    bar_width: float = 0.72,
    edge: str = "white",
) -> None:
    """Stacked proportion bars: one bar per row of ``proportions``."""
    x = np.arange(proportions.shape[0], dtype=float)
    bottom = np.zeros(proportions.shape[0])
    for ct in proportions.columns:
        vals = proportions[ct].to_numpy(dtype=float)
        ax.bar(
            x, vals, bottom=bottom, width=bar_width, label=str(ct),
            color=palette.get(str(ct), "#999999"), edgecolor=edge, linewidth=0.2,
        )
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(
        list(x_labels) if x_labels is not None else [str(i) for i in proportions.index]
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("proportion of border cells")


#: Colors used by :func:`mingl.pl.spatial_loc_region`, re-exported so the figure
#: can build a matching legend without redrawing the map itself.
BORDER_MAP_COLORS = {"other": "lightgray", "only_1": "plum", "only_2": "blue", "both": "red"}


def annotate_significance(
    ax: plt.Axes,
    positions: Sequence[float],
    labels: Sequence[str],
    *,
    y: float | None = None,
    pad: float = 0.02,
    fontsize: float = 7.0,
) -> None:
    """Write significance markers above the given x positions."""
    lo, hi = ax.get_ylim()
    y = hi - pad * (hi - lo) if y is None else y
    for pos, text in zip(positions, labels):
        if text:
            ax.text(pos, y, text, ha="center", va="top", fontsize=fontsize)


def _dedup_handles(ax: plt.Axes):
    handles, labels = ax.get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            h2.append(h)
            l2.append(l)
    return h2, l2


def compact_legend(
    ax: plt.Axes, *, ncol: int = 1, loc: str = "center left", bbox=(1.01, 0.5), title: str | None = None
) -> None:
    """Legend outside the axes, deduplicated, in first-seen order."""
    h2, l2 = _dedup_handles(ax)
    if h2:
        ax.legend(h2, l2, ncol=ncol, loc=loc, bbox_to_anchor=bbox, title=title,
                  handlelength=1.1, borderaxespad=0.0)


def legend_from(
    source: plt.Axes, target: plt.Axes, *, ncol: int = 1, title: str | None = None,
    fontsize: float = 6.5,
) -> None:
    """Draw ``source``'s legend into its own (blank) axes.

    Keeps long cell-type legends from overlapping the neighbouring panel, which
    is what happens when a 15-entry legend is anchored outside a dense subplot.
    """
    target.axis("off")
    h2, l2 = _dedup_handles(source)
    if h2:
        target.legend(h2, l2, ncol=ncol, loc="center left", title=title,
                      handlelength=1.1, borderaxespad=0.0, fontsize=fontsize)
