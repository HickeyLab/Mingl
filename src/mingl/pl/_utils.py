# MINGL/src/MINGL/pl/_utils.py

from __future__ import annotations
from typing import Union
from pathlib import Path

import matplotlib.pyplot as plt

try:
    import scanpy as sc
except ImportError:
    sc = None


def get_figdir() -> Path:
    """Return directory where figures should be saved."""
    if sc is not None and getattr(sc.settings, "figdir", None):
        return Path(sc.settings.figdir)
    # fallback
    return Path("figures")


def save_figure(fig: plt.Figure, base: str, save: Union[bool, str]):
    """
    Save a figure to disk, scanpy-style.

    Parameters
    ----------
    fig
        Matplotlib Figure.
    base
        Base name for the plot (e.g. 'neighborhood_summary').
    save
        True or a string; same semantics as in the plotting functions.
    """
    figdir = get_figdir()
    figdir.mkdir(parents=True, exist_ok=True)

    if isinstance(save, str) and save is not True:
        # mimic scanpy: string is suffix or filename
        if save.startswith(".") or save.startswith("_"):
            # suffix
            filename = f"{base}{save}"
        else:
            # explicit filename
            filename = save
    else:
        filename = f"{base}.png"

    out = figdir / filename
    fig.savefig(out, bbox_inches="tight")
