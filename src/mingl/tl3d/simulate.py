"""Synthetic transition tissues in 2D or 3D with a known transition sharpness.

Mirrors :func:`mingl.tl.simulate.simulate_transition_tissue` and adds
``depth`` (a third axis) and ``transition_axis`` (which axis the unit1<->unit2
transition runs along). ``depth=None, transition_axis="x"`` reproduces the 2D
generator draw-for-draw for the same seed.

The key 3D validation is a transition along ``z``: it is invisible to a 2D
pipeline (every x/y neighborhood mixes both units), and a correct 3D pipeline
must recover it.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd

from ..tl.simulate import _LOGISTIC_10_90, SHARPNESS_PRESETS, _sigmoid

__all__ = ["simulate_transition_tissue", "SHARPNESS_PRESETS"]


def simulate_transition_tissue(
    *,
    n_cells: int = 3000,
    sharpness: Union[str, float] = "medium",
    unit1: str = "Inner Follicle",
    unit2: str = "Outer Follicle",
    width: float = 100.0,
    height: float = 100.0,
    depth: Optional[float] = None,
    transition_axis: str = "x",
    n_regions: int = 1,
    region_prefix: str = "sim_region",
    cell_types: Sequence[str] = ("CT_unit1", "CT_unit2", "CT_shared_a", "CT_shared_b", "CT_shared_c"),
    prob_noise: float = 0.03,
    prob_clip: float = 1e-3,
    seed: int = 0,
) -> ad.AnnData:
    """Generate a synthetic 2D/3D tissue with a known unit1<->unit2 transition.

    Parameters
    ----------
    width, height, depth
        Extents along x, y and z. ``depth=None`` gives a 2D tissue (no ``z``
        column).
    transition_axis
        ``"x"`` (default, as in 2D), ``"y"``, or ``"z"`` (requires ``depth``).
        The logistic is centered on that axis's midline and ``sharpness`` is a
        fraction of that axis's extent.

    Other parameters and outputs are as in
    :func:`mingl.tl.simulate.simulate_transition_tissue`; ``uns["sim_params"]``
    additionally records ``depth``, ``transition_axis`` and ``n_dims``.
    """
    if isinstance(sharpness, str):
        key = sharpness.strip().lower()
        if key not in SHARPNESS_PRESETS:
            raise ValueError(
                f"Unknown sharpness preset {sharpness!r}. "
                f"Choose from {sorted(SHARPNESS_PRESETS)} or pass a float width fraction."
            )
        width_frac = float(SHARPNESS_PRESETS[key])
        sharpness_label = key
    else:
        width_frac = float(sharpness)
        if not (0.0 < width_frac <= 1.0):
            raise ValueError("Numeric sharpness (transition width fraction) must be in (0, 1].")
        sharpness_label = f"width_frac={width_frac:g}"

    if n_cells < 2:
        raise ValueError("n_cells must be >= 2.")
    if n_regions < 1:
        raise ValueError("n_regions must be >= 1.")
    cell_types = list(cell_types)
    if len(cell_types) < 2:
        raise ValueError("Need at least 2 cell types (first two are the polar types).")

    extents = {"x": float(width), "y": float(height)}
    if depth is not None:
        if depth <= 0:
            raise ValueError("depth must be positive (or None for a 2D tissue).")
        extents["z"] = float(depth)
    if transition_axis not in extents:
        raise ValueError(f"transition_axis must be one of {sorted(extents)} (got {transition_axis!r}).")

    axis_extent = extents[transition_axis]
    beta = _LOGISTIC_10_90 / (width_frac * axis_extent)
    center = axis_extent / 2.0

    rng = np.random.default_rng(seed)
    frames = []
    for r in range(n_regions):
        # Draw order x, y, (z), noise, cell type -- keeps 2D draws identical to mingl.tl.simulate.
        axes = {
            "x": rng.uniform(0.0, width, size=n_cells),
            "y": rng.uniform(0.0, height, size=n_cells),
        }
        if depth is not None:
            axes["z"] = rng.uniform(0.0, depth, size=n_cells)

        logits = beta * (axes[transition_axis] - center)
        if prob_noise > 0:
            logits = logits + rng.normal(0.0, prob_noise, size=n_cells)
        p1 = np.clip(_sigmoid(logits), prob_clip, 1.0 - prob_clip)
        p2 = 1.0 - p1

        neigh = np.where(p1 >= 0.5, unit1, unit2)

        n_ct = len(cell_types)
        w = np.empty((n_cells, n_ct), dtype=float)
        w[:, 0] = p1
        w[:, 1] = p2
        if n_ct > 2:
            w[:, 2:] = 0.3
        w = w / w.sum(axis=1, keepdims=True)
        cum = np.cumsum(w, axis=1)
        u = rng.uniform(0.0, 1.0, size=(n_cells, 1))
        ct_idx = np.clip((u > cum).sum(axis=1), 0, n_ct - 1)
        ct = np.asarray(cell_types)[ct_idx]

        frames.append(
            pd.DataFrame(
                {
                    **axes,
                    "unique_region": f"{region_prefix}_{r}",
                    "Neighborhood": neigh,
                    "Community": neigh,
                    "Tissue Unit": neigh,
                    "Cell Type": ct,
                    unit1: p1,
                    unit2: p2,
                }
            )
        )

    obs = pd.concat(frames, axis=0, ignore_index=True)
    obs.index = obs.index.astype(str)

    adata = ad.AnnData(obs=obs)
    adata.uns["sim_params"] = {
        "sharpness_label": sharpness_label,
        "transition_width_frac": width_frac,
        "ground_truth_sharpness": 1.0 / width_frac,
        "logistic_beta": float(beta),
        "unit1": unit1,
        "unit2": unit2,
        "width": float(width),
        "height": float(height),
        "depth": None if depth is None else float(depth),
        "transition_axis": transition_axis,
        "n_dims": len(extents),
        "n_cells": int(n_cells),
        "n_regions": int(n_regions),
        "prob_noise": float(prob_noise),
        "seed": int(seed),
    }
    return adata
