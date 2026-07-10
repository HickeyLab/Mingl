"""Synthetic transition-tissue generator for validating MINGL gradient/transition
clustering (addresses reviewer comment R1.5).

The reviewer asked for "simulations with known sharp and gradual gradients to show
that the metric recovers ground truth" and for robustness tests to bin number,
neighbor number, cluster number, and spatial sampling density.

This module builds spatial tissues in which the transition between two
organizational units has a *known* sharpness. Because the ground-truth transition
width is set by the caller, the steepness that MINGL recovers can be compared
against it directly, and the pipeline can be re-run while sweeping the analysis
parameters.

The generated ``AnnData`` matches the exact columns consumed by
:func:`mingl.tl.grad.mingl_neighborhoods_scverse`, so the full gradient/transition
clustering path can be run on it without any extra preprocessing.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import pandas as pd
import anndata as ad

__all__ = ["simulate_transition_tissue", "SHARPNESS_PRESETS"]


# Preset transition widths expressed as a fraction of the tissue width over which
# the unit-1 membership probability moves from ~0.1 to ~0.9 (the logistic 10-90
# width). Smaller fraction => sharper (more abrupt) transition.
SHARPNESS_PRESETS = {
    "sharp": 0.05,
    "medium": 0.25,
    "gradual": 0.70,
}

# 10-90 width of a standard logistic in logit units (log(0.9/0.1) - log(0.1/0.9)).
_LOGISTIC_10_90 = 2.0 * np.log(9.0)  # ~= 4.3944


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def simulate_transition_tissue(
    *,
    n_cells: int = 3000,
    sharpness: Union[str, float] = "medium",
    unit1: str = "Inner Follicle",
    unit2: str = "Outer Follicle",
    width: float = 100.0,
    height: float = 100.0,
    n_regions: int = 1,
    region_prefix: str = "sim_region",
    cell_types: Sequence[str] = ("CT_unit1", "CT_unit2", "CT_shared_a", "CT_shared_b", "CT_shared_c"),
    prob_noise: float = 0.03,
    prob_clip: float = 1e-3,
    seed: int = 0,
) -> ad.AnnData:
    """Generate a synthetic tissue with a known unit1<->unit2 transition sharpness.

    A single spatial axis (``x``) drives the transition. The unit-1 membership
    probability follows a logistic curve centered at the tissue midline; the
    ``sharpness`` argument sets how abruptly it switches, which is the ground-truth
    quantity the steepness metric should recover.

    Parameters
    ----------
    n_cells
        Number of cells per region.
    sharpness
        Either a preset name (``"sharp"``, ``"medium"``, ``"gradual"``) or a float
        giving the transition width as a fraction of ``width`` (smaller = sharper).
    unit1, unit2
        Names of the two organizational units. Used both as the probability column
        names and as the discrete ``Neighborhood``/``Community``/``Tissue Unit``
        labels (argmax assignment), matching what MINGL expects downstream.
    width, height
        Tissue dimensions (arbitrary spatial units).
    n_regions
        Number of independent regions to concatenate (each an i.i.d. draw with the
        same sharpness). Useful for reproducibility/subsampling tests.
    cell_types
        Cell-type labels. The first is enriched toward ``unit1``, the second toward
        ``unit2``, the rest are roughly uniform, so composition shifts along the
        gradient (feeds the downstream cell-type-distribution plots).
    prob_noise
        Std of Gaussian noise added in logit space before the sigmoid, so
        probabilities are not perfectly deterministic.
    prob_clip
        Probabilities are clipped to ``[prob_clip, 1 - prob_clip]`` so the MINGL
        log-ratio Score stays finite.
    seed
        Random seed for full reproducibility.

    Returns
    -------
    AnnData
        ``adata.obs`` has columns ``x, y, unique_region, Neighborhood, Community,
        Tissue Unit, Cell Type, <unit1>, <unit2>`` (the last two are membership
        probabilities). Ground-truth parameters are stored in
        ``adata.uns["sim_params"]``, including ``ground_truth_sharpness`` (the
        monotone target the recovered steepness should track).
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

    # Logistic slope so the 10-90 transition spans width_frac * width in x.
    beta = _LOGISTIC_10_90 / (width_frac * width)
    center = width / 2.0

    rng = np.random.default_rng(seed)
    frames = []
    for r in range(n_regions):
        x = rng.uniform(0.0, width, size=n_cells)
        y = rng.uniform(0.0, height, size=n_cells)

        logits = beta * (x - center)
        if prob_noise > 0:
            logits = logits + rng.normal(0.0, prob_noise, size=n_cells)
        p1 = _sigmoid(logits)
        p1 = np.clip(p1, prob_clip, 1.0 - prob_clip)
        p2 = 1.0 - p1

        neigh = np.where(p1 >= 0.5, unit1, unit2)

        # Cell-type sampling: type 0 tracks p1, type 1 tracks p2, rest ~uniform.
        n_ct = len(cell_types)
        w = np.empty((n_cells, n_ct), dtype=float)
        w[:, 0] = p1
        w[:, 1] = p2
        if n_ct > 2:
            w[:, 2:] = 0.3
        w = w / w.sum(axis=1, keepdims=True)
        # Per-cell categorical draw via inverse-CDF on cumulative weights.
        cum = np.cumsum(w, axis=1)
        u = rng.uniform(0.0, 1.0, size=(n_cells, 1))
        ct_idx = (u > cum).sum(axis=1)
        ct_idx = np.clip(ct_idx, 0, n_ct - 1)
        ct = np.asarray(cell_types)[ct_idx]

        region_name = f"{region_prefix}_{r}"
        frames.append(
            pd.DataFrame(
                {
                    "x": x,
                    "y": y,
                    "unique_region": region_name,
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
        # Monotone ground-truth target: sharper transition => larger value.
        "ground_truth_sharpness": 1.0 / width_frac,
        "logistic_beta": float(beta),
        "unit1": unit1,
        "unit2": unit2,
        "width": float(width),
        "height": float(height),
        "n_cells": int(n_cells),
        "n_regions": int(n_regions),
        "prob_noise": float(prob_noise),
        "seed": int(seed),
    }
    return adata
