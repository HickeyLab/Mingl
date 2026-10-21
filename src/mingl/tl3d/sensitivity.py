"""3D-capable ground-truth validation and sensitivity sweeps for the steepness metric.

Mirrors :mod:`mingl.tl.sensitivity`, routing the pipeline through the 3D
:func:`mingl.tl3d.grad.mingl_neighborhoods_scverse` (``z_key``) and the 3D
simulator (``depth``, ``transition_axis``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ..tl.gb import steepness_score
from ..tl.sensitivity import _make_bin_labels, _quiet
from ._spatial import spatial_keys

__all__ = [
    "run_gradient_transition",
    "gradient_sensitivity_analysis",
    "validate_ground_truth_recovery",
]

_NON_COORD_EXTRA_COLS = ("unique_region", "Neighborhood", "Cell Type", "Tissue Unit", "Community")


def run_gradient_transition(
    adata,
    *,
    unit1: str = "Inner Follicle",
    unit2: str = "Outer Follicle",
    neigh_key: str = "Neighborhood",
    x_key: str = "x",
    y_key: str = "y",
    z_key: Optional[str] = None,
    k: int = 20,
    n_bins: int = 5,
    n_clusters: int = 5,
    use_quantiles: bool = True,
    subsample_frac: float = 1.0,
    seed: int = 0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the gradient/transition pipeline once (2D or 3D) and return its steepness.

    Same contract as :func:`mingl.tl.sensitivity.run_gradient_transition`
    (never mutates ``adata``; errors are recorded, not raised); the result also
    records ``z_key``.
    """
    from .grad import mingl_neighborhoods_scverse

    base = {
        "k": int(k),
        "n_bins": int(n_bins),
        "n_clusters": int(n_clusters),
        "subsample_frac": float(subsample_frac),
        "use_quantiles": bool(use_quantiles),
        "seed": int(seed),
        "z_key": z_key,
    }
    fail = {
        "steepness": float("nan"),
        "slope": float("nan"),
        "slope_norm": float("nan"),
        "score_iqr": float("nan"),
        "n_clusters_effective": 0,
        "n_cells": 0,
        **base,
    }

    try:
        ad_run = adata.copy()
        if subsample_frac < 1.0:
            rng = np.random.default_rng(seed)
            n = ad_run.n_obs
            keep = np.sort(rng.choice(n, size=max(2, int(round(n * subsample_frac))), replace=False))
            ad_run = ad_run[keep].copy()

        n_cells = int(ad_run.n_obs)

        def _call():
            return mingl_neighborhoods_scverse(
                ad_run,
                tu1=unit1,
                tu2=unit2,
                extra_cols=(*spatial_keys(x_key, y_key, z_key), *_NON_COORD_EXTRA_COLS),
                ks=(int(k),),
                k=int(k),
                distance_max="none",
                X=x_key,
                Y=y_key,
                Z=z_key,
                neigh=neigh_key,
                target_neighborhoods=(unit1, unit2),
                n_bins=int(n_bins),
                bin_labels=_make_bin_labels(n_bins),
                use_quantiles=bool(use_quantiles),
                n_clusters=int(n_clusters),
                random_state=int(seed),
            )

        out = _call() if verbose else _quiet(_call)
        st = steepness_score(out[0])
        return {
            "steepness": st["steepness"],
            "slope": st["slope"],
            "slope_norm": st["slope_norm"],
            "score_iqr": st["score_iqr"],
            "n_clusters_effective": int(st["n_clusters"]),
            "n_cells": n_cells,
            "error": None,
            **base,
        }
    except Exception as exc:  # keep the sweep going; record the failure
        fail["error"] = f"{type(exc).__name__}: {exc}"
        return fail


def gradient_sensitivity_analysis(
    adata,
    *,
    unit1: str = "Inner Follicle",
    unit2: str = "Outer Follicle",
    neigh_key: str = "Neighborhood",
    x_key: str = "x",
    y_key: str = "y",
    z_key: Optional[str] = None,
    baseline: Optional[Dict[str, Any]] = None,
    param_grid: Optional[Dict[str, Sequence]] = None,
    transition_label: Optional[str] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """One-at-a-time sensitivity sweep of the steepness output, in 2D or 3D.

    See :func:`mingl.tl.sensitivity.gradient_sensitivity_analysis` for the
    defaults and output columns.
    """
    baseline = {"k": 20, "n_bins": 5, "n_clusters": 5, "subsample_frac": 1.0, **(baseline or {})}
    if param_grid is None:
        param_grid = {
            "k": [10, 20, 30, 50],
            "n_bins": [3, 4, 5, 6, 7],
            "n_clusters": [3, 4, 5, 6, 8],
            "subsample_frac": [0.25, 0.5, 0.75, 1.0],
        }

    if transition_label is None and hasattr(adata, "uns"):
        transition_label = adata.uns.get("sim_params", {}).get("sharpness_label", "unknown")

    def _run(settings):
        return run_gradient_transition(
            adata, unit1=unit1, unit2=unit2, neigh_key=neigh_key,
            x_key=x_key, y_key=y_key, z_key=z_key, seed=seed, **settings,
        )

    rows = [{"transition": transition_label, "param": "baseline", "value": np.nan, **_run(baseline)}]
    for pname, values in param_grid.items():
        for v in values:
            settings = dict(baseline)
            settings[pname] = v
            rows.append({"transition": transition_label, "param": pname, "value": v, **_run(settings)})

    cols = [
        "transition", "param", "value", "steepness", "slope", "slope_norm",
        "score_iqr", "n_clusters_effective", "n_cells", "k", "n_bins",
        "n_clusters", "subsample_frac", "z_key", "error",
    ]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]


def validate_ground_truth_recovery(
    *,
    sharpness_levels: Sequence = ("sharp", "medium", "gradual"),
    n_cells: int = 3000,
    unit1: str = "Inner Follicle",
    unit2: str = "Outer Follicle",
    depth: Optional[float] = None,
    transition_axis: str = "x",
    use_z: Optional[bool] = None,
    k: int = 20,
    n_bins: int = 5,
    n_clusters: int = 5,
    n_repeats: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Check that recovered steepness tracks the known sharpness, in 2D or 3D.

    ``depth`` / ``transition_axis`` configure the simulated tissue. ``use_z``
    controls whether the pipeline sees the ``z`` axis (default: whenever the
    tissue is 3D); ``use_z=False`` on a z-transition is the negative control.
    """
    from .simulate import simulate_transition_tissue

    if use_z is None:
        use_z = depth is not None
    if use_z and depth is None:
        raise ValueError("use_z=True requires a 3D tissue (set depth).")
    z_key = "z" if use_z else None

    rows = []
    for level in sharpness_levels:
        for rep in range(n_repeats):
            adata = simulate_transition_tissue(
                n_cells=n_cells, sharpness=level, unit1=unit1, unit2=unit2,
                depth=depth, transition_axis=transition_axis, seed=seed + rep,
            )
            sim = adata.uns["sim_params"]
            res = run_gradient_transition(
                adata, unit1=unit1, unit2=unit2, z_key=z_key,
                k=k, n_bins=n_bins, n_clusters=n_clusters, seed=seed,
            )
            rows.append(
                {
                    "sharpness_label": sim["sharpness_label"],
                    "transition_width_frac": sim["transition_width_frac"],
                    "ground_truth_sharpness": sim["ground_truth_sharpness"],
                    "transition_axis": transition_axis,
                    "n_dims": sim["n_dims"],
                    "z_key": z_key,
                    "repeat": rep,
                    "steepness": res["steepness"],
                    "slope": res["slope"],
                    "slope_norm": res["slope_norm"],
                    "n_clusters_effective": res["n_clusters_effective"],
                    "error": res["error"],
                }
            )
    return pd.DataFrame(rows)
