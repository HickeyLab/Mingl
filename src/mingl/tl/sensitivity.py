"""Ground-truth validation and sensitivity analysis for the gradient/transition
steepness metric (addresses reviewer comment R1.5).

Two things the reviewer asked for:

* **Ground-truth recovery** — on synthetic tissues with known sharp/medium/gradual
  transitions (:mod:`mingl.tl.simulate`), does the recovered steepness increase
  monotonically with the true sharpness?
* **Robustness** — how much does the steepness output move when the analysis
  parameters (neighbor count ``k``, number of probability bins, number of transition
  clusters, and spatial sampling density) change?

Both are driven by :func:`run_gradient_transition`, which runs the full pipeline
(:func:`mingl.tl.grad.mingl_neighborhoods_scverse` -> transition clusters ->
:func:`mingl.tl.gb.steepness_score`) once for a given parameter set.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .gb import steepness_score

__all__ = [
    "run_gradient_transition",
    "gradient_sensitivity_analysis",
    "validate_ground_truth_recovery",
]

_CANON5 = ("Very Low", "Low", "Medium", "High", "Very High")
_EXTRA_COLS = ("x", "y", "unique_region", "Neighborhood", "Cell Type", "Tissue Unit", "Community")


def _make_bin_labels(n_bins: int) -> list:
    """Bin labels of the requested length (canonical names when n_bins == 5).

    Label strings only name the dummy columns fed to the neighbor-composition
    clustering, so any distinct set works for n_bins != 5.
    """
    if n_bins == 5:
        return list(_CANON5)
    return [f"bin_{i}" for i in range(n_bins)]


def _quiet(func, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return func(*args, **kwargs)


def run_gradient_transition(
    adata,
    *,
    unit1: str = "Inner Follicle",
    unit2: str = "Outer Follicle",
    neigh_key: str = "Neighborhood",
    k: int = 20,
    n_bins: int = 5,
    n_clusters: int = 5,
    use_quantiles: bool = True,
    subsample_frac: float = 1.0,
    seed: int = 0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the full gradient/transition pipeline once and return its steepness.

    Copies ``adata`` (never mutates the caller's object), optionally subsamples to
    ``subsample_frac`` of the cells, runs
    :func:`mingl.tl.grad.mingl_neighborhoods_scverse` with the given parameters, and
    computes :func:`mingl.tl.gb.steepness_score`.

    Returns a dict with ``steepness``, ``slope``, ``slope_norm``, ``score_iqr``,
    ``n_clusters_effective`` (clusters that actually received cells), ``n_cells``,
    the parameter set, and ``error`` (``None`` on success, else the message).
    """
    from .grad import mingl_neighborhoods_scverse

    base = {
        "k": int(k),
        "n_bins": int(n_bins),
        "n_clusters": int(n_clusters),
        "subsample_frac": float(subsample_frac),
        "use_quantiles": bool(use_quantiles),
        "seed": int(seed),
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
            n_keep = max(2, int(round(n * subsample_frac)))
            keep = np.sort(rng.choice(n, size=n_keep, replace=False))
            ad_run = ad_run[keep].copy()

        n_cells = int(ad_run.n_obs)
        labels = _make_bin_labels(n_bins)

        def _call():
            return mingl_neighborhoods_scverse(
                ad_run,
                tu1=unit1,
                tu2=unit2,
                extra_cols=_EXTRA_COLS,
                ks=(int(k),),
                k=int(k),
                distance_max="none",
                neigh=neigh_key,
                target_neighborhoods=(unit1, unit2),
                n_bins=int(n_bins),
                bin_labels=labels,
                use_quantiles=bool(use_quantiles),
                n_clusters=int(n_clusters),
                random_state=int(seed),
            )

        out = _call() if verbose else _quiet(_call)
        ad_out = out[0]

        st = steepness_score(ad_out)
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
    baseline: Optional[Dict[str, Any]] = None,
    param_grid: Optional[Dict[str, Sequence]] = None,
    transition_label: Optional[str] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """One-at-a-time sensitivity sweep of the steepness output.

    Each parameter in ``param_grid`` is varied around ``baseline`` while the others
    are held fixed, and the pipeline is re-run. Returns a tidy DataFrame with one
    row per run (including a ``param="baseline"`` row) and columns
    ``transition, param, value, steepness, slope, slope_norm, score_iqr,
    n_clusters_effective, n_cells, error``.

    Defaults sweep ``k in {10,20,30,50}``, ``n_bins in {3,4,5,6,7}``,
    ``n_clusters in {3,4,5,6,8}``, ``subsample_frac in {0.25,0.5,0.75,1.0}``.
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
            adata,
            unit1=unit1,
            unit2=unit2,
            neigh_key=neigh_key,
            seed=seed,
            **settings,
        )

    rows = []
    # Baseline reference row.
    res = _run(baseline)
    rows.append({"transition": transition_label, "param": "baseline", "value": np.nan, **res})

    for pname, values in param_grid.items():
        for v in values:
            settings = dict(baseline)
            settings[pname] = v
            res = _run(settings)
            rows.append({"transition": transition_label, "param": pname, "value": v, **res})

    cols = [
        "transition", "param", "value", "steepness", "slope", "slope_norm",
        "score_iqr", "n_clusters_effective", "n_cells", "k", "n_bins",
        "n_clusters", "subsample_frac", "error",
    ]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]


def validate_ground_truth_recovery(
    *,
    sharpness_levels: Sequence = ("sharp", "medium", "gradual"),
    n_cells: int = 3000,
    unit1: str = "Inner Follicle",
    unit2: str = "Outer Follicle",
    k: int = 20,
    n_bins: int = 5,
    n_clusters: int = 5,
    n_repeats: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Check that recovered steepness tracks the known transition sharpness.

    Generates a synthetic tissue for each level in ``sharpness_levels`` (repeated
    ``n_repeats`` times with different seeds), runs the pipeline at the given
    parameters, and returns a tidy DataFrame with the ground-truth sharpness
    alongside the recovered steepness. A correct metric yields steepness that
    increases with ``ground_truth_sharpness`` (sharp > medium > gradual).
    """
    from .simulate import simulate_transition_tissue

    rows = []
    for level in sharpness_levels:
        for rep in range(n_repeats):
            adata = simulate_transition_tissue(
                n_cells=n_cells,
                sharpness=level,
                unit1=unit1,
                unit2=unit2,
                seed=seed + rep,
            )
            sim = adata.uns["sim_params"]
            res = run_gradient_transition(
                adata,
                unit1=unit1,
                unit2=unit2,
                k=k,
                n_bins=n_bins,
                n_clusters=n_clusters,
                seed=seed,
            )
            rows.append(
                {
                    "sharpness_label": sim["sharpness_label"],
                    "transition_width_frac": sim["transition_width_frac"],
                    "ground_truth_sharpness": sim["ground_truth_sharpness"],
                    "repeat": rep,
                    "steepness": res["steepness"],
                    "slope": res["slope"],
                    "slope_norm": res["slope_norm"],
                    "n_clusters_effective": res["n_clusters_effective"],
                    "error": res["error"],
                }
            )
    return pd.DataFrame(rows)
