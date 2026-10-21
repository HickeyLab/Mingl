"""3D-capable local Score gradients (:func:`gb_local_score_gradients`, :func:`gb`).

Mirrors :mod:`mingl.tl.gb`. With ``z_key=None`` both functions delegate to the
2D originals (including their plots). With ``z_key`` set, each cell's gradient
is the least-squares plane fit of Score differences on coordinate offsets to its
``k`` nearest neighbors in 3D -- the same local linear model as 2D, one more
axis. The Score-ordering tools (``order_transition_clusters``,
``steepness_score``, ``gb_prob_bin_cluster_plots``) are coordinate-free; use
them from :mod:`mingl.tl.gb` directly.

3D rendering of the gradient field is not implemented yet (plotting pass).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..tl.gb import gb_local_score_gradients as _gb_local_score_gradients_2d
from ..tl.gb import gb_prob_bin_cluster_plots
from ._spatial import get_coords, require_columns, spatial_keys

__all__ = ["gb_local_score_gradients", "gb", "local_linear_gradients"]

_AXES = ("x", "y", "z")


def local_linear_gradients(
    coords: np.ndarray,
    scores: np.ndarray,
    neigh_idx: np.ndarray,
    *,
    batch_size: int = 50_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Least-squares local gradient of ``scores`` at every point, in any dimension.

    For point ``i`` with neighbors ``neigh_idx[i, 1:]`` (column 0 is the point
    itself), fits ``scores[j] - scores[i] ~ g . (coords[j] - coords[i]) + c`` --
    the same model as the per-cell ``LinearRegression`` in the 2D tool, solved in
    batches (minimum-norm least squares, like ``LinearRegression``).

    Points whose neighbor offsets do not span all ``d`` axes are left invalid,
    matching the 2D tool's rank check.

    Returns ``(grads, valid)`` with ``grads`` of shape ``(n, d)`` (NaN where
    invalid).
    """
    coords = np.asarray(coords, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n, d = coords.shape
    grads = np.full((n, d), np.nan, dtype=float)
    valid = np.zeros(n, dtype=bool)

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        nb = neigh_idx[start:stop, 1:]
        pts = coords[nb] - coords[start:stop, None, :]
        vals = scores[nb] - scores[start:stop, None]

        ok = np.linalg.matrix_rank(pts) >= d
        if not ok.any():
            continue
        pc = pts[ok] - pts[ok].mean(axis=1, keepdims=True)
        vc = vals[ok] - vals[ok].mean(axis=1, keepdims=True)
        beta = np.einsum("mdk,mk->md", np.linalg.pinv(pc), vc)

        rows = np.arange(start, stop)[ok]
        grads[rows] = beta
        valid[rows] = True

    return grads, valid


def gb_local_score_gradients(
    adata,
    *,
    region_key: str = "unique_region",
    region_value: Optional[str] = None,
    x_key: str = "x",
    y_key: str = "y",
    z_key: Optional[str] = None,
    score_key: str = "Score",
    score_source: str = "obs",
    score_layer: Optional[str] = None,
    k_neighbors: int = 20,
    normalize_by: str = "iqr",
    use_progress: bool = True,
    make_plots: bool = True,
    sample_for_plot: Optional[int] = None,
    figsize: Tuple[float, float] = (8, 8),
    cmap: str = "inferno",
    point_size: float = 1.0,
    alpha_pts: float = 0.95,
    vmax_pct: float = 99.0,
    out_prefix: str = "grad",
    show: bool = True,
) -> Dict[str, Any]:
    """Local Score gradients per cell, in 2D (delegates) or 3D.

    3D writes ``obs[{out_prefix}_x/_y/_z]`` (gradient components),
    ``_mag`` (Euclidean norm), ``_mag_norm`` (magnitude x median-NN distance /
    Score IQR|range|1), and ``_valid``; stores ``uns[{out_prefix}_summary]`` and
    ``uns[{out_prefix}_params]``. ``make_plots`` is ignored in 3D (with a
    warning) until the plotting pass.
    """
    if z_key is None:
        return _gb_local_score_gradients_2d(
            adata,
            region_key=region_key, region_value=region_value,
            x_key=x_key, y_key=y_key,
            score_key=score_key, score_source=score_source, score_layer=score_layer,
            k_neighbors=k_neighbors, normalize_by=normalize_by,
            use_progress=use_progress, make_plots=make_plots,
            sample_for_plot=sample_for_plot, figsize=figsize, cmap=cmap,
            point_size=point_size, alpha_pts=alpha_pts, vmax_pct=vmax_pct,
            out_prefix=out_prefix, show=show,
        )

    del use_progress, sample_for_plot, figsize, cmap, point_size, alpha_pts, vmax_pct, show
    if make_plots:
        warnings.warn(
            "gb_local_score_gradients: 3D gradient rendering is not implemented yet; "
            "skipping plots (pass make_plots=False to silence).",
            stacklevel=2,
        )

    coord_cols = spatial_keys(x_key, y_key, z_key)
    require_columns(adata.obs, [region_key, *coord_cols])

    if region_value is None:
        idx = adata.obs_names
        region_label = "__all__"
    else:
        idx = adata.obs.index[adata.obs[region_key].astype(str) == str(region_value)]
        region_label = str(region_value)
    if len(idx) == 0:
        raise ValueError(f"No cells matched region_key='{region_key}' with region_value='{region_value}'.")

    coords = get_coords(adata.obs.loc[idx], x_key, y_key, z_key, dtype=float)
    finite_mask = np.isfinite(coords).all(axis=1)
    idx, coords = idx[finite_mask], coords[finite_mask]

    if score_source == "obs":
        require_columns(adata.obs, [score_key])
        scores = pd.to_numeric(adata.obs.loc[idx, score_key], errors="coerce").to_numpy(dtype=float)
    elif score_source == "layer":
        if score_layer is None:
            raise ValueError("score_layer must be provided when score_source='layer'.")
        if score_layer not in adata.layers:
            raise KeyError(f"score_layer='{score_layer}' not found in adata.layers.")
        arr = np.asarray(adata.layers[score_layer])[adata.obs.index.get_indexer(idx)]
        if arr.ndim == 2:
            if arr.shape[1] != 1:
                raise ValueError(f"Layer '{score_layer}' has shape {arr.shape}. Expected 1 column for scalar score.")
            arr = arr[:, 0]
        scores = arr.astype(float)
    else:
        raise ValueError("score_source must be 'obs' or 'layer'.")

    keep = np.isfinite(scores)
    idx, coords, scores = idx[keep], coords[keep], scores[keep]
    n_pts, d = coords.shape
    if n_pts < 2:
        raise ValueError("Need at least 2 cells with finite coordinates and scores to compute gradients.")

    tree_all = cKDTree(coords)
    dists_nn, _ = tree_all.query(coords, k=2)
    d_med = float(np.median(dists_nn[:, 1]))
    if not np.isfinite(d_med) or d_med <= 0:
        d_med = float(np.mean(dists_nn[:, 1]) + 1e-8)

    q1, q3 = np.percentile(scores, [25, 75])
    score_iqr = float(max(1e-8, q3 - q1))
    score_rng = float(max(1e-8, float(np.nanmax(scores) - np.nanmin(scores))))
    norms = {"iqr": score_iqr, "range": score_rng, "none": 1.0}
    if normalize_by not in norms:
        raise ValueError("normalize_by must be 'iqr', 'range', or 'none'.")
    score_norm = norms[normalize_by]

    k_use = min(k_neighbors, max(1, n_pts - 1))
    _, neigh_idx = tree_all.query(coords, k=k_use + 1)

    grads, valid = local_linear_gradients(coords, scores, neigh_idx)
    grad_mag = np.linalg.norm(grads, axis=1)
    grad_mag_norm = grad_mag * (d_med / score_norm)
    grad_mag_norm[~np.isfinite(grad_mag_norm)] = np.nan

    comp_cols = [f"{out_prefix}_{a}" for a in _AXES[:d]]
    out_mag, out_mag_norm, out_valid = f"{out_prefix}_mag", f"{out_prefix}_mag_norm", f"{out_prefix}_valid"
    for col in [*comp_cols, out_mag, out_mag_norm]:
        if col not in adata.obs.columns:
            adata.obs[col] = np.nan
    if out_valid not in adata.obs.columns:
        adata.obs[out_valid] = False

    for j, col in enumerate(comp_cols):
        adata.obs.loc[idx, col] = grads[:, j]
    adata.obs.loc[idx, out_mag] = grad_mag
    adata.obs.loc[idx, out_mag_norm] = grad_mag_norm
    adata.obs.loc[idx, out_valid] = valid

    summary = {
        "region_key": region_key,
        "region_value": region_label,
        "n_dims": int(d),
        "n_points_used": int(n_pts),
        "n_valid_grad": int(valid.sum()),
        "d_med": float(d_med),
        "score_iqr": float(score_iqr),
        "score_range": float(score_rng),
        "normalize_by": normalize_by,
        "median_grad_mag_norm": float(np.nanmedian(grad_mag_norm)) if valid.any() else float("nan"),
        "mean_grad_mag_norm": float(np.nanmean(grad_mag_norm)) if valid.any() else float("nan"),
    }
    params = {
        "x_key": x_key,
        "y_key": y_key,
        "z_key": z_key,
        "score_key": score_key,
        "score_source": score_source,
        "score_layer": score_layer,
        "k_neighbors": int(k_neighbors),
    }
    adata.uns[f"{out_prefix}_summary"] = summary
    adata.uns[f"{out_prefix}_params"] = params

    return {"summary": summary, "params": params, "figs": {}}


def gb(
    adata,
    *,
    cluster_key: str = "Probability_Bin_Cluster",
    score_key: str = "Score",
    neighborhood_key: str = "Neighborhood",
    inner_name: str = "Inner Follicle",
    outer_name: str = "Outer Follicle",
    min_cells: int = 10,
    pb_prefix: str = "pb",
    region_key: str = "unique_region",
    region_value: str = "B006_Descending - Sigmoid",
    x_key: str = "x",
    y_key: str = "y",
    z_key: Optional[str] = None,
    k_neighbors: int = 20,
    normalize_by: str = "iqr",
    grad_prefix: str = "grad",
    make_plots: bool = True,
) -> Dict[str, Any]:
    """:func:`mingl.tl.gb.gb` with ``z_key`` forwarded to the gradient step."""
    out12 = gb_prob_bin_cluster_plots(
        adata,
        cluster_key=cluster_key,
        score_key=score_key,
        neighborhood_key=neighborhood_key,
        inner_name=inner_name,
        outer_name=outer_name,
        min_cells=min_cells,
        out_prefix=pb_prefix,
        make_plots=make_plots,
    )
    out3 = gb_local_score_gradients(
        adata,
        region_key=region_key,
        region_value=region_value,
        x_key=x_key,
        y_key=y_key,
        z_key=z_key,
        score_key=score_key,
        k_neighbors=k_neighbors,
        normalize_by=normalize_by,
        make_plots=make_plots and z_key is None,
        out_prefix=grad_prefix,
    )
    return {"plot12": out12, "plot3": out3}
