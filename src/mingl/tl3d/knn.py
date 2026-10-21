"""3D-capable k-NN cell-type windows (:func:`KNN2`, legacy :func:`KNN`).

Same algorithm as :mod:`mingl.tl.knn2` / :mod:`mingl.tl.knn`; the only change is
that neighbors are found in ``(x, y)`` or ``(x, y, z)`` via ``z_key``. With
``z_key=None`` the output is identical to the originals.

A ``region_key`` value must be one contiguous tissue volume. Never split a
serial-section stack into one region per section, or neighbors are cut off at
section boundaries.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ._spatial import get_coords, require_columns, spatial_keys

__all__ = ["KNN", "KNN2"]


def KNN2(
    adata: ad.AnnData,
    *,
    x_key: str = "x",
    y_key: str = "y",
    z_key: Optional[str] = None,
    region_key: str = "unique_region",
    cluster_col: str = "cell_type",
    ks: Sequence[int] = (5, 10, 20, 100, 300),
    keep_obs_cols: Optional[Sequence[str]] = None,
) -> Dict[int, pd.DataFrame]:
    """Compute cell-type neighborhood windows using a k-NN in (x, y[, z]) per region.

    Parameters
    ----------
    adata
        AnnData with coordinates in ``.obs``, region IDs in ``.obs[region_key]``
        and cell-type labels in ``.obs[cluster_col]``. As in
        :func:`mingl.tl.knn2.KNN2`, ``obs_names`` must be integer-castable.
    x_key, y_key, z_key
        Coordinate columns. ``z_key=None`` gives the 2D windows of
        :func:`mingl.tl.knn2.KNN2`.
    keep_obs_cols
        Extra ``obs`` columns to carry into each window frame (the coordinate,
        region and cluster columns are always carried).

    Returns
    -------
    dict[int, pd.DataFrame]
        ``k -> window frame`` (rows in ``adata.obs`` order, one float32 count
        column per cell type).
    """
    ks = list(ks)
    n_neighbors = max(ks)
    coord_cols = spatial_keys(x_key, y_key, z_key)
    require_columns(adata.obs, [*coord_cols, region_key, cluster_col])

    adata = adata.copy()
    if adata.obs.columns.has_duplicates:
        counts = {}
        new_cols = []
        for c in adata.obs.columns:
            if c not in counts:
                counts[c] = 0
                new_cols.append(c)
            else:
                counts[c] += 1
                new_cols.append(f"{c}__dup{counts[c]}")
        adata.obs.columns = pd.Index(new_cols)

    if keep_obs_cols is None:
        keep_obs_cols = []

    keep_cols: List[str] = []
    for c in list(keep_obs_cols) + [*coord_cols, region_key, cluster_col]:
        if c not in keep_cols:
            keep_cols.append(c)
    keep_cols_existing = [c for c in keep_cols if c in adata.obs.columns]

    dummies = pd.get_dummies(adata.obs[cluster_col]).add_prefix(f"{cluster_col}__")
    adata.obs = pd.concat([adata.obs, dummies], axis=1)

    sum_cols = dummies.columns.to_numpy()
    values = adata.obs[sum_cols].to_numpy(dtype=np.float32)

    tissue_group = adata.obs.groupby(region_key)
    exps = list(adata.obs[region_key].unique())

    def get_windows(job, n_neighbors: int) -> np.ndarray:
        _start_time, _idx, tissue_name, indices = job

        region = adata.obs[region_key]
        if isinstance(region, pd.DataFrame):
            region = region.iloc[:, 0]

        mask = region.to_numpy() == tissue_name
        tissue = adata[mask].copy()

        coords = get_coords(tissue.obs, x_key, y_key, z_key)
        fit = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
        distances, neighbor_idx = fit.kneighbors(coords)

        args = distances.argsort(axis=1)
        add = np.arange(neighbor_idx.shape[0]) * neighbor_idx.shape[1]
        sorted_indices = neighbor_idx.flatten()[args + add[:, None]]

        obs_index = np.asarray(tissue.obs.index.to_numpy())
        neighbors = obs_index[sorted_indices]
        return neighbors.astype(np.int32)

    tissue_chunks = [
        (time.time(), exps.index(tissue_name), tissue_name, indices)
        for tissue_name, indices in tissue_group.groups.items()
        for indices in np.array_split(indices, 1)
    ]

    tissues = [get_windows(job, n_neighbors) for job in tissue_chunks]

    out_dict = {}
    for k in ks:
        for neighbors, job in zip(tissues, tissue_chunks):
            tissue_name = job[2]
            indices = job[3]
            chunk = np.arange(len(neighbors))

            window = values[neighbors[chunk, :k].flatten()]
            window = window.reshape(len(chunk), k, len(sum_cols)).sum(axis=1)
            out_dict[(tissue_name, k)] = (window.astype(np.float32), indices)

    pretty_cols = [c.removeprefix(f"{cluster_col}__") for c in sum_cols]
    windows: Dict[int, pd.DataFrame] = {}
    for k in ks:
        dfs = []
        for exp in exps:
            data, idx = out_dict[(exp, k)]
            dfs.append(pd.DataFrame(data, index=idx.astype(int), columns=pretty_cols))

        window_df = pd.concat(dfs, axis=0)
        if window_df.index.dtype != adata.obs.index.dtype:
            window_df.index = window_df.index.astype(str)
        window_df = window_df.loc[adata.obs.index]

        window_df = pd.concat([adata.obs[keep_cols_existing], window_df], axis=1)
        windows[k] = window_df

    return windows


def KNN(
    adata: ad.AnnData,
    *,
    x_key: str = "x",
    y_key: str = "y",
    z_key: Optional[str] = None,
    region_key: str = "unique_region",
    cluster_col: str = "cell_type",
    ks: Sequence[int] = (5, 10, 20),
) -> Dict[int, pd.DataFrame]:
    """Legacy window builder (:func:`mingl.tl.knn.KNN`) with an optional ``z_key``.

    Prefer :func:`KNN2`. Kept for parity: identical output to the original when
    ``z_key=None`` (including its float16 window counts).
    """
    ks = list(ks)
    n_neighbors = max(ks)
    require_columns(adata.obs, [*spatial_keys(x_key, y_key, z_key), region_key, cluster_col])

    adata = adata.copy()

    dummies = pd.get_dummies(adata.obs[cluster_col])
    adata.obs = pd.concat([adata.obs, dummies], axis=1)

    sum_cols = adata.obs[cluster_col].unique()
    values = adata.obs[sum_cols].values

    tissue_group = adata.obs.groupby(region_key)
    exps = list(adata.obs[region_key].unique())

    def get_windows(job, n_neighbors: int) -> np.ndarray:
        _start_time, _idx, tissue_name, indices = job
        tissue = adata[adata.obs[region_key] == tissue_name]
        coords = get_coords(tissue.obs, x_key, y_key, z_key)
        fit = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
        distances, neighbor_idx = fit.kneighbors(coords)

        args = distances.argsort(axis=1)
        add = np.arange(neighbor_idx.shape[0]) * neighbor_idx.shape[1]
        sorted_indices = neighbor_idx.flatten()[args + add[:, None]]

        neighbors = tissue.obs.index.values[sorted_indices]
        return neighbors.astype(np.int32)

    tissue_chunks = [
        (time.time(), exps.index(tissue_name), tissue_name, indices)
        for tissue_name, indices in tissue_group.groups.items()
        for indices in np.array_split(indices, 1)
    ]

    tissues = [get_windows(job, n_neighbors) for job in tissue_chunks]

    out_dict = {}
    for k in ks:
        for neighbors, job in zip(tissues, tissue_chunks):
            tissue_name = job[2]
            indices = job[3]
            chunk = np.arange(len(neighbors))

            window = values[neighbors[chunk, :k].flatten()]
            window = window.reshape(len(chunk), k, len(sum_cols)).sum(axis=1)
            out_dict[(tissue_name, k)] = (window.astype(np.float16), indices)

    windows: Dict[int, pd.DataFrame] = {}
    for k in ks:
        dfs = []
        for exp in exps:
            data, idx = out_dict[(exp, k)]
            dfs.append(pd.DataFrame(data, index=idx.astype(int), columns=sum_cols))

        window_df = pd.concat(dfs, axis=0)
        if window_df.index.dtype != adata.obs.index.dtype:
            window_df.index = window_df.index.astype(str)
        window_df = window_df.loc[adata.obs.index]

        window_df = pd.concat([adata.obs[[region_key, cluster_col]], window_df], axis=1)
        windows[k] = window_df

    return windows
