"""3D-capable neighborhood centroids and GMM membership probabilities.

Mirrors :func:`mingl.tl.centroids.centroid_Calculation`,
:func:`mingl.tl.gmm.cpu_gmm_probability` and
:func:`mingl.tl.gmm_gpu.gpu_gmm_probability`. The originals call ``KNN2``
without coordinate keys (so they only work with columns literally named
``x``/``y``); these versions take ``x_key``/``y_key``/``z_key`` and pass them
through. The probability math is imported unchanged from :mod:`mingl.tl.gmm`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd

from ..tl.gmm import _compute_probability_batch, _default_batch_size
from .knn import KNN2

__all__ = ["centroid_Calculation", "cpu_gmm_probability", "gpu_gmm_probability"]

# KNN2's default ks; the originals relied on it implicitly.
_KNN2_DEFAULT_KS = (5, 10, 20, 100, 300)


def centroid_Calculation(
    adata: ad.AnnData,
    *,
    k: int = 10,
    cluster_col: str = "cell_type",
    neighborhood_col: str = "neighborhood",
    region_col: str = "unique_region",
    x_key: str = "x",
    y_key: str = "y",
    z_key: Optional[str] = None,
    ks: Sequence[int] = _KNN2_DEFAULT_KS,
    store_key: Optional[str] = None,
) -> ad.AnnData:
    """Per-neighborhood mean and std of k-NN cell-type counts, in 2D or 3D.

    Returns an AnnData with neighborhoods as ``obs`` and ``<cell type>_mean`` /
    ``<cell type>_std`` features as ``var``; optionally stored in
    ``adata.uns[store_key]``.
    """
    windows = KNN2(
        adata,
        x_key=x_key,
        y_key=y_key,
        z_key=z_key,
        region_key=region_col,
        cluster_col=cluster_col,
        ks=ks,
    )
    if k not in windows:
        raise ValueError(f"k={k} not in available ks from KNN2: {list(windows.keys())}")

    windows_k = windows[k]
    windows_k[cluster_col] = adata.obs[cluster_col]

    filtered_cells = adata.obs.copy()
    cell_type_columns = adata.obs[cluster_col].unique()
    windows_k[cell_type_columns] = windows_k[cell_type_columns].astype("float32")

    all_results = []
    for neighborhood in adata.obs[neighborhood_col].unique():
        in_neighborhood = filtered_cells[filtered_cells[neighborhood_col] == neighborhood].index.values
        matching_cells_df = windows_k.loc[in_neighborhood]

        mean_std_results = {neighborhood_col: neighborhood}
        for column in cell_type_columns:
            if column in matching_cells_df.columns:
                col_values = matching_cells_df[column]
                mean_std_results[f"{column}_mean"] = col_values.mean()
                mean_std_results[f"{column}_std"] = col_values.std()
        all_results.append(mean_std_results)

    results_df = pd.DataFrame(all_results).set_index(neighborhood_col)
    feature_cols = results_df.columns.tolist()
    X = results_df[feature_cols].to_numpy(dtype=np.float64)

    obs = pd.DataFrame(index=results_df.index)
    obs[neighborhood_col] = results_df.index
    var = pd.DataFrame(index=feature_cols)
    centroid_adata = ad.AnnData(X=X, obs=obs, var=var)

    if store_key is not None:
        adata.uns[store_key] = centroid_adata
    return centroid_adata


def cpu_gmm_probability(
    CELLS_ADATA: ad.AnnData,
    CENTROIDS_ADATA: ad.AnnData,
    *,
    cluster_col: str = "cell_type",
    neighborhood_col: str = "neighborhood",
    region_key: str = "unique_region",
    x_key: str = "x",
    y_key: str = "y",
    z_key: Optional[str] = None,
    ks: Sequence[int] = (10, 20, 100, 300),
    k: int = 10,
    threshold: float = 0.25,
    num_processes: Optional[int] = None,
    prob_key: str = "neighborhood_probabilities",
    prob_variable_key: str = "neighborhood_probability_neighborhoods",
) -> ad.AnnData:
    """Per-cell neighborhood membership probabilities on CPU, in 2D or 3D.

    Same outputs as :func:`mingl.tl.gmm.cpu_gmm_probability`:
    ``obsm[prob_key]`` (cells x neighborhoods) and ``uns[prob_variable_key]``.
    """
    del threshold  # retained for API compatibility

    if num_processes is not None and num_processes < 1:
        raise ValueError("num_processes must be at least 1 when provided.")
    if neighborhood_col not in CELLS_ADATA.obs or cluster_col not in CELLS_ADATA.obs:
        raise KeyError(f"One or more required columns ({neighborhood_col}, {cluster_col}) are missing in obs.")

    windows = KNN2(
        CELLS_ADATA,
        x_key=x_key,
        y_key=y_key,
        z_key=z_key,
        cluster_col=cluster_col,
        region_key=region_key,
        ks=ks,
    )
    if k not in windows:
        raise ValueError(f"k={k} not in available ks from KNN2: {list(windows.keys())}")

    windows_k = windows[k].copy()
    windows_k[cluster_col] = CELLS_ADATA.obs[cluster_col].values

    cell_type_features = list(pd.Index(CELLS_ADATA.obs[cluster_col]).unique())
    centroid_df = CENTROIDS_ADATA.to_df()
    mean_cols = [f"{cell_type}_mean" for cell_type in cell_type_features]
    std_cols = [f"{cell_type}_std" for cell_type in cell_type_features]

    missing_centroid_cols = [col for col in mean_cols + std_cols if col not in centroid_df.columns]
    if missing_centroid_cols:
        raise ValueError(f"Missing centroid feature columns: {', '.join(missing_centroid_cols[:10])}")
    missing_window_cols = [ct for ct in cell_type_features if ct not in windows_k.columns]
    if missing_window_cols:
        raise ValueError(f"Missing KNN window columns for cell types: {', '.join(map(str, missing_window_cols[:10]))}")

    window_values = np.ascontiguousarray(windows_k.loc[:, cell_type_features].to_numpy(dtype=np.float64, copy=False))
    centroid_means = np.ascontiguousarray(centroid_df.loc[:, mean_cols].to_numpy(dtype=np.float64, copy=False))
    centroid_stds = np.ascontiguousarray(centroid_df.loc[:, std_cols].to_numpy(dtype=np.float64, copy=False))
    neighborhood_names = centroid_df.index.tolist()

    n_cells = window_values.shape[0]
    if n_cells == 0:
        probabilities = np.empty((0, len(neighborhood_names)), dtype=np.float64)
    else:
        batch_size = _default_batch_size(
            n_cells=n_cells,
            n_centroids=centroid_means.shape[0],
            n_features=centroid_means.shape[1],
        )
        batches = [(start, min(start + batch_size, n_cells)) for start in range(0, n_cells, batch_size)]

        if num_processes is None or num_processes == 1 or len(batches) == 1:
            outputs = [
                _compute_probability_batch(window_values[start:stop], centroid_means, centroid_stds)
                for start, stop in batches
            ]
        else:
            with ThreadPoolExecutor(max_workers=min(num_processes, len(batches))) as executor:
                futures = [
                    executor.submit(_compute_probability_batch, window_values[start:stop], centroid_means, centroid_stds)
                    for start, stop in batches
                ]
                outputs = [future.result() for future in futures]
        probabilities = np.vstack(outputs)

    CELLS_ADATA.obsm[prob_key] = probabilities
    CELLS_ADATA.uns[prob_variable_key] = neighborhood_names
    return CELLS_ADATA


def gpu_gmm_probability(
    cells: ad.AnnData,
    centroids: ad.AnnData,
    *,
    cluster_col: str = "cell_type",
    neighborhood_col: str = "neighborhood",
    region_key: str = "unique_region",
    x_key: str = "x",
    y_key: str = "y",
    z_key: Optional[str] = None,
    ks: Sequence[int] = _KNN2_DEFAULT_KS,
    k: int = 10,
    batch_size: int = 20000,
    prob_key: str = "neighborhood_probability",
    prob_variable_key: str = "neighborhood_probability_neighborhoods",
):
    """GPU (CuPy) version of :func:`cpu_gmm_probability`, in 2D or 3D.

    Mirrors :func:`mingl.tl.gmm_gpu.gpu_gmm_probability`, including its output
    keys and the per-neighborhood probability columns written to ``obs``.
    CuPy is imported lazily so this module loads on machines without a GPU.
    """
    import cupy as cp

    del neighborhood_col  # accepted for API symmetry, as in the original

    windows = KNN2(
        cells,
        x_key=x_key,
        y_key=y_key,
        z_key=z_key,
        cluster_col=cluster_col,
        region_key=region_key,
        ks=ks,
    )
    win = windows[k].copy()
    win[cluster_col] = cells.obs[cluster_col].values

    cell_types = cells.obs[cluster_col].unique().tolist()
    mean_cols = [f"{ct}_mean" for ct in cell_types]
    std_cols = [f"{ct}_std" for ct in cell_types]

    missing = [c for c in mean_cols + std_cols if c not in centroids.var_names]
    if missing:
        raise ValueError(f"Missing centroid columns: {missing[:10]}")
    missing_win = [ct for ct in cell_types if ct not in win.columns]
    if missing_win:
        raise ValueError(f"Missing window columns: {missing_win[:10]}")

    win_features = win[cell_types].copy()
    means = cp.array(centroids[:, mean_cols].X)
    stds = cp.array(centroids[:, std_cols].X)
    nb_names = list(centroids.to_df().index)

    def compute_batch(df):
        data = cp.array(df.values)
        exp_cell = data[:, cp.newaxis, :]
        exp_mean = means[cp.newaxis, :, :]
        exp_std = stds[cp.newaxis, :, :]

        safe_std = cp.where(exp_std == 0, 1e-10, exp_std)
        coeff = 1.0 / (safe_std * cp.sqrt(2 * cp.pi))
        exponent = -0.5 * ((exp_cell - exp_mean) / safe_std) ** 2
        pdf = coeff * cp.exp(exponent)

        zero_mask = exp_std == 0
        eq_mask = exp_cell == exp_mean
        pdf = cp.where(zero_mask & eq_mask, 1, pdf)
        pdf = cp.where(zero_mask & (~eq_mask), 0, pdf)

        tp = cp.prod(pdf, axis=2)
        norm = cp.sum(tp, axis=1, keepdims=True)
        norm = cp.where(norm == 0, 1e-10, norm)
        return (tp / norm).get()

    outputs = []
    n = len(win_features)
    for start in range(0, n, batch_size):
        batch_df = win_features.iloc[start:min(start + batch_size, n)]
        outputs.append(pd.DataFrame(compute_batch(batch_df), index=batch_df.index, columns=nb_names))

    final = pd.concat(outputs).loc[cells.obs_names]

    cells.obsm[prob_key] = final.values
    cells.uns[prob_variable_key] = nb_names
    cells.obs[nb_names] = pd.DataFrame(final.values, index=cells.obs_names, columns=nb_names)
    return cells
