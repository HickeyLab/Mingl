import anndata as ad
from typing import Dict, Optional, Sequence
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time

def KNN(
    adata: ad.AnnData,
    *,
    x_key: str = "x",
    y_key: str = "y",
    region_key: str = "unique_region",
    cluster_col: str = "cell_type",
    ks: Sequence[int] = (5, 10, 20),
) -> Dict[int, pd.DataFrame]:
    """
    Compute cell-type neighborhood windows using a k-NN in (x,y) per region.

    Parameters
    ----------
    adata
        AnnData with spatial coordinates in `.obs[x_key]`, `.obs[y_key]`,
        region IDs in `.obs[region_key]`, and cluster labels in `.obs[cluster_col]`.
    x_key, y_key
        Names of the columns in `adata.obs` containing x/y coordinates.
    region_key
        Column in `adata.obs` defining regions / tissues (e.g. 'unique_region').
    cluster_col
        Column in `adata.obs` with cell type / cluster labels.
    ks
        Sequence of neighborhood sizes to compute.

    Returns
    -------
    dict[int, pd.DataFrame]
        Mapping k -> DataFrame of neighborhood summaries.
    """
    ks = list(ks)
    n_neighbors = max(ks)

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
        coords = tissue.obs[[x_key, y_key]].values
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
            df = pd.DataFrame(data, index=idx.astype(int), columns=sum_cols)
            dfs.append(df)

        window_df = pd.concat(dfs, axis=0)

        # Ensure the index types match
        if window_df.index.dtype != adata.obs.index.dtype:
            window_df.index = window_df.index.astype(str)  # Convert to string if needed

        window_df = window_df.loc[adata.obs.index]  # Re-align the index with adata.obs.index

        window_df = pd.concat(
            [adata.obs[[region_key, cluster_col]], window_df],
            axis=1,
        )
        windows[k] = window_df

    return windows