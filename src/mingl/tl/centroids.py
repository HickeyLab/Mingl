import anndata as ad
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from typing import Dict, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd

from .knn2 import KNN2


def centroid_Calculation(
    adata: ad.AnnData,
    *,
    k: int = 10,
    cluster_col: str = "cell_type",
    neighborhood_col: str = "neighborhood",
    region_col: str = "unique_region",
    store_key: Optional[str] = None,
) -> ad.AnnData:
    """
    Compute per-neighborhood mean and std of cell-type counts in k-NN windows.

    This is the same computation you had before, but:

    - keeps indices aligned (no reset_index)
    - allows choosing k
    - optionally stores the centroid AnnData in `adata.uns[store_key]`
      so you can reuse it for many plots.

    Parameters
    ----------
    adata
        AnnData with:
          - `.obs[cluster_col]` (cell type labels)
          - `.obs[neighborhood_col]` (neighborhood assignment)
          - spatial info needed for KNN (x/y & region columns).
    k
        Neighborhood size to use from the KNN windows (must be in the `ks`
        list passed to KNN; default 10).
    cluster_col
        Column in `adata.obs` with cell-type labels.
    neighborhood_col
        Column in `adata.obs` with neighborhood IDs.
    store_key
        If not None, store the centroid AnnData in `adata.uns[store_key]`.

    Returns
    -------
    AnnData
        AnnData with:
          - obs: neighborhoods
          - var: centroid features (means/stds per cell type)
          - X: numeric matrix (n_neighborhoods x n_features)
    """
    # get KNN windows
    windows = KNN2(adata, region_key=region_col, cluster_col=cluster_col)
    print(windows)
    if k not in windows:
        raise ValueError(f"k={k} not in available ks from KNN: {list(windows.keys())}")

    windows_k = windows[k]

    # ensure we have the cluster column on windows_k
    windows_k[cluster_col] = adata.obs[cluster_col]

    # use obs directly; keep original indices (no reset_index)
    filtered_cells = adata.obs.copy()

    # cell types → columns we created in KNN
    cell_type_columns = adata.obs[cluster_col].unique()
    print(cell_type_columns)
    print(type(cell_type_columns))
    windows_k[cell_type_columns] = windows_k[cell_type_columns].astype("float32")

    neighborhoods_to_loop = adata.obs[neighborhood_col].unique()
    all_results = []

    for neighborhood in neighborhoods_to_loop:
        # cells in this neighborhood (indices are original obs index)
        filtered_neighborhood_df = filtered_cells[
            filtered_cells[neighborhood_col] == neighborhood
        ]
        cell_numbers_in_neighborhood = filtered_neighborhood_df.index.values

        # take matching rows from windows_k
        matching_cells_df = windows_k.loc[cell_numbers_in_neighborhood]

        mean_std_results = {neighborhood_col: neighborhood}

        for column in cell_type_columns:
            if column in matching_cells_df.columns:
                col_values = matching_cells_df[column]
                mean_std_results[f"{column}_mean"] = col_values.mean()
                mean_std_results[f"{column}_std"] = col_values.std()

        all_results.append(mean_std_results)

    # neighborhood × feature table
    results_df = pd.DataFrame(all_results).set_index(neighborhood_col)

    feature_cols = results_df.columns.tolist()
    X = results_df[feature_cols].to_numpy(dtype=np.float64)

    # obs: neighborhoods
    obs = pd.DataFrame(index=results_df.index)
    obs[neighborhood_col] = results_df.index

    # var: feature names
    var = pd.DataFrame(index=feature_cols)

    centroid_adata = ad.AnnData(X=X, obs=obs, var=var)

    if store_key is not None:
        adata.uns[store_key] = centroid_adata

    return centroid_adata