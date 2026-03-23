from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import anndata as ad
from anndata import AnnData


def mergeGMM(
    GMM_adata: AnnData,
    cell_adata: AnnData,
    *,
    join: str = "outer",
) -> AnnData:
    """
    Merge GMM results AnnData with the main cell AnnData along observations.

    This is the scverse-style equivalent of the original `mergeGMM`,
    but implemented with `anndata.concat` and a clear API.

    Parameters
    ----------
    GMM_adata
        AnnData containing GMM results (e.g. neighborhood probabilities, cluster labels).
        Its `.obs_names` must match (a subset of) `cell_adata.obs_names`.
    cell_adata
        AnnData containing the main annotated dataset (cell metadata, coordinates, etc.).
    join
        How to join variables (columns) from the two AnnData objects.
        Passed to `anndata.concat(join=...)`. Common choices:
        - "outer" (default): union of variables
        - "inner": intersection of variables

    Returns
    -------
    AnnData
        New AnnData with:
          - obs: aligned cells (by `obs_names`)
          - var: combined variables from `cell_adata` and `GMM_adata`
          - X/obsm/obsp/misc merged according to `anndata.concat` rules.
    """
    # We want to align on obs (cells) and concatenate along variables.
    # axis=1 → concatenate vars, match obs by name.
    merged = ad.concat(
        {"cell": cell_adata, "gmm": GMM_adata},
        axis=1,
        join=join,
        label=None,
        merge="unique",
    )
    return merged


def findPositives(
    adata: AnnData,
    *,
    prob_key: str = "neighborhood_probabilities",
    threshold: float = 0.25,
    result_key: str = "Count_Above_Threshold",
) -> AnnData:
    """
    Count, per cell, how many neighborhood probabilities exceed a threshold.

    This is the scverse-compatible version of your original `findPositives`,
    but vectorized and using `adata.obsm[prob_key]` instead of `.obs` columns.

    Parameters
    ----------
    adata
        AnnData with a per-cell neighborhood probability matrix stored in
        `adata.obsm[prob_key]`. Shape should be (n_cells, n_neighborhoods).
    prob_key
        Key in `adata.obsm` where the neighborhood probability matrix is stored.
        Can be a NumPy array or a pandas DataFrame. If a DataFrame, its index
        should align with `adata.obs_names`.
    threshold
        Probability threshold. For each cell, we count how many neighborhood
        probabilities are strictly greater than this value.
    result_key
        Name of the column to create in `adata.obs` where the counts are stored.

    Returns
    -------
    AnnData
        The same AnnData object (mutated in-place) with a new column
        `adata.obs[result_key]` containing the counts.
    """
    if prob_key not in adata.obsm:
        raise KeyError(f"{prob_key!r} not found in adata.obsm")

    prob_raw = adata.obsm[prob_key]

    # Align with obs and get a 2D array
    if isinstance(prob_raw, pd.DataFrame):
        probs = prob_raw.reindex(adata.obs_names).to_numpy()
    else:
        probs = np.asarray(prob_raw)
        if probs.shape[0] != adata.n_obs:
            raise ValueError(
                f"adata.obsm[{prob_key!r}] has shape {probs.shape}, "
                f"but expected first dimension == n_obs ({adata.n_obs})"
            )

    # Vectorized count: how many neighborhoods per cell have p > threshold
    counts = (probs > threshold).sum(axis=1)

    # Store result in .obs
    adata.obs[result_key] = counts.astype(int)

    return adata