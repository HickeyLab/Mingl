import math
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd

from .knn2 import KNN2


def _compute_probability_batch(
    window_batch: np.ndarray,
    centroid_means: np.ndarray,
    centroid_stds: np.ndarray,
) -> np.ndarray:
    """Score one batch of cells against all centroid profiles."""
    batch_size = window_batch.shape[0]
    n_centroids, n_features = centroid_means.shape
    probabilities = np.ones((batch_size, n_centroids), dtype=np.float64)

    for feature_index in range(n_features):
        cell_values = window_batch[:, feature_index][:, np.newaxis]
        means = centroid_means[:, feature_index][np.newaxis, :]
        stds = centroid_stds[:, feature_index][np.newaxis, :]

        zero_std_mask = stds == 0
        safe_stds = np.where(zero_std_mask, 1.0, stds)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            pdf = (
                1.0
                / (safe_stds * math.sqrt(2 * math.pi))
                * np.exp(-0.5 * ((cell_values - means) / safe_stds) ** 2)
            )

        exact_match_mask = cell_values == means
        pdf = np.where(zero_std_mask & exact_match_mask, 1.0, pdf)
        pdf = np.where(zero_std_mask & ~exact_match_mask, 0.0, pdf)

        probabilities *= pdf

    probability_sums = probabilities.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        probabilities = np.divide(
            probabilities,
            probability_sums,
            out=np.zeros_like(probabilities),
            where=probability_sums != 0,
        )

    return probabilities


def _default_batch_size(n_cells: int, n_centroids: int, n_features: int) -> int:
    """Keep each temporary 3D block reasonably small for typical laptops/workstations."""
    target_elements = 4_000_000
    elements_per_row = max(1, n_centroids * n_features)
    batch_size = target_elements // elements_per_row
    return max(256, min(n_cells, min(8_192, batch_size or 256)))


def cpu_gmm_probability(
    CELLS_ADATA: ad.AnnData,
    CENTROIDS_ADATA: ad.AnnData,
    *,
    cluster_col: str = "cell_type",
    neighborhood_col: str = "neighborhood",
    region_key: str = "unique_region",
    ks: Sequence[int] = (10, 20, 100, 300),
    k: int = 10,
    threshold: float = 0.25,
    num_processes: Optional[int] = None,
    prob_key: str = "neighborhood_probabilities",
    prob_variable_key: str = "neighborhood_probability_neighborhoods",
) -> ad.AnnData:
    """
    Calculate per-cell neighborhood membership probabilities on CPU.

    The public API is unchanged, but the implementation now uses batched NumPy
    evaluation instead of shipping the full window table to a separate process
    for every cell. This keeps the default path safe on Windows while still
    allowing optional parallel workers for users who want them.
    """
    del threshold  # retained for API compatibility

    if num_processes is not None and num_processes < 1:
        raise ValueError("num_processes must be at least 1 when provided.")

    if neighborhood_col not in CELLS_ADATA.obs or cluster_col not in CELLS_ADATA.obs:
        raise KeyError(f"One or more required columns ({neighborhood_col}, {cluster_col}) are missing in obs.")

    windows = KNN2(CELLS_ADATA, cluster_col=cluster_col, region_key=region_key, ks=ks)
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
        preview = ", ".join(missing_centroid_cols[:10])
        raise ValueError(f"Missing centroid feature columns: {preview}")

    missing_window_cols = [cell_type for cell_type in cell_type_features if cell_type not in windows_k.columns]
    if missing_window_cols:
        preview = ", ".join(map(str, missing_window_cols[:10]))
        raise ValueError(f"Missing KNN window columns for cell types: {preview}")

    window_values = np.ascontiguousarray(
        windows_k.loc[:, cell_type_features].to_numpy(dtype=np.float64, copy=False)
    )
    centroid_means = np.ascontiguousarray(
        centroid_df.loc[:, mean_cols].to_numpy(dtype=np.float64, copy=False)
    )
    centroid_stds = np.ascontiguousarray(
        centroid_df.loc[:, std_cols].to_numpy(dtype=np.float64, copy=False)
    )
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
            max_workers = min(num_processes, len(batches))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        _compute_probability_batch,
                        window_values[start:stop],
                        centroid_means,
                        centroid_stds,
                    )
                    for start, stop in batches
                ]
                outputs = [future.result() for future in futures]

        probabilities = np.vstack(outputs)

    probabilities_df = pd.DataFrame(
        probabilities,
        index=windows_k.index,
        columns=neighborhood_names,
    )

    CELLS_ADATA.obsm[prob_key] = probabilities_df.values
    CELLS_ADATA.uns[prob_variable_key] = neighborhood_names

    return CELLS_ADATA
