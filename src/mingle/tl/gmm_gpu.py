import anndata as ad
import pandas as pd
import numpy as np
import cupy as cp
from .knn2 import KNN2

def gpu_gmm_probability(
    cells: ad.AnnData,
    centroids: ad.AnnData,
    *,
    cluster_col: str = "Cell Type",
    neighborhood_col: str = "Neighborhood",
    region_key: str = "unique_region",
    k: int = 10,
    batch_size: int = 20000,
):
    windows = KNN2(cells, cluster_col=cluster_col, region_key=region_key)
    win = windows[k].copy()

    win[cluster_col] = cells.obs[cluster_col].values
    cell_types = cells.obs[cluster_col].unique().tolist()

    # centroids.var.index contains names like "NK_mean", "NK_std"
    mean_cols = [f"{ct}_mean" for ct in cell_types]
    std_cols = [f"{ct}_std" for ct in cell_types]

    means = cp.array(centroids[:, mean_cols].X)
    stds = cp.array(centroids[:, std_cols].X)
    nb_names = centroids.obs.index.tolist()

    def compute_batch(df):
        data = cp.array(df[cell_types].values)

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

        return (tp / norm).get()

    n = len(win)
    batches = (n + batch_size - 1) // batch_size

    outputs = []
    for i in range(batches):
        start, end = i * batch_size, min((i + 1) * batch_size, n)
        print(f"GPU Processing batch {i+1}/{batches}...")
        df = win.iloc[start:end]
        probs = compute_batch(df)
        outputs.append(pd.DataFrame(probs, index=df.index, columns=nb_names))

    final = pd.concat(outputs).sort_index()

    cells.obsm["neighborhood_probability"] = final.values
    cells.uns["neighborhood_probability_neighborhoods"] = nb_names

    nb_names = cells.uns['neighborhood_probability_neighborhoods']
    cells.obs[nb_names] = pd.DataFrame(cells.obsm['neighborhood_probability'], index=cells.obs_names, columns=nb_names)

    return cells