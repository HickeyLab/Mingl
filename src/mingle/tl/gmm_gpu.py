import anndata as ad
import pandas as pd
import numpy as np
import cupy as cp
from .knn2 import KNN2


def gpu_gmm_probability(
    cells: ad.AnnData,
    centroids: ad.AnnData,
    *,
    cluster_col: str = "cell_type",
    neighborhood_col: str = "neighborhood",
    region_key: str = "unique_region",
    k: int = 10,
    batch_size: int = 20000,
    prob_key: str = "neighborhood_probability",
    prob_variable_key: str = "neighborhood_probability_neighborhoods",
):
    # -----------------------------
    # 1. Compute windows (same as CPU)
    # -----------------------------
    windows = KNN2(cells, cluster_col=cluster_col, region_key=region_key)
    win = windows[k].copy()

    # Attach cluster labels (same as CPU)
    win[cluster_col] = cells.obs[cluster_col].values

    # -----------------------------
    # 2. CRITICAL: match CPU feature order EXACTLY
    # -----------------------------
    cell_types = cells.obs[cluster_col].unique().tolist()

    mean_cols = [f"{ct}_mean" for ct in cell_types]
    std_cols  = [f"{ct}_std"  for ct in cell_types]

    # Safety check (important)
    missing = [c for c in mean_cols + std_cols if c not in centroids.var_names]
    if missing:
        raise ValueError(f"Missing centroid columns: {missing[:10]}")

    # Ensure window has all required columns
    missing_win = [ct for ct in cell_types if ct not in win.columns]
    if missing_win:
        raise ValueError(f"Missing window columns: {missing_win[:10]}")

    # Enforce exact ordering (VERY IMPORTANT)
    win_features = win[cell_types].copy()

    # -----------------------------
    # 3. Load centroids (aligned to CPU order)
    # -----------------------------
    means = cp.array(centroids[:, mean_cols].X)
    stds  = cp.array(centroids[:, std_cols].X)

    # -----------------------------
    # 4. Neighborhood names (match CPU iterrows)
    # -----------------------------
    # CPU uses index from iterrows()
    nb_names = list(centroids.to_df().index)

    # -----------------------------
    # 5. Batch computation (same math)
    # -----------------------------
    def compute_batch(df):
        data = cp.array(df.values)

        exp_cell = data[:, cp.newaxis, :]
        exp_mean = means[cp.newaxis, :, :]
        exp_std  = stds[cp.newaxis, :, :]

        safe_std = cp.where(exp_std == 0, 1e-10, exp_std)

        coeff = 1.0 / (safe_std * cp.sqrt(2 * cp.pi))
        exponent = -0.5 * ((exp_cell - exp_mean) / safe_std) ** 2
        pdf = coeff * cp.exp(exponent)

        # Handle std == 0 exactly like CPU logic
        zero_mask = exp_std == 0
        eq_mask   = exp_cell == exp_mean

        pdf = cp.where(zero_mask & eq_mask, 1, pdf)
        pdf = cp.where(zero_mask & (~eq_mask), 0, pdf)

        # Product across features
        tp = cp.prod(pdf, axis=2)

        # Normalize (same as CPU)
        norm = cp.sum(tp, axis=1, keepdims=True)
        norm = cp.where(norm == 0, 1e-10, norm)

        return (tp / norm).get()

    # -----------------------------
    # 6. Run batches
    # -----------------------------
    outputs = []
    n = len(win_features)
    batches = (n + batch_size - 1) // batch_size

    for i in range(batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n)

        print(f"GPU Processing batch {i+1}/{batches}...")

        batch_df = win_features.iloc[start:end]
        probs = compute_batch(batch_df)

        outputs.append(
            pd.DataFrame(probs, index=batch_df.index, columns=nb_names)
        )

    final = pd.concat(outputs)

    # -----------------------------
    # 7. CRITICAL: enforce row alignment with cells
    # -----------------------------
    final = final.loc[cells.obs_names]

    # -----------------------------
    # 8. Store results (same structure as before)
    # -----------------------------
    cells.obsm[prob_key] = final.values
    cells.uns[prob_variable_key] = nb_names

    cells.obs[nb_names] = pd.DataFrame(
        final.values,
        index=cells.obs_names,
        columns=nb_names,
    )

    return cells