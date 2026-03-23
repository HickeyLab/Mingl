import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import anndata as ad

def crd(
    cells: ad.AnnData,
    windows2: pd.DataFrame,
    probabilities_df: pd.DataFrame,
    cell_type_features,
    *,
    out_probs_path: str = "local_probs.csv",
    out_delta_path: str = "delta_probs.csv",
    copy_cells: pd.DataFrame | None = None,
):
    """
    crd

    Compare, for each region, LOCAL (region-only) MINGL probabilities vs GLOBAL (all-cells) MINGL probabilities.

    Inputs
    ------
    cells : anndata.AnnData
        AnnData with required metadata in `.obs` including:
        - 'cellid', 'region', 'neigh_name'
        (and any other columns you use elsewhere).
    windows2 : pd.DataFrame
        Output window dataframe for a chosen k (e.g., windows[k]) containing:
        - 'cellid', 'region', and all columns in `cell_type_features`
        Index should align to `cells.obs.index` (same as your pipeline).
    probabilities_df : pd.DataFrame
        Global MINGL probabilities computed over all cells together.
        Must have a 'cellid' column plus neighborhood columns.
    cell_type_features : list[str]
        Feature columns used for likelihood computation (cell-type composition features).
    out_probs_path : str
        Where to save local probabilities CSV.
    out_delta_path : str
        Where to save delta probabilities CSV.
    copy_cells : pd.DataFrame | None
        If provided, used for index-aligned neigh_name assignment.
        If None, uses cells.obs.copy().

    Returns
    -------
    (final_probs_df, final_deltas_df) : tuple[pd.DataFrame, pd.DataFrame]
        final_probs_df columns: neighborhoods + ['cellid','region','neigh_name']
        final_deltas_df columns: [f"{n}_delta" ...] + ['cellid','region','neigh_name']
    """
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    # neighborhoods list (same logic)
    neighborhoods = probabilities_df.columns.tolist()
    if 'cellid' in neighborhoods:
        neighborhoods.remove('cellid')
    if 'Unnamed: 0' in neighborhoods:
        neighborhoods.remove('Unnamed: 0')

    # assigned_df validation (same logic)
    assigned_df = cells.obs.set_index("cellid")
    required_assigned_cols = {"neigh_name", "region"}
    if not required_assigned_cols.issubset(set(assigned_df.columns)):
        raise ValueError(f"assigned_df must contain columns: {required_assigned_cols}")

    # CPU helper: logsumexp (same logic)
    def cp_logsumexp(a, axis=1, keepdims=True):
        a_max = np.max(a, axis=axis, keepdims=True)
        a_max = np.where(np.isfinite(a_max), a_max, -1e300)
        s = np.sum(np.exp(a - a_max), axis=axis, keepdims=True)
        s = np.where(s == 0, np.nan, s)
        return a_max + np.log(s)

    # Prepare storage (same logic)
    all_region_probs = []
    all_region_deltas = []
    global_idxed = probabilities_df.set_index('cellid')

    # Use copy_cells DataFrame (obs copy) for neigh_name assignment
    if copy_cells is None:
        copy_cells = cells.obs.copy()

    # Run loop (same logic)
    unique_regions = windows2['region'].unique()
    print("Processing regions:", len(unique_regions))

    for region in tqdm(unique_regions, desc="Processing regions (CPU log-space)"):
        region_cells = windows2[windows2['region'] == region].copy()
        region_cell_ids = region_cells['cellid'].values

        cell_data = region_cells[cell_type_features].copy()
        C = cell_data.shape[0]
        if C == 0:
            print(f"  - Region {region}: no cells, skipping")
            continue

        # neigh_name assignment (same logic)
        try:
            region_cells['neigh_name'] = copy_cells.loc[region_cells.index, 'neigh_name'].values
        except Exception:
            if 'cellid' in copy_cells.columns:
                mapping = copy_cells.set_index('cellid')['neigh_name'].to_dict()
                region_cells['neigh_name'] = region_cells['cellid'].map(mapping).values
            else:
                raise

        # region-specific centroids (same logic)
        region_results = []
        neigh_counts = []
        for neighborhood in neighborhoods:
            neighborhood_cells = region_cells[region_cells['neigh_name'] == neighborhood]
            matching_cell_ids = neighborhood_cells.index
            neigh_counts.append(len(matching_cell_ids))
            stats = {"Neighborhood": neighborhood}

            if len(matching_cell_ids) <= 1:
                for col in cell_type_features:
                    stats[f"{col}_mean"] = np.nan
                    stats[f"{col}_std"] = np.nan
            else:
                for col in cell_type_features:
                    stats[f"{col}_mean"] = neighborhood_cells[col].mean()
                    std_val = neighborhood_cells[col].std(ddof=0)
                    stats[f"{col}_std"] = np.nan if pd.isna(std_val) else std_val

            region_results.append(stats)

        df_region_centroids = pd.DataFrame(region_results)

        centroid_means = df_region_centroids[[f"{c}_mean" for c in cell_type_features]].values.astype(float)
        centroid_stds  = df_region_centroids[[f"{c}_std"  for c in cell_type_features]].values.astype(float)

        global_mean_fallback = np.nanmean(centroid_means, axis=0)
        global_std_fallback = np.nanmedian(np.where(np.isnan(centroid_stds), np.nan, centroid_stds), axis=0)
        global_std_fallback = np.where(
            np.isnan(global_std_fallback) | (global_std_fallback <= 0),
            1e-1,
            global_std_fallback,
        )

        neigh_counts = np.array(neigh_counts)
        low_mask = neigh_counts <= 1
        if low_mask.any():
            centroid_means[low_mask, :] = global_mean_fallback[None, :]
            centroid_stds[low_mask, :] = global_std_fallback[None, :]

        centroid_stds = np.where(centroid_stds <= 0, 1e-6, centroid_stds)

        region_means_cp = np.array(centroid_means, dtype=np.float64)
        region_stds_cp  = np.array(centroid_stds,  dtype=np.float64)
        cell_array_cp   = np.array(cell_data.values.astype(np.float64), dtype=np.float64)

        Xarr = cell_array_cp[:, None, :]
        Marr = region_means_cp[None, :, :]
        Sarr = region_stds_cp[None, :, :]

        log_coeff = -0.5 * np.log(2.0 * np.pi * (Sarr ** 2))
        exponent  = -0.5 * ((Xarr - Marr) / Sarr) ** 2
        log_pdf   = log_coeff + exponent

        log_total = np.sum(log_pdf, axis=2)

        row_logsum = cp_logsumexp(log_total, axis=1, keepdims=True)
        log_prob   = log_total - row_logsum
        probs_cp   = np.exp(log_prob)

        local_probs_np = probs_cp

        row_sums = np.nansum(local_probs_np, axis=1)
        n_nan_rows = int(np.sum(np.isnan(row_sums)))
        n_zero_rows = int(np.sum(np.isclose(row_sums, 0.0, atol=1e-12)))
        print(f"Region {region}: cells={C}, NaN-sum-rows={n_nan_rows}, zero-sum-rows={n_zero_rows}")

        # Global probs alignment (same logic)
        common_ids = [cid for cid in region_cell_ids if cid in global_idxed.index]
        if len(common_ids) != len(region_cell_ids):
            global_probs = np.full((len(region_cell_ids), len(neighborhoods)), np.nan, dtype=float)
            present_mask = [cid in global_idxed.index for cid in region_cell_ids]
            if any(present_mask):
                for i, cid in enumerate(region_cell_ids):
                    if cid in global_idxed.index:
                        global_probs[i, :] = global_idxed.loc[cid, neighborhoods].values
        else:
            global_probs = global_idxed.loc[region_cell_ids, neighborhoods].values

        # Local probs df (same logic)
        local_probs_df = pd.DataFrame(local_probs_np, columns=neighborhoods)
        local_probs_df['cellid'] = region_cells['cellid'].values
        local_probs_df['region'] = region
        local_probs_df['neigh_name'] = region_cells['neigh_name'].values
        all_region_probs.append(local_probs_df)

        if global_probs.shape != local_probs_np.shape:
            pass

        delta_values = local_probs_np - global_probs
        delta_df = pd.DataFrame(delta_values, columns=[f"{n}_delta" for n in neighborhoods])
        delta_df["cellid"] = region_cells['cellid'].values
        delta_df["region"] = region
        delta_df["neigh_name"] = region_cells["neigh_name"].values
        all_region_deltas.append(delta_df)

    # Save outputs (same logic)
    final_probs_df = pd.concat(all_region_probs, ignore_index=True)
    final_deltas_df = pd.concat(all_region_deltas, ignore_index=True)

    final_probs_df.to_csv(out_probs_path, index=False)
    final_deltas_df.to_csv(out_delta_path, index=False)

    print("✅ Done. Saved region-level local probs to:", out_probs_path)
    print("✅ Done. Saved region-level delta probs to:", out_delta_path)

    return final_probs_df, final_deltas_df

def crd2(
    *,
    windows2: pd.DataFrame,
    probabilities_df: pd.DataFrame,
    assigned_df: pd.DataFrame,
    neighborhoods: list,
    cell_type_features: list,
    copy_cells: pd.DataFrame,
    out_probs_path: str = "local_probs.csv",
    out_delta_path: str = "delta_probs.csv",
    # PARAMETERS YOU REQUESTED (defaults preserved)
    min_count: int = 10,
    min_floor_abs: float = 0.05,
    use_region_frac: float = 0.25,
    shrink_alpha: float = 5.0,
):
    """
    crd

    Scverse-compatible wrapper for your region-vs-global MINGL comparison.
    - Accepts scverse-friendly data structures (AnnData-derived obs as DataFrames).
    - Preserves your computation logic exactly (including diagnostics, exclusion rules, NaN semantics).
    - Outputs:
        local_probs.csv and delta_probs.csv (or custom paths)

    Required inputs (must already exist upstream, unchanged logic):
    - windows2: DataFrame with per-cell rows and columns:
        ['cellid','region', (optionally 'x','y'), plus all cell_type_features, plus 'neigh_name' if present]
      Index should align to copy_cells index OR have 'cellid' to map.
    - probabilities_df: DataFrame with 'cellid' column + neighborhood probability columns (global MINGL).
    - assigned_df: DataFrame indexed by cellid with columns including 'neigh_name' and 'region'
    - neighborhoods: ordered list of all neighborhood labels (matches probabilities_df column order, excluding 'cellid'/Unnamed)
    - cell_type_features: list of feature columns (F features)
    - copy_cells: DataFrame with 'neigh_name' (index-aligned to windows2 OR has 'cellid' for mapping)

    Returns
    -------
    (final_probs_df, final_deltas_df) : tuple[pd.DataFrame, pd.DataFrame]
    """
    # Updated GPU region-level probability computation (CuPy-like, log-space, robust to underflow)
    # NOTE: to keep scverse-compat and avoid GPU hard-deps, we use numpy as `cp` exactly as you wrote.
    import numpy as np
    import pandas as pd
    import numpy as cp
    from tqdm import tqdm

    # small alias used in diagnostics
    _np = np

    # Basic validation (unchanged)
    required_assigned_cols = {"neigh_name", "region"}
    if not required_assigned_cols.issubset(set(assigned_df.columns)):
        raise ValueError(f"assigned_df must contain columns: {required_assigned_cols}")

    # Sanitizer utility (define once) (unchanged)
    def sanitize_centroids(means, stds, counts, region_cells,
                           min_floor_abs=min_floor_abs, use_region_frac=use_region_frac, shrink_alpha=shrink_alpha):
        K, F = means.shape

        region_mean = np.nanmean(means, axis=0) if np.any(np.isfinite(means)) else np.zeros(F, dtype=float)
        region_std  = np.nanstd(region_cells[cell_type_features].values.astype(float), axis=0, ddof=1)
        region_std  = np.where(np.isnan(region_std) | (region_std <= 0), 0.1, region_std)
        min_std_vec = np.maximum(region_std * use_region_frac, min_floor_abs)

        means_clean = np.where(np.isfinite(means), means, region_mean[None, :])

        stds_clean = np.where(np.isfinite(stds), stds, min_std_vec[None, :])
        stds_clean = np.maximum(stds_clean, min_std_vec[None, :])

        n = counts.astype(float) + 1e-8
        neigh_var = stds_clean ** 2
        region_var = (region_std ** 2)[None, :]
        combined_var = (n[:, None] * neigh_var + shrink_alpha * region_var) / (n[:, None] + shrink_alpha)
        stds_shrunk = np.sqrt(np.maximum(combined_var, min_std_vec[None, :]**2))
        stds_shrunk = np.maximum(stds_shrunk, min_std_vec[None, :])

        return means_clean, stds_shrunk

    # Helper: stable logsumexp on cupy (here cp==numpy) (unchanged)
    def cp_logsumexp(a, axis=1, keepdims=True):
        a_max = cp.max(a, axis=axis, keepdims=True)
        a_max = cp.where(cp.isfinite(a_max), a_max, -1e300)
        s = cp.sum(cp.exp(a - a_max), axis=axis, keepdims=True)
        s = cp.where(s == 0, cp.nan, s)
        return a_max + cp.log(s)

    # Prepare storage (unchanged)
    all_region_probs = []
    all_region_deltas = []

    unique_regions = windows2['region'].unique()
    print("Processing regions:", len(unique_regions))

    for region in tqdm(unique_regions, desc="Processing regions (log-space GPU)"):
        # 1) Filter cells for this region (unchanged)
        region_cells = windows2[windows2['region'] == region].copy()
        region_cell_ids = region_cells['cellid'].values
        cell_data = region_cells[cell_type_features].copy()
        C = cell_data.shape[0]
        if C == 0:
            print(f"  - Region {region}: no cells, skipping")
            continue

        # Replace infinities in raw cell_data (unchanged)
        cell_data = cell_data.replace([np.inf, -np.inf], np.nan)

        # 2) Add assigned neighborhood (unchanged)
        try:
            region_cells['neigh_name'] = copy_cells.loc[region_cells.index, 'neigh_name'].values
        except Exception:
            if 'cellid' in copy_cells.columns:
                mapping = copy_cells.set_index('cellid')['neigh_name'].to_dict()
                region_cells['neigh_name'] = region_cells['cellid'].map(mapping).values
            else:
                raise

        # 3) Build region-specific centroids (unchanged)
        region_results = []
        neigh_counts = []
        for neighborhood in neighborhoods:
            neighborhood_cells = region_cells[region_cells['neigh_name'] == neighborhood]
            matching_cell_ids = neighborhood_cells.index
            neigh_counts.append(len(matching_cell_ids))
            stats = {"Neighborhood": neighborhood}
            if len(matching_cell_ids) <= 1:
                for col in cell_type_features:
                    stats[f"{col}_mean"] = np.nan
                    stats[f"{col}_std"] = np.nan
            else:
                for col in cell_type_features:
                    stats[f"{col}_mean"] = neighborhood_cells[col].mean()
                    std_val = neighborhood_cells[col].std(ddof=0)
                    stats[f"{col}_std"] = np.nan if pd.isna(std_val) else std_val
            region_results.append(stats)

        df_region_centroids = pd.DataFrame(region_results)
        K = df_region_centroids.shape[0]

        centroid_means = df_region_centroids[[f"{c}_mean" for c in cell_type_features]].values.astype(float)
        centroid_stds  = df_region_centroids[[f"{c}_std"  for c in cell_type_features]].values.astype(float)
        counts = np.array(neigh_counts, dtype=int)

        # DIAGNOSTICS (unchanged)
        neigh_counts_arr = _np.array(neigh_counts)
        flat_stds = centroid_stds.flatten()
        pct = lambda q: float(_np.nanpercentile(flat_stds, q))
        print(f"  Region '{region}': cell_count={C}, neighborhoods={K}")
        print(f"   neigh_counts: min={neigh_counts_arr.min()}, median={_np.median(neigh_counts_arr)}, max={neigh_counts_arr.max()}")
        print("   centroid std percentiles (1,5,25,50,75,95,99,100):",
              pct(1), pct(5), pct(25), pct(50), pct(75), pct(95), pct(99), _np.nanmax(flat_stds))
        small_frac = float(_np.nanmean(flat_stds <= 1e-3))
        print(f"   fraction of centroid stds <= 1e-3: {small_frac:.3f}")

        # SANITIZE (unchanged)
        means_clean, stds_shrunk = sanitize_centroids(
            centroid_means.copy(), centroid_stds.copy(), counts, region_cells,
            min_floor_abs=min_floor_abs, use_region_frac=use_region_frac, shrink_alpha=shrink_alpha
        )

        # Exclude neighborhoods with too few cells (unchanged)
        low_mask = counts < min_count
        if low_mask.any():
            means_clean[low_mask, :] = np.nan
            stds_shrunk[low_mask, :] = np.nan

        # Region-level fallbacks (unchanged)
        region_mean  = np.nanmean(means_clean, axis=0)
        alt_region_mean = np.nanmean(cell_data.values.astype(float), axis=0)
        region_mean = np.where(np.isfinite(region_mean), region_mean,
                               np.where(np.isfinite(alt_region_mean), alt_region_mean, 0.0))

        region_std  = np.nanstd(cell_data.values.astype(float), axis=0, ddof=1)
        region_std  = np.where(np.isnan(region_std) | (region_std <= 0), 0.1, region_std)
        min_std_vec = np.maximum(region_std * use_region_frac, min_floor_abs)

        compute_means = np.where(np.isfinite(means_clean), means_clean, region_mean[None, :])
        compute_stds  = np.where(np.isfinite(stds_shrunk), stds_shrunk, min_std_vec[None, :])

        counts = np.array(neigh_counts, dtype=int)
        count_mask = counts >= min_count
        finite_mask = np.all(np.isfinite(compute_means) & np.isfinite(compute_stds), axis=1)
        valid_mask = count_mask & finite_mask

        valid_idx = np.where(valid_mask)[0]
        invalid_idx = np.where(~valid_mask)[0]

        K_full = len(neighborhoods)

        # If none valid (unchanged)
        if valid_idx.size == 0:
            local_probs_np = np.full((C, K_full), np.nan, dtype=float)
            print(f"  Region {region}: NO valid neighborhoods (all excluded) -> local_probs all NaN")
        else:
            means_valid = compute_means[valid_idx, :]
            stds_valid  = compute_stds[valid_idx, :]

            if not (np.isfinite(means_valid).all() and np.isfinite(stds_valid).all()):
                local_probs_np = np.full((C, K_full), np.nan, dtype=float)
                print(f"  Region {region}: unexpected non-finite in valid subset -> skipping GPU for safety")
            else:
                # "GPU" arrays (cp == numpy here; unchanged structure)
                region_means_cp = cp.asarray(means_valid, dtype=cp.float64)
                region_stds_cp  = cp.asarray(stds_valid, dtype=cp.float64)
                cell_array_cp   = cp.asarray(cell_data.values.astype(np.float64), dtype=cp.float64)

                X = cell_array_cp[:, None, :]
                M = region_means_cp[None, :, :]
                S = region_stds_cp[None, :, :]

                log_coeff = -0.5 * cp.log(2.0 * cp.pi * (S ** 2))
                exponent = -0.5 * ((X - M) / S) ** 2
                log_pdf = log_coeff + exponent

                log_total = cp.sum(log_pdf, axis=2)

                row_logsum = cp_logsumexp(log_total, axis=1, keepdims=True)
                log_prob_valid = log_total - row_logsum
                probs_valid_cp = cp.exp(log_prob_valid)

                try:
                    probs_valid = probs_valid_cp.get()
                except Exception:
                    probs_valid = np.array(probs_valid_cp)

                local_probs_np = np.full((C, K_full), np.nan, dtype=float)
                local_probs_np[:, valid_idx] = probs_valid

                if invalid_idx.size:
                    local_probs_np[:, invalid_idx] = np.nan

                del region_means_cp, region_stds_cp, cell_array_cp, X, M, S, log_pdf, log_total, row_logsum, log_prob_valid, probs_valid_cp
                try:
                    cp._default_memory_pool.free_all_blocks()
                except Exception:
                    pass

        # Diagnostics (unchanged)
        rows_all_nan = np.all(np.isnan(local_probs_np), axis=1)
        row_sums = np.nansum(local_probs_np, axis=1)
        n_all_nan_rows = int(rows_all_nan.sum())
        n_zero_rows = int(((~rows_all_nan) & np.isclose(row_sums, 0.0, atol=1e-12)).sum())
        print(f"  Region {region}: cells={C}, rows_all_nan={n_all_nan_rows}, zero-sum-rows={n_zero_rows}")

        col_to_idx = {col: i for i, col in enumerate(neighborhoods)}
        assigned_idx = [col_to_idx.get(n, None) for n in region_cells['neigh_name'].values]
        assigned_idx_arr = _np.array([i if i is not None else -1 for i in assigned_idx], dtype=int)

        assigned_probs = _np.full(len(local_probs_np), _np.nan)
        valid_rows = assigned_idx_arr >= 0
        if valid_rows.any():
            assigned_probs[valid_rows] = local_probs_np[valid_rows, assigned_idx_arr[valid_rows]]

        print("  assigned-prob: mean, median, min, max:",
              _np.nanmean(assigned_probs), _np.nanmedian(assigned_probs),
              _np.nanmin(assigned_probs), _np.nanmax(assigned_probs))

        with np.errstate(divide='ignore', invalid='ignore'):
            p = local_probs_np.copy()
            p = np.where((p >= 0) & np.isfinite(p), p, 0.0)
            mask = p > 0.0
            ent = -np.nansum(p * np.log2(np.where(mask, p, 1.0)), axis=1)
            ent[rows_all_nan] = np.nan

        print("  entropy (bits) mean/median/min/max:",
              _np.nanmean(ent), _np.nanmedian(ent), _np.nanmin(ent), _np.nanmax(ent))

        low_count_neighs = [neigh for neigh, cnt in zip(neighborhoods, neigh_counts) if cnt <= 1]
        if len(low_count_neighs) > 0:
            print(f"  neighborhoods with <=1 cell in region: {len(low_count_neighs)} (examples):", low_count_neighs[:6])

        # 6) Retrieve global probabilities (unchanged)
        global_idxed = probabilities_df.set_index('cellid')
        common_ids = [cid for cid in region_cell_ids if cid in global_idxed.index]
        if len(common_ids) != len(region_cell_ids):
            global_probs = np.full((len(region_cell_ids), K_full), np.nan, dtype=float)
            present_mask = [cid in global_idxed.index for cid in region_cell_ids]
            if any(present_mask):
                for i, cid in enumerate(region_cell_ids):
                    if cid in global_idxed.index:
                        global_probs[i, :] = global_idxed.loc[cid, neighborhoods].values
        else:
            global_probs = global_idxed.loc[region_cell_ids, neighborhoods].values

        # Prepare x/y for output (unchanged)
        x_vals = region_cells.get('x', region_cells.get('X', np.nan))
        y_vals = region_cells.get('y', region_cells.get('Y', np.nan))

        # 7) local probs df (unchanged)
        local_probs_df = pd.DataFrame(local_probs_np, columns=neighborhoods)
        local_probs_df['cellid'] = region_cells['cellid'].values
        local_probs_df['region'] = region
        local_probs_df['neigh_name'] = region_cells['neigh_name'].values
        local_probs_df['x'] = np.array(x_vals)
        local_probs_df['y'] = np.array(y_vals)
        all_region_probs.append(local_probs_df)

        # 8) delta df (unchanged)
        delta_values = local_probs_np - global_probs
        delta_df = pd.DataFrame(delta_values, columns=[f"{n}_delta" for n in neighborhoods])
        delta_df["cellid"] = region_cells['cellid'].values
        delta_df["region"] = region
        delta_df["neigh_name"] = region_cells["neigh_name"].values
        delta_df['x'] = np.array(x_vals)
        delta_df['y'] = np.array(y_vals)
        all_region_deltas.append(delta_df)

    # After loop: combine and save (unchanged)
    final_probs_df = pd.concat(all_region_probs, ignore_index=True)
    final_deltas_df = pd.concat(all_region_deltas, ignore_index=True)

    final_probs_df.to_csv(out_probs_path, index=False)
    final_deltas_df.to_csv(out_delta_path, index=False)

    print("✅ Done. Saved region-level local probs to:", out_probs_path)
    print("✅ Done. Saved region-level delta probs to:", out_delta_path)

    return final_probs_df, final_deltas_df
