import os
import re
import time
import numpy as np
import pandas as pd
import anndata as ad

def ccd(
    cells_path: str,
    probs_paths: dict,
    pp,  # pass in your scverse reader module, e.g. mg.pp or your pp namespace
    cellid_key: str = "cellid",
    assigned_neigh_key: str = "neigh_name",
    min_count: int = 10,
    save_deltas: bool = False,
    out_dir: str | None = None,
    out_prefix: str = "delta_prob",
):
    """
    Scverse-compatible pipeline:
      1) loads cells (AnnData) via pp.read_file(cells_path)
      2) loads combined + context probability CSVs
      3) aligns columns
      4) computes delta = context - combined
      5) masks to assigned neighborhood per cell (vectorized)
      6) stores deltas in adata.layers['delta_<ContextLabel>'] with neighborhoods in adata.var_names
      7) TRUE melts wide deltas, merges assigned + region_full, filters to assigned row
      8) MIN_COUNT filter on Neighborhood×Context (unique cellids)

    Required:
      probs_paths must include key 'combined' and may include: 'tumor','normal','metaplasia','dysplasia'
    Returns:
      adata, combined_melted_filtered, combo_counts
    """

    start_time = time.time()

    # ---- helpers ----
    def extract_full_region(cellid):
        if pd.isna(cellid):
            return np.nan
        s = str(cellid)
        m = re.search(r"([A-Za-z0-9]+_reg\d+)", s)
        if m:
            return m.group(1)
        parts = s.split("_")
        if len(parts) >= 2 and str(parts[-1]).startswith("reg"):
            return parts[-2] + "_" + parts[-1]
        return np.nan

    def read_probs_csv(path):
        df = pd.read_csv(path)
        if cellid_key not in df.columns:
            raise KeyError(f"Probability CSV missing required column '{cellid_key}': {path}")
        df = df.set_index(cellid_key)
        bad = [c for c in df.columns if str(c).startswith("Unnamed")]
        if bad:
            df = df.drop(columns=bad)
        return df

    def keep_only_assigned(delta_df, assigned_series):
        assigned_series = assigned_series.reindex(delta_df.index)
        neigh_array = assigned_series.to_numpy()
        col_array = np.array(delta_df.columns)

        mask = col_array[None, :] != neigh_array[:, None]
        arr = delta_df.to_numpy(copy=True)
        arr[mask] = 0.0
        return pd.DataFrame(arr, index=delta_df.index, columns=delta_df.columns)

    # ---- 0) Load cells into AnnData ----
    print("🔹 Loading cells (AnnData) ...")
    adata0 = pp.read_file(cells_path)

    # Ensure obs_names are cellids (string)
    if cellid_key in adata0.obs.columns:
        adata0.obs_names = pd.Index(adata0.obs[cellid_key].astype(str).tolist(), name=cellid_key)
    else:
        adata0.obs_names = pd.Index(adata0.obs_names.astype(str).tolist(), name=cellid_key)

    if assigned_neigh_key not in adata0.obs.columns:
        raise KeyError(f"adata.obs must have '{assigned_neigh_key}' (assigned neighborhood per cell)")

    assigned_series0 = adata0.obs[assigned_neigh_key].astype(str).copy()
    assigned_series0.index = adata0.obs_names

    # ---- 1) Load probability CSVs ----
    if "combined" not in probs_paths:
        raise ValueError("probs_paths must include a 'combined' entry")

    print("🔹 Loading probability CSVs ...")
    combined_df = read_probs_csv(probs_paths["combined"])
    
    context_map = {
        "tumor": "Tumor",
        "normal": "Normal",
        "metaplasia": "Metaplasia",
        "dysplasia": "Dysplasia",
    }
    context_dfs = {}
    for key, label in context_map.items():
        if key in probs_paths and probs_paths[key] is not None:
            context_dfs[label] = read_probs_csv(probs_paths[key])

    print(f"  - combined_df: {combined_df.shape}")
    for label, df in context_dfs.items():
        print(f"  - {label}_df: {df.shape}")

    # ---- 2) Align columns across all dfs ----
    neighborhood_cols = list(combined_df.columns)
    if len(neighborhood_cols) != len(set(neighborhood_cols)):
        raise ValueError("combined_df has duplicate neighborhood columns")

    print("🔹 Aligning columns...")
    for label in list(context_dfs.keys()):
        df = context_dfs[label]
        missing_cols = [c for c in neighborhood_cols if c not in df.columns]
        if missing_cols:
            print(f"  - {label}: adding {len(missing_cols)} missing columns as 0.0")
            df = df.copy()
            for c in missing_cols:
                df[c] = 0.0
        df = df.reindex(columns=neighborhood_cols, fill_value=0.0)
        context_dfs[label] = df
        print(f"  - {label}: final shape {df.shape}")

    # ---- 3) Restrict to shared cellids between AnnData + combined ----
    shared = adata0.obs_names.intersection(combined_df.index)
    if len(shared) == 0:
        raise ValueError("No shared cellids between AnnData and combined probability CSV")

    adata0 = adata0[shared].copy()
    combined_df = combined_df.loc[shared]
    assigned_series0 = assigned_series0.loc[shared]
    print(f"🔹 Cells after intersection (adata ∩ combined): {adata0.n_obs}")

    # ---- 4) Create scverse-style AnnData container with neighborhoods as vars ----
    print("🔹 Creating scverse-style AnnData container...")
    adata = ad.AnnData(
        X=np.zeros((adata0.n_obs, len(neighborhood_cols)), dtype=np.float32),
        obs=adata0.obs.copy(),
        var=pd.DataFrame(index=pd.Index(neighborhood_cols, name="Neighborhood")),
    )
    adata.obs_names = adata0.obs_names.copy()  # keep same ids

    # add region_full to obs (handy)
    adata.obs["region_full"] = pd.Series(adata.obs_names, index=adata.obs_names).map(extract_full_region)

    # normalize assigned neigh strings
    adata.obs[assigned_neigh_key] = adata.obs[assigned_neigh_key].astype(str).str.strip()
    assigned_series = adata.obs[assigned_neigh_key].copy()
    assigned_series.index = adata.obs_names

    # ---- 5) Compute wide delta per context, store in layers, optionally save ----
    if save_deltas and out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    delta_wide_by_context = {}

    for label, ctx_df in context_dfs.items():
        print(f"\n▶ Processing condition: {label}")
        t0 = time.time()

        common_ids = shared.intersection(ctx_df.index)
        print(f"  - {label}: {len(common_ids)} cells in common")
        if len(common_ids) == 0:
            print(f"  - {label}: skipping (no overlap)")
            continue

        comb = combined_df.loc[common_ids, neighborhood_cols]
        ctx = ctx_df.loc[common_ids, neighborhood_cols]

        delta = ctx - comb
        delta = keep_only_assigned(delta, assigned_series)

        delta_wide_by_context[label] = delta

        # layer matrix: align to all adata cells, NaN where missing
        wide = pd.DataFrame(np.nan, index=adata.obs_names, columns=neighborhood_cols)
        wide.loc[delta.index, :] = delta.values
        adata.layers[f"delta_{label}"] = wide.to_numpy(dtype=np.float32)

        print(f"  - Vectorized neighborhood filtering applied ({delta.shape[0]} cells)")

        if save_deltas and out_dir is not None:
            outpath = os.path.join(out_dir, f"{out_prefix}_{label.lower()}_vs_combined.csv")
            out_df = delta.copy()
            out_df.index.name = cellid_key
            out_df = out_df.reset_index()
            out_df.to_csv(outpath, index=False)
            print(f"  - {label} done. Saved {out_df.shape[0]} rows × {out_df.shape[1]} cols to {outpath}")

        print(f"  - Elapsed for {label}: {time.time() - t0:.1f} sec")

    if len(delta_wide_by_context) == 0:
        raise ValueError("No contexts produced deltas (check probs_paths keys and overlaps)")

    print(f"\n✅ All conditions processed in {time.time() - start_time:.1f} sec total")

    # ---- 6) Melt each delta + merge assigned + region_full + filter to assigned ----
    print("\n🔹 Melting + filtering ...")
    melted_dfs = []

    for label, delta in delta_wide_by_context.items():
        print(f"\n▶ Melting {label} ...")

        delta2 = delta.copy()
        delta2.index.name = cellid_key  # critical for reset_index column name

        df_long = delta2.reset_index().melt(
            id_vars=[cellid_key],
            value_vars=neighborhood_cols,
            var_name="Neighborhood",
            value_name="Delta",
        )
        df_long["Context"] = label

        # map in region_full + assigned neigh from adata (safe: same index universe)
        df_long["region_full"] = df_long[cellid_key].map(adata.obs["region_full"])
        df_long["neigh_name"] = df_long[cellid_key].map(assigned_series)

        # normalize strings to avoid silent mismatches
        df_long["Neighborhood"] = df_long["Neighborhood"].astype(str).str.strip()
        df_long["neigh_name"] = df_long["neigh_name"].astype(str).str.strip()

        before = len(df_long)
        df_long = df_long[df_long["Neighborhood"] == df_long["neigh_name"]].copy()
        after = len(df_long)

        print(f"  - rows before filter: {before}; after assigned filter: {after} (kept {after/before:.2%})")

        n_missing_assigned = df_long["neigh_name"].isna().sum()
        if n_missing_assigned > 0:
            print(f"  - WARNING: {n_missing_assigned} rows missing assigned neigh_name")

        melted_dfs.append(df_long)

    combined_melted = pd.concat(melted_dfs, ignore_index=True)
    print(f"\n✅ Combined melted (raw): {combined_melted.shape[0]} rows, {combined_melted.shape[1]} cols")

    # ---- 7) MIN_COUNT filter on Neighborhood×Context ----
    combo_counts = (
        combined_melted.groupby(["Neighborhood", "Context"])[cellid_key]
        .nunique()
        .reset_index(name="count")
    )

    excluded = combo_counts[combo_counts["count"] < min_count]
    print("\n⚠️ Excluded Neighborhood × Context pairs (count < MIN_COUNT):")
    print(excluded.sort_values(["Context", "Neighborhood"]).head(50))
    print(f"  -> total excluded combos: {len(excluded)}")

    valid = combo_counts[combo_counts["count"] >= min_count][["Neighborhood", "Context"]]
    combined_melted_filtered = combined_melted.merge(valid, on=["Neighborhood", "Context"], how="inner")

    print(f"✅ Final combined_melted: {combined_melted_filtered.shape[0]} rows, {combined_melted_filtered.shape[1]} cols")
    print(f"Unique full regions represented after filtering: {combined_melted_filtered['region_full'].nunique()}")
    print("Top 10 Neighborhood×Context pairs by count:")
    print(combo_counts.sort_values("count", ascending=False).head(10))

    # stash diagnostics in adata.uns
    adata.uns["combo_counts"] = combo_counts
    adata.uns["min_count"] = int(min_count)

    return adata, combined_melted_filtered, combo_counts
