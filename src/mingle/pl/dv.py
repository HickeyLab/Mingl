import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import anndata as ad


def plot_log2fc_vs_mean_abundance(
    data,  # pd.DataFrame OR AnnData
    neighborhood,
    bucket_map,
    cell_type_color_map,
    min_count=10,
    subset_region=None,
    subset_patient=None,
    subset_context=None,
    patient_split_sep="_",
    figsize=(10, 6),
    eps_pct=0.01,
    size_scale=300,
    color_by_bucket=False,
    fc_threshold=1.0,
    abundance_threshold_pct=1.0,
    fontsize=12,
    annotate_sectors=True,
    min_marker_size=40,
    # scverse key params
    region_key="region",
    neigh_key="neigh_name",
    cluster_key="Cell Type",
    context_key="Context",
):
    """
    scverse-compatible version:
      - `data` can be AnnData (uses data.obs) OR a pandas DataFrame.
      - keys are configurable (region_key, neigh_key, cluster_key, context_key)

    Plot all cell types on one panel:
      x = log2(subset % / global %)
      y = mean neighborhood % abundance (linear)
    Dot size is proportional to subset % only (with minimum size).
    """

    # ---- get df from AnnData or DataFrame (no helper)
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        if not hasattr(data, "obs"):
            raise TypeError("`data` must be a pandas DataFrame or an AnnData object with `.obs`.")
        df = data.obs

    # ---- checks
    required_cols = [region_key, neigh_key, cluster_key]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns in obs/df: {missing}")

    has_context = (context_key in df.columns)

    df_neigh = df[df[neigh_key].astype(str) == str(neighborhood)].copy()
    if df_neigh.empty:
        raise ValueError(f"No data found for neighborhood '{neighborhood}'")

    # ---- subset selection
    if subset_region is not None:
        subset_df = df_neigh[df_neigh[region_key].astype(str) == str(subset_region)].copy()
        subset_label = f"{subset_region}"
    elif subset_patient is not None:
        pat_series = df_neigh[region_key].astype(str).str.split(patient_split_sep, n=1).str[0]
        subset_df = df_neigh[pat_series == str(subset_patient)].copy()
        subset_label = f"{subset_patient}"
    elif subset_context is not None:
        if not has_context:
            raise ValueError(f"subset_context provided, but '{context_key}' not found in obs/df.")
        subset_df = df_neigh[df_neigh[context_key].astype(str) == str(subset_context)].copy()
        subset_label = f"{subset_context}"
    else:
        raise ValueError("Provide subset_region or subset_patient or subset_context.")

    context_mode = subset_context is not None
    denom_global = max(len(df_neigh), 1)
    denom_subset = max(len(subset_df), 1)

    # ---- counts
    counts_global = df_neigh.groupby(cluster_key).size()
    counts_subset = subset_df.groupby(cluster_key).size()

    # filter by min_count (global OR subset)
    global_pass = set(counts_global[counts_global > min_count].index)
    subset_pass = set(counts_subset[counts_subset > min_count].index)
    ct_union = sorted(
        global_pass.union(subset_pass),
        key=lambda x: counts_global.get(x, 0),
        reverse=True
    )

    if len(ct_union) == 0:
        raise ValueError(f"No cell types > {min_count} to plot.")

    # ---- compute metrics
    rows = []
    for ct in ct_union:
        g = int(counts_global.get(ct, 0))
        s = int(counts_subset.get(ct, 0))
        pct_g = g / denom_global * 100.0
        pct_s = s / denom_subset * 100.0
        mean_pct = (pct_g + pct_s) / 2.0
        log2fc = np.log2((pct_s + eps_pct) / (pct_g + eps_pct))
        rows.append(
            {
                "ct": ct,
                "log2fc": log2fc,
                "mean_pct": mean_pct,
                "subset_pct": pct_s,
                "global_pct": pct_g,
                "subset_count": s,
                "global_count": g,
            }
        )

    plot_df = pd.DataFrame(rows).set_index("ct")
    if context_mode:
        plot_df["context_pct"] = plot_df["subset_pct"]
        plot_df["context_count"] = plot_df["subset_count"]

    # ---- colors
    if color_by_bucket:
        ct_to_bucket = {}
        for b in bucket_map:
            for ct in bucket_map[b]:
                if ct not in ct_to_bucket:
                    ct_to_bucket[ct] = b

        buckets = list(dict.fromkeys(ct_to_bucket.values()))
        cmap = plt.get_cmap("tab10")
        bucket_colors = {b: cmap(i % 10) for i, b in enumerate(buckets)}
        plot_df["color"] = [
            bucket_colors.get(ct_to_bucket.get(ct), "#999999") for ct in plot_df.index
        ]
    else:
        plot_df["color"] = [cell_type_color_map.get(ct, "#999999") for ct in plot_df.index]

    # ---- marker sizes (subset % only), with explicit minimum
    max_subset_pct = float(max(plot_df["subset_pct"].max(), eps_pct))
    marker_sizes = (plot_df["subset_pct"] / max_subset_pct) * size_scale
    marker_sizes = np.maximum(marker_sizes, min_marker_size)

    # ---- plotting
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        plot_df["log2fc"],
        plot_df["mean_pct"],
        s=marker_sizes,
        c=plot_df["color"],
        alpha=0.85,
        edgecolor="k",
        linewidth=0.3,
        marker="o",
        zorder=2,
    )

    ax.tick_params(axis="both", which="major", labelsize=fontsize)

    # guides
    ax.axvline(0, color="0.5", linestyle="--", linewidth=1)
    ax.axvline(fc_threshold, color="lightgray", linestyle=":", linewidth=1)
    ax.axvline(-fc_threshold, color="lightgray", linestyle=":", linewidth=1)
    ax.axhline(abundance_threshold_pct, color="lightgray", linestyle=":", linewidth=1)

    # labels
    if context_mode:
        ax.set_xlabel("log2(Cell Type % of Group / Global)", fontsize=fontsize)
        ax.set_ylabel("Mean % of CN", fontsize=fontsize)
        ax.set_title(f"{neighborhood}: {subset_label}", fontsize=fontsize)
    else:
        ax.set_xlabel("log₂(subset % / global %)", fontsize=fontsize)
        ax.set_ylabel("Mean CN Abundance (%)", fontsize=fontsize)
        ax.set_title(f"{neighborhood} {subset_label}", fontsize=fontsize)

    # ---- annotate sector points (top-left and top-right)
    if annotate_sectors:
        tr_mask = (plot_df["log2fc"] >= fc_threshold) & (plot_df["mean_pct"] >= abundance_threshold_pct)
        tl_mask = (plot_df["log2fc"] <= -fc_threshold) & (plot_df["mean_pct"] >= abundance_threshold_pct)
        to_label = plot_df[tr_mask | tl_mask]

        y_offset = 0.02 * float(plot_df["mean_pct"].max() if len(plot_df) else 1.0)

        for ct, row in to_label.iterrows():
            ax.text(
                float(row["log2fc"]),
                float(row["mean_pct"] + y_offset),
                str(ct),
                fontsize=max(fontsize - 12, 6),
                ha="center",
                va="bottom",
                color="black",
                zorder=4,
            )

    plt.tight_layout()
    plt.show()
    return plot_df


# --- Example (AnnData)
# plot_df = plot_log2fc_vs_mean_abundance_scverse(
#     adata,
#     neighborhood="Neigh_12",
#     bucket_map=bucket_map,
#     cell_type_color_map=cell_type_color_map,
#     subset_region="B004_Ascending",
#     region_key="region",
#     neigh_key="neigh_name",
#     cluster_key="Cell Type",
#     context_key="Context",
# )
