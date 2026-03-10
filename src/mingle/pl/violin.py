import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def _robust_qcut_to_bins(
    series: pd.Series,
    canonical_bins,
    desired_q=(0, 0.2, 0.4, 0.6, 0.8, 1.0),
):
    """
    Returns: mapped_series (object with strings/nan), edges_used (np.array), method_used (str), labels_used (list)
    """
    s = series.dropna()
    if s.empty:
        return pd.Series([np.nan] * len(series), index=series.index), np.array([]), "empty", []

    vmin = float(s.min())
    vmax = float(s.max())
    if np.isclose(vmin, vmax):
        mapped = pd.Series(["Medium"] * len(series), index=series.index)
        return mapped, np.array([vmin, vmax]), "constant", ["Medium"]

    # Try qcut -> then convert to cut with unique edges so labels are stable
    try:
        cat_q = pd.qcut(series, q=list(desired_q), duplicates="drop")
        if hasattr(cat_q.dtype, "categories"):
            n_intervals = len(cat_q.dtype.categories)
        else:
            n_intervals = len(pd.unique(cat_q.dropna()))

        labels_use = list(canonical_bins)[:n_intervals]

        edges = series.quantile(np.linspace(0, 1, n_intervals + 1)).values
        unique_edges = np.unique(edges)
        if len(unique_edges) < 2:
            raise ValueError("quantile edges collapsed")

        mapped = pd.cut(series, bins=unique_edges, labels=labels_use, include_lowest=True)
        mapped = mapped.astype(object).where(~pd.isna(mapped), other=np.nan)
        return mapped, unique_edges, "qcut_adjusted", labels_use

    except Exception:
        # equal-width fallback
        try:
            n_bins = len(canonical_bins)
            edges = np.linspace(vmin, vmax, n_bins + 1)
            unique_edges = np.unique(edges)
            n_intervals = len(unique_edges) - 1
            if n_intervals <= 0:
                mapped = pd.Series(["Medium"] * len(series), index=series.index)
                return mapped, np.array([vmin, vmax]), "constant_fallback", ["Medium"]

            labels_use = list(canonical_bins)[:n_intervals]
            mapped = pd.cut(series, bins=unique_edges, labels=labels_use, include_lowest=True)
            mapped = mapped.astype(object).where(~pd.isna(mapped), other=np.nan)
            return mapped, unique_edges, "equal_width", labels_use
        except Exception:
            mapped = pd.Series([np.nan] * len(series), index=series.index)
            return mapped, np.array([]), "failed", []


def plot_pooled_violin(
    adata,
    neighborhood_key,
    *,
    neighborhoods_to_plot=("Inner Follicle", "Outer Follicle"),
    aggregation_mode="pooled",
    min_cells=10,
    showfliers=False,     # kept for parity; not used for violinplot
    plot_means=True,
    xtick_fontsize=15,
    y_label="",
    cluster_key="Probability_Bin_Cluster",
    score_key="Score",
    canonical_bins=("Very Low", "Low", "Medium", "High", "Very High"),
    ax=None,
    figsize=None,
    dpi=300,
    cmap_name="GnBu",
):
    df = adata.obs.copy()

    if aggregation_mode != "pooled":
        raise NotImplementedError("Only aggregation_mode='pooled' is implemented (matches your script).")

    # keys exist?
    if cluster_key not in df.columns:
        raise KeyError(f"cluster_key='{cluster_key}' not found in columns.")
    if score_key not in df.columns:
        raise KeyError(f"score_key='{score_key}' not found in columns.")

    neigh_col = neighborhood_key

    # normalize cluster_str
    df["cluster_str"] = df[cluster_key].astype(str).str.strip()

    # ---- bin mapping to canonical bins (from score if numeric) ----
    canonical_bins = list(canonical_bins)
    weights = {b: 2 ** i for i, b in enumerate(canonical_bins)}

    # ensure numeric series for binning
    score_numeric = pd.to_numeric(df[score_key], errors="coerce")

    mapped, edges_used, method_used, labels_used = _robust_qcut_to_bins(
        score_numeric, canonical_bins=canonical_bins
    )
    df["_bin_mapped"] = mapped

    # normalize labels just in case
    _map = {
        "very low": "Very Low", "very_low": "Very Low", "very-low": "Very Low",
        "low": "Low",
        "medium": "Medium", "med": "Medium",
        "high": "High",
        "very high": "Very High", "very_high": "Very High", "very-high": "Very High",
    }

    def _normalize_label(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        return _map.get(s, str(x).strip())

    df["_bin_mapped"] = df["_bin_mapped"].apply(_normalize_label)

    # ---- rank clusters by size-normalized weighted_prop ----
    cluster_keys = df["cluster_str"].dropna().unique().tolist()
    try:
        cluster_keys = sorted(cluster_keys, key=lambda x: int(x))
    except Exception:
        cluster_keys = sorted(cluster_keys, key=str)

    rank_rows = []
    for cl in cluster_keys:
        mask = df["cluster_str"] == cl
        n_total = int(mask.sum())
        if n_total == 0:
            continue

        vc = df.loc[mask, "_bin_mapped"].value_counts(dropna=False).to_dict()
        counts = {b: int(vc.get(b, 0)) for b in canonical_bins}
        unmapped_count = int(sum(v for k, v in vc.items() if (k not in canonical_bins) or pd.isna(k)))

        props = {b: (counts[b] / n_total) if n_total > 0 else 0.0 for b in canonical_bins}
        weighted_prop = float(sum(props[b] * weights[b] for b in canonical_bins))
        weighted_sum = int(sum(counts[b] * weights[b] for b in canonical_bins))

        row = {
            "cluster": cl,
            "total": n_total,
            "weighted_sum": weighted_sum,
            "weighted_prop": weighted_prop,
            "unmapped": unmapped_count,
        }
        for b in canonical_bins:
            row[f"count_{b.replace(' ', '_')}"] = counts[b]
        rank_rows.append(row)

    rank_df = pd.DataFrame(rank_rows)
    rank_df = rank_df.sort_values(["weighted_prop", "total"], ascending=[False, False]).reset_index(drop=True)
    global_cluster_order = rank_df["cluster"].tolist()

    # ---- filter clusters by min_cells per neighborhood ----
    neighborhoods = [n for n in neighborhoods_to_plot if isinstance(n, str) and len(n) > 0]
    if len(neighborhoods) == 0:
        raise ValueError("neighborhoods_to_plot is empty.")

    neighborhoods_lower = [n.lower() for n in neighborhoods]
    neigh_lower = df[neigh_col].astype(str).str.lower()

    clusters_ok = []
    for cl in global_cluster_order:
        ok = True
        for nlow in neighborhoods_lower:
            cnt = int(((df["cluster_str"] == cl) & (neigh_lower == nlow)).sum())
            if cnt < min_cells:
                ok = False
                break
        if ok:
            clusters_ok.append(cl)

    if len(clusters_ok) == 0:
        raise ValueError(f"No clusters meet min_cells={min_cells} in every neighborhood: {neighborhoods_to_plot}")

    cluster_order_to_plot = clusters_ok

    # ---- pooled arrays (scores across selected neighborhoods) ----
    def pooled_collect_arrays(full_df, cluster_order):
        arrays = []
        for cl in cluster_order:
            subset = full_df[full_df["cluster_str"] == cl]
            subset = subset[subset[neigh_col].astype(str).str.lower().isin(neighborhoods_lower)]
            vals = pd.to_numeric(subset[score_key], errors="coerce").dropna().values
            arrays.append(np.asarray(vals) if vals.size > 0 else np.array([]))
        return arrays

    combined_arrays = pooled_collect_arrays(df, cluster_order_to_plot)

    # ---- plotting ----
    num_clusters = len(cluster_order_to_plot)
    spacing = 1.2
    x_positions = np.arange(num_clusters) * spacing

    if ax is None:
        if figsize is None:
            fig_w = max(4, (num_clusters * 0.35) + 3)
            fig_h = 3
            figsize = (fig_w, fig_h)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    cmap = cm.get_cmap(cmap_name)
    colors = [cmap(0.3 + 0.6 * (i / max(1, num_clusters - 1))) for i in range(num_clusters)]

    wide = 4
    violin_width = wide * max(0.1, 0.9 / max(1, num_clusters))
    edge_color = "k"
    edge_lw = 0.6

    for idx, (xi, arr) in enumerate(zip(x_positions, combined_arrays)):
        if isinstance(arr, np.ndarray) and arr.size > 0:
            vp = ax.violinplot([arr], positions=[xi], widths=violin_width,
                               showmeans=False, showmedians=False, showextrema=False)
            body = vp["bodies"][0]
            body.set_facecolor(colors[idx])
            body.set_edgecolor(edge_color)
            body.set_linewidth(edge_lw)
            body.set_alpha(0.95)

    # y-limits
    all_arrays = [a for a in combined_arrays if isinstance(a, np.ndarray) and a.size > 0]
    if len(all_arrays) > 0:
        all_vals = np.concatenate(all_arrays)
        y_min = min(0.0, float(np.nanmin(all_vals)))
        y_max = max(1.0, float(np.nanmax(all_vals)))
    else:
        y_min, y_max = 0.0, 1.0
    rng = (y_max - y_min) if (y_max - y_min) > 0 else 1.0
    ax.set_ylim(y_min - 0.02 * rng, y_max + 0.05 * rng)

    if plot_means:
        for xi, arr in zip(x_positions, combined_arrays):
            if isinstance(arr, np.ndarray) and arr.size > 0:
                ax.plot(xi, float(np.nanmean(arr)), marker="o", color="k", markersize=4, zorder=5)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(c) for c in cluster_order_to_plot],
                       fontsize=xtick_fontsize, rotation=0, ha="center")

    ax.set_xlabel("")
    ax.set_ylabel(y_label, fontsize=12)
    ax.tick_params(axis="y", labelsize=15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    left_limit = x_positions[0] - violin_width
    right_limit = x_positions[-1] + violin_width
    span = right_limit - left_limit
    pad = span * 0.01
    ax.set_xlim(left_limit - pad, right_limit + pad)

    # attach useful metadata (scverse-y convenience)
    ax._mingle_rank_df = rank_df
    ax._mingle_bin_method = method_used
    ax._mingle_bin_edges = edges_used
    ax._mingle_neighborhood_key = neigh_col
    ax._mingle_cluster_order = cluster_order_to_plot
    plt.tight_layout()
    plt.show()
    return ax, rank_df