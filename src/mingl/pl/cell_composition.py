import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def cell_type_distributions(
    adata,
    *,
    neighborhoods_to_plot=("Inner Follicle", "Outer Follicle"),
    min_cells=10,
    cluster_key="Probability_Bin_Cluster",
    score_key="Score",
    cell_type_key="Cell Type",
    neighborhood_key="Neighborhood",
    canonical_bins=("Very Low", "Low", "Medium", "High", "Very High"),
    width=0.8,
    spacing=1.0,
    dpi=200,
    figsize=None,
    ax=None,
    store_key=None,
    return_fig=False,
):
    # ---------------------------
    # checks
    # ---------------------------
    for k in [cluster_key, score_key, cell_type_key, neighborhood_key]:
        if k not in adata.obs.columns:
            raise KeyError(f"Missing required column '{k}' in adata.obs.")

    df = adata.obs.copy()

    neighborhoods = [n for n in neighborhoods_to_plot if isinstance(n, str) and n.strip()]
    if len(neighborhoods) == 0:
        raise ValueError("neighborhoods_to_plot must contain at least one non-empty string.")

    # ---------------------------
    # prep
    # ---------------------------
    df["_cluster_str"] = df[cluster_key].astype(str).str.strip()
    df["cluster_str"] = df["_cluster_str"]

    # ---------------------------
    # robust qcut binning
    # ---------------------------
    canonical_bins = list(canonical_bins)
    weights = {b: 2 ** i for i, b in enumerate(canonical_bins)}

    def robust_qcut_to_bins(series, desired_q=(0, 0.2, 0.4, 0.6, 0.8, 1.0)):
        s = series.dropna()
        if s.empty:
            return pd.Series([np.nan] * len(series), index=series.index), None, "empty", []
        vmin, vmax = float(s.min()), float(s.max())
        if np.isclose(vmin, vmax):
            return pd.Series(["Medium"] * len(series), index=series.index), None, "constant", ["Medium"]
        try:
            cat = pd.qcut(series, q=list(desired_q), duplicates="drop")
            n_intervals = len(cat.dtype.categories)
            edges = series.quantile(np.linspace(0, 1, n_intervals + 1)).values
            edges_u = np.unique(edges)
            labels = canonical_bins[: len(edges_u) - 1]
            mapped = pd.cut(series, bins=edges_u, labels=labels, include_lowest=True)
            return mapped.astype(object), edges_u, "qcut_adjusted", labels
        except Exception:
            edges = np.linspace(vmin, vmax, len(canonical_bins) + 1)
            edges_u = np.unique(edges)
            labels = canonical_bins[: len(edges_u) - 1]
            mapped = pd.cut(series, bins=edges_u, labels=labels, include_lowest=True)
            return mapped.astype(object), edges_u, "equal_width", labels

    score_numeric = pd.to_numeric(df[score_key], errors="coerce")
    mapped, edges_used, method_used, _ = robust_qcut_to_bins(score_numeric)
    df["_bin_mapped"] = mapped

    _map = {
        "very low": "Very Low",
        "low": "Low",
        "medium": "Medium",
        "high": "High",
        "very high": "Very High",
    }

    df["_bin_mapped"] = df["_bin_mapped"].apply(
        lambda x: _map.get(str(x).strip().lower(), x) if pd.notna(x) else np.nan
    )

    # ---------------------------
    # ranking
    # ---------------------------
    cluster_keys = sorted(df["cluster_str"].dropna().unique(), key=str)

    rank_rows = []
    for cl in cluster_keys:
        sub = df[df["cluster_str"] == cl]
        n = len(sub)
        if n == 0:
            continue
        vc = sub["_bin_mapped"].value_counts(dropna=False).to_dict()
        props = {b: vc.get(b, 0) / n for b in canonical_bins}
        weighted_prop = sum(props[b] * weights[b] for b in canonical_bins)
        rank_rows.append({"cluster": cl, "weighted_prop": weighted_prop, "total": n})

    rank_df = (
        pd.DataFrame(rank_rows)
        .sort_values(["weighted_prop", "total"], ascending=[False, False])
        .reset_index(drop=True)
    )

    global_cluster_order = rank_df["cluster"].tolist()

    # ---------------------------
    # min_cells filter
    # ---------------------------
    neigh_lower = df[neighborhood_key].astype(str).str.lower()
    neighborhoods_lower = [n.lower() for n in neighborhoods]

    clusters_ok = []
    for cl in global_cluster_order:
        ok = True
        for nlow in neighborhoods_lower:
            cnt = ((df["_cluster_str"] == cl) & (neigh_lower == nlow)).sum()
            if cnt < min_cells:
                ok = False
                break
        if ok:
            clusters_ok.append(cl)

    if not clusters_ok:
        raise ValueError("No clusters pass min_cells filter.")

    cluster_order_to_plot = clusters_ok

    # ---------------------------
    # pooled subset
    # ---------------------------
    combined_subset = df[neigh_lower.isin(neighborhoods_lower)].copy()

    # ---------------------------
    # NEW COLOR LOGIC (exactly as requested)
    # ---------------------------
    cell_types = sorted(combined_subset[cell_type_key].dropna().astype(str).unique())
    n = len(cell_types)

    palette_names = ['tab20', 'Set3', 'Set2', 'Paired', 'Dark2', 'Accent']
    combined_colors = []

    for name in palette_names:
        combined_colors.extend(sns.color_palette(name))

    if len(combined_colors) < n:
        raise ValueError(
            f"Not enough distinct colors for {n} cell types. "
            f"Max supported is {len(combined_colors)}."
        )

    final_palette = combined_colors[:n]
    color_dict = dict(zip(cell_types, final_palette))

    # ---------------------------
    # percent composition
    # ---------------------------
    def compute_percent_matrix(full_df_subset, cluster_order, cell_types):
        ct_series = full_df_subset[cell_type_key].astype(str)
        perc = {ct: np.zeros(len(cluster_order)) for ct in cell_types}
        for i, cl in enumerate(cluster_order):
            mask = full_df_subset["_cluster_str"] == cl
            total = mask.sum()
            if total == 0:
                continue
            for ct in cell_types:
                perc[ct][i] = ((mask) & (ct_series == ct)).sum() / total * 100
        return perc

    combined_perc = compute_percent_matrix(
        combined_subset, cluster_order_to_plot, cell_types
    )

    # ---------------------------
    # plotting
    # ---------------------------
    num_clusters = len(cluster_order_to_plot)
    x = np.arange(num_clusters) * spacing

    if ax is None:
        if figsize is None:
            figsize = (max(2, 0.6 * num_clusters), 3)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    bottom = np.zeros(num_clusters)
    for ct in cell_types:
        vals = combined_perc[ct]
        ax.bar(
            x,
            vals,
            bottom=bottom,
            width=width,
            color=color_dict[ct],
            edgecolor="k",
            linewidth=0.15,
            zorder=3,
        )
        bottom += vals

    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_order_to_plot, fontsize=10)
    ax.tick_params(axis="y", labelsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    left = x[0] - width / 2
    right = x[-1] + width / 2
    pad = 0.02 * (right - left)
    ax.set_xlim(left - pad, right + pad)

    fig.subplots_adjust(bottom=0.20, left=0.15, right=0.98)

    # ---------------------------
    # store
    # ---------------------------
    if store_key is not None:
        adata.uns[store_key] = {
            "rank_df": rank_df,
            "cluster_order": cluster_order_to_plot,
            "cell_types": cell_types,
            "combined_perc": {ct: combined_perc[ct].tolist() for ct in cell_types},
        }
    plt.show()
    if return_fig:
        return fig, ax, rank_df, combined_perc
    return ax, rank_df, combined_perc



