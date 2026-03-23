from __future__ import annotations

from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ============================================================
# Helpers
# ============================================================
def _coerce_cluster_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def _normalize_prob_bin_labels(series: pd.Series) -> pd.Series:
    canonical = ["Very Low", "Low", "Medium", "High", "Very High"]
    _map = {
        "very low": "Very Low", "very_low": "Very Low", "very-low": "Very Low",
        "low": "Low",
        "medium": "Medium", "med": "Medium",
        "high": "High",
        "very high": "Very High", "very_high": "Very High", "very-high": "Very High",
    }
    def f(x):
        if pd.isna(x):
            return np.nan
        sx = str(x).strip().lower()
        return _map.get(sx, str(x).strip())
    out = series.apply(f)
    # keep only canonical or NaN
    out = out.where(out.isin(canonical), other=np.nan)
    return out

def _safe_assign_bins_from_score(
    series: pd.Series,
    labels: List[str],
    n_bins: int = 5,
    prefer_quantiles: bool = True,
) -> Tuple[pd.Series, np.ndarray, str]:
    """
    Return (cat_series, edges, method) where cat_series contains labels low->high.
    """
    s = pd.to_numeric(series, errors="coerce")
    s_non = s.dropna()
    if s_non.empty:
        cat = pd.Series(pd.Categorical([np.nan] * len(s), categories=labels, ordered=True), index=s.index)
        return cat, np.array([]), "empty"

    vmin = float(s_non.min())
    vmax = float(s_non.max())
    if np.isclose(vmin, vmax):
        cat = pd.Series(["Medium"] * len(s), index=s.index, dtype="category")
        cat = cat.astype(pd.CategoricalDtype(categories=labels, ordered=True))
        return cat, np.array([vmin, vmax]), "constant"

    if prefer_quantiles:
        try:
            cat_q = pd.qcut(s, q=n_bins, labels=labels, duplicates="raise")
            edges = s.quantile(np.linspace(0, 1, n_bins + 1)).values
            return cat_q.astype(pd.CategoricalDtype(categories=labels, ordered=True)), np.asarray(edges), "quantiles"
        except Exception:
            pass

    # equal-width fallback
    edges = np.linspace(vmin, vmax, n_bins + 1)
    edges = np.unique(edges)
    if len(edges) < 2:
        cat = pd.Series(["Medium"] * len(s), index=s.index, dtype="category")
        cat = cat.astype(pd.CategoricalDtype(categories=labels, ordered=True))
        return cat, np.array([vmin, vmax]), "constant_fallback"

    n_intervals = len(edges) - 1
    labels_use = labels[:n_intervals]
    cat = pd.cut(s, bins=edges, labels=labels_use, include_lowest=True)
    cat = cat.astype(pd.CategoricalDtype(categories=labels_use, ordered=True))
    return cat, edges, "equal_width"


# ============================================================
# Plot 1 + Plot 2: Ranking + Inner/Outer Bars + Pooled Violin
# ============================================================
def gb_prob_bin_cluster_plots(
    adata,
    *,
    cluster_key: str = "Probability_Bin_Cluster",
    score_key: str = "Score",
    neighborhood_key: str = "Neighborhood",
    inner_name: str = "Inner Follicle",
    outer_name: str = "Outer Follicle",
    canonical_bins: List[str] = ("Very Low", "Low", "Medium", "High", "Very High"),
    min_cells: int = 10,

    # Plot 1 (bar) styling
    bar_spacing: float = 0.6,
    bar_width: float = 0.28,
    bar_fig_h: float = 3.0,
    xtick_fontsize_bar: int = 15,
    ytick_fontsize_bar: int = 15,
    bar_colors: Tuple[str, str] = ("teal", "orange"),
    legend_fontsize: int = 25,
    legend_title: str = "Neighborhood",
    legend_figsize: Tuple[float, float] = (1.6, 0.5),

    # Plot 2 (violin) styling
    violin_fig_h: float = 3.0,
    xtick_fontsize_violin: int = 15,
    ytick_fontsize_violin: int = 15,
    violin_cmap: str = "GnBu",
    violin_edge_lw: float = 0.6,
    violin_alpha: float = 0.95,
    mean_marker_size: float = 4.0,
    plot_means: bool = True,

    # Output keys
    out_prefix: str = "pb",
    make_plots: bool = True,
) -> Dict[str, Any]:
    """
    scverse-compatible implementation of your Plot 1 (inner/outer bar by ranked cluster)
    and Plot 2 (pooled violin of Score across Inner+Outer, same ranked order, min_cells filter).

    Reads from adata.obs: cluster_key, score_key, neighborhood_key.
    Writes to adata.uns:
      - f"{out_prefix}_rank_df" (DataFrame)
      - f"{out_prefix}_agg_df"  (DataFrame)
      - f"{out_prefix}_cluster_order_global" (list)
      - f"{out_prefix}_cluster_order_plot"   (list)
      - f"{out_prefix}_meta" (dict)

    Returns dict with DataFrames + figs (if make_plots).
    """

    # ---- checks
    for k in (cluster_key, neighborhood_key):
        if k not in adata.obs.columns:
            raise KeyError(f"'{k}' not found in adata.obs.")
    if score_key not in adata.obs.columns:
        raise KeyError(f"'{score_key}' not found in adata.obs.")

    obs = adata.obs.copy()
    obs["cluster_str"] = _coerce_cluster_str(obs[cluster_key])

    # ---- bin mapping used ONLY for ranking (same logic as your script)
    weights = {b: 2 ** i for i, b in enumerate(list(canonical_bins))}

    cat_series, edges_used, method_used = _safe_assign_bins_from_score(
        obs[score_key],
        labels=list(canonical_bins),
        n_bins=len(canonical_bins),
        prefer_quantiles=True,
    )
    obs["_bin_mapped"] = pd.Series(cat_series, index=obs.index).astype(object)
    obs["_bin_mapped"] = _normalize_prob_bin_labels(obs["_bin_mapped"])

    # ---- rank clusters by size-normalized weighted_prop
    # Prefer numeric cluster sort if possible
    try:
        cluster_keys_sorted = sorted(obs["cluster_str"].unique(), key=lambda x: int(x))
    except Exception:
        cluster_keys_sorted = sorted(obs["cluster_str"].unique(), key=str)

    rank_rows = []
    for cl in cluster_keys_sorted:
        mask = obs["cluster_str"] == cl
        n_total = int(mask.sum())
        if n_total == 0:
            continue

        vc = obs.loc[mask, "_bin_mapped"].value_counts(dropna=False).to_dict()
        counts = {b: int(vc.get(b, 0)) for b in canonical_bins}
        unmapped_count = int(sum(v for k, v in vc.items() if (k not in canonical_bins) or pd.isna(k)))

        weighted_sum = sum(counts[b] * weights[b] for b in canonical_bins)
        props = {b: (counts[b] / n_total) if n_total > 0 else 0.0 for b in canonical_bins}
        weighted_prop = sum(props[b] * weights[b] for b in canonical_bins)

        row = {
            "cluster": cl,
            "total": n_total,
            "weighted_sum": int(weighted_sum),
            "weighted_prop": float(weighted_prop),
            "unmapped": unmapped_count,
        }
        for b in canonical_bins:
            row[f"count_{b.replace(' ', '_')}"] = counts[b]
        rank_rows.append(row)

    rank_df = pd.DataFrame(rank_rows)
    if rank_df.empty:
        raise ValueError("No clusters found to rank (rank_df is empty).")

    rank_df = rank_df.sort_values(["weighted_prop", "total"], ascending=[False, False]).reset_index(drop=True)
    global_cluster_order = rank_df["cluster"].tolist()

    # ---- Plot 1 aggregation: inner/outer % within each cluster (filtered to those two neighborhoods)
    df_filt = obs[obs[neighborhood_key].isin([inner_name, outer_name])].copy()

    agg_rows = []
    for cl in global_cluster_order:
        mask_cl = df_filt["cluster_str"] == cl
        total = int(mask_cl.sum())
        if total == 0:
            continue
        inner_count = int(((df_filt[neighborhood_key] == inner_name) & mask_cl).sum())
        outer_count = int(((df_filt[neighborhood_key] == outer_name) & mask_cl).sum())

        inner_pct = 100.0 * inner_count / total if total > 0 else 0.0
        outer_pct = 100.0 * outer_count / total if total > 0 else 0.0

        agg_rows.append({"cluster": cl, "inner_pct": inner_pct, "outer_pct": outer_pct, "total": total})

    agg_df = pd.DataFrame(agg_rows)
    if agg_df.empty:
        raise ValueError(f"No rows found after filtering neighborhoods to '{inner_name}'/'{outer_name}'.")

    # Keep only clusters that survive filter, preserve ranked order
    clusters_plot1 = agg_df["cluster"].tolist()

    figs = {}

    if make_plots:
        # ---- Plot 1: bar plot
        spacing = float(bar_spacing)
        x = np.arange(len(clusters_plot1)) * spacing

        inner_percentages = agg_df["inner_pct"].to_numpy()
        outer_percentages = agg_df["outer_pct"].to_numpy()

        fig_w = max(6, 0.6 * len(clusters_plot1))
        fig_h = float(bar_fig_h)
        fig1 = plt.figure(figsize=(fig_w, fig_h), dpi=300)
        ax = plt.gca()

        inner_bars = ax.bar(
            x - bar_width / 2,
            inner_percentages,
            width=bar_width,
            color=bar_colors[0],
            edgecolor="k",
            linewidth=0.15,
            zorder=3,
            label=inner_name,
        )
        outer_bars = ax.bar(
            x + bar_width / 2,
            outer_percentages,
            width=bar_width,
            color=bar_colors[1],
            edgecolor="k",
            linewidth=0.15,
            zorder=3,
            label=outer_name,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(clusters_plot1, fontsize=xtick_fontsize_bar, rotation=0, ha="right")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="y", labelsize=ytick_fontsize_bar)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")

        # small x-padding
        left_limit = x[0] - bar_width
        right_limit = x[-1] + bar_width
        span = right_limit - left_limit if len(x) > 1 else 1.0
        ax.set_xlim(left_limit - span * 0.02, right_limit + span * 0.02)

        plt.tight_layout()
        plt.show()
        figs["plot1_bars"] = fig1

        # ---- Plot 1 legend as a separate tiny figure
        fig_leg = plt.figure(figsize=legend_figsize, dpi=300)
        fig_leg.patch.set_alpha(0.0)
        leg_ax = fig_leg.add_subplot(111)
        leg_ax.axis("off")

        handles = [inner_bars, outer_bars]
        labels = [inner_name, outer_name]
        leg = leg_ax.legend(
            handles,
            labels,
            loc="center",
            frameon=False,
            ncol=1,
            fontsize=legend_fontsize,
            handlelength=1.2,
            title=legend_title,
            title_fontsize=legend_fontsize,
        )
        if hasattr(leg, "get_frame"):
            leg.get_frame().set_alpha(0.0)
        plt.tight_layout()
        plt.show()
        figs["plot1_legend"] = fig_leg

    # ---- Plot 2: pooled violin of Score across inner+outer, using ranked order + min_cells filter
    # Keep clusters with >= min_cells in each neighborhood
    clusters_ok = []
    neigh_lower = obs[neighborhood_key].astype(str).str.lower()
    for cl in global_cluster_order:
        mask_cl = obs["cluster_str"] == cl
        cnt_inner = int(((neigh_lower == inner_name.lower()) & mask_cl).sum())
        cnt_outer = int(((neigh_lower == outer_name.lower()) & mask_cl).sum())
        if (cnt_inner >= min_cells) and (cnt_outer >= min_cells):
            clusters_ok.append(cl)

    if len(clusters_ok) == 0:
        raise ValueError(f"No clusters meet min_cells={min_cells} in BOTH '{inner_name}' and '{outer_name}'.")

    # pooled arrays (combine inner+outer)
    arrays = []
    neigh_set = {inner_name.lower(), outer_name.lower()}
    for cl in clusters_ok:
        sub = obs[obs["cluster_str"] == cl]
        sub = sub[sub[neighborhood_key].astype(str).str.lower().isin(neigh_set)]
        vals = pd.to_numeric(sub[score_key], errors="coerce").dropna().to_numpy(dtype=float)
        arrays.append(vals)

    if make_plots:
        num_clusters = len(clusters_ok)
        spacing = 1.2
        x_positions = np.arange(num_clusters) * spacing

        fig_w = max(4, (num_clusters * 0.35) + 3)
        fig_h = float(violin_fig_h)
        fig2, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

        cmap = cm.get_cmap(violin_cmap)
        colors = [cmap(0.3 + 0.6 * (i / max(1, num_clusters - 1))) for i in range(num_clusters)]

        wide = 4
        violin_width = wide * max(0.1, 0.9 / max(1, num_clusters))
        edge_color = "k"

        for idx, (xi, arr) in enumerate(zip(x_positions, arrays)):
            if isinstance(arr, np.ndarray) and arr.size > 0:
                vp = ax.violinplot(
                    [arr],
                    positions=[xi],
                    widths=violin_width,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                body = vp["bodies"][0]
                body.set_facecolor(colors[idx])
                body.set_edgecolor(edge_color)
                body.set_linewidth(violin_edge_lw)
                body.set_alpha(violin_alpha)

        # y-limits
        all_vals = np.concatenate([a for a in arrays if isinstance(a, np.ndarray) and a.size > 0])
        y_min = min(0.0, float(np.nanmin(all_vals))) if all_vals.size else 0.0
        y_max = max(1.0, float(np.nanmax(all_vals))) if all_vals.size else 1.0
        rng = (y_max - y_min) if (y_max - y_min) > 0 else 1.0
        ax.set_ylim(y_min - 0.02 * rng, y_max + 0.05 * rng)

        if plot_means:
            for xi, arr in zip(x_positions, arrays):
                if isinstance(arr, np.ndarray) and arr.size > 0:
                    ax.plot(xi, float(np.nanmean(arr)), marker="o", color="k", markersize=mean_marker_size, zorder=5)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(c) for c in clusters_ok], fontsize=xtick_fontsize_violin, rotation=0, ha="center")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=ytick_fontsize_violin)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")

        left_limit = x_positions[0] - violin_width
        right_limit = x_positions[-1] + violin_width
        span = right_limit - left_limit
        pad = span * 0.01
        ax.set_xlim(left_limit - pad, right_limit + pad)

        plt.tight_layout()
        plt.show()
        figs["plot2_violin"] = fig2

    # ---- write to adata.uns
    meta = {
        "cluster_key": cluster_key,
        "score_key": score_key,
        "neighborhood_key": neighborhood_key,
        "inner_name": inner_name,
        "outer_name": outer_name,
        "canonical_bins": list(canonical_bins),
        "binning_method": method_used,
        "bin_edges_used": edges_used.tolist() if isinstance(edges_used, np.ndarray) else [],
        "min_cells": int(min_cells),
    }

    adata.uns[f"{out_prefix}_rank_df"] = rank_df
    adata.uns[f"{out_prefix}_agg_df"] = agg_df
    adata.uns[f"{out_prefix}_cluster_order_global"] = global_cluster_order
    adata.uns[f"{out_prefix}_cluster_order_plot"] = clusters_ok
    adata.uns[f"{out_prefix}_meta"] = meta

    return {
        "rank_df": rank_df,
        "agg_df": agg_df,
        "cluster_order_global": global_cluster_order,
        "cluster_order_plot": clusters_ok,
        "meta": meta,
        "figs": figs,
    }


# ============================================================
# Plot 3: Local gradient magnitude map (same as before)
# ============================================================
def gb_local_score_gradients(
    adata,
    *,
    region_key: str = "unique_region",
    region_value: Optional[str] = None,
    x_key: str = "x",
    y_key: str = "y",
    score_key: str = "Score",
    score_source: str = "obs",         # "obs" or "layer"
    score_layer: Optional[str] = None, # used if score_source="layer"
    k_neighbors: int = 20,
    normalize_by: str = "iqr",         # "iqr" | "range" | "none"
    use_progress: bool = True,
    make_plots: bool = True,
    sample_for_plot: Optional[int] = None,
    figsize: Tuple[float, float] = (8, 8),
    cmap: str = "inferno",
    point_size: float = 1.0,
    alpha_pts: float = 0.95,
    vmax_pct: float = 99.0,
    out_prefix: str = "grad",
) -> Dict[str, Any]:

    if region_key not in adata.obs.columns:
        raise KeyError(f"region_key='{region_key}' not found in adata.obs.")
    for k in (x_key, y_key):
        if k not in adata.obs.columns:
            raise KeyError(f"'{k}' not found in adata.obs (required for coordinates).")

    if region_value is None:
        idx = adata.obs_names
        region_label = "__all__"
    else:
        m = adata.obs[region_key].astype(str) == str(region_value)
        idx = adata.obs.index[m]
        region_label = str(region_value)

    if len(idx) == 0:
        raise ValueError(f"No cells matched region_key='{region_key}' with region_value='{region_value}'.")

    coords = adata.obs.loc[idx, [x_key, y_key]].to_numpy(dtype=float)
    if np.any(~np.isfinite(coords)):
        finite_mask = np.isfinite(coords).all(axis=1)
        idx = idx[finite_mask]
        coords = coords[finite_mask]
    n_pts = coords.shape[0]
    if n_pts < 2:
        raise ValueError("Need at least 2 cells with finite coordinates to compute gradients.")

    # scores
    if score_source == "obs":
        if score_key not in adata.obs.columns:
            raise KeyError(f"score_key='{score_key}' not found in adata.obs.")
        scores = pd.to_numeric(adata.obs.loc[idx, score_key], errors="coerce").to_numpy(dtype=float)
    elif score_source == "layer":
        if score_layer is None:
            raise ValueError("score_layer must be provided when score_source='layer'.")
        if score_layer not in adata.layers:
            raise KeyError(f"score_layer='{score_layer}' not found in adata.layers.")
        layer = adata.layers[score_layer]
        if layer.ndim == 1:
            scores = np.asarray(layer)[adata.obs.index.get_indexer(idx)].astype(float)
        else:
            arr = np.asarray(layer)[adata.obs.index.get_indexer(idx)]
            if arr.shape[1] != 1:
                raise ValueError(
                    f"Layer '{score_layer}' has shape {arr.shape}. Expected 1 column for scalar score."
                )
            scores = arr[:, 0].astype(float)
    else:
        raise ValueError("score_source must be 'obs' or 'layer'.")

    keep = np.isfinite(scores)
    idx = idx[keep]
    coords = coords[keep]
    scores = scores[keep]
    n_pts = coords.shape[0]
    if n_pts < 2:
        raise ValueError("After dropping NaN scores, fewer than 2 points remain.")

    tree_all = cKDTree(coords)
    dists_nn, _ = tree_all.query(coords, k=2)
    d_med = float(np.median(dists_nn[:, 1]))
    if not np.isfinite(d_med) or d_med <= 0:
        d_med = float(np.mean(dists_nn[:, 1]) + 1e-8)

    q1, q3 = np.percentile(scores, [25, 75])
    score_iqr = float(max(1e-8, q3 - q1))
    score_rng = float(max(1e-8, float(np.nanmax(scores) - np.nanmin(scores))))
    if normalize_by == "iqr":
        score_norm = score_iqr
    elif normalize_by == "range":
        score_norm = score_rng
    elif normalize_by == "none":
        score_norm = 1.0
    else:
        raise ValueError("normalize_by must be 'iqr', 'range', or 'none'.")

    k_use = min(k_neighbors, max(1, n_pts - 1))
    _, neigh_idx = tree_all.query(coords, k=k_use + 1)

    grad_x = np.full(n_pts, np.nan, dtype=float)
    grad_y = np.full(n_pts, np.nan, dtype=float)
    grad_mag = np.full(n_pts, np.nan, dtype=float)
    valid = np.zeros(n_pts, dtype=bool)

    iterator = range(n_pts)
    if use_progress and (tqdm is not None):
        iterator = tqdm(iterator, desc=f"Local gradients ({region_label})", unit="pt")

    for i in iterator:
        neighbors = neigh_idx[i, 1:]
        pts = coords[neighbors] - coords[i]
        vals = scores[neighbors] - scores[i]

        if np.linalg.matrix_rank(pts) < 2:
            continue

        lr = LinearRegression().fit(pts, vals)
        gx = float(lr.coef_[0])
        gy = float(lr.coef_[1])
        mag = float(np.hypot(gx, gy))

        grad_x[i] = gx
        grad_y[i] = gy
        grad_mag[i] = mag
        valid[i] = True

    grad_mag_norm = grad_mag * (d_med / score_norm)
    grad_mag_norm[~np.isfinite(grad_mag_norm)] = np.nan

    out_x = f"{out_prefix}_x"
    out_y = f"{out_prefix}_y"
    out_mag = f"{out_prefix}_mag"
    out_valid = f"{out_prefix}_valid"
    out_mag_norm = f"{out_prefix}_mag_norm"

    if out_x not in adata.obs.columns:
        adata.obs[out_x] = np.nan
        adata.obs[out_y] = np.nan
        adata.obs[out_mag] = np.nan
        adata.obs[out_mag_norm] = np.nan
        adata.obs[out_valid] = False

    adata.obs.loc[idx, out_x] = grad_x
    adata.obs.loc[idx, out_y] = grad_y
    adata.obs.loc[idx, out_mag] = grad_mag
    adata.obs.loc[idx, out_mag_norm] = grad_mag_norm
    adata.obs.loc[idx, out_valid] = valid

    summary = {
        "region_key": region_key,
        "region_value": region_label,
        "n_points_used": int(n_pts),
        "n_valid_grad": int(np.nansum(valid)),
        "d_med": float(d_med),
        "score_iqr": float(score_iqr),
        "score_range": float(score_rng),
        "normalize_by": normalize_by,
        "median_grad_mag_norm": float(np.nanmedian(grad_mag_norm)),
        "mean_grad_mag_norm": float(np.nanmean(grad_mag_norm)),
    }
    params = {
        "x_key": x_key,
        "y_key": y_key,
        "score_key": score_key,
        "score_source": score_source,
        "score_layer": score_layer,
        "k_neighbors": int(k_neighbors),
        "vmax_pct": float(vmax_pct),
        "cmap": cmap,
        "point_size": float(point_size),
        "alpha_pts": float(alpha_pts),
    }
    adata.uns[f"{out_prefix}_summary"] = summary
    adata.uns[f"{out_prefix}_params"] = params

    figs = {}
    if make_plots:
        plot_idx = idx
        if sample_for_plot is not None and sample_for_plot < len(plot_idx):
            rng = np.random.default_rng(0)
            plot_idx = rng.choice(np.array(plot_idx), size=int(sample_for_plot), replace=False)

        plot_obs = adata.obs.loc[plot_idx, [x_key, y_key, out_mag_norm]].copy()
        plot_obs = plot_obs[np.isfinite(plot_obs[out_mag_norm].to_numpy())]

        vmax = float(np.nanpercentile(plot_obs[out_mag_norm].to_numpy(), vmax_pct)) if len(plot_obs) else 1.0
        vmin = 0.0

        fig1 = plt.figure(figsize=figsize, dpi=300)
        ax1 = plt.gca()
        ax1.scatter(
            plot_obs[x_key], plot_obs[y_key],
            c=plot_obs[out_mag_norm],
            cmap=cmap,
            s=point_size,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha_pts,
            linewidths=0,
            edgecolors="none",
        )
        ax1.set_aspect("equal", "box")
        ax1.axis("off")
        plt.tight_layout()
        plt.show()
        figs["scatter"] = fig1

        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        fig2 = plt.figure(figsize=(4, 4), dpi=300)
#        fig2.subplots_adjust(right=0.85)

        cb_ax = fig2.add_axes([0.15, 0.05, 0.06, 0.9])
        cbar = fig2.colorbar(sm, cax=cb_ax, orientation="vertical")
        cb_labelsize = 15
        cbar.ax.tick_params(labelsize=cb_labelsize)
        cbar.set_label("Normalized Gradient Magnitude\n(IQR per median-NN distance)", fontsize=cb_labelsize)
        #plt.tight_layout()
        plt.show()
        figs["colorbar"] = fig2

    return {"summary": summary, "params": params, "figs": figs}

def gb(
    adata,
    *,
    # Plot1/2 inputs
    cluster_key: str = "Probability_Bin_Cluster",
    score_key: str = "Score",
    neighborhood_key: str = "Neighborhood",
    inner_name: str = "Inner Follicle",
    outer_name: str = "Outer Follicle",
    min_cells: int = 10,
    pb_prefix: str = "pb",
    # Plot3 inputs
    region_key: str = "unique_region",
    region_value: str = "B006_Descending - Sigmoid",
    x_key: str = "x",
    y_key: str = "y",
    k_neighbors: int = 20,
    normalize_by: str = "iqr",
    grad_prefix: str = "grad",
    make_plots: bool = True,
) -> Dict[str, Any]:
    out12 = gb_prob_bin_cluster_plots(
        adata,
        cluster_key=cluster_key,
        score_key=score_key,
        neighborhood_key=neighborhood_key,
        inner_name=inner_name,
        outer_name=outer_name,
        min_cells=min_cells,
        out_prefix=pb_prefix,
        make_plots=make_plots,
    )
    out3 = gb_local_score_gradients(
        adata,
        region_key=region_key,
        region_value=region_value,
        x_key=x_key,
        y_key=y_key,
        score_key=score_key,
        k_neighbors=k_neighbors,
        normalize_by=normalize_by,
        make_plots=make_plots,
        out_prefix=grad_prefix,
    )
    return {"plot12": out12, "plot3": out3}




