# =========================
# rnd: Region × Neighborhood Δ dot+bar plot (scverse-compatible)
# =========================

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def rnd(
    delta: Union[pd.DataFrame, str],
    *,
    # required columns (long format)
    region_col: str = "region",
    neighborhood_col: str = "Neighborhood",
    delta_col: str = "Delta",
    cellid_col: str = "cellid",
    neigh_name_col: str = "neigh_name",
    # if wide csv/path is provided, infer these:
    wide_delta_suffix: str = "_delta",
    wide_cellid_col: str = "cellid",
    wide_neigh_name_col: str = "neigh_name",
    # optionally derive region from cellid if missing
    derive_region_if_missing: bool = True,
    region_regex: str = r"_([^_]+)$",
    # filtering
    min_cells_per_region_neigh: int = 10,
    keep_only_assigned_neighborhood: bool = True,
    # context coloring for the bar plot
    palette: Optional[Dict[str, str]] = None,
    normal_regions: Optional[Iterable[str]] = None,
    tumor_regions: Optional[Iterable[str]] = None,
    metaplasia_regions: Optional[Iterable[str]] = None,
    dysplasia_regions: Optional[Iterable[str]] = None,
    unknown_label: str = "Unknown",
    # sizing
    min_area: float = 40.0,
    max_area: float = 1200.0,
    orig_fig_width: float = 36.0,
    new_fig_width: float =30.0,
    fig_height: float = 24.0,
    dpi: int = 35,
    # colormap
    cmap_name: str = "vlag",
    # legend
    legend_props: Tuple[float, float, float] = (0.05, 0.25, 0.50),
    legend_title: str = "Proportion of\nRegion",
    # show/return
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes, plt.Figure, pd.DataFrame]:
    # -------------------------
    # Defaults: palette + region lists
    # -------------------------
    if palette is None:
        palette = {
            "Normal": "skyblue",
            "Tumor": "red",
            "Metaplasia": "orange",
            "Dysplasia": "purple",
            unknown_label: "gray",
        }

    if normal_regions is None:
        normal_regions = ["E08_reg002", "E08_reg003", "E17_reg001"]
    if tumor_regions is None:
        tumor_regions = [
            "E08_reg004", "E08_reg005", "E11_reg001", "E19_reg003", "E19_reg004",
            "E11_reg005", "E11_reg006", "E17_reg005",
        ]
    if metaplasia_regions is None:
        metaplasia_regions = [
            "E08_reg006", "E08_reg007", "E11_reg002", "E11_reg003", "E11_reg004",
            "E12_reg002", "E12_reg003", "E17_reg002", "E17_reg003", "E17_reg004",
            "E19_reg001", "E12_reg004", "E12_reg005", "E17_reg006",
        ]
    if dysplasia_regions is None:
        dysplasia_regions = ["E08_reg001", "E12_reg001", "E19_reg002"]

    normal_regions = set(normal_regions)
    tumor_regions = set(tumor_regions)
    metaplasia_regions = set(metaplasia_regions)
    dysplasia_regions = set(dysplasia_regions)

    def map_region_to_context(r: str) -> str:
        if r in normal_regions:
            return "Normal"
        if r in tumor_regions:
            return "Tumor"
        if r in metaplasia_regions:
            return "Metaplasia"
        if r in dysplasia_regions:
            return "Dysplasia"
        return unknown_label

    # -------------------------
    # Load / coerce to LONG format
    # -------------------------
    if isinstance(delta, str):
        df = pd.read_csv(delta)
    else:
        df = delta.copy()

    # detect whether df is long or wide
    is_long = {region_col, neighborhood_col, delta_col}.issubset(df.columns)
    is_wide = any(str(c).endswith(wide_delta_suffix) for c in df.columns)

    if not is_long:
        if not is_wide:
            raise ValueError(
                "Input does not look like LONG (needs region/Neighborhood/Delta) "
                "or WIDE (needs *_delta columns)."
            )
        # WIDE -> LONG (compatible with your original script)
        wide = df
        if derive_region_if_missing and region_col not in wide.columns:
            if wide_cellid_col not in wide.columns:
                raise ValueError(f"Missing '{wide_cellid_col}' needed to derive region.")
            wide[region_col] = wide[wide_cellid_col].astype(str).str.extract(region_regex)

        delta_cols = [c for c in wide.columns if str(c).endswith(wide_delta_suffix)]
        long_df = wide.melt(
            id_vars=[wide_cellid_col, region_col, wide_neigh_name_col],
            value_vars=delta_cols,
            var_name=neighborhood_col,
            value_name=delta_col,
        )
        long_df[neighborhood_col] = long_df[neighborhood_col].astype(str).str.replace(
            wide_delta_suffix, "", regex=False
        )
        # match user’s names for filtering later
        long_df = long_df.rename(
            columns={
                wide_cellid_col: cellid_col,
                wide_neigh_name_col: neigh_name_col,
            }
        )
    else:
        long_df = df

    # sanity checks for long_df
    needed = {region_col, neighborhood_col, delta_col, cellid_col}
    if keep_only_assigned_neighborhood:
        needed.add(neigh_name_col)
    missing = [c for c in needed if c not in long_df.columns]
    if missing:
        raise ValueError(f"LONG df missing required columns: {missing}")

    if derive_region_if_missing and region_col not in long_df.columns:
        long_df[region_col] = long_df[cellid_col].astype(str).str.extract(region_regex)

    # Keep only assigned neighborhood deltas (Neighborhood == neigh_name)
    if keep_only_assigned_neighborhood:
        long_df = long_df[long_df[neighborhood_col] == long_df[neigh_name_col]]

    long_df = long_df.dropna(subset=[delta_col, region_col, neighborhood_col])

    # -------------------------
    # Filter region×neighborhood combos with >= min_cells
    # -------------------------
    counts = (
        long_df.groupby([region_col, neighborhood_col])[cellid_col]
        .nunique()
        .reset_index()
        .rename(columns={cellid_col: "count"})
    )
    valid = counts[counts["count"] >= int(min_cells_per_region_neigh)][[region_col, neighborhood_col]]
    long_df = long_df.merge(valid, on=[region_col, neighborhood_col], how="inner")

    # -------------------------
    # Compute mean Δ + n_cells
    # -------------------------
    mean_delta = (
        long_df.groupby([region_col, neighborhood_col])[delta_col]
        .mean()
        .reset_index(name="mean_delta")
    )
    cell_counts = (
        long_df.groupby([region_col, neighborhood_col])
        .size()
        .reset_index(name="n_cells")
    )
    plot_df = mean_delta.merge(cell_counts, on=[region_col, neighborhood_col])

    # -------------------------
    # Pivot for region ordering by sum(|mean Δ|)
    # -------------------------
    pivot_df = plot_df.pivot(index=region_col, columns=neighborhood_col, values="mean_delta")

    all_regions = long_df[region_col].unique()
    all_neighborhoods = long_df[neighborhood_col].unique()
    pivot_df = pivot_df.reindex(index=all_regions, columns=all_neighborhoods)

    sum_abs = pivot_df.abs().sum(axis=1)
    region_order: List[str] = sum_abs.sort_values(ascending=False).index.tolist()
    neighborhood_order: List[str] = list(pivot_df.columns)

    # Categorical ordering for plotting
    plot_df[region_col] = pd.Categorical(plot_df[region_col], categories=region_order, ordered=True)
    plot_df[neighborhood_col] = pd.Categorical(plot_df[neighborhood_col], categories=neighborhood_order, ordered=True)
    plot_df = plot_df.dropna(subset=[region_col, neighborhood_col]).reset_index(drop=True)

    # Context mapping
    plot_df["Context"] = plot_df[region_col].astype(str).apply(map_region_to_context)

    # Grid positions
    x_map = {n: i for i, n in enumerate(neighborhood_order)}
    y_map = {r: i for i, r in enumerate(region_order)}
    plot_df["x"] = plot_df[neighborhood_col].astype(str).map(x_map)
    plot_df["y"] = plot_df[region_col].astype(str).map(y_map)

    # Dot size scaling normalized per region (row)
    region_totals = plot_df.groupby(region_col)["n_cells"].transform("sum")
    plot_df["proportion"] = np.where(region_totals > 0, plot_df["n_cells"] / region_totals, 0.0)
    plot_df["area"] = plot_df["proportion"] * (max_area - min_area) + min_area
    plot_df["area"] = plot_df["area"].clip(lower=min_area, upper=max_area)

    # Color normalization
    max_abs_val = float(np.nanmax(np.abs(plot_df["mean_delta"].to_numpy())))
    norm = Normalize(vmin=-max_abs_val, vmax=max_abs_val)
    cmap = sns.color_palette(cmap_name, as_cmap=True)

    # Bar colors by region context
    bar_colors_by_region = [palette.get(map_region_to_context(r), palette[unknown_label]) for r in region_order]

    # =========================
    # MAIN DOT + BAR PLOT
    # =========================
    marker_scale = 1#(orig_fig_width / new_fig_width) ** 2

    plt.close("all")
    fig, ax = plt.subplots(figsize=(new_fig_width, fig_height), dpi=dpi)
    fig.subplots_adjust(right=0.80)
    fig.subplots_adjust(bottom=0.25)
    pos = ax.get_position()
    ax_bar = fig.add_axes([0.82, pos.y0, 0.16, pos.height])  # fixed right panel

    ax.scatter(
        plot_df["x"], plot_df["y"],
        s=plot_df["area"] * marker_scale,
        c=plot_df["mean_delta"],
        cmap=cmap, norm=norm,
        edgecolors="black", linewidths=0.5,
    )

    ax.set_xticks(range(len(neighborhood_order)))
    ax.set_xticklabels(neighborhood_order, rotation=90, fontsize=20)
    ax.set_yticks(range(len(region_order)))
    ax.set_yticklabels(region_order, fontsize=25)

    n_rows = len(region_order)
    ax.set_xlim(-0.5, len(neighborhood_order) - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.invert_yaxis()

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.draw()

    # -------------------------
    # Right-side bar axis
    # -------------------------
    bar_axis_width = 0.15
    bar_pad = 0.01

    pos = ax.get_position()
    ax_bar = fig.add_axes([pos.x1 + bar_pad, pos.y0, bar_axis_width, pos.height])

    ax_bar.barh(
        np.arange(n_rows),
        sum_abs.reindex(region_order).values,
        color=bar_colors_by_region,
        height=0.95,
    )

    ax_bar.set_ylim(-0.5, n_rows - 0.5)
    ax_bar.invert_yaxis()
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("Sum |Mean Δ|", fontsize=32)
    ax_bar.tick_params(axis="x", labelsize=30)

    for s in ax_bar.spines.values():
        s.set_visible(False)

    # -------------------------
    # Dot size legend (proportion)
    # -------------------------
    legend_handles = []
    legend_labels = []
    for p in legend_props:
        area = float(np.clip(p * (max_area - min_area) + min_area, min_area, max_area))
        h = plt.scatter(
            [], [], s=area * marker_scale,
            facecolors="lightgray",
            edgecolors="black",
            linewidths=0.6,
        )
        legend_handles.append(h)
        legend_labels.append(f"{int(p * 100)}%")

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        title=legend_title,
        loc="lower right",
        bbox_to_anchor=(1.08, -0.1),
        frameon=False,
        fontsize=25,
        title_fontsize=25,
    )

    # -------------------------
    # Separate colorbar figure
    # -------------------------
    fig_cb, ax_cb = plt.subplots(figsize=(2, 6), dpi=dpi)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig_cb.colorbar(sm, ax=ax_cb)
    cbar.set_label("Mean Δ Probability", fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    ax_cb.remove()

    if show:
        plt.show()

    return fig, ax, fig_cb, plot_df