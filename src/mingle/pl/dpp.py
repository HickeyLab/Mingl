from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def dpp(
    delta: Union[pd.DataFrame, str],
    *,
    # ---- wide -> long settings (if input is a wide csv/df) ----
    wide_delta_suffix: str = "_delta",
    cellid_col: str = "cellid",
    neigh_name_col: str = "neigh_name",
    region_col: str = "region",
    region_regex: str = r"_([^_]+)$",
    derive_region_if_missing: bool = True,
    keep_only_assigned_neighborhood: bool = True,
    min_cells_per_region_neigh: int = 10,
    # ---- long-format names (if you already pass long df) ----
    neighborhood_col: str = "Neighborhood",
    delta_col: str = "Delta",
    # ---- patient parsing ----
    patient_split_char: str = "_",
    patient_from_region: bool = True,     # matches your original: patient = region.split("_")[0]
    # ---- region -> context mapping ----
    palette: Optional[Dict[str, str]] = None,
    normal_regions: Optional[Iterable[str]] = None,
    tumor_regions: Optional[Iterable[str]] = None,
    metaplasia_regions: Optional[Iterable[str]] = None,
    dysplasia_regions: Optional[Iterable[str]] = None,
    unknown_label: str = "Unknown",
    # ---- plot params ----
    bar_height: float = 0.8,
    tick_font: int = 25,
    dpi: int = 150,
    # ---- show/return ----
    show: bool = True,
) -> Dict[str, object]:
    """
    Compute + plot patient divergence summaries from region×neighborhood mean deltas
    (scverse-friendly: takes either a wide csv/df or a long df).

    Produces:
      Figure 1: stacked patient bars = sum over regions of sum over neighborhoods |mean_delta|
      Figure 2: average divergence per patient = total_divergence / n_regions

    Returns dict with:
      fig_total, ax_total, fig_avg, ax_avg,
      plot_df, mean_delta, region_scores, patient_totals_total, patient_totals_avg,
      region_order, neighborhood_order, patient_order_total, patient_order_avg
    """

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
    # Load df (csv path or df)
    # -------------------------
    if isinstance(delta, str):
        df = pd.read_csv(delta)
    else:
        df = delta.copy()

    # detect if already LONG with required columns
    is_long = {region_col, neighborhood_col, delta_col}.issubset(df.columns)

    if not is_long:
        # expect WIDE format with *_delta cols
        delta_cols = [c for c in df.columns if str(c).endswith(wide_delta_suffix)]
        if not delta_cols:
            raise ValueError(
                "Input is not LONG (needs region/Neighborhood/Delta) and has no *_delta columns for WIDE."
            )

        if derive_region_if_missing and region_col not in df.columns:
            if cellid_col not in df.columns:
                raise ValueError(f"Missing '{cellid_col}' needed to derive region.")
            df[region_col] = df[cellid_col].astype(str).str.extract(region_regex)

        long_df = df.melt(
            id_vars=[cellid_col, region_col, neigh_name_col],
            value_vars=delta_cols,
            var_name=neighborhood_col,
            value_name=delta_col,
        )
        long_df[neighborhood_col] = long_df[neighborhood_col].astype(str).str.replace(
            wide_delta_suffix, "", regex=False
        )
    else:
        long_df = df

    # -------------------------
    # Filters: assigned neighborhood only + min cell threshold
    # -------------------------
    if keep_only_assigned_neighborhood:
        if neigh_name_col not in long_df.columns:
            raise ValueError(f"Missing '{neigh_name_col}' required for keep_only_assigned_neighborhood=True.")
        long_df = long_df[long_df[neighborhood_col] == long_df[neigh_name_col]]

    long_df = long_df.dropna(subset=[region_col, neighborhood_col, delta_col])

    # Count unique cells per region×neighborhood (matches your code)
    if cellid_col not in long_df.columns:
        raise ValueError(f"Missing '{cellid_col}' column needed for counting cells.")
    counts = (
        long_df.groupby([region_col, neighborhood_col])[cellid_col]
        .nunique()
        .reset_index()
        .rename(columns={cellid_col: "count"})
    )
    valid = counts[counts["count"] >= int(min_cells_per_region_neigh)][[region_col, neighborhood_col]]
    long_df = long_df.merge(valid, on=[region_col, neighborhood_col], how="inner")

    # -------------------------
    # mean_delta + plot_df (mean + n_cells)
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
    # Pivot for sorting like your original
    # -------------------------
    pivot_df = mean_delta.pivot(index=region_col, columns=neighborhood_col, values="mean_delta")

    all_regions = long_df[region_col].unique()
    all_neighborhoods = long_df[neighborhood_col].unique()
    pivot_df = pivot_df.reindex(index=all_regions, columns=all_neighborhoods)

    # patient + enrichment (clip lower=0, sum)
    if patient_from_region:
        patients = pivot_df.index.to_series().astype(str).str.split(patient_split_char).str[0]
    else:
        # fallback (kept for completeness)
        patients = pd.Series(pivot_df.index, index=pivot_df.index, dtype=str)

    enrichment_score = pivot_df.clip(lower=0).sum(axis=1)

    sort_df = pd.DataFrame(
        {"region": pivot_df.index, "patient": patients.values, "enrichment": enrichment_score.values}
    ).set_index("region")

    region_order = (
        sort_df.sort_values(["patient", "enrichment"], ascending=[True, False])
        .index
        .tolist()
    )
    neighborhood_order = list(pivot_df.columns)

    # -------------------------
    # Region divergence scores
    # -------------------------
    region_scores = plot_df[[region_col, "mean_delta"]].copy()
    region_scores["region_score"] = region_scores["mean_delta"].abs()
    region_scores = (
        region_scores.groupby(region_col, as_index=False)
        .agg(region_score=("region_score", "sum"))
    )
    region_scores["patient"] = region_scores[region_col].astype(str).str.split(patient_split_char, n=1).str[0]

    patient_totals = (
        region_scores.groupby("patient", as_index=False)
        .agg(total_divergence=("region_score", "sum"), n_regions=(region_col, "nunique"))
    )
    patient_totals["normalized_divergence"] = patient_totals["total_divergence"] / patient_totals["n_regions"]

    # orderings
    patient_totals_total = patient_totals.sort_values("total_divergence", ascending=False).reset_index(drop=True)
    patient_order_total = patient_totals_total["patient"].tolist()

    patient_totals_avg = patient_totals.sort_values("normalized_divergence", ascending=False).reset_index(drop=True)
    patient_order_avg = patient_totals_avg["patient"].tolist()

    # region ordering within patient (for stack)
    region_scores_sorted = (
        region_scores.set_index(["patient", region_col])
        .sort_values(["patient", "region_score"], ascending=[True, False])
        .reset_index()
    )

    # ========================
    # Figure 1: stacked contributions ordered by TOTAL divergence
    # ========================
    plt.close("all")
    fig_total, ax_total = plt.subplots(
        figsize=(8, max(4, 0.35 * len(patient_order_total))),
        dpi=dpi,
    )

    max_total = patient_totals_total["total_divergence"].max() if not patient_totals_total.empty else 1.0
    y_pos1 = np.arange(len(patient_order_total))

    for i, patient in enumerate(patient_order_total):
        parts = region_scores_sorted[region_scores_sorted["patient"] == patient].sort_values("region_score", ascending=False)
        left = 0.0
        for _, r in parts.iterrows():
            val = float(r["region_score"])
            ctx = map_region_to_context(str(r[region_col]))
            color = palette.get(ctx, palette.get(unknown_label, "gray"))
            ax_total.barh(
                i, val, left=left, height=bar_height,
                color=color, edgecolor="black", linewidth=0.3
            )
            left += val

    ax_total.set_yticks(y_pos1)
    ax_total.set_yticklabels(patient_order_total, fontsize=tick_font)
    ax_total.invert_yaxis()
    ax_total.set_xlim(0, max_total * 1.08 if max_total > 0 else 1.0)
    ax_total.set_xlabel("Total Divergence", fontsize=25)
    ax_total.tick_params(axis="x", labelsize=tick_font)
    ax_total.grid(False)
    for s in ["top", "right"]:
        ax_total.spines[s].set_visible(False)

    plt.tight_layout()
    if show:
        plt.show()

    # ========================
    # Figure 2: normalized divergence ordered by AVERAGE divergence
    # ========================
    plt.close("all")
    fig_avg, ax_avg = plt.subplots(
        figsize=(5, max(4, 0.35 * len(patient_order_avg))),
        dpi=dpi,
    )

    y_pos2 = np.arange(len(patient_order_avg))
    norm_vals = patient_totals_avg.set_index("patient").loc[patient_order_avg, "normalized_divergence"].values

    ax_avg.barh(
        y_pos2,
        norm_vals,
        height=bar_height,
        color=sns.color_palette("Dark2", n_colors=max(3, len(patient_order_avg))),
        edgecolor="black",
        linewidth=0.3,
    )

    ax_avg.set_yticks(y_pos2)
    ax_avg.set_yticklabels(patient_order_avg, fontsize=tick_font)
    ax_avg.invert_yaxis()
    ax_avg.set_xlim(0, norm_vals.max() * 1.08 if norm_vals.size and norm_vals.max() > 0 else 1.0)
    ax_avg.set_xlabel("Average Divergence", fontsize=25)
    ax_avg.tick_params(axis="x", labelsize=tick_font)
    ax_avg.grid(False)
    for s in ["top", "right"]:
        ax_avg.spines[s].set_visible(False)

    plt.tight_layout()
    if show:
        plt.show()

    return {
        "fig_total": fig_total,
        "ax_total": ax_total,
        "fig_avg": fig_avg,
        "ax_avg": ax_avg,
        "plot_df": plot_df,
        "mean_delta": mean_delta,
        "region_scores": region_scores_sorted,
        "patient_totals_total": patient_totals_total,
        "patient_totals_avg": patient_totals_avg,
        "region_order": region_order,
        "neighborhood_order": neighborhood_order,
        "patient_order_total": patient_order_total,
        "patient_order_avg": patient_order_avg,
    }
