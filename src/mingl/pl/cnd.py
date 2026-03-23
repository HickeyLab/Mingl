import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from .. import pp  # scverse-style (AnnData in / AnnData out)


# -----------------------------
# scverse-compatible pipeline
# -----------------------------
def cnd(
    *,
    cells_path: str,
    probs_paths: dict,
    x_key: str = "x",
    y_key: str = "y",
    region_key: str = "region",
    assigned_neigh_key: str = "neigh_name",
    cellid_key: str = "cellid",
    out_dir: str | None = None,
    out_prefix: str = "delta_prob",
    min_count: int = 10,
    save_deltas: bool = True,
    make_plot: bool = True,
    figsize=(48, 16),
    dpi=60,
):
    """
    Converts your CSV-based script into a scverse-compatible function:
      - Reads cells via mg.pp.read_file(...) (AnnData)
      - Loads probability CSVs (combined + per-context)
      - Aligns columns
      - Computes per-cell delta = context - combined
      - Keeps ONLY the assigned neighborhood per cell (vectorized)
      - Melts long, MIN_COUNT filters Neighborhood×Context
      - Makes the same dot+bar plot

    probs_paths must include:
      probs_paths["combined"] = <csv>
      probs_paths["tumor"] = <csv> (optional)
      probs_paths["normal"] = <csv> (optional)
      probs_paths["metaplasia"] = <csv> (optional)
      probs_paths["dysplasia"] = <csv> (optional)

    Returns:
      dict with:
        - adata
        - delta_wide_by_context (dict[str, DataFrame])
        - combined_melted (DataFrame)
        - combo_counts (DataFrame)
        - fig, fig_cb (matplotlib figures or None)
    """

    t_start = time.time()
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    # ---- helpers ----
    def extract_full_region(cellid):
        if pd.isna(cellid):
            return np.nan
        s = str(cellid)
        m = re.search(r"([A-Za-z0-9]+_reg\d+)", s)
        if m:
            return m.group(1)
        parts = s.split("_")
        if len(parts) >= 2 and parts[-1].startswith("reg"):
            return parts[-2] + "_" + parts[-1]
        return np.nan

    def read_probs_csv(path):
        df = pd.read_csv(path)
        if cellid_key not in df.columns:
            raise KeyError(f"Probability CSV missing required column '{cellid_key}': {path}")
        df = df.set_index(cellid_key)
        # drop unnamed columns if present
        bad = [c for c in df.columns if str(c).startswith("Unnamed")]
        if bad:
            df = df.drop(columns=bad)
        return df

    def keep_only_assigned(delta_df, assigned_series):
        """
        delta_df: (cells x neighborhoods)
        assigned_series: indexed by cellid, value = assigned neighborhood
        """
        assigned_series = assigned_series.reindex(delta_df.index)
        neigh_array = assigned_series.to_numpy()
        col_array = np.array(delta_df.columns)

        # mask True where column != assigned neighborhood
        mask = col_array[None, :] != neigh_array[:, None]
        arr = delta_df.to_numpy(copy=True)
        arr[mask] = 0.0
        return pd.DataFrame(arr, index=delta_df.index, columns=delta_df.columns)

    # ---- 0) Load cells into AnnData (scverse-style) ----
    print("🔹 Loading cells (AnnData) ...")
    adata = pp.read_file(cells_path)

    # Ensure we have cellid available + consistent
    if cellid_key in adata.obs.columns:
        cellids = adata.obs[cellid_key].astype(str).tolist()
        adata.obs_names = pd.Index(cellids, name=cellid_key)
    else:
        # assume obs_names are already cellids
        cellids = adata.obs_names.astype(str).tolist()

    if assigned_neigh_key not in adata.obs.columns:
        raise KeyError(f"adata.obs must have '{assigned_neigh_key}' (assigned neighborhood per cell)")

    assigned_series = adata.obs[assigned_neigh_key].copy()
    assigned_series.index = adata.obs_names

    # ---- 1) Load probability CSVs ----
    if "combined" not in probs_paths:
        raise ValueError("probs_paths must include a 'combined' entry")

    print("🔹 Loading probability CSVs ...")
    combined_df = read_probs_csv(probs_paths["combined"])

    # Load contexts present in dict (skip missing)
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

    # ---- 2) Align columns across all dfs (important fix vs your script) ----
    neighborhood_cols = list(combined_df.columns)

    # Ensure combined has no duplicates
    if len(neighborhood_cols) != len(set(neighborhood_cols)):
        raise ValueError("combined_df has duplicate neighborhood columns")

    for label in list(context_dfs.keys()):
        df = context_dfs[label]
        # add missing columns as 0.0
        missing_cols = [c for c in neighborhood_cols if c not in df.columns]
        if missing_cols:
            df = df.copy()
            for c in missing_cols:
                df[c] = 0.0
        # enforce same order
        df = df.reindex(columns=neighborhood_cols, fill_value=0.0)
        context_dfs[label] = df

    # ---- 3) Restrict to shared cellids between AnnData + combined ----
    shared = adata.obs_names.intersection(combined_df.index)
    if len(shared) == 0:
        raise ValueError("No shared cellids between AnnData and combined probability CSV")

    adata = adata[shared].copy()
    combined_df = combined_df.loc[shared]
    assigned_series = assigned_series.loc[shared]

    # ---- 4) Compute wide delta per context (and optionally save) ----
    delta_wide_by_context = {}

    for label, ctx_df in context_dfs.items():
        common_ids = shared.intersection(ctx_df.index)
        if len(common_ids) == 0:
            print(f"⚠️ {label}: no overlapping cellids with combined/adata. Skipping.")
            continue

        comb = combined_df.loc[common_ids, neighborhood_cols]
        ctx = ctx_df.loc[common_ids, neighborhood_cols]

        delta = ctx - comb
        delta = keep_only_assigned(delta, assigned_series)

        delta_wide_by_context[label] = delta

        if save_deltas and out_dir is not None:
            outpath = os.path.join(out_dir, f"{out_prefix}_{label.lower()}_vs_combined.csv")
            # match your original: include cellid column
            out_df = delta.copy()
            out_df[cellid_key] = out_df.index.astype(str)
            out_df = out_df.reset_index(drop=True)
            out_df.to_csv(outpath, index=False)
            print(f"  - Saved {label} delta -> {outpath} ({out_df.shape[0]}×{out_df.shape[1]})")

    if len(delta_wide_by_context) == 0:
        raise ValueError("No contexts produced deltas (check probs_paths keys and overlaps)")

    # ---- 5) Melt each delta + merge assigned + region_full + filter to assigned ----
    print("🔹 Melting + filtering ...")
    region_full = pd.Series(adata.obs_names, index=adata.obs_names).map(extract_full_region)

    melted_dfs = []
    for label, delta in delta_wide_by_context.items():
        df_long = (
            delta.reset_index()
                 .rename(columns={"index": cellid_key})
                 .melt(
                     id_vars=[cellid_key],
                     value_vars=neighborhood_cols,
                     var_name="Neighborhood",
                     value_name="Delta",
                 )
        )
        df_long["Context"] = label
        df_long["region_full"] = df_long[cellid_key].map(region_full)
        df_long["neigh_name"] = df_long[cellid_key].map(assigned_series)

        # keep only assigned neigh row
        before = len(df_long)
        df_long = df_long[df_long["Neighborhood"] == df_long["neigh_name"]].copy()
        after = len(df_long)
        print(f"  - {label}: kept {after}/{before} ({after/before:.2%}) after assigned filter")
        melted_dfs.append(df_long)

    combined_melted = pd.concat(melted_dfs, ignore_index=True)
    print(f"✅ combined_melted (raw): {combined_melted.shape}")

    # ---- 6) MIN_COUNT filter on Neighborhood×Context (unique cellids) ----
    combo_counts = (
        combined_melted.groupby(["Neighborhood", "Context"])[cellid_key]
        .nunique().reset_index(name="count")
    )
    valid = combo_counts[combo_counts["count"] >= min_count][["Neighborhood", "Context"]]
    combined_melted = combined_melted.merge(valid, on=["Neighborhood", "Context"], how="inner")
    print(f"✅ combined_melted (MIN_COUNT≥{min_count}): {combined_melted.shape}")

    # ---- 7) Plot (dot + bar) ----
    fig = fig_cb = None
    if make_plot:
        palette = {
            "Normal": "skyblue",
            "Tumor": "red",
            "Metaplasia": "orange",
            "Dysplasia": "purple",
            "Unknown": "gray",
        }

        # mean Δ and counts
        mean_delta = (
            combined_melted.groupby(["Context", "Neighborhood"])["Delta"]
            .mean().reset_index(name="mean_delta")
        )
        cell_counts = (
            combined_melted.groupby(["Context", "Neighborhood"])
            .size().reset_index(name="n_cells")
        )
        plot_df = mean_delta.merge(cell_counts, on=["Context", "Neighborhood"])

        # pivot for row ordering by sum |mean Δ|
        # --- preserve your original neighborhood order (first-seen in combined_melted) ---
        all_contexts = combined_melted["Context"].unique()
        all_neighborhoods = combined_melted["Neighborhood"].unique()

        pivot_df = plot_df.pivot(index="Context", columns="Neighborhood", values="mean_delta")

        # force row/col order to match your script (prevents alphabetical pivot sorting)
        pivot_df = pivot_df.reindex(index=all_contexts, columns=all_neighborhoods)

        # order contexts by signal strength (your script behavior)
        sum_abs = pivot_df.abs().sum(axis=1)
        region_order = sum_abs.sort_values(ascending=False).index.tolist()

        # x-axis neighborhood order stays as "first-seen"
        neighborhood_order = list(pivot_df.columns)


        plot_df["Context"] = pd.Categorical(plot_df["Context"], categories=region_order, ordered=True)
        plot_df["Neighborhood"] = pd.Categorical(plot_df["Neighborhood"], categories=neighborhood_order, ordered=True)
        plot_df = plot_df.dropna(subset=["Context", "Neighborhood"]).reset_index(drop=True)

        # positions
        x_map = {n: i for i, n in enumerate(neighborhood_order)}
        y_map = {c: i for i, c in enumerate(region_order)}
        plot_df["x"] = plot_df["Neighborhood"].map(x_map)
        plot_df["y"] = plot_df["Context"].map(y_map)

        # dot size scaling (per context)
        min_area, max_area = 25, 1000
        context_totals = plot_df.groupby("Context")["n_cells"].transform("sum")
        plot_df["proportion"] = np.where(context_totals > 0, plot_df["n_cells"] / context_totals, 0.0)
        plot_df["area"] = (plot_df["proportion"] * (max_area - min_area) + min_area).clip(min_area, max_area)

        # color normalization
        max_abs_val = np.nanmax(np.abs(plot_df["mean_delta"])) if len(plot_df) else 1.0
        norm = Normalize(vmin=-max_abs_val, vmax=max_abs_val)

        # bar colors
        def map_context_to_key(context):
            for k in palette:
                if k in str(context):
                    return k
            return "Unknown"

        bar_colors_by_context = [palette[map_context_to_key(c)] for c in region_order]

        plt.close("all")
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.subplots_adjust(right=0.80)
        ax.scatter(
            plot_df["x"], plot_df["y"],
            s=plot_df["area"],
            c=plot_df["mean_delta"],
            cmap="vlag", norm=norm,   # same look as your seaborn vlag
            edgecolors="black", linewidths=0.5
        )

        ax.set_xticks(range(len(neighborhood_order)))
        #ax.set_xticklabels(neighborhood_order, rotation=90, fontsize=25)
        ax.set_yticks(range(len(region_order)))
        #ax.set_yticklabels(region_order, fontsize=25)
        ax.set_xticklabels(neighborhood_order, rotation=90, fontsize=18)
        ax.set_yticklabels(region_order, fontsize=20)
        n_rows = len(region_order)
        ax.set_xlim(-0.5, len(neighborhood_order) - 0.5)
        ax.set_ylim(-0.5, n_rows - 0.5)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # bar axis aligned to main axis bbox
        pos = ax.get_position()
        bar_axis_width = 0.08
        bar_pad = 0.005
        ax_bar = fig.add_axes([pos.x1 + bar_pad, pos.y0, bar_axis_width, pos.height])

        ax_bar.barh(np.arange(n_rows), sum_abs.values, height=0.95, color=bar_colors_by_context)
        ax_bar.set_ylim(-0.5, n_rows - 0.5)
        ax_bar.invert_yaxis()
        ax_bar.set_yticks([])
        ax_bar.set_xlabel("Sum |Mean Δ|", fontsize=25)
        ax_bar.tick_params(axis="x", labelsize=25)
        for s in ax_bar.spines.values():
            s.set_visible(False)

        # dot size legend
        legend_props = [0.05, 0.25, 0.50]
        handles, labels = [], []
        for p in legend_props:
            area = float(np.clip(p * (max_area - min_area) + min_area, min_area, max_area))
            h = ax.scatter([], [], s=area, facecolors="lightgray", edgecolors="black", linewidths=0.6)
            handles.append(h)
            labels.append(f"{int(p * 100)}%")

        fig.legend(
            handles=handles,
            labels=labels,
            title="Proportion of\nContext",
            loc="lower right",
            bbox_to_anchor=(0.90, 0.08),
            frameon=False,
            fontsize=25,
            title_fontsize=25
        )

        # separate horizontal colorbar figure
        fig_cb, ax_cb = plt.subplots(figsize=(6, 1.5), dpi=dpi)
        sm = ScalarMappable(norm=norm, cmap="vlag")
        sm.set_array([])
        cbar = fig_cb.colorbar(sm, ax=ax_cb, orientation="horizontal", fraction=0.8, pad=0.3)
        cbar.set_label("Mean Δ Probability", fontsize=25)
        cbar.ax.tick_params(labelsize=25)
        ax_cb.remove()

        if out_dir is not None:
            fig_path = os.path.join(out_dir, f"{out_prefix}_dotbar.png")
            cb_path = os.path.join(out_dir, f"{out_prefix}_colorbar.png")
            fig.savefig(fig_path, bbox_inches="tight")
            fig_cb.savefig(cb_path, bbox_inches="tight")
            print(f"  - Saved plot -> {fig_path}")
            print(f"  - Saved colorbar -> {cb_path}")

    print(f"✅ Done in {time.time() - t_start:.1f} sec")
    plt.show()
    return {
        "adata": adata,
        "delta_wide_by_context": delta_wide_by_context,
        "combined_melted": combined_melted,
        "combo_counts": combo_counts,
        "fig": fig,
        "fig_cb": fig_cb,
    }




