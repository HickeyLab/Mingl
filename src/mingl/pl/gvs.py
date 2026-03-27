import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad

 
def plot_global_vs_subset_horizontal_buckets(
    data: Union[pd.DataFrame, ad.AnnData],
    neighborhood: str,
    bucket_map: Dict[str, List[str]],
    cell_type_color_map: Dict[str, str],
    min_count: int = 10,
    subset_region: Optional[str] = None,
    subset_patient: Optional[str] = None,
    subset_context: Optional[str] = None,
    patient_split_sep: str = "_",
    figsize: Tuple[int, int] = (18, 5),
    title_fontsize: int = 25,
    label_fontsize: int = 25,
    show_context: bool = False,
    # scverse key params
    region_key: str = "region",
    neigh_key: str = "neigh_name",
    cluster_key: str = "Cell Type",
    context_key: str = "Context",
) -> None:
    """
    AnnData/scverse compatible:
      - `data` can be AnnData (uses data.obs) OR a pandas DataFrame.
      - keys are configurable (region_key, neigh_key, cluster_key, context_key)
    """

    # ---- get dataframe from AnnData or df directly (no helper)
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        if not hasattr(data, "obs"):
            raise TypeError("`data` must be a pandas DataFrame or an AnnData object with `.obs`.")
        df = data.obs

    # required columns check
    required_cols = [region_key, neigh_key, cluster_key]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns in obs/df: {missing}")

    # optional context presence
    has_context = (context_key in df.columns)

    # filter to neighborhood
    df_neigh = df[df[neigh_key].astype(str) == str(neighborhood)].copy()
    if df_neigh.empty:
        raise ValueError(f"No data found for neighborhood '{neighborhood}'")

    # define subset (subset_df is subset of df_neigh)
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
        raise ValueError("Provide subset_region or subset_patient or subset_context (region takes precedence).")

    context_mode = subset_context is not None

    # optional context label display only
    if show_context and has_context and not subset_df.empty and not context_mode:
        uniq_ctx = subset_df[context_key].dropna().astype(str).unique()
        ctx_label = uniq_ctx[0] if len(uniq_ctx) == 1 else "Mixed"
        subset_label = f"{subset_label}\n({ctx_label})"

    # neighborhood-level denominators
    denom_global_neigh = max(len(df_neigh), 1)
    denom_subset_neigh = max(len(subset_df), 1)

    # plotting layout
    bucket_names = list(bucket_map.keys())
    fig, axes = plt.subplots(1, len(bucket_names), figsize=figsize, sharey=True)
    if len(bucket_names) == 1:
        axes = [axes]

    # iterate buckets
    for ax, bucket_name in zip(axes, bucket_names):
        candidates = bucket_map[bucket_name]

        # only candidate CTs actually present anywhere in the neighborhood
        neigh_cts = df_neigh[cluster_key].dropna().unique().tolist()
        present_cts = [ct for ct in candidates if ct in neigh_cts]

        # axis basics
        ax.set_title(bucket_name, fontsize=title_fontsize)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Global", subset_label], fontsize=label_fontsize)
        ax.tick_params(axis="x", labelsize=label_fontsize)
        ax.tick_params(axis="y", labelsize=label_fontsize)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        if len(present_cts) == 0:
            ax.text(
                0.5, 0.5, "No cell types\nin bucket",
                ha="center", va="center", fontsize=14, transform=ax.transAxes
            )
            ax.grid(False)
            continue

        # counts computed within the neighborhood (global) and within subset (subset)
        counts_global = (
            df_neigh[df_neigh[cluster_key].isin(present_cts)]
            .groupby(cluster_key)
            .size()
        )
        counts_subset = (
            subset_df[subset_df[cluster_key].isin(present_cts)]
            .groupby(cluster_key)
            .size()
        )

        # union of CTs that pass min_count in global OR subset
        global_pass = set(counts_global[counts_global > min_count].index.tolist())
        subset_pass = set(counts_subset[counts_subset > min_count].index.tolist())
        ct_union = sorted(
            global_pass.union(subset_pass),
            key=lambda x: counts_global.get(x, 0),
            reverse=True
        )

        if len(ct_union) == 0:
            empty_msg = (
                f"No cell types > {min_count} cells"
                if context_mode
                else f"No cell types > {min_count} in neighborhood or subset"
            )
            ax.text(
                0.5, 0.5, empty_msg,
                ha="center", va="center", fontsize=14, style="italic", transform=ax.transAxes
            )
            ax.grid(False)
            continue

        # compute percentages using NEIGHBORHOOD-LEVEL denominators
        pct_global = []
        pct_subset = []
        for ct in ct_union:
            g = int(counts_global.get(ct, 0))
            s = int(counts_subset.get(ct, 0))
            pct_global.append((g / denom_global_neigh * 100) if (g > min_count) else 0.0)
            pct_subset.append((s / denom_subset_neigh * 100) if (s > min_count) else 0.0)

        plot_df = pd.DataFrame(
            [pct_global, pct_subset],
            index=["Global", subset_label],
            columns=ct_union
        )

        colors = [cell_type_color_map.get(ct, "#999999") for ct in plot_df.columns]

        plot_df.plot(kind="barh", stacked=True, ax=ax, color=colors, width=0.65, legend=False)
        ax.grid(False)

        # dynamic x-axis scaling chosen from {25, 50, 100}
        raw_max = float(plot_df.sum(axis=1).max())
        if context_mode:
            if raw_max <= 25:
                nice_max = 25
            elif raw_max <= 50:
                nice_max = 50
            else:
                nice_max = 100

            n_ticks = 5 if nice_max % 4 == 0 else 4
            tick_vals = np.linspace(0, nice_max, n_ticks)
            tick_vals = np.unique(np.round(tick_vals).astype(int))
        else:
            if raw_max <= 25:
                nice_max = 25
                tick_vals = np.array([0, 8, 17, 25])
            elif raw_max <= 50:
                nice_max = 50
                tick_vals = np.array([0, 17, 33, 50])
            else:
                nice_max = 100
                tick_vals = np.array([0, 25, 50, 75, 100])

        padding = 0.02 * nice_max
        ax.set_xlim(0, nice_max + padding)

        tick_vals = tick_vals[(tick_vals >= 0) & (tick_vals <= nice_max)]
        ax.set_xticks(tick_vals)
        ax.set_xticklabels([f"{int(t)}" for t in tick_vals], fontsize=label_fontsize)

    fig.supxlabel("Proportion of Neighborhood (%)", fontsize=label_fontsize)
    plt.tight_layout()
    plt.show()


def auto_assign_buckets(unique_cts: Sequence[str]) -> Tuple[List[str], List[str], List[str]]:
    epikeys = [r"\bepithel", r"\benter", r"\bgoblet", r"\bkeratin", r"\bcolono", r"\bpaneth"]
    mesokeys = [r"\bfibro", r"\bfibrobla", r"\bendothel", r"\bpericyte", r"\bmesench"]
    immunekeys = [r"\bt\s?cell", r"\bb\s?cell", r"\bmacroph", r"\bdendrit", r"\bnk\b", r"\bneutroph", r"\bmonocyte"]

    epithelial, mesenchymal, immune = [], [], []

    for ct in unique_cts:
        if ct is None or (isinstance(ct, float) and np.isnan(ct)):
            continue
        name = str(ct).lower()
        assigned = False

        for p in epikeys:
            if re.search(p, name):
                epithelial.append(str(ct)); assigned = True; break
        if assigned:
            continue

        for p in mesokeys:
            if re.search(p, name):
                mesenchymal.append(str(ct)); assigned = True; break
        if assigned:
            continue

        for p in immunekeys:
            if re.search(p, name):
                immune.append(str(ct)); assigned = True; break

    return sorted(set(epithelial)), sorted(set(mesenchymal)), sorted(set(immune))


# --- Example (AnnData)
# unique_cts = adata.obs["Cell Type"].unique()
# epi, mes, imm = auto_assign_buckets(unique_cts)
# bucket_map = {"Epithelial": epi, "Mesenchymal": mes, "Immune": imm}
# plot_global_vs_subset_horizontal_buckets_scverse(
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
