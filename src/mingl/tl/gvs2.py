import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
import MINGL as mg
 
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
        # --- explicit region -> context mapping
        normal_regions = ['E08_reg002','E08_reg003','E17_reg001']
        tumor_regions = ['E08_reg004','E08_reg005','E11_reg001','E19_reg003','E19_reg004',
                         'E11_reg005','E11_reg006','E17_reg005']
        metaplasia_regions = ['E08_reg006','E08_reg007','E11_reg002','E11_reg003','E11_reg004',
                              'E12_reg002','E12_reg003','E17_reg002','E17_reg003','E17_reg004',
                              'E19_reg001','E12_reg004','E12_reg005','E17_reg006']
        dysplasia_regions = ['E08_reg001','E12_reg001','E19_reg002']

        def map_region_to_context(region: str) -> str:
            if region in normal_regions:
                return "Normal"
            elif region in tumor_regions:
                return "Tumor"
            elif region in metaplasia_regions:
                return "Metaplasia"
            elif region in dysplasia_regions:
                return "Dysplasia"
            else:
                return "Unknown"

        # ---- ensure Context exists on the *source* AnnData.obs (persistent)
        if not isinstance(data, pd.DataFrame):
            # df is data.obs in this case
            if context_key not in df.columns:
                df[context_key] = df[region_key].astype(str).apply(map_region_to_context)
            has_context = True  # update flag

        # ---- also ensure Context exists on df_neigh (local copy used for plotting/subsetting)
        if context_key not in df_neigh.columns:
            df_neigh[context_key] = df_neigh[region_key].astype(str).apply(map_region_to_context)

        # ---- now subset by context
        subset_df = df_neigh[df_neigh[context_key].astype(str) == str(subset_context)].copy()
        subset_label = f"{subset_context}"


    else:
        raise ValueError("Provide subset_region or subset_patient or subset_context (region takes precedence).")

    # optional context label display only
    if show_context and has_context and not subset_df.empty:
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
            ax.text(
                0.5, 0.5, f"No cell types > {min_count} in neighborhood or subset",
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

import MINGL as mg

file_path = r"/Volumes/data/MINGLE/Data/Esophagus/all_regions_from_h5mu.csv"

# Read the CSV file
df = mg.pp.read_file(file_path)

epithelial_cell_types = [
 'Squamous Annexin A1+','Squamous p63+','Squamou p63+ EGFRhi','Epithelial',
 'Epithelial Ki67+ p53+','Epithelial MUC1+ Ki67+','Epithelial CK7+ p53+',
 'Epithelial pH2AX+','Epithelial CD73hi','Epithelial p53+','Epithelial CK7+',
 'Epithelial HLADR+','Foveloar','Foveloar Ki67+ p53+','Foveloar p53+',
 'Goblet','Goblet p53+','Goblet Ki67+ p53+','Paneth','Chief','Parietal',
 'Neck','Neck p53+','Neck Ki67+ p53+','Neuroendocrine'
]

mesenchymal_cell_types = [
 'Endothelial','Endothelial CD36hi','Endothelial aSMAhi',
 'Lymphatic','Lymphatic CD73+','Stroma','Stroma CD73+','Smooth Muscle','Nerve'
]

immune_cell_types = [
 'Neutrophil','M1 Macrophage','M2 Macrophage','CD4+ Treg','CD4+ T cell PD1+',
 'CD4+ T cell','CD8+ T cell','CD8+ T cell PD1+','B cell','Plasma','DC'
]

# Example usage of auto-assignment (uncomment if you want to use it)
# epithelial_cell_types_auto, mesenchymal_cell_types_auto, immune_cell_types_auto = auto_assign_buckets(unique_cell_types)
# print("Epithelial (auto):", epithelial_cell_types_auto)
# print("Mesenchymal (auto):", mesenchymal_cell_types_auto)
# print("Immune (auto):", immune_cell_types_auto)

# If you used auto-assign and want to use those lists, set:
# epithelial_cell_types = epithelial_cell_types_auto
# mesenchymal_cell_types = mesenchymal_cell_types_auto
# immune_cell_types = immune_cell_types_auto

# === 2) Consolidate into a mapping for plotting ===
bucket_map = {
    "Epithelial": epithelial_cell_types,
    "Mesenchymal": mesenchymal_cell_types,
    "Immune": immune_cell_types
}

cell_type_color_map = {
  "B cell": "#00ff00",
  "CD4+ T cell": "#ff00ff",
  "CD4+ T cell PD1+": "#0080ff",
  "CD4+ Treg": "#ff8000",
  "CD8+ T cell": "#80bf80",
  "CD8+ T cell PD1+": "#4c06b1",
  "Chief": "#c40129",
  "DC": "#fc82cb",
  "Endothelial": "#ebff1d",
  "Endothelial CD36hi": "#3d8004",
  "Endothelial aSMAhi": "#00ffff",
  "Epithelial": "#00ff80",
  "Epithelial CD73hi": "#008080",
  "Epithelial CK7+": "#87e8fc",
  "Epithelial CK7+ p53+": "#945576",
  "Epithelial HLADR+": "#8080ff",
  "Epithelial Ki67+ p53+": "#81e103",
  "Epithelial MUC1+ Ki67+": "#383049",
  "Epithelial p53+": "#f7d689",
  "Epithelial pH2AX+": "#ae33de",
  "Foveloar": "#f32387",
  "Foveloar Ki67+ p53+": "#0000ff",
  "Foveloar p53+": "#33bfca",
  "Goblet": "#11c23f",
  "Goblet Ki67+ p53+": "#a16411",
  "Goblet p53+": "#d99063",
  "Lymphatic": "#0c44c5",
  "Lymphatic CD73+": "#5bf953",
  "M1 Macrophage": "#a70183",
  "M2 Macrophage": "#b2acc8",
  "Neck": "#ef4028",
  "Neck Ki67+ p53+": "#60020f",
  "Neck p53+": "#5736f8",
  "Nerve": "#000080",
  "Neuroendocrine": "#bdac04",
  "Neutrophil": "#5b855c",
  "Paneth": "#b0ff80",
  "Parietal": "#516bac",
  "Plasma": "#5ff8b0",
  "Smooth Muscle": "#8000ff",
  "Squamou p63+ EGFRhi": "#8f2c33",
  "Squamous Annexin A1+": "#fe44d4",
  "Squamous p63+": "#05410f",
  "Stroma": "#f3c0fb",
  "Stroma CD73+": "#be68b5"
}


plot_global_vs_subset_horizontal_buckets(
    data=df,
    neighborhood="Mature Intestinal and Immune",
    bucket_map=bucket_map,
    cell_type_color_map=cell_type_color_map,
    min_count=10,
    subset_region="E19_reg003",
    figsize=(11, 3)
)

# Choose ONE neighborhood and ONE context
neighborhood_name = "Mature Intestinal and Immune"
context_name = "Tumor"

# Minimum number of cells required to be plotted
min_cells = 10

plot_global_vs_subset_horizontal_buckets(
    data=df,
    neighborhood=neighborhood_name,
    subset_context=context_name,
    bucket_map=bucket_map,
    cell_type_color_map=cell_type_color_map,
    min_count=min_cells,
    figsize=(10, 3),
    title_fontsize=25,
    label_fontsize=25
)