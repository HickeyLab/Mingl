import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
# Load the wide-format delta DataFrame
delta_df = pd.read_csv(r"/Volumes/data/MINGLE/Data/Esophagus/20251217_all_regions_delta_probs.csv")

# Add 'region' column if missing
if "region" not in delta_df.columns:
    delta_df["region"] = delta_df["cellid"].astype(str).str.extract(r'_([^_]+)$')

# Identify delta columns and melt to long format
delta_cols = [col for col in delta_df.columns if col.endswith("_delta")]

delta_long = delta_df.melt(
    id_vars=["cellid", "region", "neigh_name"],
    value_vars=delta_cols,
    var_name="Neighborhood",
    value_name="Delta"
)

# Clean up Neighborhood column names
delta_long["Neighborhood"] = delta_long["Neighborhood"].str.replace("_delta", "", regex=False)
 
# Keep only assigned neighborhood deltas
delta_long = delta_long[delta_long["Neighborhood"] == delta_long["neigh_name"]].dropna(subset=["Delta"])
# Count cells per region × neighborhood
counts = (
    delta_long.groupby(["region", "Neighborhood"])["cellid"]
    .nunique()
    .reset_index()
    .rename(columns={"cellid": "count"})
)

# Filter valid combinations
valid_combos = counts[counts["count"] >= 10][["region", "Neighborhood"]]
delta_long = delta_long.merge(valid_combos, on=["region", "Neighborhood"], how="inner")
# -------------------------
# Inputs: delta_long
# must contain columns: "region", "Neighborhood", "Delta"
# -------------------------

# -------------------------
# Context color palette
# -------------------------
palette = {
    "Normal": "skyblue",
    "Tumor": "red",
    "Metaplasia": "orange",
    "Dysplasia": "purple",
    "Unknown": "gray"
}

# -------------------------
# Explicit region → context mapping
# -------------------------
normal_regions = ['E08_reg002','E08_reg003','E17_reg001']
tumor_regions = ['E08_reg004','E08_reg005','E11_reg001','E19_reg003','E19_reg004',
                 'E11_reg005','E11_reg006','E17_reg005']
metaplasia_regions = ['E08_reg006','E08_reg007','E11_reg002','E11_reg003','E11_reg004',
                      'E12_reg002','E12_reg003','E17_reg002','E17_reg003','E17_reg004',
                      'E19_reg001','E12_reg004','E12_reg005','E17_reg006']
dysplasia_regions = ['E08_reg001','E12_reg001','E19_reg002']

def map_region_to_context(region):
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

# -------------------------
# 1) Compute mean Δ and counts
# -------------------------
mean_delta = (
    delta_long
    .groupby(["region", "Neighborhood"])["Delta"]
    .mean()
    .reset_index(name="mean_delta")
)

cell_counts = (
    delta_long
    .groupby(["region", "Neighborhood"])
    .size()
    .reset_index(name="n_cells")
)

plot_df = mean_delta.merge(cell_counts, on=["region", "Neighborhood"])

# -------------------------
# 2) Pivot for |mean Δ| aggregation
# -------------------------
pivot_df = plot_df.pivot(index="region", columns="Neighborhood", values="mean_delta")

all_regions = delta_long["region"].unique()
all_neighborhoods = delta_long["Neighborhood"].unique()
pivot_df = pivot_df.reindex(index=all_regions, columns=all_neighborhoods)

# -------------------------
# 3) Order regions by descending sum(|mean Δ|)
# -------------------------
sum_abs = pivot_df.abs().sum(axis=1)
region_order = sum_abs.sort_values(ascending=False).index.tolist()
neighborhood_order = list(pivot_df.columns)

# -------------------------
# 4) Reorder plotting dataframe
# -------------------------
plot_df["region"] = pd.Categorical(plot_df["region"], categories=region_order, ordered=True)
plot_df["Neighborhood"] = pd.Categorical(plot_df["Neighborhood"], categories=neighborhood_order, ordered=True)
plot_df = plot_df.dropna(subset=["region", "Neighborhood"]).reset_index(drop=True)

# attach context (useful for debugging / future legends)
plot_df["Context"] = plot_df["region"].apply(map_region_to_context)


print(plot_df)