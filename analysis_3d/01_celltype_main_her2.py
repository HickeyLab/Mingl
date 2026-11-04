"""Cell typing for the Kuett 2022 Main HER2 3D IMC volume.

The deposit has per-cell mean marker intensities but no cell-type labels, so we
phenotype here:

1. arcsinh(x / 1) transform (standard IMC cofactor), clip each marker at its
   99.9th percentile (not the usual 99th: immune cells are <1% of this volume
   and a 99th-percentile clip would flatten them), z-score.
2. Leiden clustering on a kNN graph of the lineage markers only (state markers
   such as Ki-67/pS6/cPARP and the DNA/histone channels are excluded so cells
   cluster by identity, not by cycling/death state).
3. Clusters are named from their marker profiles via ``CLUSTER_LABELS`` below
   (filled in after inspecting ``cluster_marker_means.csv`` / the heatmap).

Run:  .venv/bin/python analysis_3d/01_celltype_main_her2.py
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

DATA = Path.home() / "Desktop/Mingle_data/kuett2022_3dimc"
IN_H5AD = DATA / "processed/MainHer2BreastCancerModel.h5ad"
OUT = DATA / "results/main_her2"

LINEAGE_MARKERS = [
    "panCK", "CK5", "CK7", "CK8/18", "CK14", "CK19", "E/P-Cadherin", "HER2 (bis)",
    "CD45", "CD3", "CD8a", "CD20", "CD68", "CD138",
    "SMA", "Vimentin", "Collagen I", "vWF+ CD31",
]
STATE_MARKERS = ["Ki-67", "phospho-H3", "pS6", "cPARP+cCasp3", "CD44"]

LEIDEN_RESOLUTION = 1.0
SEED = 0

# Leiden cluster -> cell type, assigned from cluster_marker_means.csv (z-scores).
# Notes on the judgment calls:
# - 16 / 19 / 20 are keratin+ tumor-like cells carrying a single immune marker
#   (CD68 / CD8a / CD3) *without* CD45 (11% / 7% / 0% CD45+, vs 92% in true
#   T cells). Likely spillover between touching cells; kept as their own
#   "ambiguous" type rather than forced into tumor or immune.
# - 7 is luminal-keratin+ HER2+ but also SMA+/CK5+: tumor at the myoepithelial
#   interface (5 is the clean CK5+ SMA+ myoepithelial population).
# - 21 is CD138+ with low keratins -> plasma cells (CD138 is also epithelial,
#   hence the keratin check).
# - 12 is low for every marker; 2 (53 cells) is high for every marker (artifact).
CLUSTER_LABELS: dict[str, str] = {
    "14": "Tumor HER2-high",
    "13": "Tumor HER2+ CK7-",
    "17": "Tumor CK7+ CK19+",
    "15": "Tumor CK7+ CK19+",
    "18": "Tumor CK14+",
    "11": "Tumor marker-low",
    "7": "Tumor SMA+ CK5+",
    "5": "Myoepithelial",
    "4": "Fibroblasts (Vimentin+)",
    "6": "Fibroblasts (Collagen+)",
    "10": "Stroma (other)",
    "1": "Endothelial / vascular",
    "0": "T cells",
    "3": "B cells",
    "8": "Macrophages",
    "9": "Macrophages",
    "21": "Plasma cells",
    "16": "Mixed epithelial/immune (ambiguous)",
    "19": "Mixed epithelial/immune (ambiguous)",
    "20": "Mixed epithelial/immune (ambiguous)",
    "12": "Unassigned (low signal / artifact)",
    "2": "Unassigned (low signal / artifact)",
}


def transform(adata: ad.AnnData) -> np.ndarray:
    X = np.arcsinh(np.asarray(adata.X, dtype=np.float64))
    hi = np.percentile(X, 99.9, axis=0)
    X = np.minimum(X, hi)
    return (X - X.mean(axis=0)) / X.std(axis=0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(IN_H5AD)

    Z = transform(adata)
    adata.layers["arcsinh_z"] = Z.astype(np.float32)

    lin = [adata.var_names.get_loc(m) for m in LINEAGE_MARKERS]
    adata.obsm["X_lineage"] = Z[:, lin]

    sc.pp.neighbors(adata, use_rep="X_lineage", n_neighbors=15, random_state=SEED)
    sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION, flavor="igraph", n_iterations=2,
                 random_state=SEED, key_added="leiden")

    means = pd.DataFrame(Z, columns=adata.var_names).groupby(adata.obs["leiden"].to_numpy()).mean()
    means.index.name = "leiden"
    means.insert(0, "n_cells", adata.obs["leiden"].value_counts().reindex(means.index).to_numpy())
    means = means.sort_values("n_cells", ascending=False)
    means.round(2).to_csv(OUT / "cluster_marker_means.csv")

    fig, ax = plt.subplots(figsize=(14, 0.35 * len(means) + 2))
    sns.heatmap(means[LINEAGE_MARKERS + STATE_MARKERS], cmap="RdBu_r", center=0, vmin=-2, vmax=2,
                ax=ax, cbar_kws={"label": "mean z-score"})
    ax.set_yticklabels([f"{i}  (n={n})" for i, n in zip(means.index, means["n_cells"])], rotation=0)
    ax.set_title("Main HER2: Leiden clusters x markers (arcsinh, z-scored)")
    fig.tight_layout()
    fig.savefig(OUT / "cluster_marker_heatmap.png", dpi=150)

    if CLUSTER_LABELS:
        missing = set(adata.obs["leiden"].unique()) - set(CLUSTER_LABELS)
        if missing:
            raise ValueError(f"Unlabeled Leiden clusters: {sorted(missing, key=int)}")
        adata.obs["cell_type"] = adata.obs["leiden"].map(CLUSTER_LABELS).astype("category")

        ct_means = pd.DataFrame(Z, columns=adata.var_names).groupby(adata.obs["cell_type"].to_numpy()).mean()
        counts = adata.obs["cell_type"].value_counts()
        ct_means = ct_means.loc[counts.index]
        fig, ax = plt.subplots(figsize=(14, 0.4 * len(ct_means) + 2))
        sns.heatmap(ct_means[LINEAGE_MARKERS + STATE_MARKERS], cmap="RdBu_r", center=0, vmin=-2, vmax=2,
                    ax=ax, cbar_kws={"label": "mean z-score"})
        ax.set_yticklabels([f"{c}  (n={counts[c]})" for c in ct_means.index], rotation=0)
        ax.set_title("Main HER2: cell types x markers")
        fig.tight_layout()
        fig.savefig(OUT / "celltype_marker_heatmap.png", dpi=150)
        print(counts.to_string())

    del adata.obsp["distances"], adata.obsp["connectivities"]
    adata.write_h5ad(OUT / "main_her2_celltyped.h5ad", compression="gzip")
    print(f"{adata.n_obs} cells, {adata.obs['leiden'].nunique()} Leiden clusters -> {OUT}")


if __name__ == "__main__":
    main()
