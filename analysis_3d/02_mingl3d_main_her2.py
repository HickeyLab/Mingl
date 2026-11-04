"""3D MINGL on the Kuett 2022 Main HER2 volume.

Same workflow as the 2D analyses, with neighbors found in (x, y, z):

1. 3D k-NN cell-type windows                       (tl3d.KNN2, z_key="z")
2. Sweep the neighborhood count n = 1..40 and pick it with the elbow +
   plateau method                                  (tl3d.run_mingl_over_n_clusters,
                                                    tl.find_elbow_point,
                                                    tl.find_best_unsupervised_plateau)
   Windows are clustered with full k-means (10 restarts; common.CLUSTERER)
   rather than MINGL's default MiniBatchKMeans, which is seed-unstable here.
3. Neighborhood centroids                          (tl3d.centroid_Calculation)
4. Per-cell membership probabilities               (tl3d.cpu_gmm_probability)
5. Border cells: >= 2 neighborhoods above 0.25     (tl.findPositives)

k is larger than the 2D default (10) so windows reach across depth: at k=30
about 27% of neighbors are > 10 µm (one cell) away in z vs 8% at k=10, and
windows span ~30 µm of depth. k=50 is run as a sensitivity check.

Steps 3-5 are run at the elbow + plateau selection and at the probability-curve
elbow for each k, and additionally at any n selected for the other k, so every
k has outputs at a common n for the 2D-vs-3D comparison. Outputs are tagged
``k{k}_n{n}``.

Input:  results/main_her2/main_her2_celltyped.h5ad (from 01_celltype_main_her2.py)
Run:    .venv/bin/python analysis_3d/02_mingl3d_main_her2.py
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import adjusted_rand_score

from common import CLUSTERER, CT, N_INIT, OUT, membership_and_borders, name_neighborhoods, select_n_neighborhoods

IN_H5AD = OUT / "main_her2_celltyped.h5ad"
K_VALUES = (30, 50)


def plot_composition(comp, enrich, names, sizes, tag, out: Path):
    order = sizes.sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(16, 0.45 * len(order) + 3))
    sns.heatmap(enrich.loc[order], cmap="RdBu_r", center=0, vmin=-3, vmax=3, ax=ax,
                annot=(comp.loc[order] * 100).round(0).astype(int), fmt="d", annot_kws={"size": 7},
                cbar_kws={"label": "log2 enrichment vs volume (numbers = % of window)"})
    ax.set_yticklabels([f"N{nb}: {names[nb]}  (n={sizes[nb]})" for nb in order], rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
    ax.set_title(f"Main HER2 3D neighborhoods ({tag}): window composition")
    fig.tight_layout()
    fig.savefig(out / f"{tag}_neighborhood_composition.png", dpi=150)
    plt.close(fig)


def plot_slices(cells: ad.AnnData, col: str, tag: str, out: Path):
    """Neighborhoods in three xy slices (top/middle/bottom) and an xz side view."""
    obs = cells.obs
    cats = sorted(obs[col].unique(), key=lambda s: int(s.split(":")[0][1:]))
    pal = dict(zip(cats, sns.color_palette("tab20", len(cats))))
    zs = obs["z"].quantile([0.2, 0.5, 0.8]).to_numpy()
    y_mid = obs["y"].median()

    fig, axes = plt.subplots(1, 4, figsize=(28, 8), gridspec_kw={"width_ratios": [1, 1, 1, 1.3]})
    for ax, z0 in zip(axes[:3], zs):
        s = obs[(obs["z"] - z0).abs() <= 3]
        ax.scatter(s["x"], s["y"], c=[pal[c] for c in s[col]], s=4, linewidths=0)
        ax.set_title(f"xy slice, z ≈ {z0:.0f} µm", fontsize=12)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
    s = obs[(obs["y"] - y_mid).abs() <= 6]
    axes[3].scatter(s["x"], s["z"], c=[pal[c] for c in s[col]], s=5, linewidths=0)
    axes[3].set_title(f"side view (xz), y ≈ {y_mid:.0f} µm", fontsize=12)
    axes[3].set_aspect("equal"); axes[3].set_xlabel("x (µm)"); axes[3].set_ylabel("z (µm)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=pal[c], label=c) for c in cats]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11, frameon=False)
    fig.suptitle(f"Main HER2 3D neighborhoods ({tag})", fontsize=14)
    fig.tight_layout(rect=(0, 0.12, 1, 0.97))
    fig.savefig(out / f"{tag}_neighborhood_slices.png", dpi=130)
    plt.close(fig)


def run_downstream(cells, k, n, how, tag, labels, windows, cell_types, ll_n, prob_n) -> dict:
    """Steps 3-5 for one (k, n): name, centroids, GMM probabilities, border cells."""
    nb_col = f"neighborhood_{tag}"
    cells.obs[nb_col] = labels

    comp, enrich, names = name_neighborhoods(windows, labels, cell_types)
    sizes = pd.Series(labels).value_counts()
    plot_composition(comp, enrich, names, sizes, tag, OUT)
    cells.obs[f"{nb_col}_name"] = [f"N{l}: {names[l]}" for l in labels]
    plot_slices(cells, f"{nb_col}_name", tag, OUT)

    n_pos = membership_and_borders(cells, nb_col, k, tag)

    table = pd.DataFrame({
        "name": pd.Series(names),
        "n_cells": sizes,
        "pct_cells": (100 * sizes / len(labels)).round(1),
        "pct_border": (100 * (n_pos >= 2).groupby(labels).mean()).round(1),
    })
    table.index.name = "neighborhood"
    table = table.sort_values("n_cells", ascending=False)
    table.to_csv(OUT / f"{tag}_neighborhoods.csv")
    comp.round(4).to_csv(OUT / f"{tag}_neighborhood_composition.csv")
    print(f"\n=== {tag} ({how}; elbows: LL {ll_n}, prob {prob_n}) ===")
    print(table.to_string())
    return {
        "tag": tag, "k": k, "n_neighborhoods": n, "how_chosen": how, "ll_elbow_n": ll_n, "prob_elbow_n": prob_n,
        "clusterer": CLUSTERER, "n_init": N_INIT,
        "pct_border_cells": round(100 * float((n_pos >= 2).mean()), 1),
        "pct_no_neighborhood_above_thr": round(100 * float((n_pos == 0).mean()), 1),
    }


def main() -> None:
    cells = ad.read_h5ad(IN_H5AD)
    cells.obs[CT] = cells.obs[CT].astype(str)
    cell_types = sorted(cells.obs[CT].unique())
    report = []

    sweeps = {k: select_n_neighborhoods(cells, k, cell_types, OUT) for k in K_VALUES}
    # best_n: elbow+plateau selection (falls back to the composite-score peak when no
    # plateau lies between the two elbows); prob_n: the probability-curve elbow.
    # Every n selected for any k is also run for the other k ("shared").
    all_n = sorted({n for (_, _, best_n, _, prob_n) in sweeps.values() for n in (best_n, prob_n)})
    for k in K_VALUES:
        wa, windows, best_n, ll_n, prob_n = sweeps[k]
        for n in all_n:
            how = "selected" if n == best_n else "prob_elbow" if n == prob_n else "shared"
            tag = f"k{k}_n{n}"
            labels = wa.obs[f"Neighborhood_{n}"].astype(str).to_numpy()
            report.append(run_downstream(cells, k, n, how, tag, labels, windows, cell_types, ll_n, prob_n))

    tags = [r["tag"] for r in report]
    ari = pd.DataFrame(index=tags, columns=tags, dtype=float)
    for t1 in tags:
        for t2 in tags:
            ari.loc[t1, t2] = adjusted_rand_score(cells.obs[f"neighborhood_{t1}"], cells.obs[f"neighborhood_{t2}"])
    ari.round(3).to_csv(OUT / "neighborhood_agreement_ari.csv")
    print("\nNeighborhood agreement (adjusted Rand index):\n" + ari.round(2).to_string())

    pd.DataFrame(report).to_csv(OUT / "run_summary.csv", index=False)
    print("\n" + pd.DataFrame(report).to_string(index=False))
    cells.write_h5ad(OUT / "main_her2_mingl3d.h5ad", compression="gzip")


if __name__ == "__main__":
    main()
