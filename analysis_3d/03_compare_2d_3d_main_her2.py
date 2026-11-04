"""2D vs 3D MINGL on the same cells of the Main HER2 volume.

Question: does running MINGL in 3D change what we find, compared with the usual
2D analysis of a tissue section?

Simulated 2D sections: a plane every 10 µm through the volume; each section is
the cells within ±4 µm of the plane (about one cell radius -- the cells a real
section would cut through). 29 sections, ~2,840 cells each, no cell in two
sections. Each section is analysed with ordinary 2D MINGL (x, y only; each
section its own region), then compared with the 3D result for the same cells.

2D and 3D windows are matched by *physical size*: 2D k=10 has the same median
radius (18.2 µm) as 3D k=30, and 2D k=15 (22.3 µm) matches 3D k=50 (21.7 µm).
Both use the same n: the 3D probability-curve elbow of the k=30 run, read from
run_summary.csv. Windows are clustered with the same clusterer as the 3D run
(common.CLUSTERER, full k-means), including for the seed-only noise floor.

Comparisons (same cells):
1. Local windows: how different is each cell's cell-type mix in 2D vs 3D
   (total variation distance; how often the most common neighbor type differs).
2. Neighborhoods: ARI, and % of cells whose 2D neighborhood matches its 3D one
   after optimally pairing 2D with 3D neighborhoods. Reference points: 3D vs
   3D with only the k-means seed changed (noise floor), and 3D k=30 vs k=50.
3. Border cells: overlap of the 2D and 3D border sets.
4. Depth dependence: how many of the neighborhoods a single 2D section contains.

Input:  results/main_her2/main_her2_mingl3d.h5ad (from 02_mingl3d_main_her2.py)
Run:    .venv/bin/python analysis_3d/03_compare_2d_3d_main_her2.py
"""

from __future__ import annotations

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from mingl import tl3d
from common import CLUSTERER, CT, N_INIT, OUT, REG, membership_and_borders, name_neighborhoods, select_n_neighborhoods

IN_H5AD = OUT / "main_her2_mingl3d.h5ad"
OUT2 = OUT / "2d_vs_3d"

PAIRS = ((30, 10), (50, 15))  # (3D k, 2D k) with matched window radius


def _n_from_3d_run(k: int = 30) -> int:
    """The probability-curve elbow n of the 3D k=30 run (02_mingl3d_main_her2.py)."""
    rs = pd.read_csv(OUT / "run_summary.csv")
    return int(rs.loc[rs["k"] == k, "prob_elbow_n"].iloc[0])


N = _n_from_3d_run()
SECTION_PLANES = np.arange(10.0, 300.0, 10.0)  # µm
SECTION_HALF_WIDTH = 4.0  # µm, ~one cell radius
SEEDS = (1, 2, 3)  # noise floor: k-means seed only


def build_sections(cells: ad.AnnData) -> ad.AnnData:
    z = cells.obs["z"].to_numpy(float)
    dist = np.abs(z[:, None] - SECTION_PLANES[None, :])
    nearest = dist.argmin(axis=1)
    keep = dist[np.arange(len(z)), nearest] <= SECTION_HALF_WIDTH

    obs = cells.obs.loc[keep, ["x", "y", "z", CT]].copy()
    obs["cell_id"] = obs.index.to_numpy()
    obs["section_z"] = SECTION_PLANES[nearest[keep]]
    obs[REG] = [f"z{v:03.0f}" for v in obs["section_z"]]
    obs.index = np.arange(len(obs)).astype(str)  # KNN2 needs 0..n-1
    return ad.AnnData(obs=obs)


def fractions(windows: pd.DataFrame, cell_types: list[str]) -> np.ndarray:
    w = windows[cell_types].to_numpy(float)
    return w / w.sum(axis=1, keepdims=True)


def matched_agreement(lab3: np.ndarray, lab2: np.ndarray):
    """Pair 2D with 3D neighborhoods to maximize shared cells; % of cells that agree."""
    ct = pd.crosstab(lab3, lab2)
    r, c = linear_sum_assignment(-ct.to_numpy())
    mapping = {ct.columns[j]: ct.index[i] for i, j in zip(r, c)}
    return float(ct.to_numpy()[r, c].sum() / len(lab3)), mapping


def seed_noise_floor(cells: ad.AnnData, k3: int, cell_types: list[str], ids: np.ndarray):
    """Re-cluster the 3D windows exactly as run_mingl_over_n_clusters does, changing only the seed."""
    win = tl3d.KNN2(cells, z_key="z", cluster_col=CT, region_key=REG, ks=(k3,))[k3]
    X = StandardScaler().fit_transform(win[cell_types].to_numpy(np.float32)).astype(np.float32)

    def fit(seed):
        return tl3d.fit_neighborhood_labels(X, N, clusterer=CLUSTERER, random_state=seed, n_init=N_INIT).astype(str)

    base = fit(0)
    stored = cells.obs[f"neighborhood_k{k3}_n{N}"].astype(str).to_numpy()
    if adjusted_rand_score(base, stored) != 1.0:
        raise RuntimeError("Could not reproduce the stored 3D labels; noise floor would be invalid.")
    pos = cells.obs_names.get_indexer(ids)
    reruns = {s: fit(s) for s in SEEDS}
    return [adjusted_rand_score(base[pos], lab[pos]) for lab in reruns.values()], reruns[SEEDS[0]]


def plot_crosstab(lab3_named, lab2_named, title, path):
    ct = pd.crosstab(lab3_named, lab2_named, normalize="index") * 100
    order3 = pd.Series(lab3_named).value_counts().index
    order2 = pd.Series(lab2_named).value_counts().index
    fig, ax = plt.subplots(figsize=(0.75 * len(order2) + 8, 0.5 * len(order3) + 4))
    sns.heatmap(ct.loc[order3, order2], cmap="Blues", vmin=0, vmax=100, annot=True, fmt=".0f",
                annot_kws={"size": 7}, ax=ax, cbar_kws={"label": "% of the 3D neighborhood's cells"})
    ax.set_xlabel("2D neighborhood (per-section MINGL)")
    ax.set_ylabel("3D neighborhood")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_depth(sec_z, lab3_named, lab2_mapped_named, title, path):
    """Neighborhood mix per section, 3D labels vs 2D labels (2D colored by matched 3D neighborhood)."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    cats = pd.Series(lab3_named).value_counts().index.tolist()
    pal = dict(zip(cats, sns.color_palette("tab20", len(cats))))
    for ax, labs, sub in ((axes[0], lab3_named, "3D MINGL"), (axes[1], lab2_mapped_named, "2D MINGL, one section at a time")):
        tab = pd.crosstab(sec_z, labs, normalize="index").reindex(columns=cats, fill_value=0)
        left = np.zeros(len(tab))
        for c in cats:
            ax.barh(tab.index, tab[c], left=left, height=8, color=pal[c], label=c)
            left += tab[c].to_numpy()
        ax.set_title(sub)
        ax.set_xlabel("fraction of section's cells")
    axes[0].set_ylabel("section depth z (µm)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=pal[c]) for c in cats]
    fig.legend(handles, cats, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.14, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT2.mkdir(parents=True, exist_ok=True)
    cells = ad.read_h5ad(IN_H5AD)
    cells.obs[CT] = cells.obs[CT].astype(str)
    cell_types = sorted(cells.obs[CT].unique())

    sec = build_sections(cells)
    ids = sec.obs["cell_id"].to_numpy()
    sec_z = sec.obs["section_z"].to_numpy()
    print(f"{len(SECTION_PLANES)} sections, {sec.n_obs} cells "
          f"({np.median(sec.obs[REG].value_counts()):.0f} per section)")

    rows, fracs = [], {}
    for k3, k2 in PAIRS:
        other_k3 = [k for k, _ in PAIRS if k != k3][0]
        tag2 = f"2d_k{k2}_n{N}"
        wa2, win2, best2, ll2, pr2 = select_n_neighborhoods(sec, k2, cell_types, OUT2, z_key=None, prefix=f"2d_k{k2}")
        lab2 = wa2.obs[f"Neighborhood_{N}"].astype(str).to_numpy()
        sec.obs[f"neighborhood_{tag2}"] = lab2
        b2 = membership_and_borders(sec, f"neighborhood_{tag2}", k2, tag2, z_key=None).to_numpy() >= 2

        lab3 = cells.obs.loc[ids, f"neighborhood_k{k3}_n{N}"].astype(str).to_numpy()
        lab3_named = cells.obs.loc[ids, f"neighborhood_k{k3}_n{N}_name"].astype(str).to_numpy()
        b3 = cells.obs.loc[ids, f"n_positive_k{k3}_n{N}"].to_numpy() >= 2

        # 1. local windows
        win3 = tl3d.KNN2(cells, z_key="z", cluster_col=CT, region_key=REG, ks=(k3,))[k3].loc[ids]
        f3, f2 = fractions(win3, cell_types), fractions(win2, cell_types)
        fracs[("3D", k3)], fracs[("2D", k2)] = f3, f2
        tvd = 0.5 * np.abs(f3 - f2).sum(axis=1)
        dominant_differs = f3.argmax(axis=1) != f2.argmax(axis=1)

        # 2. neighborhoods
        agree, mapping = matched_agreement(lab3, lab2)
        names3 = dict(zip(lab3, lab3_named))
        lab2_mapped_named = np.array([names3[mapping[l]] if l in mapping else "unmatched" for l in lab2])
        _, _, names2 = name_neighborhoods(win2, lab2, cell_types)
        lab2_named = np.array([f"2D-N{l}: {names2[l]}" for l in lab2])
        noise, seed_labels = seed_noise_floor(cells, k3, cell_types, ids)
        seed_tag = f"k{k3}_n{N}_seed{SEEDS[0]}"
        cells.obs[f"neighborhood_{seed_tag}"] = seed_labels
        b3_seed = membership_and_borders(cells, f"neighborhood_{seed_tag}", k3, seed_tag).loc[ids].to_numpy() >= 2
        b3_otherk = cells.obs.loc[ids, f"n_positive_k{other_k3}_n{N}"].to_numpy() >= 2
        ari_k = adjusted_rand_score(lab3, cells.obs.loc[ids, f"neighborhood_k{other_k3}_n{N}"].astype(str))

        # 3. borders
        def jaccard(a, b):
            return float((a & b).sum() / max((a | b).sum(), 1))
        jacc = jaccard(b2, b3)

        # 4. depth: neighborhoods present (>= 1% of cells) per section
        present3 = pd.crosstab(sec_z, lab3, normalize="index").ge(0.01).sum(axis=1)
        present2 = pd.crosstab(sec_z, lab2, normalize="index").ge(0.01).sum(axis=1)

        plot_crosstab(lab3_named, lab2_named, f"3D (k={k3}) vs 2D (k={k2}) neighborhoods, same cells",
                      OUT2 / f"crosstab_3d_k{k3}_vs_2d_k{k2}.png")
        plot_depth(sec_z, lab3_named, lab2_mapped_named,
                   f"Neighborhood mix by depth: 3D (k={k3}) vs 2D (k={k2})\n(2D neighborhoods colored by their best-matching 3D neighborhood)",
                   OUT2 / f"depth_mix_3d_k{k3}_vs_2d_k{k2}.png")

        rows.append({
            "pair": f"3D k={k3} vs 2D k={k2}",
            "2D own n-selection (fallback n / prob elbow)": f"{best2} / {pr2}",
            "window: median TVD 2D vs 3D": round(float(np.median(tvd)), 3),
            "window: % cells TVD > 0.25": round(100 * float((tvd > 0.25).mean()), 1),
            "window: % cells most-common neighbor type differs": round(100 * float(dominant_differs.mean()), 1),
            "neighborhoods: ARI 2D vs 3D": round(adjusted_rand_score(lab3, lab2), 3),
            "neighborhoods: % cells matched 2D=3D": round(100 * agree, 1),
            "ref: ARI 3D seed change only (mean)": round(float(np.mean(noise)), 3),
            f"ref: ARI 3D k={k3} vs k={other_k3}": round(float(ari_k), 3),
            "borders: % cells 3D": round(100 * float(b3.mean()), 1),
            "borders: % cells 2D": round(100 * float(b2.mean()), 1),
            "borders: Jaccard 2D vs 3D": round(jacc, 3),
            "ref: borders Jaccard 3D seed change only": round(jaccard(b3, b3_seed), 3),
            "ref: borders Jaccard 3D k=30 vs k=50": round(jaccard(b3, b3_otherk), 3),
            "borders: % of 3D borders also 2D borders": round(100 * float(b2[b3].mean()), 1),
            "borders: % of 2D borders also 3D borders": round(100 * float(b3[b2].mean()), 1),
            "depth: neighborhoods per section, 3D (median, min-max)": f"{present3.median():.0f} ({present3.min()}-{present3.max()})",
            "depth: neighborhoods per section, 2D (median, min-max)": f"{present2.median():.0f} ({present2.min()}-{present2.max()})",
        })

    (k3a, k2a), (k3b, k2b) = PAIRS
    for dim, ka, kb in (("3D", k3a, k3b), ("2D", k2a, k2b)):
        ref = 0.5 * np.abs(fracs[(dim, ka)] - fracs[(dim, kb)]).sum(axis=1)
        print(f"ref: window median TVD, {dim} k={ka} vs k={kb}: {np.median(ref):.3f} "
              f"(% cells > 0.25: {100 * (ref > 0.25).mean():.1f})")

    summary = pd.DataFrame(rows).set_index("pair").T
    summary.to_csv(OUT2 / "summary.csv")
    print(summary.to_string())
    sec.write_h5ad(OUT2 / "sections_2d_mingl.h5ad", compression="gzip")


if __name__ == "__main__":
    main()
