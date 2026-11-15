"""Boundary effects and border robustness on the Main HER2 volume.

Two questions left open by 03_compare_2d_3d_main_her2.py:

A. Boundary effects. Cells near a face of the
   imaged block have truncated k-NN windows: the window reaches further and is
   one-sided. How many cells are affected, how much, and do the 2D-vs-3D
   numbers change when those cells are left out?

   Zones use the bounding box of the cell centroids and the median k-NN radius
   R of the run: "z-face" = within R of the top or bottom face, "xy-face" =
   within R of a side face only, "interior" = the rest. The box is only a
   proxy for the tissue edge, so each zone also reports a data-driven check:
   the offset between a cell and the centroid of its window, as a fraction of
   the window radius (0 for a symmetric window, 3/8 for a half ball).

B. Border robustness. 03 compares border sets by exact-cell Jaccard at one
   threshold. That number is 0.09 for 2D vs 3D but only 0.15 for 3D k=30 vs
   k=50, so on its own it cannot separate "2D misses 3D borders" from "border
   sets are fragile". Here every comparison is repeated with measures that do
   not depend on exact-cell identity or on one threshold, next to the same
   references (seed only, k only) and to chance:

   - chance Jaccard for two independent sets of the same sizes;
   - the border score itself. A cell is a border cell when >= 2 membership
     probabilities exceed the threshold, i.e. when its second-largest
     probability p2 does, so p2 is the continuous border score: Spearman
     correlation of p2, and AUC of one run's p2 for the other run's borders;
   - Jaccard across thresholds, and at matched prevalence (top q% of p2);
   - distance-tolerant overlap: share of one run's border cells with a border
     cell of the other run within r µm, against a null that mirrors the other
     run's border cells across the block (same spatial clumping, no
     correspondence). Shuffling border status instead would spread the border
     cells evenly and overstate chance, because real borders are clumped;
   - which cell types are enriched among border cells (correlation of the
     per-type log2 enrichment).

   Everything is reported on all section cells and on interior cells only.
   The package threshold tools (tl3d.threshold_sensitivity_analysis,
   tl3d.spatial_border_clustering_null) are also run on each stored result.

Nothing is re-selected: neighborhoods and probabilities are read from the 02
and 03 outputs. The only refit is the seed-only reference (k-means seed 1).

Inputs: results/main_her2/main_her2_mingl3d.h5ad            (02)
        results/main_her2/2d_vs_3d/sections_2d_mingl.h5ad   (03)
Output: results/main_her2/boundary_and_borders/
Run:    .venv/bin/python analysis_3d/05_boundary_and_border_robustness_main_her2.py
"""

from __future__ import annotations

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from mingl import tl3d
from border_metrics import THRESHOLDS, border_score, compare_border_scores, fractions, plot_robustness
from common import CLUSTERER, CT, N_INIT, OUT, REG, THRESHOLD, membership_and_borders, quiet

IN_3D = OUT / "main_her2_mingl3d.h5ad"
IN_2D = OUT / "2d_vs_3d" / "sections_2d_mingl.h5ad"
OUT5 = OUT / "boundary_and_borders"

PAIRS = ((30, 10), (50, 15))  # (3D k, 2D k) with matched window radius, as in 03
N_VALUES = (4, 9, 15)  # every n stored by 02
SEED = 1  # seed-only reference
DEPTH_BINS = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 152])  # µm from the nearest z-face


def _n_from_3d_run(k: int = 30) -> int:
    """The probability-curve elbow n of the 3D k=30 run, the n used by 03."""
    rs = pd.read_csv(OUT / "run_summary.csv")
    return int(rs.loc[rs["k"] == k, "prob_elbow_n"].iloc[0])


N = _n_from_3d_run()


# ----------------------------------------------------------------------------
# A. boundary effects
# ----------------------------------------------------------------------------

def window_geometry(xyz: np.ndarray, ks) -> dict[int, pd.DataFrame]:
    """Per cell and k: window radius, one-sidedness and depth reach of the k-NN window.

    The window is the cell plus its k-1 nearest neighbors, as in KNN2.
    """
    dist, idx = cKDTree(xyz).query(xyz, k=max(ks))
    out = {}
    for k in ks:
        nb = xyz[idx[:, :k]]
        radius = dist[:, k - 1]
        offset = np.linalg.norm(nb.mean(axis=1) - xyz, axis=1)
        out[k] = pd.DataFrame({
            "radius": radius,
            "offset_ratio": offset / radius,
            "z_reach": np.abs(nb[:, :, 2] - xyz[:, None, 2]).max(axis=1),
        })
    return out


def face_distances(xyz: np.ndarray) -> pd.DataFrame:
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    d = np.minimum(xyz - lo, hi - xyz)
    return pd.DataFrame({"d_xy_face": d[:, :2].min(axis=1), "d_z_face": d[:, 2], "d_face": d.min(axis=1)})


def zone_of(faces: pd.DataFrame, radius: float) -> np.ndarray:
    return np.where(faces["d_z_face"] < radius, "z-face",
                    np.where(faces["d_xy_face"] < radius, "xy-face", "interior"))


def label_tvd(a: pd.Series, b: pd.Series) -> float:
    """Total variation distance between two neighborhood-label distributions."""
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    return float(0.5 * pa.subtract(pb, fill_value=0).abs().sum())


def boundary_tables(cells: ad.AnnData, faces: pd.DataFrame, geom: dict[int, pd.DataFrame]):
    zone_rows, depth_rows = [], []
    depth_bin = pd.cut(faces["d_z_face"], DEPTH_BINS, right=False, include_lowest=True)
    for k, g in geom.items():
        radius = float(g["radius"].median())
        zone = zone_of(faces, radius)
        border = {n: cells.obs[f"n_positive_k{k}_n{n}"].to_numpy() >= 2 for n in N_VALUES}
        labels = cells.obs[f"neighborhood_k{k}_n{N}"].astype(str).reset_index(drop=True)
        interior = zone == "interior"
        for z in ("z-face", "xy-face", "interior"):
            m = zone == z
            row = {
                "k": k, "zone": z, "zone_width_um": round(radius, 1), "n_cells": int(m.sum()),
                "pct_cells": round(100 * m.mean(), 1),
                "median_radius_um": round(float(g.loc[m, "radius"].median()), 1),
                "radius_vs_interior": round(float(g.loc[m, "radius"].median() / g.loc[interior, "radius"].median()), 2),
                "median_offset_ratio": round(float(g.loc[m, "offset_ratio"].median()), 3),
                "median_z_reach_um": round(float(g.loc[m, "z_reach"].median()), 1),
            }
            for n in N_VALUES:
                row[f"pct_border_n{n}"] = round(100 * float(border[n][m].mean()), 1)
            row[f"neighborhood_mix_tvd_vs_interior_n{N}"] = round(label_tvd(labels[m], labels[interior]), 3)
            zone_rows.append(row)
        for b, sub in g.groupby(depth_bin, observed=True):
            pos = sub.index.to_numpy()
            depth_rows.append({
                "k": k, "d_z_face_from": b.left, "d_z_face_to": b.right, "n_cells": len(sub),
                "median_radius_um": float(sub["radius"].median()),
                "median_offset_ratio": float(sub["offset_ratio"].median()),
                **{f"pct_border_n{n}": 100 * float(border[n][pos].mean()) for n in N_VALUES},
            })
    return pd.DataFrame(zone_rows), pd.DataFrame(depth_rows)


def plot_depth_profile(depth: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    panels = (("median_radius_um", "median window radius (µm)"),
              ("median_offset_ratio", "median window offset / radius"),
              (f"pct_border_n{N}", f"border cells (%), n={N}"))
    for k, sub in depth.groupby("k"):
        ticks = np.arange(len(sub))
        labels = [f"{a:.0f}–{b:.0f}" for a, b in zip(sub["d_z_face_from"], sub["d_z_face_to"])]
        for ax, (col, ylabel) in zip(axes, panels):
            ax.plot(ticks, sub[col], marker="o", label=f"k={k}")
            ax.set_ylabel(ylabel)
            ax.set_xticks(ticks, labels, rotation=45, ha="right", fontsize=8)
    axes[1].axhline(0.375, color="grey", lw=0.8, ls=":")
    axes[1].text(len(ticks) - 1, 0.375, "half ball", ha="right", va="bottom", fontsize=8, color="grey")
    axes[2].set_ylim(0, None)
    for ax in axes:
        ax.set_xlabel("distance to nearest z-face (µm)")
        ax.legend(frameon=False)
    fig.suptitle("Main HER2: 3D window geometry and border rate by distance to the top/bottom face")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# B. border robustness
# ----------------------------------------------------------------------------

def second_prob(adata: ad.AnnData, tag: str) -> np.ndarray:
    """Border score (second-largest membership probability) of a stored result."""
    return border_score(adata.obsm[f"neighborhood_probabilities_{tag}"])


def seed_rerun(cells: ad.AnnData, k: int, cell_types: list[str]):
    """3D windows, plus labels and border scores with only the k-means seed changed."""
    win = tl3d.KNN2(cells, z_key="z", cluster_col=CT, region_key=REG, ks=(k,))[k]
    X = StandardScaler().fit_transform(win[cell_types].to_numpy(np.float32)).astype(np.float32)

    def fit(seed):
        return tl3d.fit_neighborhood_labels(X, N, clusterer=CLUSTERER, random_state=seed, n_init=N_INIT).astype(str)

    if adjusted_rand_score(fit(0), cells.obs[f"neighborhood_k{k}_n{N}"].astype(str)) != 1.0:
        raise RuntimeError("Could not reproduce the stored 3D labels; the seed reference would be invalid.")
    tag = f"k{k}_n{N}_seed{SEED}"
    cells.obs[f"neighborhood_{tag}"] = fit(SEED)
    quiet(membership_and_borders, cells, f"neighborhood_{tag}", k, tag)
    return win, tag


def compare_runs(name, kind, a, b, subset, subset_name, xyz, cell_type):
    """All border-agreement measures for runs ``a`` and ``b`` (dicts with p2 and labels) on ``subset``."""
    key = {"comparison": name, "kind": kind, "cells": subset_name}
    ari = adjusted_rand_score(a["labels"][subset], b["labels"][subset])
    return compare_border_scores(a["p2"][subset], b["p2"][subset], xyz[subset], cell_type[subset], key,
                                 extra={"neighborhood_ari": round(ari, 3)})


def package_threshold_tools(adata, tag, coord_keys, label) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the package's threshold sweep and spatial null on one stored result."""
    keys = dict(prob_key=f"neighborhood_probabilities_{tag}",
                prob_variable_key=f"neighborhood_probability_neighborhoods_{tag}",
                region_key=REG, coord_keys=coord_keys)
    res = tl3d.threshold_sensitivity_analysis(adata, thresholds=THRESHOLDS, cell_type_col=CT, **keys)
    null = tl3d.spatial_border_clustering_null(adata, threshold=THRESHOLD, n_permutations=200, seed=0, **keys)
    null_row = {"run": label, **{k: v for k, v in null.items() if np.isscalar(v)}}
    return res["summary"].assign(run=label), res["stability"].assign(run=label), null_row


def main() -> None:
    OUT5.mkdir(parents=True, exist_ok=True)
    cells = ad.read_h5ad(IN_3D)
    cells.obs[CT] = cells.obs[CT].astype(str)
    cell_types = sorted(cells.obs[CT].unique())
    sec = ad.read_h5ad(IN_2D)
    sec.obs[CT] = sec.obs[CT].astype(str)
    sec.obs[REG] = sec.obs[REG].astype(str)
    ids = sec.obs["cell_id"].astype(str).to_numpy()
    pos = cells.obs_names.get_indexer(ids)
    if (pos < 0).any():
        raise RuntimeError("Section cells are missing from the 3D result.")

    # ---- A. boundary effects (all cells, 3D) ----
    xyz = cells.obs[["x", "y", "z"]].to_numpy(float)
    ks3 = [k for k, _ in PAIRS]
    faces, geom = face_distances(xyz), window_geometry(xyz, ks3)
    zones, depth = boundary_tables(cells, faces, geom)
    zones.to_csv(OUT5 / "boundary_zones.csv", index=False)
    depth.to_csv(OUT5 / "boundary_depth_profile.csv", index=False)
    plot_depth_profile(depth, OUT5 / "boundary_depth_profile.png")
    per_cell = faces.copy()
    for k, g in geom.items():
        per_cell[f"radius_k{k}"] = g["radius"]
        per_cell[f"offset_ratio_k{k}"] = g["offset_ratio"]
        per_cell[f"zone_k{k}"] = zone_of(faces, float(g["radius"].median()))
    per_cell.insert(0, "cell_id", cells.obs_names.to_numpy())
    per_cell.round(3).to_csv(OUT5 / "cell_boundary_flags.csv.gz", index=False)
    print("\nBoundary zones (3D, all cells)\n" + zones.to_string(index=False))

    # One interior definition for every comparison: further than the larger window radius from any face.
    interior_width = max(float(g["radius"].median()) for g in geom.values())
    interior = (faces["d_face"].to_numpy() >= interior_width)[pos]
    subsets = {"all": np.ones(len(pos), bool), "interior": interior}
    print(f"\nSection cells: {len(pos)}; interior (>= {interior_width:.1f} µm from every face): "
          f"{interior.sum()} ({100 * interior.mean():.1f}%)")

    # ---- B. border robustness (section cells) ----
    runs, win3 = {}, {}
    for k3, k2 in PAIRS:
        win, seed_tag = seed_rerun(cells, k3, cell_types)
        win3[k3] = fractions(win.iloc[pos], cell_types)
        runs[f"3D k={k3}"] = {"p2": second_prob(cells, f"k{k3}_n{N}")[pos],
                              "labels": cells.obs[f"neighborhood_k{k3}_n{N}"].astype(str).to_numpy()[pos]}
        runs[f"3D k={k3} seed {SEED}"] = {"p2": second_prob(cells, seed_tag)[pos],
                                          "labels": cells.obs[f"neighborhood_{seed_tag}"].astype(str).to_numpy()[pos]}
        runs[f"2D k={k2}"] = {"p2": second_prob(sec, f"2d_k{k2}_n{N}"),
                              "labels": sec.obs[f"neighborhood_2d_k{k2}_n{N}"].astype(str).to_numpy()}

    (k3a, k2a), (k3b, k2b) = PAIRS
    comparisons = [
        (f"2D k={k2a} vs 3D k={k3a}", "2D vs 3D", f"2D k={k2a}", f"3D k={k3a}"),
        (f"2D k={k2b} vs 3D k={k3b}", "2D vs 3D", f"2D k={k2b}", f"3D k={k3b}"),
        (f"3D k={k3a} vs k={k3b}", "k only", f"3D k={k3a}", f"3D k={k3b}"),
        (f"2D k={k2a} vs k={k2b}", "k only", f"2D k={k2a}", f"2D k={k2b}"),
        (f"3D k={k3a}, seed 0 vs {SEED}", "seed only", f"3D k={k3a}", f"3D k={k3a} seed {SEED}"),
        (f"3D k={k3b}, seed 0 vs {SEED}", "seed only", f"3D k={k3b}", f"3D k={k3b} seed {SEED}"),
    ]
    sec_xyz = xyz[pos]
    sec_ct = cells.obs[CT].to_numpy()[pos]
    rows, sweeps, tols = [], [], []
    for subset_name, subset in subsets.items():
        for name, kind, a, b in comparisons:
            row, sweep, tol = compare_runs(name, kind, runs[a], runs[b], subset, subset_name, sec_xyz, sec_ct)
            rows.append(row)
            sweeps.append(sweep)
            tols.append(tol)
    table, sweeps, tols = pd.DataFrame(rows), pd.concat(sweeps, ignore_index=True), pd.concat(tols, ignore_index=True)
    table.to_csv(OUT5 / "border_comparisons.csv", index=False)
    sweeps.round(4).to_csv(OUT5 / "border_threshold_sweep.csv", index=False)
    tols.round(4).to_csv(OUT5 / "border_distance_tolerance.csv", index=False)
    for subset_name in subsets:
        plot_robustness(sweeps[sweeps["cells"] == subset_name], tols[tols["cells"] == subset_name],
                        f"Main HER2 border agreement, n={N} ({subset_name} section cells)",
                        OUT5 / f"border_robustness_{subset_name}.png")
    print("\nBorder agreement\n" + table.set_index(["cells", "comparison"]).drop(columns="kind").T.to_string())

    # ---- windows: 2D vs 3D cell-type mix by zone (does the edge drive the window difference?) ----
    win_rows = []
    for k3, k2 in PAIRS:
        win2 = fractions(tl3d.KNN2(sec, z_key=None, cluster_col=CT, region_key=REG, ks=(k2,))[k2], cell_types)
        tvd = 0.5 * np.abs(win3[k3] - win2).sum(axis=1)
        zone = zone_of(faces, float(geom[k3]["radius"].median()))[pos]
        for z in ("all", "z-face", "xy-face", "interior"):
            m = np.ones(len(pos), bool) if z == "all" else zone == z
            win_rows.append({"pair": f"2D k={k2} vs 3D k={k3}", "zone": z, "n_cells": int(m.sum()),
                             "median_tvd": round(float(np.median(tvd[m])), 3),
                             "pct_tvd_gt_0.25": round(100 * float((tvd[m] > 0.25).mean()), 1)})
    windows = pd.DataFrame(win_rows)
    windows.to_csv(OUT5 / "window_tvd_by_zone.csv", index=False)
    print("\nWindow cell-type mix, 2D vs 3D, by zone\n" + windows.to_string(index=False))

    # ---- package threshold tools on each stored result ----
    summaries, stabilities, nulls = [], [], []
    for k3, k2 in PAIRS:
        for adata, tag, coords, label in ((cells, f"k{k3}_n{N}", ("x", "y", "z"), f"3D k={k3}"),
                                          (sec, f"2d_k{k2}_n{N}", ("x", "y"), f"2D k={k2}")):
            s, st, nl = package_threshold_tools(adata, tag, coords, label)
            summaries.append(s)
            stabilities.append(st)
            nulls.append(nl)
    pd.concat(summaries, ignore_index=True).to_csv(OUT5 / "threshold_sensitivity_summary.csv", index=False)
    pd.concat(stabilities, ignore_index=True).to_csv(OUT5 / "threshold_sensitivity_stability.csv", index=False)
    nulls = pd.DataFrame(nulls)
    nulls.to_csv(OUT5 / "spatial_border_clustering_null.csv", index=False)
    print("\nAre border cells spatially clustered? (mean distance to nearest other border cell, µm)\n"
          + nulls.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
