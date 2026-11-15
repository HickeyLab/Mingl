"""2D vs 3D borders with the neighborhoods held fixed (Main HER2).

05 found that border-cell sets differ strongly between 2D and 3D, and almost
as strongly between two window sizes in 2D. In those comparisons every run
also had its own neighborhood clustering, so two things changed at once: which
neighborhoods exist (the partition) and what each cell's window looks like.

Here the partition is held fixed. Every cell keeps one set of neighborhood
labels, and only the windows change. For a label set L and a window type W,
MINGL's membership step is applied unchanged: neighborhood centroids are the
mean and std of the W windows of the cells labelled L, and each cell's
membership probabilities come from its own W window scored against those
centroids. Comparing two window types under the same L then isolates the
effect of the window; comparing with each run's own neighborhoods (the 05
setting) shows what the partition adds.

Label sets: the four stored partitions (3D k=30, 3D k=50, 2D k=10, 2D k=15,
all at the same n). Window types: the same four. Comparisons, as in 05:
2D vs 3D at matched window radius, and k only within 3D and within 2D.

With fixed labels the membership columns are the same neighborhoods in both
runs, so the full posteriors can be compared as well: how often the most
probable neighborhood agrees, and the mean total variation distance.

Everything is restricted to the section cells (the only cells with 2D
labels), so centroids are computed from those cells. The stored 2D results are
reproduced exactly as a check, and so are the stored 3D results when the
centroids use every cell of the volume, as 02 does.

Centroid reference: the 3D border set changes noticeably when the centroids
are estimated from the section cells instead of the whole volume, with labels
and windows untouched. To put a number on that, centroids are re-estimated
from random subsets of the cells and all cells are re-scored. This is the
floor for any border comparison in which the centroids are re-estimated.

Inputs: results/main_her2/main_her2_mingl3d.h5ad            (02)
        results/main_her2/2d_vs_3d/sections_2d_mingl.h5ad   (03)
Output: results/main_her2/fixed_neighborhoods/
Run:    .venv/bin/python analysis_3d/06_fixed_neighborhoods_main_her2.py
"""

from __future__ import annotations

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from mingl import tl3d
from mingl.tl.gmm import _compute_probability_batch, _default_batch_size
from border_metrics import KIND_STYLE, border_score, compare_border_scores, jaccard, plot_robustness
from common import CT, OUT, REG, THRESHOLD

IN_3D = OUT / "main_her2_mingl3d.h5ad"
IN_2D = OUT / "2d_vs_3d" / "sections_2d_mingl.h5ad"
OUT6 = OUT / "fixed_neighborhoods"

PAIRS = ((30, 10), (50, 15))  # (3D k, 2D k) with matched window radius, as in 03
OWN = "each run's own"
SUBSAMPLE_FRACTIONS = (0.9, 0.75, 0.5)  # share of cells used to estimate centroids
N_REPEATS = 5


def _n_from_3d_run(k: int = 30) -> int:
    """The probability-curve elbow n of the 3D k=30 run, the n used by 03."""
    rs = pd.read_csv(OUT / "run_summary.csv")
    return int(rs.loc[rs["k"] == k, "prob_elbow_n"].iloc[0])


N = _n_from_3d_run()


def membership(windows: pd.DataFrame, labels: np.ndarray, cell_types: list[str], fit_rows=None) -> pd.DataFrame:
    """MINGL membership probabilities of ``windows`` for the neighborhoods in ``labels``.

    Same arithmetic as tl3d.centroid_Calculation (float32 window counts, mean
    and sample std per neighborhood and cell type) followed by
    tl3d.cpu_gmm_probability (float64 windows scored with the package's
    probability function). Columns are the neighborhood names. ``fit_rows``
    (row positions) restricts the cells used for the centroids; every row is scored.
    """
    w32 = windows[cell_types].astype("float32")
    if fit_rows is not None:
        w32, labels = w32.iloc[fit_rows], labels[fit_rows]
    names = sorted(pd.unique(labels), key=int)
    means = np.array([[w32.loc[labels == nb, c].mean() for c in cell_types] for nb in names], dtype=np.float64)
    stds = np.array([[w32.loc[labels == nb, c].std() for c in cell_types] for nb in names], dtype=np.float64)

    values = np.ascontiguousarray(windows[cell_types].to_numpy(dtype=np.float64))
    step = _default_batch_size(n_cells=len(values), n_centroids=len(names), n_features=len(cell_types))
    probs = np.vstack([_compute_probability_batch(values[i:i + step], means, stds) for i in range(0, len(values), step)])
    return pd.DataFrame(probs, index=windows.index, columns=names)


def centroid_subsample_reference(windows: pd.DataFrame, labels: np.ndarray, stored: np.ndarray,
                                 cell_types: list[str], run: str, rng) -> list[dict]:
    """Border agreement with the stored result when only the cells behind the centroids change."""
    rows = []
    for frac in SUBSAMPLE_FRACTIONS:
        for rep in range(N_REPEATS):
            fit_rows = np.sort(rng.choice(len(labels), int(frac * len(labels)), replace=False))
            score = border_score(membership(windows, labels, cell_types, fit_rows=fit_rows))
            rows.append({"run": run, "centroid_cells_fraction": frac, "repeat": rep,
                         "border_jaccard": jaccard(score > THRESHOLD, stored > THRESHOLD),
                         "p2_spearman": float(spearmanr(score, stored)[0]),
                         "pct_border": 100 * float((score > THRESHOLD).mean())})
    return rows


def posterior_agreement(pa: pd.DataFrame, pb: pd.DataFrame) -> dict:
    """Agreement of two posteriors over the same neighborhoods (same column names)."""
    a, b = pa.to_numpy(), pb[pa.columns].to_numpy()
    return {
        "pct_same_top_neighborhood": round(100 * float((a.argmax(axis=1) == b.argmax(axis=1)).mean()), 1),
        "mean_posterior_tvd": round(float((0.5 * np.abs(a - b).sum(axis=1)).mean()), 3),
    }


def plot_summary(table: pd.DataFrame, label_sets: list[str], path) -> None:
    """One point per comparison and label set: does fixing the neighborhoods restore agreement?"""
    order = [OWN, *label_sets]
    panels = (("border_jaccard", "border Jaccard"), ("p2_spearman", "border-score Spearman"),
              ("pct_same_top_neighborhood", "same most-probable neighborhood (%)"))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, (col, ylabel) in zip(axes, panels):
        seen = set()
        for (name, kind), sub in table.groupby(["comparison", "kind"], sort=False):
            sub = sub.set_index("neighborhoods").reindex(order)
            ls = "-" if kind not in seen else "--"
            seen.add(kind)
            ax.plot(range(len(order)), sub[col], marker="o", ls=ls, label=name, **KIND_STYLE[kind])
        if col == "border_jaccard":
            chance = table.groupby("neighborhoods")["border_jaccard_chance"].mean().reindex(order)
            ax.plot(range(len(order)), chance, color="black", lw=1, ls=":", label="chance")
        ax.set_xticks(range(len(order)), order, rotation=20, ha="right")
        ax.axvline(0.5, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("neighborhood labels used for both runs")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 100 if col.startswith("pct") else 1)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Main HER2, n={N}: agreement between window types, own neighborhoods vs fixed neighborhoods")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT6.mkdir(parents=True, exist_ok=True)
    cells = ad.read_h5ad(IN_3D)
    cells.obs[CT] = cells.obs[CT].astype(str)
    cell_types = sorted(cells.obs[CT].unique())
    sec = ad.read_h5ad(IN_2D)
    sec.obs[CT] = sec.obs[CT].astype(str)
    sec.obs[REG] = sec.obs[REG].astype(str)
    pos = cells.obs_names.get_indexer(sec.obs["cell_id"].astype(str).to_numpy())
    if (pos < 0).any():
        raise RuntimeError("Section cells are missing from the 3D result.")
    xyz = cells.obs[["x", "y", "z"]].to_numpy(float)[pos]
    cell_type = cells.obs[CT].to_numpy()[pos]

    # Windows, labels and stored border scores of the four runs, all on the section cells.
    ks3, ks2 = [k for k, _ in PAIRS], [k for _, k in PAIRS]
    win3 = tl3d.KNN2(cells, z_key="z", cluster_col=CT, region_key=REG, ks=ks3)
    win2 = tl3d.KNN2(sec, z_key=None, cluster_col=CT, region_key=REG, ks=ks2)
    windows, labels, stored = {}, {}, {}
    for k in ks3:
        windows[f"3D k={k}"] = win3[k].iloc[pos].reset_index(drop=True)
        labels[f"3D k={k}"] = cells.obs[f"neighborhood_k{k}_n{N}"].astype(str).to_numpy()[pos]
        stored[f"3D k={k}"] = border_score(cells.obsm[f"neighborhood_probabilities_k{k}_n{N}"])[pos]
    for k in ks2:
        windows[f"2D k={k}"] = win2[k].reset_index(drop=True)
        labels[f"2D k={k}"] = sec.obs[f"neighborhood_2d_k{k}_n{N}"].astype(str).to_numpy()
        stored[f"2D k={k}"] = border_score(sec.obsm[f"neighborhood_probabilities_2d_k{k}_n{N}"])
    runs = list(windows)

    # Membership for every (label set, window type).
    post = {(lab, win): membership(windows[win], labels[lab], cell_types) for lab in runs for win in runs}
    score = {key: border_score(p) for key, p in post.items()}

    # Checks against the stored pipeline output.
    for k in ks2:
        run = f"2D k={k}"
        if not np.allclose(score[(run, run)], stored[run], atol=1e-6):
            raise RuntimeError(f"Could not reproduce the stored {run} membership probabilities.")
    print("Stored 2D probabilities reproduced exactly.")
    rng = np.random.default_rng(0)
    reference = []
    for k in ks3:
        run = f"3D k={k}"
        full_windows = win3[k].reset_index(drop=True)
        full_labels = cells.obs[f"neighborhood_k{k}_n{N}"].astype(str).to_numpy()
        full_stored = border_score(cells.obsm[f"neighborhood_probabilities_k{k}_n{N}"])
        if not np.allclose(border_score(membership(full_windows, full_labels, cell_types)), full_stored, atol=1e-6):
            raise RuntimeError(f"Could not reproduce the stored {run} membership probabilities.")
        j = jaccard(score[(run, run)] > THRESHOLD, stored[run] > THRESHOLD)
        print(f"{run}: stored probabilities reproduced exactly with whole-volume centroids; "
              f"with section-cell centroids the border Jaccard to the stored set is {j:.3f}")
        reference += centroid_subsample_reference(full_windows, full_labels, full_stored, cell_types, run, rng)
    reference = pd.DataFrame(reference)
    reference.round(4).to_csv(OUT6 / "centroid_subsample_reference.csv", index=False)
    print("\nCentroids from a random subset of cells, same labels and windows (mean of "
          f"{N_REPEATS} repeats, min-max Jaccard)\n"
          + reference.groupby(["run", "centroid_cells_fraction"], sort=False)
          .agg(border_jaccard=("border_jaccard", "mean"), jaccard_min=("border_jaccard", "min"),
               jaccard_max=("border_jaccard", "max"), p2_spearman=("p2_spearman", "mean"),
               pct_border=("pct_border", "mean")).round(3).to_string())

    # How each window type behaves under each label set.
    cell_rows = []
    for (lab, win), p in post.items():
        top = p.columns.to_numpy()[p.to_numpy().argmax(axis=1)]
        cell_rows.append({"neighborhoods": lab, "windows": win,
                          "pct_border": round(100 * float((score[(lab, win)] > THRESHOLD).mean()), 1),
                          "pct_top_neighborhood_is_label": round(100 * float((top == labels[lab]).mean()), 1)})
    by_run = pd.DataFrame(cell_rows)
    by_run.to_csv(OUT6 / "membership_by_labels_and_windows.csv", index=False)
    print("\nMembership by label set and window type\n" + by_run.to_string(index=False))

    # Comparisons between window types: own neighborhoods (as in 05), then each fixed label set.
    (k3a, k2a), (k3b, k2b) = PAIRS
    comparisons = [
        (f"2D k={k2a} vs 3D k={k3a}", "2D vs 3D", f"2D k={k2a}", f"3D k={k3a}"),
        (f"2D k={k2b} vs 3D k={k3b}", "2D vs 3D", f"2D k={k2b}", f"3D k={k3b}"),
        (f"3D k={k3a} vs k={k3b}", "k only", f"3D k={k3a}", f"3D k={k3b}"),
        (f"2D k={k2a} vs k={k2b}", "k only", f"2D k={k2a}", f"2D k={k2b}"),
    ]
    rows, sweeps, tols = [], [], []
    for name, kind, a, b in comparisons:
        key = {"neighborhoods": OWN, "comparison": name, "kind": kind}
        row, sweep, tol = compare_border_scores(stored[a], stored[b], xyz, cell_type, key)
        rows.append(row)
        sweeps.append(sweep)
        tols.append(tol)
        for lab in runs:
            key = {"neighborhoods": lab, "comparison": name, "kind": kind}
            row, sweep, tol = compare_border_scores(score[(lab, a)], score[(lab, b)], xyz, cell_type, key,
                                                    extra=posterior_agreement(post[(lab, a)], post[(lab, b)]))
            rows.append(row)
            sweeps.append(sweep)
            tols.append(tol)
    table, sweeps, tols = pd.DataFrame(rows), pd.concat(sweeps, ignore_index=True), pd.concat(tols, ignore_index=True)
    table.to_csv(OUT6 / "fixed_neighborhood_comparisons.csv", index=False)
    sweeps.round(4).to_csv(OUT6 / "fixed_neighborhood_threshold_sweep.csv", index=False)
    tols.round(4).to_csv(OUT6 / "fixed_neighborhood_distance_tolerance.csv", index=False)

    plot_summary(table, runs, OUT6 / "fixed_vs_own_neighborhoods.png")
    for lab in (f"3D k={k3a}", f"2D k={k2a}"):
        plot_robustness(sweeps[sweeps["neighborhoods"] == lab], tols[tols["neighborhoods"] == lab],
                        f"Main HER2 border agreement between window types, neighborhoods fixed to the {lab} labels (n={N})",
                        OUT6 / f"border_robustness_fixed_{lab.replace(' ', '_').replace('=', '')}.png")

    show = ["border_jaccard", "border_jaccard_chance", "p2_spearman", "p2_auc_for_other_border",
            "celltype_enrichment_r", "within_10um", "within_10um_null", "pct_same_top_neighborhood", "mean_posterior_tvd"]
    for name, *_ in comparisons:
        sub = table[table["comparison"] == name].set_index("neighborhoods")[show]
        print(f"\n{name} windows\n" + sub.to_string())


if __name__ == "__main__":
    main()
