"""Reviewer response: border threshold sensitivity (Task 3).

Runs the regular MINGL pipeline across a wider threshold range
``(0.01, 0.1, 0.25, 0.4, 0.49)`` and reports how the four border outputs move:

* number of border cells,
* border location (per-region fraction, spatial centroid/spread, and the Jaccard
  stability of the border-cell set between thresholds),
* border composition (cell-type make-up of the border population),
* border cell-type enrichment (log2 enrichment vs overall abundance).

Optionally runs a spatial-clustering **null** to test whether border location is
non-random (supporting the intestine null-condition analysis).

Works on any AnnData that already carries a MINGL posterior in
``obsm[prob_key]``; if none is present (or ``--recompute`` is given) it first
scores the cells with a chosen emission model (default ``diagonal_gaussian`` ==
the regular MINGL pipeline).

Local synthetic demo (no lab files):

    python tools/reviewer_threshold_sensitivity.py --synthetic --null

Real datasets (executed by the user on the lab server):

    python tools/reviewer_threshold_sensitivity.py --dataset intestine --data intestine_results.h5ad --null
    python tools/reviewer_threshold_sensitivity.py --dataset spatial   --data <spatial>.h5ad
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mingl as mg
import mingl.tl as tl

warnings.filterwarnings("ignore")

DATASET_PRESETS = {
    "intestine": dict(cluster_col="Cell Type", neighborhood_col="Neighborhood", region_key="unique_region", x_key="x", y_key="y"),
    "melanoma": dict(cluster_col="Cell_Type", neighborhood_col="Neighborhood", region_key="filename", x_key="x", y_key="y"),
    "spatial": dict(cluster_col="Cell Type", neighborhood_col="neigh_name", region_key="region", x_key="x", y_key="y"),
}
DEFAULT_THRESHOLDS = [0.01, 0.1, 0.25, 0.4, 0.49]


def _quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _synthetic_multiunit(n_per=1200, seed=0):
    import anndata as ad

    rng = np.random.default_rng(seed)
    types = [f"CT{i}" for i in range(6)]
    comps = {
        "N0": [0.55, 0.15, 0.10, 0.08, 0.07, 0.05],
        "N1": [0.10, 0.55, 0.15, 0.08, 0.07, 0.05],
        "N2": [0.08, 0.10, 0.50, 0.20, 0.07, 0.05],
        "N3": [0.05, 0.07, 0.10, 0.15, 0.33, 0.30],
    }
    frames = []
    for i, (nb, p) in enumerate(comps.items()):
        x = rng.uniform(i * 100.0, i * 100.0 + 100.0, n_per)
        y = rng.uniform(0.0, 100.0, n_per)
        ct = rng.choice(types, size=n_per, p=p)
        frames.append(pd.DataFrame({"x": x, "y": y, "unique_region": "R1", "Neighborhood": nb, "Cell Type": ct}))
    obs = pd.concat(frames, ignore_index=True)
    obs.index = obs.index.astype(str)
    return ad.AnnData(X=np.zeros((len(obs), 0), dtype=np.float32), obs=obs)


def count_figure(summary, out_dir):
    fig, ax1 = plt.subplots(figsize=(5.5, 4), dpi=200)
    ax1.plot(summary["threshold"], summary["n_border"], "o-", color="#1f77b4", label="n_border")
    ax1.set_xlabel("threshold")
    ax1.set_ylabel("number of border cells", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(summary["threshold"], summary["frac_border"], "s--", color="#d62728", label="frac_border")
    ax2.set_ylabel("border fraction", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title("Border cells vs threshold")
    fig.tight_layout()
    path = os.path.join(out_dir, "border_count_vs_threshold.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def enrichment_heatmap(composition, out_dir):
    if composition.empty:
        return None
    piv = composition.pivot(index="cell_type", columns="threshold", values="log2_enrichment")
    piv = piv.reindex(piv.mean(axis=1).sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(1.2 * len(piv.columns) + 3, 0.4 * len(piv) + 2), dpi=200)
    vmax = float(np.nanmax(np.abs(piv.to_numpy()))) or 1.0
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_xlabel("threshold")
    ax.set_title("Border cell-type log2 enrichment")
    fig.colorbar(im, ax=ax, label="log2 enrichment")
    fig.tight_layout()
    path = os.path.join(out_dir, "border_enrichment_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def stability_figure(stability, out_dir):
    if stability.empty:
        return None
    labels = [f"{a:g}->{b:g}" for a, b in zip(stability["threshold_low"], stability["threshold_high"])]
    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=200)
    ax.plot(labels, stability["jaccard_border_cells"], "o-", color="#2ca02c")
    ax.set_ylabel("Jaccard overlap of border-cell set")
    ax.set_xlabel("threshold step")
    ax.set_ylim(0, 1)
    ax.set_title("Border-cell set stability across thresholds")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = os.path.join(out_dir, "border_stability.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None)
    ap.add_argument("--dataset", choices=sorted(DATASET_PRESETS), default="intestine")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--cluster-col", default=None)
    ap.add_argument("--neighborhood-col", default=None)
    ap.add_argument("--region-key", default=None)
    ap.add_argument("--x-key", default=None)
    ap.add_argument("--y-key", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 20, 100, 300])
    ap.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--prob-key", default="neighborhood_probabilities")
    ap.add_argument("--prob-variable-key", default="neighborhood_probability_neighborhoods")
    ap.add_argument("--model", default="diagonal_gaussian", help="Emission model to compute probabilities if absent.")
    ap.add_argument("--recompute", action="store_true", help="Recompute probabilities even if present.")
    ap.add_argument("--null", action="store_true", help="Also run the spatial border-clustering null.")
    ap.add_argument("--n-permutations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "reviewer_threshold_sensitivity_outputs"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    preset = dict(DATASET_PRESETS[args.dataset])
    cluster_col = args.cluster_col or preset["cluster_col"]
    neighborhood_col = args.neighborhood_col or preset["neighborhood_col"]
    region_key = args.region_key or preset["region_key"]
    x_key = args.x_key or preset["x_key"]
    y_key = args.y_key or preset["y_key"]
    ks = tuple(args.ks)

    if args.synthetic:
        print("Loading synthetic multi-neighborhood tissue (no lab data).")
        adata = _synthetic_multiunit(seed=args.seed)
        cluster_col, neighborhood_col, region_key, x_key, y_key = "Cell Type", "Neighborhood", "unique_region", "x", "y"
        ks = (10, 20)
    elif args.data:
        print(f"Loading {args.data} (dataset preset: {args.dataset})")
        adata = mg.pp.read_file(args.data)
    else:
        ap.error("Provide --data PATH or --synthetic.")

    have_probs = args.prob_key in adata.obsm
    if args.recompute or not have_probs:
        print(f"Scoring memberships with model={args.model} (regular MINGL = diagonal_gaussian).")
        _quiet(
            tl.mingl_membership_probabilities,
            adata,
            model=args.model,
            cluster_col=cluster_col,
            neighborhood_col=neighborhood_col,
            region_key=region_key,
            x_key=x_key,
            y_key=y_key,
            ks=ks,
            k=args.k,
            prob_key=args.prob_key,
            prob_variable_key=args.prob_variable_key,
        )
    else:
        print(f"Using existing posterior in obsm[{args.prob_key!r}].")

    print(f"Cells: {adata.n_obs}  Thresholds: {args.thresholds}")
    print(f"Output directory: {args.out_dir}\n")

    res = tl.threshold_sensitivity_analysis(
        adata,
        thresholds=args.thresholds,
        prob_key=args.prob_key,
        prob_variable_key=args.prob_variable_key,
        cell_type_col=cluster_col,
        region_key=region_key,
        coord_keys=(x_key, y_key),
    )
    res["summary"].to_csv(os.path.join(args.out_dir, "threshold_summary.csv"), index=False)
    res["composition"].to_csv(os.path.join(args.out_dir, "threshold_composition.csv"), index=False)
    res["stability"].to_csv(os.path.join(args.out_dir, "threshold_stability.csv"), index=False)

    print("== Border count / location vs threshold ==")
    print(res["summary"].round(4).to_string(index=False))
    print("\n== Border-cell set stability (Jaccard) ==")
    print(res["stability"].round(4).to_string(index=False))

    nb = res["summary"]["n_border"].to_numpy()
    print(f"\nn_border monotone non-increasing in threshold: {bool(np.all(np.diff(nb) <= 0))}")

    fig1 = count_figure(res["summary"], args.out_dir)
    fig2 = enrichment_heatmap(res["composition"], args.out_dir)
    fig3 = stability_figure(res["stability"], args.out_dir)

    if args.null:
        print("\n== Spatial border-clustering null ==")
        null_rows = []
        for t in args.thresholds:
            out = tl.spatial_border_clustering_null(
                adata,
                threshold=t,
                prob_key=args.prob_key,
                prob_variable_key=args.prob_variable_key,
                region_key=region_key,
                coord_keys=(x_key, y_key),
                n_permutations=args.n_permutations,
                seed=args.seed,
            )
            null_rows.append(out)
        null_df = pd.DataFrame(null_rows)
        null_df.to_csv(os.path.join(args.out_dir, "border_spatial_null.csv"), index=False)
        print(null_df.round(4).to_string(index=False))

    print(f"\nArtifacts written to {args.out_dir}")
    for f in [fig1, fig2, fig3]:
        if f:
            print(f"  - {os.path.basename(f)}")
    print("  - threshold_summary.csv / threshold_composition.csv / threshold_stability.csv")


if __name__ == "__main__":
    main()
