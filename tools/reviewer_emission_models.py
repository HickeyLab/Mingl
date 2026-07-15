"""Reviewer response: alternative emission models for MINGL (Tasks 1 & 2).

Compares the current diagonal-covariance Gaussian scorer against models that
respect the compositional-count structure of MINGL's k-NN windows:

* ``full_gaussian``          -- full covariance (Task 1, first priority),
* ``multinomial``            -- Multinomial(n=k, p_c),
* ``dirichlet_multinomial``  -- over-dispersed multinomial,
* ``logistic_normal``        -- ALR transform + full-covariance Gaussian.

For each model it reports the held-out neighborhood-label log-loss / accuracy
(comparable across all models), membership entropy, the resulting border-cell
fraction, and within-family fit (BIC). It writes CSV tables and comparison
figures and prints a summary.

Nothing here modifies the shipped pipeline. The neighborhood definition is held
fixed; only the emission model is swapped.

Run locally with synthetic data (no lab files needed):

    python tools/reviewer_emission_models.py --synthetic

Run on a real dataset (executed by the user on the lab server):

    python tools/reviewer_emission_models.py --dataset intestine --data intestine_results.h5ad
    python tools/reviewer_emission_models.py --dataset melanoma  --data melanoma_all_information.csv
    python tools/reviewer_emission_models.py --dataset spatial   --data <spatial_transcriptomics>.h5ad
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

# Column-name presets for the three manuscript datasets. `--data` supplies the path.
DATASET_PRESETS = {
    "intestine": dict(cluster_col="Cell Type", neighborhood_col="Neighborhood", region_key="unique_region", x_key="x", y_key="y"),
    "melanoma": dict(cluster_col="Cell_Type", neighborhood_col="Neighborhood", region_key="filename", x_key="x", y_key="y"),
    "spatial": dict(cluster_col="Cell Type", neighborhood_col="neigh_name", region_key="region", x_key="x", y_key="y"),
}

MODEL_ORDER = ["diagonal_gaussian", "full_gaussian", "multinomial", "dirichlet_multinomial", "logistic_normal"]
MODEL_COLORS = {
    "diagonal_gaussian": "#7f7f7f",
    "full_gaussian": "#1f77b4",
    "multinomial": "#2ca02c",
    "dirichlet_multinomial": "#ff7f0e",
    "logistic_normal": "#9467bd",
}


def _quiet(func, *args, **kwargs):
    """Silence the verbose prints from KNN2 / centroid_Calculation."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _synthetic_multiunit(n_per=1200, seed=0):
    """A multi-neighborhood compositional tissue for a self-contained local demo."""
    rng = np.random.default_rng(seed)
    types = [f"CT{i}" for i in range(6)]
    comps = {
        "N0": [0.55, 0.15, 0.10, 0.08, 0.07, 0.05],
        "N1": [0.10, 0.55, 0.15, 0.08, 0.07, 0.05],
        "N2": [0.08, 0.10, 0.50, 0.20, 0.07, 0.05],
        "N3": [0.05, 0.07, 0.10, 0.15, 0.33, 0.30],
    }
    import anndata as ad

    frames = []
    for i, (nb, p) in enumerate(comps.items()):
        x = rng.uniform(i * 100.0, i * 100.0 + 100.0, n_per)
        y = rng.uniform(0.0, 100.0, n_per)
        ct = rng.choice(types, size=n_per, p=p)
        frames.append(pd.DataFrame({"x": x, "y": y, "unique_region": "R1", "Neighborhood": nb, "Cell Type": ct}))
    obs = pd.concat(frames, ignore_index=True)
    obs.index = obs.index.astype(str)
    return ad.AnnData(X=np.zeros((len(obs), 0), dtype=np.float32), obs=obs)


def comparison_figure(df, out_dir):
    df = df.set_index("model").reindex([m for m in MODEL_ORDER if m in df.index])
    colors = [MODEL_COLORS[m] for m in df.index]
    panels = [
        ("test_logloss", "Held-out neighborhood log-loss (lower is better)"),
        ("test_accuracy", "Held-out neighborhood accuracy"),
        ("mean_entropy", "Mean membership entropy (nats)"),
        ("frac_border", "Border-cell fraction"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2), dpi=200)
    for ax, (col, title) in zip(axes, panels):
        ax.bar(range(len(df)), df[col].to_numpy(), color=colors)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("MINGL emission-model comparison")
    fig.tight_layout()
    path = os.path.join(out_dir, "emission_model_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def gaussian_bic_figure(df, out_dir):
    """Within-Gaussian-family BIC: diagonal vs full covariance (directly comparable)."""
    sub = df[df["model"].isin(["diagonal_gaussian", "full_gaussian"])].set_index("model")
    if not {"diagonal_gaussian", "full_gaussian"}.issubset(sub.index):
        return None
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    models = ["diagonal_gaussian", "full_gaussian"]
    ax.bar(models, [sub.loc[m, "bic"] for m in models], color=[MODEL_COLORS[m] for m in models])
    ax.set_ylabel("BIC (lower is better)")
    ax.set_title("Gaussian family: diagonal vs full covariance")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = os.path.join(out_dir, "gaussian_diagonal_vs_full_bic.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def per_model_border_summary(adata, cols, threshold, out_dir):
    """Attach every model's posterior and summarize border cells for each."""
    _quiet(
        tl.attach_all_model_probabilities,
        adata,
        cluster_col=cols["cluster_col"],
        neighborhood_col=cols["neighborhood_col"],
        region_key=cols["region_key"],
        x_key=cols["x_key"],
        y_key=cols["y_key"],
        ks=cols["ks"],
        k=cols["k"],
    )
    rows = []
    for model in MODEL_ORDER:
        pk = f"mingl_prob_{model}"
        if pk not in adata.obsm:
            continue
        m = tl.border_metrics_at_threshold(
            adata,
            threshold=threshold,
            prob_key=pk,
            prob_variable_key=f"{pk}_neighborhoods",
            cell_type_col=cols["cluster_col"],
            region_key=cols["region_key"],
            coord_keys=(cols["x_key"], cols["y_key"]),
        )
        rows.append({"model": model, "threshold": threshold, "n_border": m["n_border"], "frac_border": m["frac_border"]})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "per_model_border_summary.csv"), index=False)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="Path to a .h5ad or .csv file (real dataset).")
    ap.add_argument("--dataset", choices=sorted(DATASET_PRESETS), default="intestine")
    ap.add_argument("--synthetic", action="store_true", help="Use a built-in synthetic tissue (no lab data).")
    ap.add_argument("--cluster-col", default=None)
    ap.add_argument("--neighborhood-col", default=None)
    ap.add_argument("--region-key", default=None)
    ap.add_argument("--x-key", default=None)
    ap.add_argument("--y-key", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 20, 100, 300])
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument(
        "--subsample-frac",
        type=float,
        default=1.0,
        help="Fraction of cells to keep before analysis (tractability lever for very large datasets; "
        "windows are recomputed on the subsample, so <1.0 is an approximation).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=MODEL_ORDER)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "reviewer_emission_models_outputs"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    preset = dict(DATASET_PRESETS[args.dataset])
    cols = {
        "cluster_col": args.cluster_col or preset["cluster_col"],
        "neighborhood_col": args.neighborhood_col or preset["neighborhood_col"],
        "region_key": args.region_key or preset["region_key"],
        "x_key": args.x_key or preset["x_key"],
        "y_key": args.y_key or preset["y_key"],
        "k": args.k,
        "ks": tuple(args.ks),
    }

    if args.synthetic:
        print("Loading synthetic multi-neighborhood tissue (no lab data).")
        adata = _synthetic_multiunit(seed=args.seed)
        cols.update(cluster_col="Cell Type", neighborhood_col="Neighborhood", region_key="unique_region", x_key="x", y_key="y", ks=(10, 20))
    elif args.data:
        print(f"Loading {args.data} (dataset preset: {args.dataset})")
        adata = mg.pp.read_file(args.data)
    else:
        ap.error("Provide --data PATH or --synthetic.")

    if args.subsample_frac < 1.0:
        rng = np.random.default_rng(args.seed)
        n_keep = max(2, int(round(adata.n_obs * args.subsample_frac)))
        keep = np.sort(rng.choice(adata.n_obs, size=n_keep, replace=False))
        adata = adata[keep].copy()
        print(f"Subsampled to {adata.n_obs} cells (frac={args.subsample_frac}).")

    print(f"Cells: {adata.n_obs}  Columns: {cols}")
    print(f"Output directory: {args.out_dir}\n")

    print("== Held-out model comparison ==")
    cmp = _quiet(
        tl.compare_emission_models,
        adata,
        models=args.models,
        cluster_col=cols["cluster_col"],
        neighborhood_col=cols["neighborhood_col"],
        region_key=cols["region_key"],
        x_key=cols["x_key"],
        y_key=cols["y_key"],
        ks=cols["ks"],
        k=cols["k"],
        threshold=args.threshold,
        test_size=args.test_size,
        seed=args.seed,
    )
    cmp.to_csv(os.path.join(args.out_dir, "emission_model_comparison.csv"), index=False)
    show_cols = ["model", "test_logloss", "test_accuracy", "mean_entropy", "frac_border", "n_params", "bic"]
    print(cmp[show_cols].round(4).to_string(index=False))

    fig1 = comparison_figure(cmp, args.out_dir)
    fig2 = gaussian_bic_figure(cmp, args.out_dir)

    print("\n== Per-model border summary ==")
    border = per_model_border_summary(adata, cols, args.threshold, args.out_dir)
    print(border.round(4).to_string(index=False))

    print(f"\nArtifacts written to {args.out_dir}")
    print(f"  - emission_model_comparison.csv / {os.path.basename(fig1)}")
    if fig2:
        print(f"  - {os.path.basename(fig2)}")
    print("  - per_model_border_summary.csv")


if __name__ == "__main__":
    main()
