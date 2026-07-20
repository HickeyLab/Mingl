"""Do the emission distributions change the downstream border CONCLUSIONS?

The emission-model comparison shows the models call very different *numbers* of
border cells. This driver asks the deeper question: do they agree on **which** cells
are borders and on **which cell types** are enriched there?

For each model it fits the posterior (all cells), then at a threshold computes:
  * n_border (number of border cells),
  * pairwise **Jaccard** of the border-cell SETS  -> do models flag the same cells?
  * pairwise **Spearman** of per-cell-type log2 enrichment -> same biology?
  * top enriched cell types per model, side by side.

Writes CSVs + prints a summary. Read-only w.r.t. any input; outputs are local.

    python tools/reviewer_model_agreement.py --dataset spatial   --data <esophagus>.h5ad
    python tools/reviewer_model_agreement.py --dataset intestine --data intestine.h5ad
    python tools/reviewer_model_agreement.py --dataset melanoma  --data melanoma.csv --subsample-frac 0.3
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import warnings

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
MODELS = ["diagonal_gaussian", "full_gaussian", "multinomial", "dirichlet_multinomial", "logistic_normal"]
SHORT = {"diagonal_gaussian": "diag", "full_gaussian": "full", "multinomial": "multi",
         "dirichlet_multinomial": "dirmult", "logistic_normal": "lognorm"}


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _spearman(x, y):
    from scipy.stats import spearmanr
    ok = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[ok], y[ok]).correlation) if ok.sum() > 2 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", choices=sorted(DATASET_PRESETS), default="intestine")
    ap.add_argument("--cluster-col", default=None)
    ap.add_argument("--neighborhood-col", default=None)
    ap.add_argument("--region-key", default=None)
    ap.add_argument("--x-key", default=None)
    ap.add_argument("--y-key", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--subsample-frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "reviewer_model_agreement_outputs"))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    p = dict(DATASET_PRESETS[args.dataset])
    cl = args.cluster_col or p["cluster_col"]
    nb = args.neighborhood_col or p["neighborhood_col"]
    rg = args.region_key or p["region_key"]
    xk = args.x_key or p["x_key"]
    yk = args.y_key or p["y_key"]
    tag = args.tag or f"{args.dataset}__{nb.replace(' ', '_')}"
    out_dir = os.path.join(args.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    adata = mg.pp.read_file(args.data)
    if args.subsample_frac < 1.0:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(adata.n_obs, int(adata.n_obs * args.subsample_frac), replace=False))
        adata = adata[keep].copy()
    print(f"{args.dataset}: {adata.n_obs:,} cells, threshold={args.threshold}, tag={tag}")

    _quiet(tl.attach_all_model_probabilities, adata, models=MODELS, cluster_col=cl,
           neighborhood_col=nb, region_key=rg, x_key=xk, y_key=yk, k=args.k)

    masks, enr = {}, {}
    for m in MODELS:
        pk = f"mingl_prob_{m}"
        res = tl.border_metrics_at_threshold(adata, threshold=args.threshold, prob_key=pk,
              prob_variable_key=f"{pk}_neighborhoods", cell_type_col=cl, region_key=rg, coord_keys=(xk, yk))
        masks[m] = np.asarray(res["mask"], bool)
        enr[m] = res["composition"].set_index("cell_type")["log2_enrichment"]

    counts = pd.DataFrame({"model": MODELS, "n_border": [int(masks[m].sum()) for m in MODELS],
                           "frac_border": [float(masks[m].mean()) for m in MODELS]})
    counts.to_csv(os.path.join(out_dir, "n_border_by_model.csv"), index=False)
    print("\n== n_border by model ==")
    print(counts.to_string(index=False))

    jac = pd.DataFrame(index=[SHORT[m] for m in MODELS], columns=[SHORT[m] for m in MODELS], dtype=float)
    for a in MODELS:
        for b in MODELS:
            u = int((masks[a] | masks[b]).sum())
            jac.loc[SHORT[a], SHORT[b]] = (int((masks[a] & masks[b]).sum()) / u) if u else np.nan
    jac.to_csv(os.path.join(out_dir, "border_set_jaccard.csv"))
    print("\n== Jaccard of border-cell SETS (same cells flagged?) ==")
    print(jac.round(2).to_string())

    cts = sorted(set().union(*[e.index for e in enr.values()]))
    E = {m: enr[m].reindex(cts).values for m in MODELS}
    sp = pd.DataFrame(index=[SHORT[m] for m in MODELS], columns=[SHORT[m] for m in MODELS], dtype=float)
    for a in MODELS:
        for b in MODELS:
            sp.loc[SHORT[a], SHORT[b]] = _spearman(E[a], E[b])
    sp.to_csv(os.path.join(out_dir, "enrichment_spearman.csv"))
    print("\n== Spearman of per-cell-type ENRICHMENT (same biology?) ==")
    print(sp.round(2).to_string())

    print("\n== Top-5 border-enriched cell types per model ==")
    rows = []
    for m in MODELS:
        top = enr[m].sort_values(ascending=False).head(5)
        print(f"   {SHORT[m]:8s}: " + ", ".join(f"{ct} ({v:.2f})" for ct, v in top.items()))
        for rank, (ct, v) in enumerate(top.items(), 1):
            rows.append({"model": m, "rank": rank, "cell_type": ct, "log2_enrichment": v})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "top_enriched_by_model.csv"), index=False)
    print(f"\nArtifacts written to {out_dir}")


if __name__ == "__main__":
    main()
