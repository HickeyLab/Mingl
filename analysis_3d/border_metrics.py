"""Border-agreement measures shared by the Main HER2 border analyses (05, 06).

A cell is a border cell when >= 2 membership probabilities exceed the
threshold, i.e. when its second-largest probability does. That second-largest
probability is used throughout as the continuous border score.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from common import THRESHOLD

THRESHOLDS = np.round(np.arange(0.05, 0.451, 0.05), 2)
PREVALENCES = (0.02, 0.05, 0.10, 0.20, 0.30)
TOLERANCES = (5.0, 10.0, 15.0, 20.0)  # µm
MIRRORS = ((0,), (1,), (0, 1))  # x, y, and both: the null border patterns
MIN_CELLS_PER_TYPE = 50


def border_score(probabilities) -> np.ndarray:
    """Second-largest membership probability per cell."""
    return np.sort(np.asarray(probabilities, dtype=float), axis=1)[:, -2]


def fractions(windows: pd.DataFrame, cell_types: list[str]) -> np.ndarray:
    w = windows[cell_types].to_numpy(float)
    return w / w.sum(axis=1, keepdims=True)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else float("nan")


def chance_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Expected Jaccard of two independent sets with these prevalences."""
    pa, pb = a.mean(), b.mean()
    return float(pa * pb / (pa + pb - pa * pb)) if (pa + pb) > 0 else float("nan")


def top_fraction(score: np.ndarray, q: float) -> np.ndarray:
    return score >= np.quantile(score, 1 - q)


def auc(score: np.ndarray, target: np.ndarray) -> float:
    return float(roc_auc_score(target, score)) if 0 < target.sum() < len(target) else float("nan")


def enrichment(border: np.ndarray, cell_type: np.ndarray) -> pd.Series:
    """log2 enrichment of each cell type among border cells."""
    total = pd.Series(cell_type).value_counts()
    total = total[total >= MIN_CELLS_PER_TYPE]
    at_border = pd.Series(cell_type[border]).value_counts().reindex(total.index, fill_value=0)
    return np.log2((at_border / max(border.sum(), 1) + 1e-4) / (total / len(cell_type) + 1e-4))


def tolerant_overlap(xyz: np.ndarray, a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    """Share of border cells with a border cell of the other run within r µm (mean of both directions).

    Null: the other run's border cells mirrored in x, in y and in both (mean of
    the three). Mirroring keeps how clumped that border pattern is and removes
    any correspondence with this run's borders. Shuffling border status instead
    would spread the border cells evenly and overstate chance.
    """
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)

    def within(src: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
        d = cKDTree(target_xyz).query(xyz[src], k=1)[0]
        return np.array([(d <= r).mean() for r in TOLERANCES])

    def mirrored(mask: np.ndarray, axes) -> np.ndarray:
        out = xyz[mask].copy()
        out[:, axes] = lo[list(axes)] + hi[list(axes)] - out[:, axes]
        return out

    observed = 0.5 * (within(a, xyz[b]) + within(b, xyz[a]))
    null = np.mean([0.5 * (within(a, mirrored(b, ax)) + within(b, mirrored(a, ax))) for ax in MIRRORS], axis=0)
    return pd.DataFrame({"tolerance_um": TOLERANCES, "overlap": observed, "overlap_null": null,
                         "overlap_excess": (observed - null) / (1 - null)})


def compare_border_scores(pa, pb, xyz, cell_type, key: dict, extra: dict | None = None):
    """All border-agreement measures for two border-score vectors on the same cells.

    ``key`` labels the comparison in every output; ``extra`` adds columns to the
    summary row. Returns ``(row, sweep, tolerance)``: one summary dict, Jaccard
    across thresholds and matched prevalences, and the distance-tolerant overlap.
    """
    ba, bb = pa > THRESHOLD, pb > THRESHOLD
    ea, eb = enrichment(ba, cell_type), enrichment(bb, cell_type)
    tol = tolerant_overlap(xyz, ba, bb)

    row = {
        **key, "n_cells": len(pa),
        "pct_border_a": round(100 * ba.mean(), 1), "pct_border_b": round(100 * bb.mean(), 1),
        **(extra or {}),
        "border_jaccard": round(jaccard(ba, bb), 3),
        "border_jaccard_chance": round(chance_jaccard(ba, bb), 3),
        "border_jaccard_top10pct": round(jaccard(top_fraction(pa, 0.10), top_fraction(pb, 0.10)), 3),
        "p2_spearman": round(float(spearmanr(pa, pb)[0]), 3),
        "p2_auc_for_other_border": round(0.5 * (auc(pa, bb) + auc(pb, ba)), 3),
        "celltype_enrichment_r": round(float(np.corrcoef(ea, eb)[0, 1]), 3),
    }
    for _, t in tol.iterrows():
        if t["tolerance_um"] in (5.0, 10.0):
            r = int(t["tolerance_um"])
            row[f"within_{r}um"] = round(t["overlap"], 3)
            row[f"within_{r}um_null"] = round(t["overlap_null"], 3)

    sweep = [{**key, "sweep": "threshold", "value": float(t), "jaccard": jaccard(pa > t, pb > t),
              "chance": chance_jaccard(pa > t, pb > t),
              "pct_border_a": 100 * (pa > t).mean(), "pct_border_b": 100 * (pb > t).mean()} for t in THRESHOLDS]
    sweep += [{**key, "sweep": "prevalence", "value": q,
               "jaccard": jaccard(top_fraction(pa, q), top_fraction(pb, q)), "chance": q / (2 - q),
               "pct_border_a": 100 * q, "pct_border_b": 100 * q} for q in PREVALENCES]
    return row, pd.DataFrame(sweep), tol.assign(**key)


KIND_STYLE = {
    "2D vs 3D": dict(color="#c0392b", lw=2.2),
    "k only": dict(color="#2c6fbb", lw=1.6),
    "seed only": dict(color="#555555", lw=1.6),
}


def plot_robustness(sweep: pd.DataFrame, tol: pd.DataFrame, title: str, path) -> None:
    """Border Jaccard across thresholds and prevalences, and distance-tolerant overlap, per comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    panels = (
        (axes[0], sweep[sweep["sweep"] == "threshold"], "value", "jaccard", "probability threshold", "border Jaccard"),
        (axes[1], sweep[sweep["sweep"] == "prevalence"], "value", "jaccard",
         "border prevalence (top fraction of border score)", "border Jaccard"),
        (axes[2], tol, "tolerance_um", "overlap_excess", "distance tolerance (µm)",
         "overlap above mirrored null\n(0 = chance, 1 = complete)"),
    )
    for ax, df, x, y, xlabel, ylabel in panels:
        seen = set()
        for (name, kind), sub in df.groupby(["comparison", "kind"], sort=False):
            ls = "-" if kind not in seen else "--"
            seen.add(kind)
            ax.plot(sub[x], sub[y], marker="o", ms=4, ls=ls, label=name, **KIND_STYLE[kind])
        if "chance" in df:
            chance = df.groupby(x)["chance"].mean()
            ax.plot(chance.index, chance.to_numpy(), color="black", lw=1, ls=":", label="chance (mean over comparisons)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(min(0, df[y].min() - 0.03), 1.02)
    axes[0].axvline(THRESHOLD, color="grey", lw=0.8, ls=":")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
