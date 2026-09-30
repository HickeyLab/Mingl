"""Threshold sensitivity of MINGL border cells (Task 3).

MINGL calls a cell a *border cell* when it is multi-membership: at least two
neighborhood probabilities exceed a threshold (the tutorials use
``Count_Above_Threshold in {2, 3}``). The border definition therefore depends on
that threshold, and the reviewer asked for a sensitivity analysis over a wider
range ``(0.01, 0.1, 0.25, 0.4, 0.49)`` covering four outputs:

* **Number of border cells** -- strictly non-increasing in the threshold.
* **Border location** -- per-region border fraction, spatial centroid/spread of
  border cells, and how stable the *set* of border cells is as the threshold moves
  (Jaccard overlap between consecutive thresholds).
* **Border composition** -- cell-type make-up of the border population.
* **Border cell-type enrichment** -- log2 enrichment of each cell type in the
  border relative to its overall abundance.

Everything here operates on a posterior matrix already stored in ``obsm`` (written
by :func:`mingl.tl.gmm.cpu_gmm_probability` or
:func:`mingl.tl.emission_models.mingl_membership_probabilities`), so it is
model-agnostic and does not re-run any clustering.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd

__all__ = [
    "get_probability_frame",
    "border_mask",
    "border_metrics_at_threshold",
    "threshold_sensitivity_analysis",
    "spatial_border_clustering_null",
]

_DEFAULT_THRESHOLDS = (0.01, 0.1, 0.25, 0.4, 0.49)
_EPS = 1e-9


def get_probability_frame(
    cells: ad.AnnData,
    *,
    prob_key: str = "neighborhood_probabilities",
    prob_variable_key: str = "neighborhood_probability_neighborhoods",
) -> pd.DataFrame:
    """Return the posterior matrix as a DataFrame (cells x neighborhoods)."""
    if prob_key not in cells.obsm:
        raise KeyError(f"{prob_key!r} not found in obsm; run a MINGL scorer first.")
    arr = np.asarray(cells.obsm[prob_key], dtype=float)
    if prob_variable_key in cells.uns:
        columns = [str(c) for c in cells.uns[prob_variable_key]]
    else:
        columns = [f"neighborhood_{i}" for i in range(arr.shape[1])]
    if len(columns) != arr.shape[1]:
        raise ValueError(
            f"{prob_variable_key} has {len(columns)} names but the matrix has {arr.shape[1]} columns."
        )
    return pd.DataFrame(arr, index=cells.obs_names, columns=columns)


def border_mask(probs: np.ndarray, threshold: float, min_memberships: int = 2) -> np.ndarray:
    """Boolean mask of border cells: ``>= min_memberships`` probabilities ``> threshold``."""
    return (np.asarray(probs, dtype=float) > threshold).sum(axis=1) >= min_memberships


def border_metrics_at_threshold(
    cells: ad.AnnData,
    *,
    threshold: float,
    prob_key: str = "neighborhood_probabilities",
    prob_variable_key: str = "neighborhood_probability_neighborhoods",
    cell_type_col: str = "Cell Type",
    region_key: Optional[str] = "unique_region",
    coord_keys: Sequence[str] = ("x", "y"),
    min_memberships: int = 2,
) -> Dict[str, object]:
    """Border metrics for a single threshold.

    Returns a dict with scalar summaries (``n_border``, ``frac_border``,
    ``mean_positive``, spatial centroid/spread, per-region fractions), a
    ``composition`` DataFrame (per cell type: border counts, share of the border
    population, per-type border rate, and log2 enrichment), and the boolean
    ``mask`` of border cells (obs-ordered).
    """
    probs = get_probability_frame(
        cells, prob_key=prob_key, prob_variable_key=prob_variable_key
    ).to_numpy()
    n_cells = probs.shape[0]

    pos_counts = (probs > threshold).sum(axis=1)
    mask = pos_counts >= min_memberships
    n_border = int(mask.sum())
    frac_border = float(n_border / n_cells) if n_cells else float("nan")

    result: Dict[str, object] = {
        "threshold": float(threshold),
        "n_cells": int(n_cells),
        "n_border": n_border,
        "frac_border": frac_border,
        "mean_positive": float(pos_counts.mean()) if n_cells else float("nan"),
        "mask": mask,
    }

    # --- Border composition & enrichment ---
    if cell_type_col in cells.obs:
        ct = cells.obs[cell_type_col].astype(str).to_numpy()
        all_types = sorted(pd.Index(ct).unique().tolist())
        total_by_type = pd.Series(ct, name="ct").value_counts().reindex(all_types).fillna(0).astype(int)
        border_by_type = (
            pd.Series(ct[mask], name="ct").value_counts().reindex(all_types).fillna(0).astype(int)
        )
        global_prop = total_by_type / max(n_cells, 1)
        prop_of_border = border_by_type / n_border if n_border else border_by_type * 0.0
        prop_of_type_border = border_by_type / total_by_type.replace(0, np.nan)
        log2_enrichment = np.log2((prop_of_border + _EPS) / (global_prop + _EPS))
        composition = pd.DataFrame(
            {
                "threshold": float(threshold),
                "cell_type": all_types,
                "n_total": total_by_type.to_numpy(),
                "n_border": border_by_type.to_numpy(),
                "prop_of_border": prop_of_border.to_numpy(),
                "prop_of_type_border": prop_of_type_border.to_numpy(),
                "global_prop": global_prop.to_numpy(),
                "log2_enrichment": log2_enrichment.to_numpy(),
            }
        )
        result["composition"] = composition

    # --- Border location: spatial centroid / spread ---
    coord_cols = [c for c in coord_keys if c in cells.obs]
    if len(coord_cols) >= 2 and n_border > 0:
        coords = cells.obs.loc[:, coord_cols].to_numpy(dtype=float)[mask]
        result["border_centroid"] = {c: float(coords[:, i].mean()) for i, c in enumerate(coord_cols)}
        result["border_spread"] = {c: float(coords[:, i].std()) for i, c in enumerate(coord_cols)}
    else:
        result["border_centroid"] = {}
        result["border_spread"] = {}

    # --- Border location: per-region fraction ---
    if region_key is not None and region_key in cells.obs:
        reg = cells.obs[region_key].astype(str).to_numpy()
        per_region = (
            pd.DataFrame({"region": reg, "border": mask})
            .groupby("region")["border"]
            .mean()
        )
        result["per_region_border_frac"] = per_region.to_dict()
    else:
        result["per_region_border_frac"] = {}

    return result


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.sum(a & b))
    union = int(np.sum(a | b))
    return float(inter / union) if union else float("nan")


def threshold_sensitivity_analysis(
    cells: ad.AnnData,
    *,
    thresholds: Sequence[float] = _DEFAULT_THRESHOLDS,
    prob_key: str = "neighborhood_probabilities",
    prob_variable_key: str = "neighborhood_probability_neighborhoods",
    cell_type_col: str = "Cell Type",
    region_key: Optional[str] = "unique_region",
    coord_keys: Sequence[str] = ("x", "y"),
    min_memberships: int = 2,
) -> Dict[str, pd.DataFrame]:
    """Sweep border metrics over ``thresholds``.

    Returns a dict of tidy DataFrames:

    * ``summary`` -- one row per threshold: ``n_border``, ``frac_border``,
      ``mean_positive``, ``n_regions_with_border``, spatial centroid/spread.
    * ``composition`` -- long table (threshold x cell_type) of border counts,
      shares, per-type border rate, and log2 enrichment.
    * ``stability`` -- consecutive-threshold pairs: Jaccard overlap of the border
      cell sets (location stability) and the Pearson/Spearman correlation of the
      per-cell-type log2 enrichment vectors.
    """
    thresholds = sorted(float(t) for t in thresholds)
    summary_rows: List[dict] = []
    comp_frames: List[pd.DataFrame] = []
    masks: List[np.ndarray] = []
    enrich_by_thr: Dict[float, pd.Series] = {}

    for t in thresholds:
        m = border_metrics_at_threshold(
            cells,
            threshold=t,
            prob_key=prob_key,
            prob_variable_key=prob_variable_key,
            cell_type_col=cell_type_col,
            region_key=region_key,
            coord_keys=coord_keys,
            min_memberships=min_memberships,
        )
        masks.append(np.asarray(m["mask"], dtype=bool))
        row = {
            "threshold": t,
            "n_border": m["n_border"],
            "frac_border": m["frac_border"],
            "mean_positive": m["mean_positive"],
            "n_regions_with_border": int(sum(v > 0 for v in m["per_region_border_frac"].values())),
        }
        for axis, val in m.get("border_centroid", {}).items():
            row[f"centroid_{axis}"] = val
        for axis, val in m.get("border_spread", {}).items():
            row[f"spread_{axis}"] = val
        summary_rows.append(row)

        if "composition" in m:
            comp_frames.append(m["composition"])
            enrich_by_thr[t] = m["composition"].set_index("cell_type")["log2_enrichment"]

    summary = pd.DataFrame(summary_rows)
    composition = (
        pd.concat(comp_frames, ignore_index=True) if comp_frames else pd.DataFrame()
    )

    stability_rows: List[dict] = []
    for i in range(1, len(thresholds)):
        t_prev, t_cur = thresholds[i - 1], thresholds[i]
        pair = {
            "threshold_low": t_prev,
            "threshold_high": t_cur,
            "jaccard_border_cells": _jaccard(masks[i - 1], masks[i]),
        }
        if t_prev in enrich_by_thr and t_cur in enrich_by_thr:
            a = enrich_by_thr[t_prev]
            b = enrich_by_thr[t_cur].reindex(a.index)
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.sum() >= 2 and a[finite].std() > 0 and b[finite].std() > 0:
                pair["pearson_enrichment"] = float(np.corrcoef(a[finite], b[finite])[0, 1])
                pair["spearman_enrichment"] = float(
                    pd.Series(a[finite]).corr(pd.Series(b[finite].to_numpy(), index=a[finite].index), method="spearman")
                )
            else:
                pair["pearson_enrichment"] = float("nan")
                pair["spearman_enrichment"] = float("nan")
        stability_rows.append(pair)

    stability = pd.DataFrame(stability_rows)
    return {"summary": summary, "composition": composition, "stability": stability}


def spatial_border_clustering_null(
    cells: ad.AnnData,
    *,
    threshold: float = 0.25,
    prob_key: str = "neighborhood_probabilities",
    prob_variable_key: str = "neighborhood_probability_neighborhoods",
    region_key: Optional[str] = "unique_region",
    coord_keys: Sequence[str] = ("x", "y"),
    min_memberships: int = 2,
    n_permutations: int = 200,
    seed: int = 0,
) -> Dict[str, object]:
    """Test whether border cells are spatially clustered (a null-condition analysis).

    Statistic = mean distance from each border cell to its nearest other border
    cell (within region). The null keeps the number of border cells per region
    fixed but reassigns which cells are border uniformly at random, ``n_permutations``
    times. A statistic below the null (borders closer together than chance) means
    the border *location* is spatially structured, not random.

    Returns the observed statistic, the null mean/std, and a one-sided empirical
    p-value ``P(null <= observed)``.
    """
    from sklearn.neighbors import NearestNeighbors

    coord_cols = [c for c in coord_keys if c in cells.obs]
    if len(coord_cols) < 2:
        raise KeyError(f"Need coordinate columns {tuple(coord_keys)} in obs for a spatial null.")

    probs = get_probability_frame(
        cells, prob_key=prob_key, prob_variable_key=prob_variable_key
    ).to_numpy()
    obs_mask = (probs > threshold).sum(axis=1) >= min_memberships
    coords_all = cells.obs.loc[:, coord_cols].to_numpy(dtype=float)
    if region_key is not None and region_key in cells.obs:
        regions = cells.obs[region_key].astype(str).to_numpy()
    else:
        regions = np.zeros(coords_all.shape[0], dtype=int)

    def _mean_nn_distance(mask: np.ndarray) -> float:
        dists: List[float] = []
        for r in np.unique(regions):
            in_r = regions == r
            border_r = mask & in_r
            if border_r.sum() < 2:
                continue
            pts = coords_all[border_r]
            nn = NearestNeighbors(n_neighbors=2).fit(pts)
            d, _ = nn.kneighbors(pts)
            dists.append(d[:, 1])  # nearest neighbor that is not self
        if not dists:
            return float("nan")
        return float(np.concatenate(dists).mean())

    observed = _mean_nn_distance(obs_mask)

    rng = np.random.default_rng(seed)
    null_stats = np.empty(n_permutations)
    region_values = np.unique(regions)
    n_border_per_region = {r: int((obs_mask & (regions == r)).sum()) for r in region_values}
    for p in range(n_permutations):
        perm_mask = np.zeros_like(obs_mask)
        for r in region_values:
            idx = np.where(regions == r)[0]
            nb = n_border_per_region[r]
            if nb > 0:
                chosen = rng.choice(idx, size=nb, replace=False)
                perm_mask[chosen] = True
        null_stats[p] = _mean_nn_distance(perm_mask)

    finite_null = null_stats[np.isfinite(null_stats)]
    if np.isfinite(observed) and finite_null.size:
        p_value = float((np.sum(finite_null <= observed) + 1) / (finite_null.size + 1))
    else:
        p_value = float("nan")

    return {
        "threshold": float(threshold),
        "observed_mean_nn_distance": observed,
        "null_mean": float(finite_null.mean()) if finite_null.size else float("nan"),
        "null_std": float(finite_null.std()) if finite_null.size else float("nan"),
        "p_value_clustered": p_value,
        "n_border": int(obs_mask.sum()),
        "n_permutations": int(n_permutations),
    }
