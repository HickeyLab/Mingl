"""Border cells for **one selected pair** of organizational units.

The shipped :mod:`mingl.tl.threshold_sensitivity` treats "border cell" globally
(any cell with >= 2 positive memberships) and scores enrichment against the whole
tissue. Both figures here need the manuscript's *pair-specific* definition
(Methods 4.5), which is what "pick one border" means:

* **border group**  -- cells positive for **both** CN1 and CN2,
* **CN1-only / CN2-only groups** -- cells positive for exactly one of them,
* **border-specific cell-type enrichment**

  .. math::

      E_t^{(CN1)} = \\log_2\\frac{p_{t,\\mathrm{border}}}{p_{t,CN1\\text{-only}}}
      \\qquad
      E_t^{(CN2)} = \\log_2\\frac{p_{t,\\mathrm{border}}}{p_{t,CN2\\text{-only}}}

  with a cell type called *enriched at the interface* only when it is positive
  against **both** single-unit groups.

It also adds the quantity neither the package nor the drivers currently compute:
a **border location** coordinate. For the chosen pair, each cell gets

.. math::  s = (d_{CN1} - d_{CN2}) / d_{NN}

where :math:`d_{CN1}` is the distance to the nearest cell *discretely assigned*
to CN1 (excluding the cell itself), and :math:`d_{NN}` is the region's median
nearest-neighbor spacing -- the same density normalization the manuscript uses
for spatial gradients (Methods 4.7). ``s`` is negative inside CN1, positive
inside CN2 and ~0 at the geometric interface, so ``|s|`` measures *how far a
called border cell actually sits from the CN1|CN2 interface*, in cell diameters,
comparably across regions and datasets.

Because the coordinate depends only on the discrete labels and the coordinates,
it is computed **once** per dataset/level and reused for every emission model and
every threshold -- which is what makes both figures affordable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

__all__ = [
    "PairBorder",
    "attach_posterior",
    "positive_matrix",
    "select_top_pair",
    "pair_masks",
    "interface_coordinate",
    "pair_composition",
    "composition_via_mingl",
    "pair_enrichment",
    "per_region_border_stats",
    "analyze_pair",
]

#: The obsm/uns keys every shipped MINGL function reads.
MINGL_PROB_KEY = "neighborhood_probabilities"
MINGL_UNITS_KEY = "neighborhood_probability_neighborhoods"


def attach_posterior(adata, probs: pd.DataFrame):
    """Write a posterior into the canonical MINGL keys and return ``adata``.

    Both figures compare several posteriors (one per emission model, or one per
    threshold) over the same cells. Rather than reimplementing MINGL's analyses
    for each, this drops the posterior into the exact ``obsm``/``uns`` keys the
    shipped functions read, so :func:`mingl.tl.compute_grouped_proportions`,
    :func:`mingl.tl.findPositives`, :func:`mingl.pl.spatial_loc_region` and
    :func:`mingl.pl.plot_border_enrichment` all operate on it unmodified.
    """
    adata.obsm[MINGL_PROB_KEY] = probs.reindex(adata.obs_names).to_numpy(dtype=float)
    adata.uns[MINGL_UNITS_KEY] = [str(c) for c in probs.columns]
    return adata

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Positive membership + pair selection
# ---------------------------------------------------------------------------
def positive_matrix(
    probs: pd.DataFrame | np.ndarray, threshold: float, *, inclusive: bool = False
) -> np.ndarray:
    """Boolean (cells x units) matrix of positive memberships.

    ``inclusive=False`` (default) uses ``p > threshold``, matching
    :func:`mingl.tl.border_mask` and the shipped tutorials. The manuscript
    Methods write ``p >= 0.25``; set ``inclusive=True`` for that convention. The
    two differ only on exact ties, which do not occur for continuous posteriors.
    """
    arr = probs.to_numpy(dtype=float) if isinstance(probs, pd.DataFrame) else np.asarray(probs, dtype=float)
    return arr >= threshold if inclusive else arr > threshold


def select_top_pair(
    probs: pd.DataFrame,
    *,
    threshold: float,
    inclusive: bool = False,
    exclude: Sequence[str] = (),
) -> tuple[str, str]:
    """Pair of units sharing the most co-positive (border) cells.

    Used when a dataset/level has no manuscript-named focus border. Ties are
    broken by the alphabetical unit names so the choice is deterministic.
    """
    pos = positive_matrix(probs, threshold, inclusive=inclusive)
    names = list(probs.columns)
    keep = [i for i, n in enumerate(names) if str(n) not in set(exclude)]
    if len(keep) < 2:
        raise ValueError("Need at least two organizational units to pick a border pair.")
    pos = pos[:, keep]
    kept_names = [str(names[i]) for i in keep]
    # co[i, j] = number of cells positive for both unit i and unit j
    co = pos.T.astype(np.int64) @ pos.astype(np.int64)
    np.fill_diagonal(co, 0)
    best, best_count = None, -1
    for i in range(len(kept_names)):
        for j in range(i + 1, len(kept_names)):
            count = int(co[i, j])
            if count > best_count:
                best, best_count = (kept_names[i], kept_names[j]), count
    if best is None or best_count <= 0:
        raise ValueError(f"No pair of units shares a border cell at threshold {threshold}.")
    return best


def pair_masks(
    probs: pd.DataFrame,
    cn1: str,
    cn2: str,
    *,
    threshold: float,
    inclusive: bool = False,
    require_exactly_two: bool = False,
) -> dict[str, np.ndarray]:
    """Border / CN1-only / CN2-only masks for one pair (Methods 4.5).

    ``require_exactly_two`` restricts the border group to cells positive for
    *only* these two units (the manuscript does not require this; the default
    ``False`` matches it).
    """
    for unit in (cn1, cn2):
        if unit not in probs.columns:
            raise KeyError(
                f"Unit {unit!r} not among the scored units: {list(probs.columns)[:12]}..."
            )
    pos = positive_matrix(probs, threshold, inclusive=inclusive)
    cols = list(probs.columns)
    i1, i2 = cols.index(cn1), cols.index(cn2)
    p1, p2 = pos[:, i1], pos[:, i2]
    n_pos = pos.sum(axis=1)

    border = p1 & p2
    if require_exactly_two:
        border = border & (n_pos == 2)
    return {
        "border": border,
        "cn1_only": p1 & ~p2,
        "cn2_only": p2 & ~p1,
        "cn1_positive": p1,
        "cn2_positive": p2,
        "any_positive": p1 | p2,
        "n_positive": n_pos,
    }


# ---------------------------------------------------------------------------
# Border location
# ---------------------------------------------------------------------------
def interface_coordinate(
    obs: pd.DataFrame,
    *,
    unit_col: str,
    cn1: str,
    cn2: str,
    region_key: str,
    x_key: str = "x",
    y_key: str = "y",
    normalize: bool = True,
) -> pd.DataFrame:
    """Signed distance to the CN1|CN2 interface, per cell.

    Returns a frame indexed like ``obs`` with

    ``d_cn1`` / ``d_cn2``
        Distance to the nearest cell discretely labeled CN1 / CN2 (self excluded).
    ``interface_coord``
        ``(d_cn1 - d_cn2) / d_NN`` -- negative on the CN1 side, positive on the
        CN2 side, and exactly 0 for a cell equidistant from both units, i.e. one
        sitting on the interface itself. ``d_NN`` is the region's median
        nearest-neighbor distance when ``normalize`` (units of cell diameters),
        else 1.
    ``dist_to_interface``
        ``|interface_coord|``.
    ``region_nn_spacing``
        The ``d_NN`` used, for reference.

    Regions that contain no cell of one of the two units yield NaN (there is no
    interface there); those regions are dropped from location statistics rather
    than silently contributing zeros.
    """
    for col in (unit_col, region_key, x_key, y_key):
        if col not in obs.columns:
            raise KeyError(f"Column {col!r} missing from obs.")

    units = obs[unit_col].astype(str).to_numpy()
    regions = obs[region_key].astype(str).to_numpy()
    coords = obs.loc[:, [x_key, y_key]].to_numpy(dtype=float)

    n = obs.shape[0]
    d1 = np.full(n, np.nan)
    d2 = np.full(n, np.nan)
    spacing = np.full(n, np.nan)

    for region in pd.unique(regions):
        in_r = regions == region
        idx = np.where(in_r)[0]
        pts = coords[idx]
        finite = np.isfinite(pts).all(axis=1)
        if finite.sum() < 2:
            continue
        idx, pts = idx[finite], pts[finite]

        tree_all = cKDTree(pts)
        nn = tree_all.query(pts, k=2)[0][:, 1]
        d_nn = float(np.median(nn[np.isfinite(nn)])) if np.isfinite(nn).any() else np.nan
        if not np.isfinite(d_nn) or d_nn <= 0:
            d_nn = 1.0
        spacing[idx] = d_nn

        for target, out in ((cn1, d1), (cn2, d2)):
            is_target = units[idx] == str(target)
            if not is_target.any():
                continue  # leave NaN: this unit is absent from the region
            tree = cKDTree(pts[is_target])
            # k=2 so a cell of the target unit does not match itself.
            dist = tree.query(pts, k=min(2, int(is_target.sum())))[0]
            if dist.ndim == 1:  # only one target cell in the region
                nearest = np.where(is_target, np.inf, dist)
            else:
                nearest = np.where(is_target, dist[:, 1], dist[:, 0])
            out[idx] = nearest

    denom = spacing if normalize else np.ones(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        coord = (d1 - d2) / denom
    return pd.DataFrame(
        {
            "d_cn1": d1,
            "d_cn2": d2,
            "interface_coord": coord,
            "dist_to_interface": np.abs(coord),
            "region_nn_spacing": spacing,
        },
        index=obs.index,
    )


# ---------------------------------------------------------------------------
# Composition + enrichment
# ---------------------------------------------------------------------------
def pair_composition(
    cell_types: Sequence[str],
    masks: dict[str, np.ndarray],
    *,
    cell_type_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Cell-type counts and proportions in the border / CN1-only / CN2-only groups.

    Long format: one row per (group, cell_type) with ``n`` and ``proportion``.

    This is the mask-level equivalent of :func:`mingl.tl.compute_grouped_proportions`,
    which returns the same three groups (``"n1 only"`` / ``"n1 + n2"`` /
    ``"n2 only"``) as proportions from an AnnData. Use
    :func:`composition_via_mingl` when you have an AnnData -- that calls the
    shipped function directly. This variant exists because the region-level
    statistics need *counts* as well as proportions, and need to restrict to an
    arbitrary subset of cells (one region at a time), which the shipped
    signature does not expose. Group definitions are identical: positive means
    ``p > threshold`` and the border group is positive for both.
    """
    ct = np.asarray(cell_types, dtype=object).astype(str)
    order = list(cell_type_order) if cell_type_order is not None else sorted(pd.unique(ct).tolist())
    rows = []
    for group in ("border", "cn1_only", "cn2_only"):
        mask = np.asarray(masks[group], dtype=bool)
        counts = pd.Series(ct[mask]).value_counts().reindex(order).fillna(0).astype(int)
        total = int(counts.sum())
        rows.append(
            pd.DataFrame(
                {
                    "group": group,
                    "cell_type": order,
                    "n": counts.to_numpy(),
                    "proportion": counts.to_numpy() / total if total else np.zeros(len(order)),
                    "group_total": total,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def composition_via_mingl(
    adata,
    *,
    cn1: str,
    cn2: str,
    threshold: float,
    cell_type_col: str = "Cell Type",
) -> pd.DataFrame:
    """Border composition straight from :func:`mingl.tl.compute_grouped_proportions`.

    Returns the shipped function's output relabeled to this module's group names
    (``border`` / ``cn1_only`` / ``cn2_only``) so it is interchangeable with
    :func:`pair_composition`. Requires the posterior to be attached with
    :func:`attach_posterior` first.
    """
    import mingl.tl as tl

    out = tl.compute_grouped_proportions(
        adata, cn1, cn2, cell_type_col=cell_type_col, threshold=float(threshold)
    )
    renames = {f"{cn1} only": "cn1_only", f"{cn1} + {cn2}": "border", f"{cn2} only": "cn2_only"}
    out = out.rename(columns={cell_type_col: "cell_type", "Proportion": "proportion"})
    out["group"] = out["Subset"].map(renames).fillna(out["Subset"])
    return out.loc[:, ["group", "cell_type", "proportion"]]


def pair_enrichment(composition: pd.DataFrame, *, min_cells: int = 5) -> pd.DataFrame:
    """Border-specific cell-type enrichment, exactly as in Methods 4.5.

    Numerically identical to what :func:`mingl.pl.plot_border_enrichment`
    computes internally -- ``log2((p_border + eps) / (p_CN1-only + eps))`` and the
    same against CN2, with ``eps = 1e-9``. That function returns *figures*, not
    the values, so this returns them as a table for the paired statistics and the
    threshold sweep; call the shipped function for the manuscript Figure 2f dot
    plot itself.

    ``log2_vs_cn1`` / ``log2_vs_cn2`` are the border proportion over each
    single-unit proportion; ``enriched_both`` is True only when a cell type is
    positive against both -- the manuscript's criterion for "enriched at the
    interface". ``min_enrichment`` = ``min(log2_vs_cn1, log2_vs_cn2)`` is that
    criterion as a continuous score, so it can be tracked across models and
    thresholds.

    Cell types with fewer than ``min_cells`` border cells get NaN enrichment:
    a log-ratio built on a handful of cells is noise, not signal.
    """
    wide = composition.pivot(index="cell_type", columns="group", values="proportion")
    counts = composition.pivot(index="cell_type", columns="group", values="n")
    for group in ("border", "cn1_only", "cn2_only"):
        if group not in wide.columns:
            wide[group] = np.nan
            counts[group] = 0

    with np.errstate(divide="ignore", invalid="ignore"):
        e1 = np.log2((wide["border"] + _EPS) / (wide["cn1_only"] + _EPS))
        e2 = np.log2((wide["border"] + _EPS) / (wide["cn2_only"] + _EPS))

    out = pd.DataFrame(
        {
            "cell_type": wide.index.astype(str),
            "n_border": counts["border"].to_numpy(),
            "n_cn1_only": counts["cn1_only"].to_numpy(),
            "n_cn2_only": counts["cn2_only"].to_numpy(),
            "prop_border": wide["border"].to_numpy(),
            "prop_cn1_only": wide["cn1_only"].to_numpy(),
            "prop_cn2_only": wide["cn2_only"].to_numpy(),
            "log2_vs_cn1": e1.to_numpy(),
            "log2_vs_cn2": e2.to_numpy(),
        }
    )
    # Exactly the filter mingl.pl.plot_border_enrichment applies:
    #   mask_ok = (c1 >= min_count) & (c2 >= min_count) & (cb >= min_count)
    # i.e. all THREE groups must clear the count, not just the border group --
    # a log-ratio whose denominator rests on a couple of cells is noise. Keeping
    # this identical means the heat map and the shipped dot plot always agree on
    # which cell types are reportable.
    keep = (
        (out["n_border"] >= int(min_cells))
        & (out["n_cn1_only"] >= int(min_cells))
        & (out["n_cn2_only"] >= int(min_cells))
    ).to_numpy()
    out.loc[~keep, ["log2_vs_cn1", "log2_vs_cn2"]] = np.nan
    out["min_enrichment"] = np.minimum(out["log2_vs_cn1"], out["log2_vs_cn2"])
    out["enriched_both"] = (out["log2_vs_cn1"] > 0) & (out["log2_vs_cn2"] > 0)
    return out.sort_values("min_enrichment", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-region tables (the unit of the paired statistics)
# ---------------------------------------------------------------------------
def per_region_border_stats(
    obs: pd.DataFrame,
    masks: dict[str, np.ndarray],
    location: pd.DataFrame,
    *,
    region_key: str,
    cell_type_col: str,
    min_border_cells: int = 20,
    cell_type_order: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Region-level summaries -- the independent units for every paired test.

    Returns

    ``summary``
        One row per region: border-cell count and fraction, median/mean distance
        to the interface, and the fraction of border cells within one cell
        diameter of it.
    ``composition``
        Region x cell-type proportion among that region's border cells.
    ``enrichment``
        Region x cell-type ``min_enrichment`` (Methods 4.5, per region).

    Regions with fewer than ``min_border_cells`` border cells, or with no
    interface (one of the two units absent), are excluded from ``composition``
    and ``enrichment`` and flagged ``usable=False`` in ``summary``.
    """
    regions = obs[region_key].astype(str).to_numpy()
    ct = obs[cell_type_col].astype(str).to_numpy()
    order = list(cell_type_order) if cell_type_order is not None else sorted(pd.unique(ct).tolist())
    dist = location["dist_to_interface"].to_numpy()
    coord = location["interface_coord"].to_numpy()

    border = np.asarray(masks["border"], dtype=bool)
    summary_rows, comp_rows, enr_rows = [], [], []

    for region in pd.unique(regions):
        in_r = regions == region
        b_r = border & in_r
        n_border = int(b_r.sum())
        has_interface = bool(np.isfinite(dist[in_r]).any())
        usable = bool(n_border >= min_border_cells and has_interface)

        d_border = dist[b_r]
        d_border = d_border[np.isfinite(d_border)]
        summary_rows.append(
            {
                "region": region,
                "n_cells": int(in_r.sum()),
                "n_border": n_border,
                "frac_border": n_border / max(int(in_r.sum()), 1),
                "median_dist_to_interface": float(np.median(d_border)) if d_border.size else np.nan,
                "mean_dist_to_interface": float(np.mean(d_border)) if d_border.size else np.nan,
                "frac_within_1_diameter": float(np.mean(d_border <= 1.0)) if d_border.size else np.nan,
                "median_signed_coord": (
                    float(np.median(coord[b_r][np.isfinite(coord[b_r])]))
                    if np.isfinite(coord[b_r]).any() else np.nan
                ),
                "has_interface": has_interface,
                "usable": usable,
            }
        )
        if not usable:
            continue

        region_masks = {g: np.asarray(masks[g], dtype=bool) & in_r for g in ("border", "cn1_only", "cn2_only")}
        comp = pair_composition(ct, region_masks, cell_type_order=order)
        border_comp = comp[comp["group"] == "border"].set_index("cell_type")["proportion"]
        comp_rows.append(border_comp.rename(region))
        enr = pair_enrichment(comp).set_index("cell_type")["min_enrichment"]
        enr_rows.append(enr.rename(region))

    summary = pd.DataFrame(summary_rows)
    composition = pd.DataFrame(comp_rows) if comp_rows else pd.DataFrame(columns=order)
    enrichment = pd.DataFrame(enr_rows) if enr_rows else pd.DataFrame(columns=order)
    composition.index.name = "region"
    enrichment.index.name = "region"
    return {"summary": summary, "composition": composition, "enrichment": enrichment}


# ---------------------------------------------------------------------------
# One-shot bundle
# ---------------------------------------------------------------------------
@dataclass
class PairBorder:
    """Everything computed for one (pair, threshold, posterior) combination."""

    cn1: str
    cn2: str
    threshold: float
    n_cells: int
    n_border: int
    n_cn1_only: int
    n_cn2_only: int
    masks: dict[str, np.ndarray]
    composition: pd.DataFrame
    enrichment: pd.DataFrame
    per_region: dict[str, pd.DataFrame]
    location_of_border: np.ndarray  # |interface coord| of the border cells only

    @property
    def frac_border(self) -> float:
        return self.n_border / self.n_cells if self.n_cells else float("nan")

    def summary_row(self, **extra) -> dict:
        """Flat dict for a tidy results table."""
        d = self.location_of_border[np.isfinite(self.location_of_border)]
        row = {
            "cn1": self.cn1,
            "cn2": self.cn2,
            "threshold": self.threshold,
            "n_cells": self.n_cells,
            "n_border": self.n_border,
            "frac_border": self.frac_border,
            "n_cn1_only": self.n_cn1_only,
            "n_cn2_only": self.n_cn2_only,
            "median_dist_to_interface": float(np.median(d)) if d.size else np.nan,
            "mean_dist_to_interface": float(np.mean(d)) if d.size else np.nan,
            "frac_within_1_diameter": float(np.mean(d <= 1.0)) if d.size else np.nan,
            "n_regions_usable": int(self.per_region["summary"]["usable"].sum()),
            "n_enriched_both": int(self.enrichment["enriched_both"].sum()),
        }
        row.update(extra)
        return row


def analyze_pair(
    obs: pd.DataFrame,
    probs: pd.DataFrame,
    location: pd.DataFrame,
    *,
    cn1: str,
    cn2: str,
    threshold: float,
    region_key: str,
    cell_type_col: str,
    inclusive: bool = False,
    require_exactly_two: bool = False,
    min_cells: int = 5,
    min_border_cells: int = 20,
    cell_type_order: Sequence[str] | None = None,
    adata=None,
) -> PairBorder:
    """Run the full pair-border analysis for one posterior at one threshold.

    ``location`` is the frame from :func:`interface_coordinate`, computed once per
    dataset/level and shared across models and thresholds.

    When ``adata`` is supplied, the reported *proportions* come from the shipped
    :func:`mingl.tl.compute_grouped_proportions` (via :func:`composition_via_mingl`)
    rather than being recomputed here, so the composition panels are MINGL's own
    numbers. The per-cell-type *counts* still come from the masks: the shipped
    function returns proportions only, and the enrichment ``min_cells`` filter and
    the per-region statistics need counts. Both use the identical group definition
    (``p > threshold``, border = positive for both), so the two agree exactly --
    pinned by ``test_mingl_composition_matches_mask_composition``.
    """
    masks = pair_masks(
        probs, cn1, cn2, threshold=threshold, inclusive=inclusive,
        require_exactly_two=require_exactly_two,
    )
    ct = obs[cell_type_col].astype(str).to_numpy()
    order = list(cell_type_order) if cell_type_order is not None else sorted(pd.unique(ct).tolist())
    composition = pair_composition(ct, masks, cell_type_order=order)

    if adata is not None:
        attach_posterior(adata, probs)
        mingl_comp = composition_via_mingl(
            adata, cn1=cn1, cn2=cn2, threshold=threshold, cell_type_col=cell_type_col
        )
        lookup = mingl_comp.set_index(["group", "cell_type"])["proportion"]
        keys = pd.MultiIndex.from_arrays([composition["group"], composition["cell_type"]])
        composition["proportion"] = lookup.reindex(keys).to_numpy()
        composition["proportion"] = composition["proportion"].fillna(0.0)
    enrichment = pair_enrichment(composition, min_cells=min_cells)
    per_region = per_region_border_stats(
        obs, masks, location, region_key=region_key, cell_type_col=cell_type_col,
        min_border_cells=min_border_cells, cell_type_order=order,
    )
    return PairBorder(
        cn1=cn1,
        cn2=cn2,
        threshold=float(threshold),
        n_cells=int(probs.shape[0]),
        n_border=int(masks["border"].sum()),
        n_cn1_only=int(masks["cn1_only"].sum()),
        n_cn2_only=int(masks["cn2_only"].sum()),
        masks=masks,
        composition=composition,
        enrichment=enrichment,
        per_region=per_region,
        location_of_border=location["dist_to_interface"].to_numpy()[masks["border"]],
    )
