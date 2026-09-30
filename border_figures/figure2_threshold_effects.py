"""Figure 2 -- how the probability threshold changes what a border *is*.

A MINGL border cell is a cell with positive membership (``p > threshold``) in
more than one organizational unit, so the threshold is a free parameter of every
border result. This figure sweeps it and asks what actually moves, across five
conditions that span a negative control, three hierarchical scales of one tissue,
and a second platform:

===========================  ==========================================
condition                    focus border
===========================  ==========================================
intestine, **null**          same units, spatially randomized identities
intestine, neighborhood      k=10  window of cell-type labels
intestine, community         k=100 window of neighborhood labels
intestine, tissue unit       k=300 window of community labels
spatial (Barrett's)          k=10  window of cell-type labels
===========================  ==========================================

Hierarchy levels use the manuscript's own k and feature labels (Methods 4.2), not
a single cell-type window re-used at every scale. One border pair is followed per
condition.

Panels
------
a  **Number of border cells** vs threshold (count, fraction, and the Jaccard
   overlap of the border-cell set with the threshold-0.25 reference set).
b  **Border location** vs threshold: median distance from border cells to the
   CN1|CN2 interface, the fraction sitting within one cell diameter of it, and
   the paired region-level gap between each real condition and the null.
c  **Border composition** vs threshold: stacked cell-type composition per
   condition, plus its Jensen-Shannon divergence from the 0.25 composition.
d  **Border cell-type enrichment** vs threshold: the manuscript's pair
   enrichment (Methods 4.5) -- how many cell types stay enriched against *both*
   single-unit groups, and the per-cell-type enrichment heat map.

The null condition is the control that makes the rest readable: it shows what
these four read-outs look like when the threshold is applied to a tissue with no
real compositional structure.

Statistics
----------
Paired across tissue regions, since the same regions are re-measured at every
threshold: Friedman omnibus (Kendall's W) then Wilcoxon signed-rank against the
0.25 reference, Holm-corrected across thresholds. Enrichment values are tested
per cell type against 0 (Wilcoxon over regions, Holm across cell types), which is
what the stars in the heat map mark. Real-vs-null comparisons are paired on the
shared regions.

Usage
-----
    python -m border_figures.figure2_threshold_effects --synthetic

    python -m border_figures.figure2_threshold_effects \\
        --data intestine=/path/05_25_HuBMAP_tunit.csv \\
        --data spatial=/path/all_regions_from_h5mu.csv \\
        --pair intestine_tissue_unit="Mucosa|Muscularis Mucosa"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from . import pair_borders as pb
from . import stats as st
from .config import (
    DEFAULT_THRESHOLDS,
    REFERENCE_THRESHOLD,
    DatasetSpec,
    LevelSpec,
    default_pair,
    get_level,
    resolve_dataset,
)
from .loading import (
    NULL_MODES,
    compute_posterior,
    load_dataset,
    make_null,
    slim_obs,
    subsample_regions,
    synthetic_tissue,
)
from matplotlib.lines import Line2D

from .plotting import (
    BORDER_MAP_COLORS,
    cell_type_palette,
    collapse_to_top,
    compact_legend,
    legend_from,
    panel_label,
    save_figure,
    save_panels,
    set_style,
    stacked_composition_bars,
    top_cell_types,
)

def _short_label(spec) -> str:
    """Human-readable dataset name for condition labels.

    ``spec.name`` is the CLI key ("spatial"), which is meaningless on a figure --
    a reader cannot tell that "Spatial · cellular neighborhood" means mouse brain.
    """
    return {
        "intestine": "Intestine",
        "melanoma": "Melanoma",
        "spatial": "Mouse brain",
        "esophagus": "Esophagus",
        "synthetic": "Synthetic",
    }.get(spec.name, spec.name.capitalize())


CONDITION_COLORS = [
    "#999999",  # null -- always grey
    "#0173b2",
    "#029e73",
    "#d55e00",
    "#cc78bc",
    "#ca9161",
]


@dataclass
class Condition:
    """One dataset x hierarchy level (or its null), with its focus border."""

    key: str
    label: str
    dataset: str
    level_name: str
    is_null: bool
    spec: DatasetSpec
    level: LevelSpec
    adata: object
    color: str = "#333333"
    pair: tuple[str, str] = ("", "")
    location: pd.DataFrame | None = None
    borders: dict[float, pb.PairBorder] = field(default_factory=dict)
    cell_type_order: list[str] = field(default_factory=list)
    probs: pd.DataFrame | None = None


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------
def run_condition(
    cond: Condition,
    *,
    model: str,
    thresholds: tuple[float, ...],
    pair_override: tuple[str, str] | None,
    cache_dir: Path | None,
    cache_tag: str,
    min_border_cells: int,
) -> Condition:
    """Score one condition and analyze its focus border at every threshold."""
    print(f"\n== {cond.label} ==")
    probs = compute_posterior(
        cond.adata, model=model, spec=cond.spec, level=cond.level,
        cache_dir=cache_dir, cache_tag=cache_tag + ("_null" if cond.is_null else ""),
    )
    cond.probs = probs

    pair = pair_override or default_pair(cond.spec.name, cond.level.name)
    if pair is not None and not set(pair).issubset(set(probs.columns)):
        print(f"  ! border {pair} not present; auto-selecting")
        pair = None
    if pair is None:
        try:
            pair = pb.select_top_pair(probs, threshold=REFERENCE_THRESHOLD)
        except ValueError:
            # A null condition can legitimately have no co-positive cells at 0.25.
            pair = pb.select_top_pair(probs, threshold=min(thresholds))
        print(f"  auto-selected focus border: {pair[0]} | {pair[1]}")
    cond.pair = tuple(pair)

    print(f"  interface coordinate for {cond.pair[0]} | {cond.pair[1]} ...")
    cond.location = pb.interface_coordinate(
        cond.adata.obs, unit_col=cond.level.unit_col, cn1=cond.pair[0], cn2=cond.pair[1],
        region_key=cond.spec.region_key, x_key=cond.spec.x_key, y_key=cond.spec.y_key,
    )
    cond.cell_type_order = sorted(cond.adata.obs[cond.spec.cell_type_col].astype(str).unique().tolist())

    for t in thresholds:
        res = pb.analyze_pair(
            cond.adata.obs, probs, cond.location, cn1=cond.pair[0], cn2=cond.pair[1],
            threshold=t, region_key=cond.spec.region_key, cell_type_col=cond.spec.cell_type_col,
            min_border_cells=min_border_cells, cell_type_order=cond.cell_type_order,
            adata=cond.adata,
        )
        # Only the border mask is needed downstream (Jaccard); drop the rest so
        # five conditions x nine thresholds stay affordable on 2.5M cells.
        res.masks = {"border": res.masks["border"]}
        cond.borders[t] = res
        print(f"    t={t:<5g} n_border={res.n_border:>10,}  frac={res.frac_border:.4f}  "
              f"median dist={res.summary_row()['median_dist_to_interface']:.2f}")
    return cond


def general_sweep(cond: Condition, thresholds) -> dict:
    """Pair-free border sweep via the shipped mingl.tl.threshold_sensitivity_analysis.

    Panels a and c compare *conditions*, so they must use one border definition
    throughout: a cell with >=2 positive memberships across all units. Using a
    different focus pair per condition (as panel b necessarily does) would make
    those panels incomparable -- five unrelated interfaces side by side.
    """
    import mingl.tl as mtl

    pb.attach_posterior(cond.adata, cond.probs)
    res = mtl.threshold_sensitivity_analysis(
        cond.adata, thresholds=list(thresholds),
        cell_type_col=cond.spec.cell_type_col, region_key=cond.spec.region_key,
        coord_keys=(cond.spec.x_key, cond.spec.y_key),
    )
    comp = (res["composition"]
            .pivot(index="threshold", columns="cell_type", values="prop_of_border")
            .reindex(columns=cond.cell_type_order).fillna(0.0))
    summary = res["summary"].assign(
        condition=cond.key, label=cond.label, dataset=cond.dataset,
        level=cond.level_name, is_null=cond.is_null,
    )
    return {"summary": summary, "composition": comp}


def condition_summary(cond: Condition) -> pd.DataFrame:
    """One row per threshold: counts, location, enrichment size, set stability."""
    ref_mask = cond.borders.get(REFERENCE_THRESHOLD)
    rows = []
    for t, res in sorted(cond.borders.items()):
        row = res.summary_row(condition=cond.key, label=cond.label, dataset=cond.dataset,
                              level=cond.level_name, is_null=cond.is_null)
        if ref_mask is not None:
            a = res.masks["border"]
            b = ref_mask.masks["border"]
            union = int((a | b).sum())
            row["jaccard_vs_reference"] = float((a & b).sum() / union) if union else np.nan
        else:
            row["jaccard_vs_reference"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def location_matrix(cond: Condition) -> pd.DataFrame:
    """(regions x thresholds) median distance from border cells to the interface."""
    cols = {
        t: res.per_region["summary"].set_index("region")["median_dist_to_interface"]
        for t, res in sorted(cond.borders.items())
    }
    return pd.DataFrame(cols)


def composition_over_thresholds(cond: Condition) -> pd.DataFrame:
    """(thresholds x cell types) border composition."""
    return pd.DataFrame(
        {
            t: res.composition.query("group == 'border'").set_index("cell_type")["proportion"]
            for t, res in sorted(cond.borders.items())
        }
    ).T.reindex(columns=cond.cell_type_order).fillna(0.0)


def enrichment_over_thresholds(cond: Condition) -> pd.DataFrame:
    """(thresholds x cell types) pair enrichment ``min(E_CN1, E_CN2)``."""
    return pd.DataFrame(
        {t: res.enrichment.set_index("cell_type")["min_enrichment"] for t, res in sorted(cond.borders.items())}
    ).T.reindex(columns=cond.cell_type_order)


def condition_statistics(
    cond: Condition, *, n_boot: int, seed: int
) -> tuple[list[st.TestResult], pd.DataFrame, pd.DataFrame]:
    """All within-condition threshold statistics.

    Returns the test list, a per-threshold table (JSD from the reference
    composition + Spearman of the enrichment vector), and a (thresholds x cell
    types) table of Holm-corrected p-values for "enrichment != 0".
    """
    tests: list[st.TestResult] = []
    thresholds = sorted(cond.borders)

    # --- location: paired across regions ---
    # High thresholds can leave no usable border in any region; those columns are
    # entirely NaN and would empty the whole table if rows were dropped first.
    loc_raw = location_matrix(cond)
    dropped = [c for c in loc_raw.columns if loc_raw[c].isna().all()]
    loc = loc_raw.drop(columns=dropped).dropna(how="any")
    note = f"thresholds with no usable border excluded: {[f'{t:g}' for t in dropped]}" if dropped else ""
    ref = REFERENCE_THRESHOLD if REFERENCE_THRESHOLD in loc.columns else (loc.columns[-1] if len(loc.columns) else None)
    if loc.shape[0] >= 3 and loc.shape[1] >= 2 and ref is not None:
        renamed = loc.rename(columns={c: f"t={c:g}" for c in loc.columns})
        loc_tests = st.paired_condition_tests(
            renamed, reference=f"t={ref:g}", label=f"{cond.label} border location"
        )
        for t in loc_tests:
            t.note = (t.note + " " + note).strip()
        tests.extend(loc_tests)
    else:
        tests.append(
            st.TestResult(
                comparison=f"{cond.label} border location: across thresholds", test="friedman",
                unit="region", n=int(loc.shape[0]), statistic=np.nan, p_value=np.nan,
                note=("not enough regions with usable borders at every threshold " + note).strip(),
            )
        )

    # --- composition + enrichment vs the reference threshold ---
    comp = composition_over_thresholds(cond)
    enr = enrichment_over_thresholds(cond)
    ref_t = REFERENCE_THRESHOLD if REFERENCE_THRESHOLD in comp.index else comp.index[-1]
    region_comp = {
        t: res.per_region["composition"].reindex(columns=cond.cell_type_order).fillna(0.0)
        for t, res in cond.borders.items()
    }
    per_threshold_rows = []
    for t in thresholds:
        jsd_point = st.jensen_shannon_divergence(comp.loc[t], comp.loc[ref_t])
        shared = sorted(set(region_comp[t].index) & set(region_comp[ref_t].index))
        if len(shared) >= 3:
            def _stat(sampled, _t=t):
                rows = list(sampled)
                return st.jensen_shannon_divergence(
                    region_comp[_t].loc[rows].mean(axis=0).to_numpy(),
                    region_comp[ref_t].loc[rows].mean(axis=0).to_numpy(),
                )

            # Use the bootstrap's point estimate so the bar and its CI describe
            # the same (region-averaged) quantity.
            jsd_point, lo, hi = st.bootstrap_ci(_stat, shared, n_boot=n_boot, seed=seed)
        else:
            lo = hi = np.nan

        a, b = enr.loc[t].to_numpy(dtype=float), enr.loc[ref_t].to_numpy(dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        rho = float(sps.spearmanr(a[ok], b[ok]).statistic) if ok.sum() > 2 else np.nan
        set_t = set(cond.borders[t].enrichment.query("enriched_both")["cell_type"])
        set_ref = set(cond.borders[ref_t].enrichment.query("enriched_both")["cell_type"])
        union = set_t | set_ref
        per_threshold_rows.append(
            {
                "condition": cond.key, "label": cond.label, "threshold": t,
                "jsd_vs_reference": jsd_point, "jsd_ci_lo": lo, "jsd_ci_hi": hi,
                "spearman_enrichment_vs_reference": rho,
                "n_enriched_both": len(set_t),
                "jaccard_enriched_both_vs_reference": len(set_t & set_ref) / len(union) if union else np.nan,
                "n_regions_bootstrap": len(shared),
            }
        )

    # --- per-cell-type enrichment != 0, per threshold ---
    pvals = pd.DataFrame(index=thresholds, columns=cond.cell_type_order, dtype=float)
    for t in thresholds:
        region_enr = cond.borders[t].per_region["enrichment"]
        if region_enr.shape[0] < 3:
            continue
        per_type = []
        for ct in cond.cell_type_order:
            values = region_enr[ct].dropna() if ct in region_enr.columns else pd.Series(dtype=float)
            per_type.append(
                st.wilcoxon_vs_reference(
                    values, np.zeros(values.size),
                    comparison=f"{cond.label} enrichment != 0 | {ct} @ t={t:g}",
                )
            )
        adjusted = st.holm([r.p_value for r in per_type])
        for res, adj in zip(per_type, adjusted):
            res.p_adjusted = None if not np.isfinite(adj) else float(adj)
            res.note = (res.note + " Holm-corrected across cell types").strip()
        pvals.loc[t] = adjusted
        tests.extend(per_type)

    return tests, pd.DataFrame(per_threshold_rows), pvals


def null_comparison(
    conditions: list[Condition], *, thresholds: tuple[float, ...]
) -> list[st.TestResult]:
    """Paired region-level comparison of each real condition against its null."""
    nulls = {c.dataset: c for c in conditions if c.is_null}
    tests: list[st.TestResult] = []
    for cond in conditions:
        if cond.is_null or cond.dataset not in nulls:
            continue
        null = nulls[cond.dataset]
        real_loc, null_loc = location_matrix(cond), location_matrix(null)
        per_t = []
        for t in thresholds:
            if t not in real_loc.columns or t not in null_loc.columns:
                continue
            shared = real_loc.index.intersection(null_loc.index)
            per_t.append(
                st.wilcoxon_vs_reference(
                    real_loc.loc[shared, t], null_loc.loc[shared, t],
                    comparison=f"{cond.label} vs null: border location @ t={t:g}",
                )
            )
        for res, adj in zip(per_t, st.holm([r.p_value for r in per_t])):
            res.p_adjusted = None if not np.isfinite(adj) else float(adj)
            res.note = (res.note + " Holm-corrected across thresholds").strip()
        tests.extend(per_t)
    return tests


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def panel_a_counts(ax_n, ax_legend, summaries: dict[str, pd.DataFrame], conditions) -> None:
    """Grouped bar chart of border-cell counts per threshold, with its own legend."""
    # The null is excluded from the bar chart: its counts collapse to a handful
    # by t=0.25, so on a log axis it compresses the real conditions without
    # adding information. It remains in every other panel and in the CSVs.
    bar_conditions = [c for c in conditions if not c.is_null]
    if not bar_conditions:
        bar_conditions = list(conditions)

    thresholds = sorted(summaries[bar_conditions[0].key]["threshold"].tolist())
    n_cond = len(bar_conditions)
    width = 0.8 / max(n_cond, 1)
    x = np.arange(len(thresholds), dtype=float)

    for i, cond in enumerate(bar_conditions):
        s = summaries[cond.key].set_index("threshold").reindex(thresholds)
        ax_n.bar(
            x + (i - (n_cond - 1) / 2) * width,
            np.maximum(s["n_border"].to_numpy(), 0.0), width=width,
            color=cond.color, label=cond.label,
            edgecolor="black", linewidth=0.3,
        )

    ax_n.set_yscale("log")
    ax_n.set_xticks(x)
    ax_n.set_xticklabels([f"{t:g}" for t in thresholds], rotation=90)
    ax_n.set_ylabel("border cells (n)")
    ax_n.set_xlabel("probability threshold")
    ax_n.set_title("Number of border cells", loc="left")
    legend_from(ax_n, ax_legend, ncol=1, title="Condition", fontsize=6.5)



def _save_mingl_enrichment(figs, out_dir, stem, *, cn1, cn2, subtitle, threshold):
    """Label and save all three figures mingl.pl.plot_border_enrichment returns.

    The shipped function draws the quadrant scatter with no title and no axis
    labels, and returns two further legend figures (dot size = border-cell count,
    colour = cell type) that are meaningless if discarded. Nothing about the plot
    itself is changed here -- only annotation is added and all three are kept.
    """
    from .plotting import save_figure

    scatter = figs[0]
    ax = scatter.axes[0]
    ax.set_xlabel(f"log2( border / {cn1}-only )")
    ax.set_ylabel(f"log2( border / {cn2}-only )")
    ax.set_title(
        f"Border cell-type enrichment: {cn1} | {cn2}\n{subtitle}  ·  threshold {threshold:g}"
        "\nupper right = enriched against both units",
        fontsize=10, loc="left",
    )
    written = save_figure(scatter, out_dir, stem, pdf=False)
    for suffix, extra in zip(("_legend_counts", "_legend_colors"), figs[1:]):
        written += save_figure(extra, out_dir, stem + suffix, pdf=False)
    return written


def catplot_thresholds(thresholds: list[float], *, max_maps: int = 5) -> list[float]:
    """Thresholds to draw spatial maps for: the reference plus an even spread.

    The manuscript shows three (0.05 / 0.15 / 0.25); more than five maps in a row
    stops being legible, so the sweep is thinned around the 0.25 reference.
    """
    thresholds = sorted(thresholds)
    if len(thresholds) <= max_maps:
        return thresholds
    picks = {thresholds[0], thresholds[-1]}
    if REFERENCE_THRESHOLD in thresholds:
        picks.add(REFERENCE_THRESHOLD)
    for idx in np.linspace(0, len(thresholds) - 1, max_maps).astype(int):
        if len(picks) >= max_maps:
            break
        picks.add(thresholds[int(idx)])
    return sorted(picks)[:max_maps]


def pick_condition_region(cond: Condition) -> str:
    """Region with the most border cells at the reference threshold."""
    regions = cond.adata.obs[cond.spec.region_key].astype(str).to_numpy()
    ref = cond.borders.get(REFERENCE_THRESHOLD) or next(iter(cond.borders.values()))
    counts = pd.Series(regions[np.asarray(ref.masks["border"], dtype=bool)]).value_counts()
    if counts.empty:
        return str(pd.Series(regions).value_counts().index[0])
    return str(counts.index[0])



def save_standalone_catplots(out_dir, *, adata, spec, posteriors_or_probs, pair, region,
                             keys, threshold_for, label_for, size_in=14.0, dpi=400):
    """Render each border catplot as its own large figure, not a crop of the grid.

    Panels cropped out of the composite inherit the grid's small size, so
    individual cells are unreadable. Here every map gets a dedicated
    ``size_in`` x ``size_in`` figure at ``dpi``, which is what makes single dots
    resolvable in a tissue with hundreds of thousands of cells.
    """
    import mingl.pl as _pl
    from .plotting import save_figure

    written = []
    sub = Path(out_dir) / "catplots"
    for key in keys:
        probs = posteriors_or_probs[key] if isinstance(posteriors_or_probs, dict) else posteriors_or_probs
        pb.attach_posterior(adata, probs)
        fig, ax = plt.subplots(figsize=(size_in, size_in), dpi=dpi)
        _pl.spatial_loc_region(
            adata, region=str(region), n1=pair[0], n2=pair[1],
            threshold=float(threshold_for(key)),
            region_key=spec.region_key, x_col=spec.x_key, y_col=spec.y_key,
            s_other=0.30, s_single=0.45, s_both=1.6, ax=ax, show=False,
        )
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        ax.set_aspect("equal", "box")
        ax.set_axis_off()
        ax.set_title(label_for(key), fontsize=13)
        written += save_figure(fig, sub, f"catplot_{key}", pdf=False)
    return written


def panel_location_catplot(axes, ax_legend, cond: Condition, thresholds, region: str) -> None:
    """Manuscript Figure 2g maps of the focus border as the threshold moves.

    Drawn by the shipped :func:`mingl.pl.spatial_loc_region` -- the posterior is
    attached to the canonical MINGL keys once and the function is called per
    threshold into our axes.
    """
    import mingl.pl as pl

    pb.attach_posterior(cond.adata, cond.probs)
    for ax, t in zip(axes, thresholds):
        pl.spatial_loc_region(
            cond.adata, region=region, n1=cond.pair[0], n2=cond.pair[1], threshold=float(t),
            region_key=cond.spec.region_key, x_col=cond.spec.x_key, y_col=cond.spec.y_key,
            s_other=0.04, s_single=0.07, s_both=0.30, ax=ax, show=False,
        )
        existing = ax.get_legend()
        if existing is not None:
            existing.remove()
        ax.set_aspect("equal", "box")
        ax.set_axis_off()
        ax.set_title(
            f"t = {t:g}\n{cond.borders[t].n_border:,} border cells", fontsize=7.5
        )

    ax_legend.axis("off")
    handles = [
        Line2D([], [], marker="o", ls="", ms=5, color=color, label=label)
        for label, color in (
            (cond.pair[0], BORDER_MAP_COLORS["only_1"]),
            (cond.pair[1], BORDER_MAP_COLORS["only_2"]),
            ("Border cells", BORDER_MAP_COLORS["both"]),
            ("Other cells", BORDER_MAP_COLORS["other"]),
        )
    ]
    ax_legend.legend(handles=handles, loc="center left", fontsize=6.5,
                     title=f"{cond.label}\nregion {region}")


def panel_b_location(ax_med, ax_near, ax_null, conditions, summaries, loc_matrices, null_tests) -> None:
    for cond in conditions:
        s = summaries[cond.key]
        ls = "--" if cond.is_null else "-"
        ax_med.plot(s["threshold"], s["median_dist_to_interface"], color=cond.color,
                    marker="o", ms=3.2, lw=1.4, ls=ls, label=cond.label)
        m = loc_matrices[cond.key]
        if not m.empty:
            lo = m.quantile(0.25)
            hi = m.quantile(0.75)
            ax_med.fill_between(list(m.columns), lo, hi, color=cond.color, alpha=0.13, linewidth=0)
        ax_near.plot(s["threshold"], s["frac_within_1_diameter"], color=cond.color,
                     marker="o", ms=3.2, lw=1.4, ls=ls, label=cond.label)

    ax_med.set_ylabel("median distance to interface\n(cell diameters)")
    ax_med.set_xlabel("probability threshold")
    ax_med.set_title("Border location (band: region IQR)", loc="left")
    # Log scale: a null condition scatters borders tens of diameters away, which
    # on a linear axis would flatten every real condition onto zero.
    ax_med.set_yscale("log", nonpositive="mask")
    ax_med.axhline(1.0, color="#888888", lw=0.6, ls=":")

    ax_near.set_ylabel("fraction within 1 cell diameter")
    ax_near.set_xlabel("probability threshold")
    ax_near.set_title("Borders sitting on the interface", loc="left")

    real = [c for c in conditions if not c.is_null]
    lookup = {t.comparison: t for t in null_tests}
    x, heights, colors, stars = [], [], [], []
    for i, cond in enumerate(real):
        key = f"{cond.label} vs null: border location @ t={REFERENCE_THRESHOLD:g}"
        res = lookup.get(key)
        m_real = loc_matrices[cond.key]
        null_key = next((c.key for c in conditions if c.is_null and c.dataset == cond.dataset), None)
        if res is None or null_key is None or REFERENCE_THRESHOLD not in m_real.columns:
            continue
        m_null = loc_matrices[null_key]
        shared = m_real.index.intersection(m_null.index)
        if not len(shared):
            continue
        delta = (m_real.loc[shared, REFERENCE_THRESHOLD] - m_null.loc[shared, REFERENCE_THRESHOLD]).median()
        x.append(i)
        heights.append(delta)
        colors.append(cond.color)
        stars.append(res.stars)
    if x:
        ax_null.bar(x, heights, color=colors, width=0.65)
        for xi, h, s in zip(x, heights, stars):
            ax_null.text(xi, h, s, ha="center",
                         va="bottom" if h >= 0 else "top", fontsize=7)
        ax_null.set_xticks(range(len(real)))
        ax_null.set_xticklabels([c.label.replace(" · ", "\n") for c in real], fontsize=6)
    ax_null.axhline(0, color="#888888", lw=0.8)
    ax_null.set_ylabel(f"Δ median distance\nvs null (t={REFERENCE_THRESHOLD:g})")
    ax_null.set_title("Real vs null, paired by region", loc="left")


def panel_c_composition(axes, ax_legend, ax_jsd, conditions, comps, per_threshold) -> None:
    """Stacked composition per condition, with one legend *per dataset*.

    Datasets do not share a cell-type vocabulary -- intestine annotates 25 types
    and esophagus 45, overlapping in only 13 names (``CD8+ T`` vs ``CD8+ T cell``
    and so on). Pooling them into a single "top N" selection would both produce a
    legend full of near-duplicates and push each dataset's own abundant types
    into "Other" because the *other* dataset's types outranked them. So the
    selection, the palette and the legend are all per dataset; conditions from
    one dataset still share a key, which is correct since their vocabulary is
    identical across hierarchy levels.
    """
    by_dataset: dict[str, list] = {}
    for cond in conditions:
        by_dataset.setdefault(cond.dataset, []).append(cond)

    orders, palettes = {}, {}
    for ds, conds in by_dataset.items():
        pooled = pd.concat([comps[c.key] for c in conds], axis=0)
        orders[ds] = top_cell_types(pooled)
        palettes[ds] = cell_type_palette(orders[ds])

    for ax, cond in zip(axes, conditions):
        collapsed = collapse_to_top(comps[cond.key], orders[cond.dataset])
        stacked_composition_bars(
            ax, collapsed, palette=palettes[cond.dataset],
            x_labels=[f"{t:g}" for t in collapsed.index],
        )
        ax.set_xlabel("threshold")
        ax.set_title(f"{cond.label}\n{cond.pair[0]} | {cond.pair[1]}",
                     loc="left", fontsize=7)
        ax.tick_params(axis="x", labelrotation=90, labelsize=6)
        if ax is not axes[0]:
            ax.set_ylabel("")

    # One legend per dataset, each in its own axes. Stacking both into a single
    # axes needs the second anchored below the first's rendered height, which is
    # not known until draw time -- guessing a fraction makes them overlap.
    legend_axes = ax_legend if isinstance(ax_legend, (list, tuple)) else [ax_legend]
    for ax, (ds, order) in zip(legend_axes, orders.items()):
        ax.axis("off")
        handles = [
            Line2D([], [], marker="s", ls="", ms=5, color=palettes[ds].get(ct, "#999999"), label=ct)
            for ct in order
        ]
        # Wrap into columns so a long vocabulary never grows taller than its
        # band: a single 25-entry column overruns any share of the legend
        # column it is given, which is what caused the two keys to collide.
        ncol = max(1, -(-len(order) // 13))
        ax.legend(
            handles=handles, loc="upper left", fontsize=5.5, ncol=ncol,
            title=f"{_short_label(resolve_dataset(ds))} cell type",
            title_fontsize=6.5, handlelength=1.0, borderaxespad=0.0,
            columnspacing=0.8, labelspacing=0.3,
        )
    for ax in legend_axes[len(orders):]:
        ax.axis("off")

    if ax_jsd is None:
        return
    for cond in conditions:
        s = per_threshold[per_threshold["condition"] == cond.key]
        ax_jsd.plot(s["threshold"], s["jsd_vs_reference"], color=cond.color, marker="o",
                    ms=3.2, lw=1.4, ls="--" if cond.is_null else "-", label=cond.label)
        if s["jsd_ci_lo"].notna().any():
            ax_jsd.fill_between(s["threshold"], s["jsd_ci_lo"], s["jsd_ci_hi"],
                                color=cond.color, alpha=0.13, linewidth=0)
    ax_jsd.set_xlabel("probability threshold")
    ax_jsd.set_ylabel(
        f"JS divergence from\nt={REFERENCE_THRESHOLD:g} composition (bits)\n"
        "band: region bootstrap 95% CI"
    )
    ax_jsd.set_title("Composition shift", loc="left")
    ax_jsd.axvline(REFERENCE_THRESHOLD, color="#888888", lw=0.7, ls=":")


def enrichment_heatmap(ax, cond: Condition, enr: pd.DataFrame, pvals: pd.DataFrame, *, top_n: int = 12) -> None:
    ref_t = REFERENCE_THRESHOLD if REFERENCE_THRESHOLD in enr.index else enr.index[-1]
    ranked = enr.loc[ref_t].dropna().sort_values(ascending=False)
    keep = list(ranked.index[:top_n])
    if not keep:
        keep = list(enr.columns[:top_n])
    block = enr.loc[:, keep].T  # cell types x thresholds

    finite = block.to_numpy()[np.isfinite(block.to_numpy())]
    vmax = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
    vmax = max(vmax, 0.25)
    im = ax.imshow(block.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(block.shape[1]))
    ax.set_xticklabels([f"{t:g}" for t in block.columns], rotation=90, fontsize=6)
    ax.set_yticks(range(block.shape[0]))
    ax.set_yticklabels(block.index, fontsize=6)
    ax.set_xlabel("probability threshold")
    ax.set_title(f"{cond.label}: {cond.pair[0]} | {cond.pair[1]}", loc="left", fontsize=7.5)
    plt.colorbar(
        im, ax=ax, fraction=0.04, pad=0.02,
        label=f"log2 min(vs {cond.pair[0]}, vs {cond.pair[1]})",
    )

    for iy, ct in enumerate(block.index):
        for ix, t in enumerate(block.columns):
            p = pvals.loc[t, ct] if (t in pvals.index and ct in pvals.columns) else np.nan
            if np.isfinite(p) and p < 0.05:
                ax.text(ix, iy, "*", ha="center", va="center", fontsize=6.5, color="black")


def panel_d_enrichment(ax_count, ax_heat, conditions, per_threshold, enrichments, pvalues, highlight) -> None:
    for cond in conditions:
        s = per_threshold[per_threshold["condition"] == cond.key]
        ax_count.plot(s["threshold"], s["n_enriched_both"], color=cond.color, marker="o",
                      ms=3.2, lw=1.4, ls="--" if cond.is_null else "-", label=cond.label)
    ax_count.set_xlabel("probability threshold")
    ax_count.set_ylabel("cell types enriched at the border\n(positive vs both units)")
    ax_count.set_title("Enrichment call stability", loc="left")
    ax_count.axvline(REFERENCE_THRESHOLD, color="#888888", lw=0.7, ls=":")

    enrichment_heatmap(ax_heat, highlight, enrichments[highlight.key], pvalues[highlight.key])


def panel_d_enrichment_grid(axes, conditions, enrichments, pvalues, *, top_n=10) -> None:
    """Pair enrichment across thresholds, one heat map per condition.

    Same fixed border per condition as the composition panel, so a reader can
    follow one interface from composition to enrichment.
    """
    for ax, cond in zip(axes, conditions):
        enrichment_heatmap(ax, cond, enrichments[cond.key], pvalues[cond.key], top_n=top_n)


def panel_e_stability(ax, conditions, per_threshold) -> None:
    """Numbers behind "the threshold does not change the biology".

    Three stability measures against the t=0.25 reference, per condition:
    composition JS divergence (lower = more stable), Spearman correlation of the
    per-cell-type enrichment vector, and Jaccard of the enriched-at-both set.
    Without these the claim rests on the panels looking flat.
    """
    for cond in conditions:
        s = per_threshold[per_threshold["condition"] == cond.key].sort_values("threshold")
        ax.plot(s["threshold"], s["spearman_enrichment_vs_reference"], color=cond.color,
                marker="o", ms=3.2, lw=1.4, ls="--" if cond.is_null else "-", label=cond.label)
    ax.axhline(1.0, color="#888888", lw=0.6, ls=":")
    ax.axvline(REFERENCE_THRESHOLD, color="#888888", lw=0.7, ls=":")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("probability threshold")
    ax.set_ylabel(f"Spearman of enrichment\nvs t={REFERENCE_THRESHOLD:g}")
    ax.set_title("Enrichment stability (1.0 = unchanged)", loc="left")


def build_figure(conditions, summaries, loc_matrices, comps, enrichments, pvalues,
                 per_threshold, null_tests, highlight, general_summaries, general_comps):
    """Three rows: border-cell counts, border location catplot, border composition."""
    n_cond = len(conditions)
    n_datasets = len({c.dataset for c in conditions})
    map_thresholds = catplot_thresholds(sorted(highlight.borders))
    n_maps = len(map_thresholds)
    map_region = pick_condition_region(highlight)

    fig = plt.figure(figsize=(17.0, 21.0))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 0.95, 1.2, 1.2], hspace=0.5)

    # a -- number of border cells: ALL border cells, no focus pair, so the
    # conditions are comparable.
    row_a = gs[0].subgridspec(1, 2, width_ratios=[2.4, 1.0], wspace=0.05)
    ax_a1 = fig.add_subplot(row_a[0])
    ax_a_leg = fig.add_subplot(row_a[1])
    panel_a_counts(ax_a1, ax_a_leg, general_summaries, conditions)
    panel_label(ax_a1, "a")

    # b -- border location, focus pair (a spatial pair-map needs one).
    row_b = gs[1].subgridspec(1, n_maps + 1, width_ratios=[1.0] * n_maps + [0.6], wspace=0.12)
    ax_b = [fig.add_subplot(row_b[i]) for i in range(n_maps)]
    ax_b_leg = fig.add_subplot(row_b[n_maps])
    panel_location_catplot(ax_b, ax_b_leg, highlight, map_thresholds, map_region)
    panel_label(ax_b[0], "b")

    # c -- border composition, focus pair, named in each subplot title.
    ds_order, ds_counts = [], []
    for c in conditions:
        if c.dataset not in ds_order:
            ds_order.append(c.dataset)
            ds_counts.append(max(int((comps[c.key] > 0).any().sum()), 1))
    n_ds_leg = len(ds_order)
    row_c = gs[2].subgridspec(1, n_cond + 1, width_ratios=[1.0] * n_cond + [0.85], wspace=0.4)
    ax_c = [fig.add_subplot(row_c[i]) for i in range(n_cond)]
    legend_grid = row_c[n_cond].subgridspec(n_ds_leg, 1, hspace=0.02, height_ratios=ds_counts)
    ax_c_leg = [fig.add_subplot(legend_grid[i]) for i in range(n_ds_leg)]
    panel_c_composition(ax_c, ax_c_leg, None, conditions, comps, per_threshold)
    panel_label(ax_c[0], "c")

    # d -- border cell-type enrichment across thresholds, same focus pair.
    row_d = gs[3].subgridspec(1, n_cond, wspace=0.55)
    ax_d = [fig.add_subplot(row_d[i]) for i in range(n_cond)]
    panel_d_enrichment_grid(ax_d, conditions, enrichments, pvalues)
    panel_label(ax_d[0], "d")

    fig.suptitle(
        "Effect of the MINGL probability threshold on border identity\n"
        "a: number of border cells (all borders) · b: border location · "
        "c: composition · d: cell-type enrichment\n"
        "c and d follow one named border per condition; "
        f"maps in b show {highlight.label}, region {map_region}",
        fontsize=11, y=0.997,
    )

    panels = {
        "a_border_count_bars": [ax_a1, ax_a_leg],
        "b_location_catplot": ax_b + [ax_b_leg],
        "c_border_composition": ax_c + list(ax_c_leg),
        "d_border_enrichment": ax_d,
    }
    for cond, ax in zip(conditions, ax_d):
        panels[f"d_border_enrichment_{cond.key}"] = ax
    for t, ax in zip(map_thresholds, ax_b):
        panels[f"b_location_catplot_t{t:g}"] = ax
    for cond, ax in zip(conditions, ax_c):
        panels[f"c_border_composition_{cond.key}"] = ax
    return fig, panels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_kv(items: list[str] | None, what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"{what} expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def build_conditions(args) -> list[Condition]:
    """Assemble the condition list from the supplied data paths."""
    conditions: list[Condition] = []
    paths = {resolve_dataset(k).name: v for k, v in _parse_kv(args.data, "--data").items()}
    level_choices = _parse_kv(args.levels, "--levels")

    if args.synthetic:
        paths = {}
        base = synthetic_tissue(
            n_regions=args.synthetic_regions,
            n_cells_per_region=args.synthetic_cells,
            seed=args.seed,
        )
        spec = resolve_dataset("synthetic")
        wanted = ["neighborhood", "community", "tissue_unit"]
        if not args.skip_null:
            null_level = get_level("synthetic", args.null_level)
            conditions.append(
                Condition(
                    key="synthetic_null", label="Synthetic · null", dataset="synthetic",
                    level_name=null_level.name, is_null=True, spec=spec, level=null_level,
                    adata=make_null(base, mode=args.null_mode, spec=spec, level=null_level, seed=args.seed),
                )
            )
        for lv in wanted:
            level = get_level("synthetic", lv)
            conditions.append(
                Condition(
                    key=f"synthetic_{lv}", label=f"Synthetic · {level.label.lower()}",
                    dataset="synthetic", level_name=lv, is_null=False, spec=spec,
                    level=level, adata=base,
                )
            )

    for ds_name, path in paths.items():
        spec = resolve_dataset(ds_name)
        wanted = [
            lv.strip() for lv in
            level_choices.get(ds_name, ",".join(spec.levels)).split(",") if lv.strip()
        ]
        wanted = [lv for lv in wanted if lv in spec.levels]
        if not wanted:
            raise SystemExit(f"No valid levels requested for {ds_name}; have {sorted(spec.levels)}")

        deepest = max((get_level(ds_name, lv) for lv in wanted), key=lambda lspec: lspec.k)
        print(f"Loading {ds_name} from {path} (levels: {wanted})")
        extra = tuple({get_level(ds_name, lv).unit_col for lv in wanted}
                      | {get_level(ds_name, lv).feature_col for lv in wanted})
        adata = load_dataset(path, ds_name, deepest, required_extra=extra)
        adata = slim_obs(
            adata, [spec.cell_type_col, spec.region_key, spec.x_key, spec.y_key, *extra]
        )
        adata = subsample_regions(
            adata, region_key=spec.region_key, frac=args.subsample_frac,
            max_cells=args.max_cells, min_per_region=deepest.k, seed=args.seed,
        )
        print(f"  {ds_name}: {adata.n_obs:,} cells, "
              f"{adata.obs[spec.region_key].astype(str).nunique()} regions")

        if not args.skip_null and ds_name == args.null_dataset:
            null_level = get_level(ds_name, args.null_level if args.null_level in spec.levels else wanted[0])
            if args.null_data:
                print(f"  loading user-supplied null from {args.null_data}")
                null_adata = load_dataset(args.null_data, ds_name, null_level)
                null_adata = slim_obs(
                    null_adata, [spec.cell_type_col, spec.region_key, spec.x_key, spec.y_key, *extra]
                )
                null_adata = subsample_regions(
                    null_adata, region_key=spec.region_key, frac=args.subsample_frac,
                    max_cells=args.max_cells, min_per_region=null_level.k, seed=args.seed,
                )
            else:
                null_adata = make_null(adata, mode=args.null_mode, spec=spec, level=null_level, seed=args.seed)
            conditions.append(
                Condition(
                    key=f"{ds_name}_null", label=f"{_short_label(spec)} · null",
                    dataset=ds_name, level_name=null_level.name, is_null=True,
                    spec=spec, level=null_level, adata=null_adata,
                )
            )

        for lv in wanted:
            level = get_level(ds_name, lv)
            conditions.append(
                Condition(
                    key=f"{ds_name}_{lv}",
                    label=f"{_short_label(spec)} · {level.label.lower()}",
                    dataset=ds_name, level_name=lv, is_null=False, spec=spec,
                    level=level, adata=adata,
                )
            )

    if not conditions:
        raise SystemExit("Provide --data DATASET=PATH (repeatable) or --synthetic.")

    # Nulls first (grey), then the rest in order.
    conditions.sort(key=lambda c: (not c.is_null, c.dataset, c.level.k))
    for i, cond in enumerate(conditions):
        cond.color = CONDITION_COLORS[0] if cond.is_null else CONDITION_COLORS[
            1 + (i - sum(1 for c in conditions if c.is_null)) % (len(CONDITION_COLORS) - 1)
        ]
    return conditions


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", action="append", metavar="DATASET=PATH",
                    help="Repeatable. e.g. --data intestine=/path/file.csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--synthetic-regions", type=int, default=12)
    ap.add_argument("--synthetic-cells", type=int, default=6000,
                    help="Cells per region in the synthetic tissue (lower it for a fast smoke test).")
    ap.add_argument("--levels", action="append", metavar="DATASET=a,b,c",
                    help="Levels per dataset (default: all defined for it).")
    ap.add_argument("--pair", action="append", metavar='CONDITION="CN1|CN2"',
                    help="Focus border per condition key, e.g. intestine_tissue_unit=\"Mucosa|Muscularis Mucosa\".")
    ap.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    ap.add_argument("--model", default="diagonal_gaussian",
                    help="Emission model; the default is the regular MINGL pipeline.")
    ap.add_argument("--null-dataset", default="intestine",
                    help="Dataset the null condition is derived from.")
    ap.add_argument("--null-level", default="neighborhood")
    ap.add_argument("--null-mode", default="celltype", choices=list(NULL_MODES))
    ap.add_argument("--null-data", default=None,
                    help="Use a null .csv/.h5ad you generated yourself instead of permuting.")
    ap.add_argument("--skip-null", action="store_true")
    ap.add_argument("--highlight", default=None,
                    help="Condition key whose enrichment heat map goes in panel d.")
    ap.add_argument("--min-border-cells", type=int, default=20)
    ap.add_argument("--subsample-frac", type=float, default=1.0)
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs" / "figure2_threshold_effects"))
    ap.add_argument("--cache-dir", default=str(Path(__file__).resolve().parent / "outputs" / "_cache"))
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = None if args.no_cache else Path(args.cache_dir)
    thresholds = tuple(sorted(float(t) for t in args.thresholds))
    set_style()

    conditions = build_conditions(args)
    overrides = {
        k: tuple(p.strip() for p in v.split("|")) for k, v in _parse_kv(args.pair, "--pair").items()
    }
    bad = {k: v for k, v in overrides.items() if len(v) != 2}
    if bad:
        raise SystemExit(f'--pair values must look like "CN1|CN2"; bad: {bad}')

    cache_tag = f"{args.model}_sub{args.subsample_frac}_max{args.max_cells}_seed{args.seed}"
    for cond in conditions:
        run_condition(
            cond, model=args.model, thresholds=thresholds,
            pair_override=overrides.get(cond.key), cache_dir=cache_dir,
            cache_tag=cache_tag + (f"_{args.null_mode}" if cond.is_null else ""),
            min_border_cells=args.min_border_cells,
        )

    # Panels a and c use the pair-free definition so conditions are comparable;
    # panel b keeps its focus pair, which a spatial pair-map requires.
    print("\n== Pair-free (general) sweep for panels a and c ==")
    general = {}
    for c in conditions:
        general[c.key] = general_sweep(c, thresholds)
        n25 = general[c.key]["summary"].query("threshold == @REFERENCE_THRESHOLD")["n_border"]
        print(f"  {c.label}: {int(n25.iloc[0]):,} border cells @0.25" if len(n25) else f"  {c.label}: n/a")

    summaries = {c.key: condition_summary(c) for c in conditions}
    general_summaries = {k: v["summary"] for k, v in general.items()}
    general_comps = {k: v["composition"] for k, v in general.items()}
    loc_matrices = {c.key: location_matrix(c) for c in conditions}
    comps = {c.key: composition_over_thresholds(c) for c in conditions}
    enrichments = {c.key: enrichment_over_thresholds(c) for c in conditions}

    all_tests: list[st.TestResult] = []
    per_threshold_frames, pvalues = [], {}
    for cond in conditions:
        tests, per_thr, pv = condition_statistics(cond, n_boot=args.n_boot, seed=args.seed)
        all_tests.extend(tests)
        per_threshold_frames.append(per_thr)
        pvalues[cond.key] = pv
    per_threshold = pd.concat(per_threshold_frames, ignore_index=True)

    null_tests = null_comparison(conditions, thresholds=thresholds)
    all_tests.extend(null_tests)

    # ---- tables ----
    pd.concat(general_summaries.values(), ignore_index=True).to_csv(
        out_dir / "threshold_summary_GENERAL.csv", index=False)
    for k, v in general_comps.items():
        v.to_csv(out_dir / f"composition_GENERAL_{k}.csv")
    summary_all = pd.concat(summaries.values(), ignore_index=True)
    summary_all.to_csv(out_dir / "threshold_summary.csv", index=False)
    per_threshold.to_csv(out_dir / "threshold_composition_enrichment_stability.csv", index=False)
    st.results_to_frame(all_tests).to_csv(out_dir / "statistics.csv", index=False)
    for cond in conditions:
        comps[cond.key].to_csv(out_dir / f"composition_{cond.key}.csv")
        enrichments[cond.key].to_csv(out_dir / f"enrichment_{cond.key}.csv")
        loc_matrices[cond.key].to_csv(out_dir / f"location_per_region_{cond.key}.csv")
    pd.DataFrame(
        [{"condition": c.key, "label": c.label, "dataset": c.dataset, "level": c.level_name,
          "unit_col": c.level.unit_col, "feature_col": c.level.feature_col, "k": c.level.k,
          "is_null": c.is_null, "cn1": c.pair[0], "cn2": c.pair[1], "n_cells": c.adata.n_obs}
         for c in conditions]
    ).to_csv(out_dir / "conditions.csv", index=False)

    print("\n== Border counts across thresholds ==")
    pivot = summary_all.pivot(index="threshold", columns="label", values="n_border")
    print(pivot.to_string())
    monotone = {
        c.key: bool(np.all(np.diff(summaries[c.key]["n_border"].to_numpy()) <= 0)) for c in conditions
    }
    print(f"\nn_border non-increasing in threshold: {monotone}")

    print("\n== Primary (region-paired) tests ==")
    frame = st.results_to_frame(all_tests)
    primary = frame[(frame["unit"] == "region") & (~frame["comparison"].str.contains("enrichment != 0"))]
    if not primary.empty:
        print(primary[["comparison", "test", "n", "p_value", "p_adjusted", "effect_name",
                       "effect_size", "stars"]].round(4).to_string(index=False))
    else:
        print(f"  (none: no region reached --min-border-cells={args.min_border_cells} "
              "at enough thresholds; lower it or use more cells)")

    # ---- figures ----
    highlight_key = args.highlight or next(
        (c.key for c in conditions if not c.is_null and c.level_name == "tissue_unit"),
        next(c.key for c in conditions if not c.is_null),
    )
    highlight = next(c for c in conditions if c.key == highlight_key)
    fig, panels = build_figure(conditions, summaries, loc_matrices, comps, enrichments,
                               pvalues, per_threshold, null_tests, highlight,
                               general_summaries, general_comps)
    panel_paths = save_panels(fig, out_dir, "fig2", panels)
    # Full-size standalone catplots, one per threshold, for every condition.
    for _cond in conditions:
        _ts = catplot_thresholds(sorted(_cond.borders))
        _region = pick_condition_region(_cond)
        panel_paths += save_standalone_catplots(
            out_dir, adata=_cond.adata, spec=_cond.spec,
            posteriors_or_probs=_cond.probs, pair=_cond.pair, region=_region,
            keys=[f"{_cond.key}_t{t:g}" for t in _ts],
            threshold_for=lambda k: float(k.rsplit("_t", 1)[1]),
            label_for=lambda k, c=_cond, r=_region: (
                f"{c.label}  |  {c.pair[0]} | {c.pair[1]}  |  "
                f"t={k.rsplit('_t', 1)[1]}  |  region {r}"),
        )
    written = save_figure(fig, out_dir, "figure2_threshold_effects")
    print(f"  saved {len(panel_paths)} individual panels to {out_dir / 'panels'}")

    # One enrichment heat map per condition, so every focus border is inspectable.
    for cond in conditions:
        f, ax = plt.subplots(figsize=(4.2, 4.6))
        enrichment_heatmap(ax, cond, enrichments[cond.key], pvalues[cond.key])
        written += save_figure(f, out_dir, f"enrichment_heatmap_{cond.key}", pdf=False)

    # The manuscript Figure 2f enrichment dot plot, drawn by the shipped
    # mingl.pl function. Swept across thresholds for the SAME border in each
    # condition, so the plots show how cell-type enrichment at one interface
    # changes as the positivity cutoff moves -- the threshold is the variable,
    # the border is held fixed.
    import mingl.pl as mpl_pl

    enr_thresholds = catplot_thresholds(sorted(conditions[0].borders))
    print(f"\nEnrichment quadrants at thresholds {[f'{t:g}' for t in enr_thresholds]}")
    for cond in conditions:
      pb.attach_posterior(cond.adata, cond.probs)
      for enr_t in enr_thresholds:
        try:
            figs = mpl_pl.plot_border_enrichment(
                adata=cond.adata, n1=cond.pair[0], n2=cond.pair[1],
                cell_type_col=cond.spec.cell_type_col,
                pos_threshold=enr_t, show=False, label_dots=True,
            )
            written += _save_mingl_enrichment(
                figs, out_dir, f"border_enrichment_dotplot_{cond.key}_t{enr_t:g}",
                cn1=cond.pair[0], cn2=cond.pair[1],
                subtitle=cond.label, threshold=enr_t,
            )
        except (ValueError, KeyError, RuntimeError) as exc:
            print(f"  ! plot_border_enrichment skipped {cond.key} @ t={enr_t:g}: {exc}")

    print(f"\nArtifacts written to {out_dir}")
    for p in written:
        print(f"  - {p.name}")
    print("  - threshold_summary.csv / threshold_composition_enrichment_stability.csv / "
          "statistics.csv / conditions.csv")
    print("  - composition_<condition>.csv / enrichment_<condition>.csv / "
          "location_per_region_<condition>.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
