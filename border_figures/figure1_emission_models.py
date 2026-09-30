"""Figure 1 -- do alternative mixture distributions change MINGL's borders?

Contrasts the shipped MINGL GMM (independent univariate Gaussians, diagonal
covariance) against four emission distributions that respect the compositional
count nature of the k-NN window: a full-covariance Gaussian, a multinomial, a
Dirichlet-multinomial, and a logistic-normal. All five score the *same* cells
against the *same* discrete organizational units, so every difference below is
attributable to the emission distribution alone.

Panels
------
a  **Border location, spatial** (one focus border; melanoma by default). A grid
   of manuscript Figure 2g-style maps, one per emission model: cells positive
   only for CN1 (pink), only for CN2 (blue), border cells positive for both
   (red), everything else grey. Same region, same cells, same border pair in
   every panel -- only the emission distribution changes.
b  **Border location, quantified.** Distance from each called border cell to the
   CN1|CN2 interface, in cell diameters, per model. A model whose border cells
   sit further from the interface is calling borders that are not where the
   tissue actually transitions -- this is what puts a number and a p-value on
   panel a.
c  **Border cell-type composition** (same focus border). Stacked composition of
   the border population per model, the Jensen-Shannon divergence of each
   model's composition from the GMM's (bootstrapped over regions), and a
   per-cell-type log2 ratio heat map with per-type paired tests.
d  **Assigned neighborhood probability distributions** (every dataset, all
   neighborhoods). The probability each cell assigns to its own annotated
   neighborhood, per model and per dataset; a neighborhood x model heat map of
   those medians; and the paired per-neighborhood deltas against the GMM.

Statistics
----------
Cells sharing k-NN windows are not independent, and all five models score the
same cells, so the primary tests are **paired across tissue regions** (panels
a, b) or **across neighborhoods** (panel c): Friedman omnibus with Kendall's W,
then Wilcoxon signed-rank against the GMM, Holm-corrected. Cell-level
Kolmogorov-Smirnov / Mann-Whitney results are also written out but flagged
descriptive. Every number in the figure is written to CSV next to it.

Usage
-----
    # local validation, no lab data
    python -m border_figures.figure1_emission_models --synthetic

    # real data (paths are read-only; outputs stay local).
    # Panels a-c: one melanoma border. Panel d: all three datasets.
    python -m border_figures.figure1_emission_models \\
        --data melanoma=/path/melanoma_all_information.csv \\
        --data intestine=/path/05_25_HuBMAP_tunit.csv \\
        --data spatial=/path/all_regions_from_h5mu.csv \\
        --pair "Inflamed Tumor|Productive T cell & Tumor"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mingl.pl as mpl_pl
import numpy as np
import pandas as pd
import seaborn as sns

from . import pair_borders as pb
from . import stats as st
from .config import (
    BASELINE_MODEL,
    manuscript_region,
    MODEL_COLORS,
    MODEL_LABELS,
    MODELS,
    REFERENCE_THRESHOLD,
    default_pair,
    get_level,
    resolve_dataset,
)
from .loading import compute_posterior, load_dataset, slim_obs, subsample_regions, synthetic_tissue
from matplotlib.lines import Line2D

from .plotting import (
    BORDER_MAP_COLORS,
    annotate_significance,
    cell_type_palette,
    collapse_to_top,
    legend_from,
    panel_label,
    save_figure,
    save_panels,
    set_style,
    stacked_composition_bars,
    top_cell_types,
)

#: Cap on the number of cells drawn in a distribution panel (statistics always
#: use every cell; this only keeps the violins from taking minutes to render).
MAX_POINTS_PER_VIOLIN = 20_000


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------
def _display_sample(values: np.ndarray, seed: int = 0, cap: int = MAX_POINTS_PER_VIOLIN) -> np.ndarray:
    values = values[np.isfinite(values)]
    if values.size <= cap:
        return values
    return np.random.default_rng(seed).choice(values, size=cap, replace=False)


def compute_focus_border(
    adata,
    *,
    dataset: str,
    level_name: str,
    models: tuple[str, ...],
    pair: tuple[str, str] | None,
    threshold: float,
    cache_dir: Path | None,
    cache_tag: str,
    min_border_cells: int,
) -> dict:
    """Posteriors + pair-border analysis for every model on the focus border."""
    spec = resolve_dataset(dataset)
    level = get_level(dataset, level_name)

    posteriors = {
        m: compute_posterior(adata, model=m, spec=spec, level=level,
                             cache_dir=cache_dir, cache_tag=cache_tag)
        for m in models
    }

    base_probs = posteriors[BASELINE_MODEL if BASELINE_MODEL in posteriors else models[0]]
    if pair is None:
        pair = default_pair(spec.name, level.name)
    if pair is not None and not set(pair).issubset(set(base_probs.columns)):
        print(
            f"  ! requested border {pair} not present at this level "
            f"(units: {list(base_probs.columns)[:8]}...); auto-selecting instead"
        )
        pair = None
    if pair is None:
        pair = pb.select_top_pair(base_probs, threshold=threshold)
        print(f"  auto-selected focus border: {pair[0]} | {pair[1]}")
    cn1, cn2 = pair

    print(f"  computing interface coordinate for {cn1} | {cn2} ...")
    location = pb.interface_coordinate(
        adata.obs, unit_col=level.unit_col, cn1=cn1, cn2=cn2,
        region_key=spec.region_key, x_key=spec.x_key, y_key=spec.y_key,
    )

    cell_type_order = sorted(adata.obs[spec.cell_type_col].astype(str).unique().tolist())
    results = {
        m: pb.analyze_pair(
            adata.obs, posteriors[m], location, cn1=cn1, cn2=cn2, threshold=threshold,
            region_key=spec.region_key, cell_type_col=spec.cell_type_col,
            min_border_cells=min_border_cells, cell_type_order=cell_type_order,
            adata=adata,
        )
        for m in models
    }
    return {
        "spec": spec, "level": level, "pair": (cn1, cn2), "threshold": threshold,
        "posteriors": posteriors, "location": location, "results": results,
        "cell_type_order": cell_type_order, "obs": adata.obs, "adata": adata,
    }


def location_statistics(focus: dict, models: tuple[str, ...]) -> tuple[pd.DataFrame, list[st.TestResult]]:
    """Region-paired tests on the median distance from border cells to the interface."""
    results = focus["results"]
    per_region = {
        m: results[m].per_region["summary"].set_index("region")["median_dist_to_interface"]
        for m in models
    }
    matrix = pd.DataFrame(per_region)
    # A model that calls no border cells has an all-NaN column; dropping rows
    # first would then empty the whole table, so drop such models explicitly and
    # say so rather than silently reporting "not enough regions".
    empty_models = [c for c in matrix.columns if matrix[c].isna().all()]
    matrix = matrix.drop(columns=empty_models).dropna(how="any")

    tests: list[st.TestResult] = []
    note = f"excluded (no usable borders): {empty_models}" if empty_models else ""
    if matrix.shape[0] >= 3 and matrix.shape[1] >= 2 and BASELINE_MODEL in matrix.columns:
        tests.extend(
            st.paired_condition_tests(matrix, reference=BASELINE_MODEL, label="border location")
        )
        for t in tests:
            t.note = (t.note + " " + note).strip()
    else:
        tests.append(
            st.TestResult(
                comparison="border location: across models", test="friedman", unit="region",
                n=int(matrix.shape[0]), statistic=float("nan"), p_value=float("nan"),
                note=("not enough regions with usable borders for a paired test " + note).strip(),
            )
        )
    # Descriptive cell-level view.
    base_vals = results[BASELINE_MODEL].location_of_border
    for m in models:
        if m == BASELINE_MODEL:
            continue
        tests.append(
            st.mannwhitney(
                results[m].location_of_border, base_vals,
                comparison=f"border location (cells): {m} vs {BASELINE_MODEL}",
            )
        )
    return matrix, tests


def composition_statistics(
    focus: dict, models: tuple[str, ...], *, seed: int = 0, n_boot: int = 1000
) -> tuple[pd.DataFrame, pd.DataFrame, list[st.TestResult]]:
    """JSD from the GMM composition (region-bootstrapped) + per-cell-type paired tests."""
    results = focus["results"]
    order = focus["cell_type_order"]

    overall = pd.DataFrame(
        {
            m: results[m].composition.query("group == 'border'").set_index("cell_type")["proportion"]
            for m in models
        }
    ).reindex(order).fillna(0.0).T  # models x cell types

    region_comp = {m: results[m].per_region["composition"].reindex(columns=order).fillna(0.0) for m in models}
    shared = None
    for m in models:
        idx = set(region_comp[m].index)
        shared = idx if shared is None else (shared & idx)
    shared_regions = sorted(shared or set())

    n_border = {m: results[m].n_border for m in models}
    jsd_rows, tests = [], []
    for m in models:
        if m == BASELINE_MODEL:
            jsd_rows.append({"model": m, "jsd_vs_gmm": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
                             "n_regions": len(shared_regions), "n_border": n_border[m]})
            continue
        if n_border[m] == 0 or n_border[BASELINE_MODEL] == 0:
            # No border population to compare -- report it as missing, not as 0.
            jsd_rows.append({"model": m, "jsd_vs_gmm": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                             "n_regions": len(shared_regions), "n_border": n_border[m]})
            continue
        point = st.jensen_shannon_divergence(overall.loc[m], overall.loc[BASELINE_MODEL])
        if len(shared_regions) >= 3:
            a = region_comp[m].loc[shared_regions]
            b = region_comp[BASELINE_MODEL].loc[shared_regions]

            def _stat(sampled_regions, _a=a, _b=b):
                rows = list(sampled_regions)
                return st.jensen_shannon_divergence(
                    _a.loc[rows].mean(axis=0).to_numpy(), _b.loc[rows].mean(axis=0).to_numpy()
                )

            # Take the bootstrap's own point estimate, not the pooled-composition
            # one: the CI is built from the region-averaged statistic, so a
            # pooled point estimate can fall outside its own interval.
            point, lo, hi = st.bootstrap_ci(_stat, shared_regions, n_boot=n_boot, seed=seed)
        else:
            lo = hi = float("nan")
        jsd_rows.append({"model": m, "jsd_vs_gmm": point, "ci_lo": lo, "ci_hi": hi,
                         "n_regions": len(shared_regions), "n_border": n_border[m]})

        # Per-cell-type paired difference across regions.
        if len(shared_regions) >= 3:
            per_type = [
                st.wilcoxon_vs_reference(
                    region_comp[m].loc[shared_regions, ct],
                    region_comp[BASELINE_MODEL].loc[shared_regions, ct],
                    comparison=f"composition: {ct} | {m} vs {BASELINE_MODEL}",
                )
                for ct in order
            ]
            for res, adj in zip(per_type, st.holm([r.p_value for r in per_type])):
                res.p_adjusted = None if not np.isfinite(adj) else float(adj)
                res.note = (res.note + " Holm-corrected across cell types").strip()
            tests.extend(per_type)

    # Descriptive: is the border composition homogeneous across models at all?
    counts = pd.DataFrame(
        {
            m: results[m].composition.query("group == 'border'").set_index("cell_type")["n"]
            for m in models
        }
    ).reindex(order).fillna(0).T
    tests.append(st.chi2_homogeneity(counts, comparison="border composition across models"))

    return overall, pd.DataFrame(jsd_rows), tests


def assigned_probability_table(
    adata, probs: pd.DataFrame, *, unit_col: str
) -> pd.DataFrame:
    """Per-cell probability of the cell's own annotated unit, plus the max probability."""
    units = adata.obs[unit_col].astype(str).to_numpy()
    columns = list(probs.columns)
    lookup = {name: i for i, name in enumerate(columns)}
    arr = probs.to_numpy(dtype=float)
    col_idx = np.array([lookup.get(u, -1) for u in units])
    valid = col_idx >= 0
    assigned = np.full(arr.shape[0], np.nan)
    assigned[valid] = arr[np.arange(arr.shape[0])[valid], col_idx[valid]]
    return pd.DataFrame(
        {"unit": units, "p_assigned": assigned, "p_max": arr.max(axis=1)}, index=probs.index
    )


def compute_probability_panel(
    datasets: dict[str, object],
    *,
    models: tuple[str, ...],
    cache_dir: Path | None,
    cache_tag: str,
) -> dict:
    """Assigned-probability distributions for every dataset x model (neighborhood level)."""
    per_cell: dict[tuple[str, str], pd.DataFrame] = {}
    per_unit_rows, tests = [], []

    for ds_name, adata in datasets.items():
        spec = resolve_dataset(ds_name)
        level = get_level(ds_name, "neighborhood")
        print(f"  [{ds_name}] assigned-probability panel ({level.unit_col}, k={level.k})")
        tables = {}
        for m in models:
            probs = compute_posterior(
                adata, model=m, spec=spec, level=level, cache_dir=cache_dir, cache_tag=cache_tag
            )
            tables[m] = assigned_probability_table(adata, probs, unit_col=level.unit_col)
            per_cell[(ds_name, m)] = tables[m]

        # Per-neighborhood medians: the paired unit for this panel.
        medians = pd.DataFrame(
            {m: tables[m].groupby("unit")["p_assigned"].median() for m in models}
        ).dropna(how="any")
        for unit, row in medians.iterrows():
            for m in models:
                per_unit_rows.append(
                    {"dataset": ds_name, "unit": unit, "model": m, "median_p_assigned": row[m]}
                )

        label = f"assigned probability [{ds_name}]"
        if medians.shape[0] >= 3 and BASELINE_MODEL in medians.columns:
            tests.extend(st.paired_condition_tests(medians, reference=BASELINE_MODEL, label=label))
        else:
            tests.append(
                st.TestResult(
                    comparison=f"{label}: across models", test="friedman", unit="neighborhood",
                    n=int(medians.shape[0]), statistic=float("nan"), p_value=float("nan"),
                    note="fewer than 3 neighborhoods with complete data",
                )
            )
        for m in models:
            if m == BASELINE_MODEL:
                continue
            tests.append(
                st.ks_vs_reference(
                    tables[m]["p_assigned"].to_numpy(),
                    tables[BASELINE_MODEL]["p_assigned"].to_numpy(),
                    comparison=f"{label} (cells): {m} vs {BASELINE_MODEL}",
                )
            )

    return {"per_cell": per_cell, "per_unit": pd.DataFrame(per_unit_rows), "tests": tests}


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def pick_display_region(focus: dict, *, model: str = BASELINE_MODEL) -> str:
    """Region to draw in the spatial panel: the one with the most border cells.

    Deterministic and independent of the models being compared (it is chosen on
    the GMM baseline), so every model's map shows the same tissue.
    """
    spec = focus["spec"]
    obs = focus["obs"]
    regions = obs[spec.region_key].astype(str).to_numpy()
    # Prefer the region the manuscript itself plots, so these panels show the
    # same tissue the paper discusses in detail.
    preferred = manuscript_region(spec.name)
    if preferred is not None and (regions == str(preferred)).any():
        return str(preferred)
    ref = focus["results"].get(model) or next(iter(focus["results"].values()))
    counts = pd.Series(regions[np.asarray(ref.masks["border"], dtype=bool)]).value_counts()
    if counts.empty:  # no border cells anywhere: fall back to the largest region
        return str(pd.Series(regions).value_counts().index[0])
    return str(counts.index[0])



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


def panel_a_spatial(
    axes, ax_legend, focus, models, region: str, *, adata, point_size: float = 1.0
) -> None:
    """Figure 2g-style border maps, one per emission model, on the same region.

    Drawn by the shipped :func:`mingl.pl.spatial_loc_region`: for each model the
    model's posterior is written into the canonical MINGL keys and the function
    is called into our axes, so these panels are the package's own plot, not a
    reimplementation of it.
    """
    import mingl.pl as pl

    spec = focus["spec"]
    cn1, cn2 = focus["pair"]

    for ax, m in zip(axes, models):
        pb.attach_posterior(adata, focus["posteriors"][m])
        pl.spatial_loc_region(
            adata, region=str(region), n1=cn1, n2=cn2, threshold=focus["threshold"],
            region_key=spec.region_key, x_col=spec.x_key, y_col=spec.y_key,
            s_other=point_size * 0.06, s_single=point_size * 0.10, s_both=point_size * 0.40,
            ax=ax, show=False,
        )
        # The shipped function draws a full-size per-axes legend sized for a
        # standalone 20x20 figure; in a grid that is unreadable, so it is
        # replaced by one shared legend built from the same colors below.
        existing = ax.get_legend()
        if existing is not None:
            existing.remove()
        ax.set_aspect("equal", "box")
        ax.set_axis_off()
        ax.set_title(
            f"{MODEL_LABELS[m].replace(chr(10), ' ')}\n"
            f"{focus['results'][m].n_border:,} border cells (all regions)",
            fontsize=8,
        )

    # A shared legend, built from the colors mingl.pl.spatial_loc_region uses.
    ax_legend.axis("off")
    handles = [
        Line2D([], [], marker="o", ls="", ms=5, color=color, label=label)
        for label, color in (
            (cn1, BORDER_MAP_COLORS["only_1"]),
            (cn2, BORDER_MAP_COLORS["only_2"]),
            ("Border cells", BORDER_MAP_COLORS["both"]),
            ("Other cells", BORDER_MAP_COLORS["other"]),
        )
    ]
    ax_legend.legend(handles=handles, loc="center left", fontsize=6.5, title="Cell category")



def panel_spatial_probability(axes, ax_cbar, focus, models, region: str, *, point_size: float = 0.05) -> None:
    """Spatial map of each cell's assigned-neighborhood probability, per model.

    Manuscript Figure 1d in style: every cell coloured by the probability MINGL
    gives to the neighborhood it is already annotated with, so low-probability
    (uncertain) cells light up at organizational boundaries.

    Written here rather than calling :func:`mingl.pl.spatial_probability_mapping`
    because that function recomputes windows, centroids and probabilities from
    scratch internally -- it cannot be handed an existing posterior, and the
    whole point of this panel is to display the five *already computed* emission
    models on identical cells.
    """
    spec, level = focus["spec"], focus["level"]
    obs = focus["obs"]
    in_region = (obs[spec.region_key].astype(str) == str(region)).to_numpy()
    x = obs.loc[in_region, spec.x_key].to_numpy(dtype=float)
    y = obs.loc[in_region, spec.y_key].to_numpy(dtype=float)
    units = obs.loc[in_region, level.unit_col].astype(str).to_numpy()

    im = None
    for ax, m in zip(axes, models):
        probs = focus["posteriors"][m]
        cols = list(probs.columns)
        idx = {n: i for i, n in enumerate(cols)}
        arr = probs.to_numpy(dtype=float)[in_region]
        col_idx = np.array([idx.get(u, -1) for u in units])
        p_assigned = np.full(arr.shape[0], np.nan)
        ok = col_idx >= 0
        p_assigned[ok] = arr[np.arange(arr.shape[0])[ok], col_idx[ok]]

        order = np.argsort(np.nan_to_num(p_assigned, nan=-1.0))  # low probability on top
        im = ax.scatter(
            x[order], y[order], c=p_assigned[order], cmap="viridis", vmin=0.0, vmax=1.0,
            s=point_size, linewidths=0, edgecolors="none", rasterized=True,
        )
        ax.set_aspect("equal", "box")
        ax.set_axis_off()
        median = float(np.nanmedian(p_assigned)) if np.isfinite(p_assigned).any() else float("nan")
        ax.set_title(f"{MODEL_LABELS[m].replace(chr(10), ' ')}\nmedian P = {median:.2f}", fontsize=8)

    ax_cbar.axis("off")
    if im is not None:
        plt.colorbar(im, ax=ax_cbar, fraction=0.5, pad=0.02,
                     label="P(assigned neighborhood)")


def panel_a_location(ax_violin, ax_paired, focus, models, matrix, tests, seed=0) -> None:
    results = focus["results"]
    cn1, cn2 = focus["pair"]

    frames = [
        pd.DataFrame(
            {
                "model": MODEL_LABELS[m],
                "dist": _display_sample(results[m].location_of_border, seed=seed),
            }
        )
        for m in models
    ]
    long = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    order = [MODEL_LABELS[m] for m in models]
    palette = {MODEL_LABELS[m]: MODEL_COLORS[m] for m in models}

    if not long.empty:
        sns.violinplot(
            data=long, x="model", y="dist", order=order, hue="model", palette=palette,
            legend=False, cut=0, density_norm="width", linewidth=0.5, inner=None, ax=ax_violin,
        )
        for coll in ax_violin.collections:
            coll.set_alpha(0.45)
        sns.boxplot(
            data=long, x="model", y="dist", order=order, width=0.16, showcaps=False,
            showfliers=False, boxprops={"facecolor": "white", "linewidth": 0.7},
            whiskerprops={"linewidth": 0.7}, medianprops={"linewidth": 1.1, "color": "black"},
            ax=ax_violin,
        )
    # Region medians on top -- the actual statistical units.
    if not matrix.empty:
        region_long = matrix.reset_index().melt(id_vars="region", var_name="model", value_name="dist")
        region_long["model"] = region_long["model"].map(MODEL_LABELS)
        sns.stripplot(
            data=region_long, x="model", y="dist", order=order, ax=ax_violin,
            color="black", size=2.2, jitter=0.14, alpha=0.75,
        )

    upper = np.nanpercentile(long["dist"], 99) if not long.empty else 1.0
    ax_violin.set_ylim(0, max(float(upper), 1.0) * 1.18)
    ax_violin.set_xlabel("")
    ax_violin.set_ylabel("distance to interface\n(cell diameters)")
    ax_violin.set_title(f"Border location: {cn1} | {cn2}", loc="left")
    ax_violin.axhline(1.0, color="#888888", lw=0.6, ls=":", zorder=0)
    # Border-cell counts belong with the model names, not floating in the panel.
    ax_violin.set_xticks(range(len(models)))
    ax_violin.set_xticklabels(
        [f"{MODEL_LABELS[m]}\nn={results[m].n_border:,}" for m in models]
    )

    posthoc = {
        t.comparison.split(": ")[-1].split(" vs ")[0]: t
        for t in tests
        if t.test == "wilcoxon_signed_rank" and t.unit == "region"
    }
    annotate_significance(
        ax_violin,
        list(range(len(models))),
        ["" if m == BASELINE_MODEL else posthoc.get(m, st.TestResult("", "", "", 0, np.nan, np.nan)).stars
         for m in models],
    )

    # Paired region view: every region, GMM -> model. Models with no usable
    # borders were dropped from the matrix upstream and simply have no points.
    if not matrix.empty and BASELINE_MODEL in matrix.columns:
        for m in models:
            if m == BASELINE_MODEL or m not in matrix.columns:
                continue
            delta = matrix[m] - matrix[BASELINE_MODEL]
            ax_paired.scatter(
                np.full(delta.size, models.index(m)) + np.random.default_rng(seed).normal(0, 0.06, delta.size),
                delta, s=6, color=MODEL_COLORS[m], alpha=0.75, linewidths=0,
            )
            ax_paired.plot([models.index(m) - 0.22, models.index(m) + 0.22],
                           [np.median(delta)] * 2, color="black", lw=1.2)
    ax_paired.axhline(0, color="#888888", lw=0.8)
    ax_paired.set_xticks(range(len(models)))
    ax_paired.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=0)
    ax_paired.set_ylabel("Δ median distance\nvs MINGL GMM")
    ax_paired.set_title("Paired by region", loc="left")


def panel_b_composition(ax_stack, ax_legend, focus, models, overall) -> None:
    """Stacked border composition per model, showing every cell type."""
    cn1, cn2 = focus["pair"]
    results = focus["results"]
    order = top_cell_types(overall)          # all cell types, no "Other"
    collapsed = collapse_to_top(overall, order)
    palette = cell_type_palette(order)

    stacked_composition_bars(
        ax_stack, collapsed, palette=palette,
        x_labels=[MODEL_LABELS[m] for m in collapsed.index],
    )
    ax_stack.set_title(f"Border composition: {cn1} | {cn2}", loc="left")
    for i, m in enumerate(collapsed.index):
        if results[m].n_border == 0:
            ax_stack.text(i, 0.5, "no border\ncells", ha="center", va="center",
                          fontsize=6, color="#666666", rotation=90)
    ncol = 2 if len(order) > 18 else 1
    legend_from(ax_stack, ax_legend, ncol=ncol, title="Cell type", fontsize=5.5)


def panel_c_probabilities(axes, ax_paired, prob_panel, models, datasets, seed=0) -> None:
    per_cell = prob_panel["per_cell"]
    per_unit = prob_panel["per_unit"]
    order = [MODEL_LABELS[m] for m in models]
    palette = {MODEL_LABELS[m]: MODEL_COLORS[m] for m in models}

    for ax, ds_name in zip(axes, datasets):
        frames = []
        for m in models:
            tbl = per_cell.get((ds_name, m))
            if tbl is None:
                continue
            frames.append(
                pd.DataFrame(
                    {"model": MODEL_LABELS[m],
                     "p": _display_sample(tbl["p_assigned"].to_numpy(), seed=seed)}
                )
            )
        if not frames:
            ax.set_visible(False)
            continue
        long = pd.concat(frames, ignore_index=True)
        sns.violinplot(
            data=long, x="model", y="p", order=order, hue="model", palette=palette, legend=False,
            cut=0, density_norm="width", linewidth=0.5, inner="quart", ax=ax,
        )
        for coll in ax.collections:
            coll.set_alpha(0.55)
        sub = per_unit[per_unit["dataset"] == ds_name]
        if not sub.empty:
            sub = sub.assign(model_label=sub["model"].map(MODEL_LABELS))
            sns.stripplot(
                data=sub, x="model_label", y="median_p_assigned", order=order, ax=ax,
                color="black", size=2.4, jitter=0.14, alpha=0.8,
            )
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("")
        ax.set_ylabel("P(assigned neighborhood)")
        n_units = sub["unit"].nunique() if not sub.empty else 0
        ax.set_title(f"{resolve_dataset(ds_name).label}  ({n_units} neighborhoods)", loc="left")

    # Per-neighborhood medians, paired to the GMM.
    if not per_unit.empty:
        wide = per_unit.pivot_table(
            index=["dataset", "unit"], columns="model", values="median_p_assigned"
        )
        markers = {ds: mk for ds, mk in zip(datasets, ["o", "s", "^", "D", "v"])}
        for ds_name in datasets:
            if ds_name not in wide.index.get_level_values(0):
                continue
            block = wide.loc[ds_name]
            for m in models:
                if m == BASELINE_MODEL or m not in block.columns:
                    continue
                delta = (block[m] - block[BASELINE_MODEL]).dropna()
                xpos = models.index(m) + np.random.default_rng(seed).normal(0, 0.07, delta.size)
                ax_paired.scatter(
                    xpos, delta, s=8, color=MODEL_COLORS[m], alpha=0.7, linewidths=0,
                    marker=markers.get(ds_name, "o"),
                    label=resolve_dataset(ds_name).label if m == models[-1] else None,
                )
        for m in models:
            if m == BASELINE_MODEL or m not in wide.columns or BASELINE_MODEL not in wide.columns:
                continue
            vals = (wide[m] - wide[BASELINE_MODEL]).dropna()
            if vals.size:
                ax_paired.plot([models.index(m) - 0.24, models.index(m) + 0.24],
                               [np.median(vals)] * 2, color="black", lw=1.2)
    ax_paired.axhline(0, color="#888888", lw=0.8)
    ax_paired.set_xticks(range(len(models)))
    ax_paired.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax_paired.set_ylabel("Δ median P(assigned)\nvs MINGL GMM")
    ax_paired.set_title("Paired by neighborhood", loc="left")
    handles, labels = ax_paired.get_legend_handles_labels()
    if handles:
        ax_paired.legend(handles, labels, loc="best", markerscale=1.4)


def panel_d_probability_heatmap(ax, prob_panel, models) -> None:
    """Neighborhood x model heat map of the median assigned probability."""
    per_unit = prob_panel["per_unit"]
    if per_unit.empty:
        ax.set_visible(False)
        return
    wide = per_unit.pivot_table(index=["dataset", "unit"], columns="model", values="median_p_assigned")
    cols = [m for m in models if m in wide.columns]
    wide = wide.loc[:, cols]
    # Group rows by dataset, and within a dataset order by the GMM baseline so
    # the neighborhoods MINGL is least confident about sit together.
    sort_col = BASELINE_MODEL if BASELINE_MODEL in wide.columns else cols[0]
    wide = wide.sort_values(["dataset", sort_col], ascending=[True, False])

    im = ax.imshow(wide.to_numpy(), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in cols], fontsize=6, rotation=90)
    labels = [f"{resolve_dataset(ds).name} · {unit}" for ds, unit in wide.index]
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=max(3.5, min(6.0, 260 / max(len(labels), 1))))
    ax.set_title("Median P(assigned) per neighborhood", loc="left")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="median P(assigned)")

    # Separate the datasets with a hairline.
    datasets_in_order = [ds for ds, _ in wide.index]
    for i in range(1, len(datasets_in_order)):
        if datasets_in_order[i] != datasets_in_order[i - 1]:
            ax.axhline(i - 0.5, color="white", lw=1.2)


def build_figure(focus, models, matrix, loc_tests, overall, jsd, comp_tests, prob_panel,
                 datasets, region, seed=0):
    """Three rows: border catplot, border composition, assigned-probability maps."""
    n_models = len(models)
    fig = plt.figure(figsize=(19.0, 21.0))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.1, 1.0, 0.95], hspace=0.42)

    # a -- mingl.pl.spatial_loc_region, one map per emission model.
    row_a = gs[0].subgridspec(1, n_models + 1, width_ratios=[1.0] * n_models + [0.5], wspace=0.12)
    ax_a = [fig.add_subplot(row_a[i]) for i in range(n_models)]
    ax_a_leg = fig.add_subplot(row_a[n_models])
    panel_a_spatial(ax_a, ax_a_leg, focus, models, region, adata=focus["adata"])
    panel_label(ax_a[0], "a")

    # b -- border cell-type composition (proportions from mingl.tl).
    row_b = gs[1].subgridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.3)
    ax_b1 = fig.add_subplot(row_b[0])
    ax_b_leg = fig.add_subplot(row_b[1])
    panel_b_composition(ax_b1, ax_b_leg, focus, models, overall)
    panel_label(ax_b1, "b")

    # c -- spatial map of assigned-neighborhood probability, per model.
    row_c = gs[2].subgridspec(1, n_models + 1, width_ratios=[1.0] * n_models + [0.5], wspace=0.12)
    ax_c = [fig.add_subplot(row_c[i]) for i in range(n_models)]
    ax_c_cbar = fig.add_subplot(row_c[n_models])
    panel_spatial_probability(ax_c, ax_c_cbar, focus, models, region)
    panel_label(ax_c[0], "c")

    # d -- the same probabilities as violin distributions, so the models can be
    # compared statistically rather than only by eye.
    n_ds = len(datasets)
    row_d = gs[3].subgridspec(1, n_ds + 1, width_ratios=[1.0] * n_ds + [1.0], wspace=0.4)
    ax_d = [fig.add_subplot(row_d[i]) for i in range(n_ds)]
    ax_d_paired = fig.add_subplot(row_d[n_ds])
    panel_c_probabilities(ax_d, ax_d_paired, prob_panel, models, datasets, seed=seed)
    panel_label(ax_d[0], "d")

    spec, level = focus["spec"], focus["level"]
    n_analyzed = int(next(iter(focus["results"].values())).n_cells)
    fig.suptitle(
        f"Emission distribution vs MINGL GMM — {focus['pair'][0]} | {focus['pair'][1]} "
        f"({spec.label}, {level.label}, k={level.k}, threshold={focus['threshold']:g}; "
        f"n = {n_analyzed:,} cells; maps show region {region})",
        fontsize=11, y=0.995,
    )

    panels = {
        "a_border_catplot": ax_a + [ax_a_leg],
        "b_border_composition": [ax_b1, ax_b_leg],
        "c_assigned_probability_map": ax_c + [ax_c_cbar],
        "d_assigned_probability_violins": ax_d + [ax_d_paired],
    }
    for ds_name, ax in zip(datasets, ax_d):
        panels[f"d_assigned_probability_violins_{ds_name}"] = ax
    for m, ax in zip(models, ax_a):
        panels[f"a_border_catplot_{m}"] = ax
    for m, ax in zip(models, ax_c):
        panels[f"c_assigned_probability_map_{m}"] = ax
    return fig, panels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_data_args(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--data expects DATASET=PATH, got {item!r}")
        name, path = item.split("=", 1)
        out[resolve_dataset(name).name] = path
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", action="append", metavar="DATASET=PATH",
                    help="Repeatable. e.g. --data intestine=/path/file.csv")
    ap.add_argument("--synthetic", action="store_true",
                    help="Run on a locally generated tissue instead of lab data.")
    ap.add_argument("--synthetic-regions", type=int, default=12)
    ap.add_argument("--synthetic-cells", type=int, default=6000,
                    help="Cells per region in the synthetic tissue (lower it for a fast smoke test).")
    ap.add_argument("--focus-dataset", default=None,
                    help="Dataset for panels a-c (default: melanoma if supplied, else the first).")
    ap.add_argument("--focus-level", default="neighborhood",
                    help="Hierarchy level for panels a-c (neighborhood|community|tissue_unit).")
    ap.add_argument("--pair", default=None, metavar='"CN1|CN2"',
                    help="Focus border. Default: the manuscript pair for this dataset/level, else auto.")
    ap.add_argument("--region", default=None,
                    help="Region drawn in the panel-a maps (default: the one with the most border cells).")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--threshold", type=float, default=REFERENCE_THRESHOLD)
    ap.add_argument("--min-border-cells", type=int, default=20,
                    help="Minimum border cells for a region to enter the paired statistics.")
    ap.add_argument("--subsample-frac", type=float, default=1.0)
    ap.add_argument("--max-cells", type=int, default=None,
                    help="Global cap per dataset (region-stratified).")
    ap.add_argument("--melanoma-max-cells", type=int, default=1_500_000,
                    help="Extra cap for the 5M-cell melanoma file; set 0 to disable.")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs" / "figure1_emission_models"))
    ap.add_argument("--cache-dir", default=str(Path(__file__).resolve().parent / "outputs" / "_cache"))
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)

    models = tuple(args.models)
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise SystemExit(f"Unknown model(s) {unknown}; choose from {list(MODELS)}")
    if BASELINE_MODEL not in models:
        raise SystemExit(f"{BASELINE_MODEL} (the MINGL GMM baseline) must be included in --models")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = None if args.no_cache else Path(args.cache_dir)
    set_style()

    # ---- load ----
    paths = _parse_data_args(args.data)
    loaded: dict[str, object] = {}
    if args.synthetic:
        print("Loading synthetic tissue (no lab data).")
        loaded["synthetic"] = synthetic_tissue(
            n_regions=args.synthetic_regions,
            n_cells_per_region=args.synthetic_cells,
            seed=args.seed,
        )
    # Resolve the focus dataset/level *before* loading: the focus level decides
    # which columns must be present and, through k, which regions are large
    # enough to keep (k=300 at the tissue-unit level, vs 10 at neighborhood).
    available = [*loaded, *paths]
    focus_ds = args.focus_dataset or ("melanoma" if "melanoma" in available else available[0])
    if focus_ds not in available:
        raise SystemExit(f"--focus-dataset {focus_ds!r} was not supplied (have {sorted(available)}).")
    focus_level = args.focus_level
    try:
        get_level(focus_ds, focus_level)
    except KeyError:
        print(f"  ! {focus_ds} has no level {focus_level!r}; falling back to 'neighborhood'")
        focus_level = "neighborhood"

    if paths:
        for ds_name, path in paths.items():
            spec = resolve_dataset(ds_name)
            # Panel c always works at the neighborhood level, panels a/b at the
            # focus level, so both must be validated and both k respected.
            levels_used = [get_level(ds_name, "neighborhood")]
            if ds_name == focus_ds:
                levels_used.append(get_level(ds_name, focus_level))
            deepest = max(levels_used, key=lambda lspec: lspec.k)
            extra = tuple({lspec.unit_col for lspec in levels_used}
                          | {lspec.feature_col for lspec in levels_used})

            print(f"Loading {ds_name} from {path}")
            adata = load_dataset(path, ds_name, deepest, required_extra=extra)
            # The lab CSVs carry dozens of marker columns; keep only what the
            # figure needs so three datasets fit in memory at once.
            adata = slim_obs(
                adata, [spec.cell_type_col, spec.region_key, spec.x_key, spec.y_key, *extra]
            )
            cap = args.max_cells
            if ds_name == "melanoma" and args.melanoma_max_cells:
                cap = min(cap or args.melanoma_max_cells, args.melanoma_max_cells)
            adata = subsample_regions(
                adata, region_key=spec.region_key, frac=args.subsample_frac,
                max_cells=cap, min_per_region=deepest.k, seed=args.seed,
            )
            loaded[ds_name] = adata
            print(f"  {ds_name}: {adata.n_obs:,} cells, "
                  f"{adata.obs[spec.region_key].astype(str).nunique()} regions")
    if not loaded:
        raise SystemExit("Provide --data DATASET=PATH (repeatable) or --synthetic.")
    if focus_ds not in loaded:
        raise SystemExit(f"--focus-dataset {focus_ds!r} was not loaded (have {sorted(loaded)}).")

    pair = tuple(p.strip() for p in args.pair.split("|")) if args.pair else None
    if pair is not None and len(pair) != 2:
        raise SystemExit('--pair must look like "CN1|CN2"')

    cache_tag = f"sub{args.subsample_frac}_max{args.max_cells}_seed{args.seed}"

    # ---- panels a/b ----
    print(f"\n== Focus border: {focus_ds} / {focus_level} ==")
    focus = compute_focus_border(
        loaded[focus_ds], dataset=focus_ds, level_name=focus_level, models=models,
        pair=pair, threshold=args.threshold, cache_dir=cache_dir, cache_tag=cache_tag,
        min_border_cells=args.min_border_cells,
    )
    matrix, loc_tests = location_statistics(focus, models)
    overall, jsd, comp_tests = composition_statistics(
        focus, models, seed=args.seed, n_boot=args.n_boot
    )

    # ---- panel c ----
    print("\n== Assigned neighborhood probabilities (all datasets) ==")
    prob_panel = compute_probability_panel(
        loaded, models=models, cache_dir=cache_dir, cache_tag=cache_tag
    )

    # ---- write tables ----
    summary = pd.DataFrame(
        [focus["results"][m].summary_row(model=m, dataset=focus_ds, level=focus_level) for m in models]
    )
    summary.to_csv(out_dir / "focus_border_summary.csv", index=False)
    matrix.to_csv(out_dir / "border_location_per_region.csv")
    overall.to_csv(out_dir / "border_composition_by_model.csv")
    jsd.to_csv(out_dir / "border_composition_jsd_vs_gmm.csv", index=False)
    prob_panel["per_unit"].to_csv(out_dir / "assigned_probability_per_neighborhood.csv", index=False)
    for m in models:
        focus["results"][m].enrichment.assign(model=m).to_csv(
            out_dir / f"border_enrichment_{m}.csv", index=False
        )
    all_tests = st.results_to_frame([*loc_tests, *comp_tests, *prob_panel["tests"]])
    all_tests.to_csv(out_dir / "statistics.csv", index=False)

    print("\n== Focus border summary ==")
    print(summary[["model", "n_border", "frac_border", "median_dist_to_interface",
                   "frac_within_1_diameter", "n_enriched_both"]].round(4).to_string(index=False))
    print("\n== Primary (region/neighborhood-paired) tests ==")
    primary = all_tests[all_tests["unit"] != "cell"]
    if not primary.empty:
        print(primary[["comparison", "test", "n", "p_value", "p_adjusted",
                       "effect_name", "effect_size", "stars"]].round(4).to_string(index=False))
    else:
        print(f"  (none: no region reached --min-border-cells={args.min_border_cells}; "
              "lower it or use more cells)")

    # ---- figure ----
    region = args.region or pick_display_region(focus)
    print(f"\nPanel-a maps drawn on region {region!r}")
    fig, panels = build_figure(
        focus, models, matrix, loc_tests, overall, jsd, comp_tests,
        prob_panel, list(loaded), region, seed=args.seed,
    )
    panel_paths = save_panels(fig, out_dir, "fig1", panels)
    # Full-size standalone catplots: the cropped grid panels are too small to
    # resolve individual cells.
    panel_paths += save_standalone_catplots(
        out_dir, adata=focus["adata"], spec=focus["spec"],
        posteriors_or_probs=focus["posteriors"], pair=focus["pair"], region=region,
        keys=list(models), threshold_for=lambda k: focus["threshold"],
        label_for=lambda k: (f"{MODEL_LABELS[k].replace(chr(10), ' ')}  |  "
                             f"{focus['pair'][0]} | {focus['pair'][1]}  |  "
                             f"t={focus['threshold']:g}  |  region {region}"),
    )
    written = save_figure(fig, out_dir, "figure1_emission_models")
    print(f"  saved {len(panel_paths)} individual panels to {out_dir / 'panels'}")

    # The manuscript Figure 2f enrichment dot plot, drawn per model by the
    # shipped mingl.pl function (it creates its own figures, so it cannot be
    # composed into the grid above).
    for m in models:
        try:
            pb.attach_posterior(focus["adata"], focus["posteriors"][m])
            figs = mpl_pl.plot_border_enrichment(
                adata=focus["adata"], n1=focus["pair"][0], n2=focus["pair"][1],
                cell_type_col=focus["spec"].cell_type_col, pos_threshold=args.threshold,
                show=False,
            )
            written += _save_mingl_enrichment(
                figs, out_dir, f"border_enrichment_dotplot_{m}",
                cn1=focus["pair"][0], cn2=focus["pair"][1],
                subtitle=f"{MODEL_LABELS[m].replace(chr(10), ' ')} · {focus['spec'].label}",
                threshold=args.threshold,
            )
        except (ValueError, KeyError, RuntimeError) as exc:
            print(f"  ! mingl.pl.plot_border_enrichment skipped for {m}: {exc}")
    print(f"\nArtifacts written to {out_dir}")
    for p in written:
        print(f"  - {p.name}")
    print("  - focus_border_summary.csv / border_location_per_region.csv / "
          "border_composition_by_model.csv / border_composition_jsd_vs_gmm.csv")
    print("  - assigned_probability_per_neighborhood.csv / border_enrichment_<model>.csv / statistics.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
