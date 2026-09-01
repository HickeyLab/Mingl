# `border_figures` — two figures on MINGL borders

Self-contained package for two figures. It only *reads* data you point it at and
only writes into `border_figures/outputs/`. Nothing under `src/mingl/` is
modified, so every previously generated figure stays reproducible.

* **Figure 1** — do alternative mixture/emission distributions change MINGL's
  borders, versus the shipped GMM?
* **Figure 2** — how does the probability threshold change what a border *is*?

Both were validated locally on a synthetic tissue (`--synthetic`); the real runs
are the commands in [Running on the lab data](#running-on-the-lab-data).

---

## Built on MINGL's own commands

The analysis and the plots are the package's, not reimplementations. The bridge
is `pair_borders.attach_posterior(adata, probs)`, which writes a posterior into
the canonical `obsm["neighborhood_probabilities"]` / `uns[...]` keys — so each
emission model (figure 1) and each threshold (figure 2) can be handed to the
shipped functions unmodified:

| what | MINGL command |
|---|---|
| per-model posteriors | `mg.tl.mingl_membership_probabilities` |
| border catplot (Fig 2g) | `mg.pl.spatial_loc_region(adata, region, n1, n2, threshold, ax=…)` |
| border composition (Fig 2e) | `mg.tl.compute_grouped_proportions(adata, n1, n2, threshold=…)` |
| border cell-type enrichment (Fig 2f) | `mg.pl.plot_border_enrichment(adata, n1=…, n2=…)` |
| positive-membership counts | `mg.tl.findPositives` |
| global threshold sweep | `mg.tl.threshold_sensitivity_analysis`, `border_metrics_at_threshold`, `spatial_border_clustering_null` |

Only four things here are *not* in MINGL, and each is written because nothing in
the package covers it:

1. **Distance to the CN1|CN2 interface** — the quantitative border-location
   metric (`pair_borders.interface_coordinate`).
2. **Region-paired statistics** (`stats.py`).
3. **The null-condition construction** (`loading.make_null`).
4. A **numeric** enrichment table — `plot_border_enrichment` returns figures, not
   values, so `pair_borders.pair_enrichment` recomputes the identical
   `log2((p_border+ε)/(p_single+ε))` for the statistics while the shipped
   function draws the dot plot.

Everything else is grid scaffolding around MINGL calls: the `mingl.pl` functions
each draw one model at one threshold, and these figures compare five models and
nine thresholds.

## What the figures show

### Figure 1 · `figure1_emission_models.py`

Five emission distributions score the **same** cells against the **same**
discrete organizational units, so every difference is attributable to the
distribution alone. `diagonal_gaussian` is the shipped MINGL scorer — the "our
GMM" baseline everything is contrasted against.

| panel | content |
|---|---|
| **a** | **Border locations catplot** — a grid of `mg.pl.spatial_loc_region` maps, one per emission model, on the same melanoma region and border: CN1 pink, CN2 blue, border cells red, rest grey. |
| **b** | **Border cell-type composition** for the same border: stacked composition per model (`mg.tl.compute_grouped_proportions`), with Jensen-Shannon divergence from the GMM composition and a region bootstrap written to CSV. |
| **c** | **Assigned neighborhood probability, mapped** — every cell coloured by P(its own annotated neighborhood), one map per model on the same region, so low-probability cells light up at organizational boundaries. |
| **d** | **Assigned neighborhood probability, distributions** — the same quantity as violins for every dataset, with the per-neighborhood paired deltas against the GMM. |

Panels a–c use **one** melanoma border (default *Inflamed Tumor | Productive
T cell & Tumor*) on one region; panel d uses **all** neighborhoods in **each**
dataset. `mg.pl.plot_border_enrichment` is also run per model and saved as
`border_enrichment_dotplot_<model>.png`.

The border-location quantification (distance to the CN1|CN2 interface, region-paired
against the GMM) is computed and written to `border_location_per_region.csv` and
`statistics.csv`, but is **not** currently drawn: `panel_a_location` and
`panel_d_probability_heatmap` are implemented and not called by `build_figure`.

### Figure 2 · `figure2_threshold_effects.py`

Sweeps the threshold across five conditions, one focus border each:

| condition | window | focus border (default) |
|---|---|---|
| intestine **null** | randomized identities | auto-selected |
| intestine · neighborhood | k=10 of cell-type labels | Plasma Cell Enriched \| Outer Follicle |
| intestine · community | k=100 of neighborhood labels | Plasma Cell Enriched \| Secretory Epithelial |
| intestine · tissue unit | k=300 of community labels | Mucosa \| Muscularis Mucosa |
| mouse brain · anatomical region | k=10 of cell-type labels | cortical layer VI \| corpus callosum |

| panel | requested quantity |
|---|---|
| **a** | **number of border cells** — grouped bar chart per threshold. Pair-free (all border cells) so the conditions are comparable; the null is excluded here because it collapses to a handful of cells by t=0.25 and compresses the log axis |
| **b** | **border location, catplot** — `mg.pl.spatial_loc_region` maps of the highlighted condition's border as the threshold moves |
| **c** | **border composition** — stacked composition per threshold (`mg.tl.compute_grouped_proportions`), one focus border per condition |
| **d** | **border cell-type enrichment** — per-cell-type enrichment heat map across thresholds, one focus border per condition |

`mg.pl.plot_border_enrichment` is saved per condition per threshold, and a separate
`enrichment_heatmap_<condition>.png` is written for every condition, so each focus
border is inspectable, not just the highlighted one.

The border-location quantification and the stability curves are computed and written
to `location_per_region_<condition>.csv`, `threshold_composition_enrichment_stability.csv`
and `statistics.csv`, but are **not** currently drawn: `panel_b_location`,
`panel_d_enrichment` and `panel_e_stability` are implemented and not called by
`build_figure`.

### Figure 3 · `figure3_network_graphs.py`

One interaction network per emission distribution — the most direct check that a
distribution has not broken known tissue architecture. Nodes are organizational
units; an edge counts cells co-positive for **exactly two** memberships at the
threshold, the manuscript Figure 3a definition. Both graphs are built and drawn by
the shipped `mingl.tl.build_neighborhood_pair_graph` and
`mingl.tl.plot_neighborhood_pair_graph`; this driver only swaps the posterior
underneath them.

Run on intestine at two levels: neighborhood (top 25 edges, spring layout) and
tissue unit (top 5 edges, circular layout).

```bash
python -m border_figures.figure3_network_graphs \
    --data intestine=/path/05_25_HuBMAP_tunit.csv \
    --levels neighborhood tissue_unit
```

Note the shipped builder reads the posterior from `obs` columns rather than `obsm`;
both are honoured.

---

## Definitions

**Hierarchy levels follow Methods 4.2 exactly.** Each level scores a window of
the *lower* level's labels, with a level-specific `k`:

| level | window feature | k |
|---|---|---|
| cellular neighborhood | cell type | 10 |
| community | neighborhood | 100 |
| tissue unit | community | 300 |

This matters: the existing `tools/reviewer_threshold_sensitivity.py` re-scores
every level with the cell-type window at `k=10`, which is not the hierarchy the
manuscript defines. Composition and enrichment read-outs stay at the **cell-type**
level at every scale, as in Figure 2k.

**Pair-specific border** (Methods 4.5) — for a chosen pair CN1, CN2:

* *border group*: cells positive (`p > threshold`) for **both**,
* *CN1-only* / *CN2-only*: positive for exactly one,
* enrichment `E_t^(CN1) = log2(p_t,border / p_t,CN1-only)`, likewise for CN2, and
  a cell type counts as enriched at the interface only when **both** are positive.
  `min_enrichment = min(E^(CN1), E^(CN2))` turns that criterion into a continuous
  score that can be tracked across models and thresholds.

**Border location** is the one quantity the package did not already have. For the
chosen pair each cell gets

```
s = (d_CN1 − d_CN2) / d_NN
```

where `d_CN1` is the distance to the nearest cell *discretely assigned* to CN1
(self excluded) and `d_NN` is the region's median nearest-neighbor spacing — the
same density normalization the manuscript uses for spatial gradients (Methods
4.7). `s` is negative inside CN1, positive inside CN2, and 0 on the interface, so
`|s|` is "how far is this called border cell from the actual CN1\|CN2 interface,
in cell diameters", comparable across regions, datasets and platforms. It depends
only on the discrete labels and coordinates, so it is computed **once** per
dataset/level and reused for every model and threshold.

**Null condition** (`--null-mode`, default `celltype`) permutes each cell's
identity labels *within each region*. Coordinates, regions, unit labels and all
label abundances are preserved — so the interface stays exactly where it was and
"are border cells still at the interface?" remains answerable — while the local
compositional structure MINGL scores is destroyed. Alternatives: `coordinates`,
`units`, or `--null-data PATH` to use a null file you generated yourself.

---

## Statistics

Cells sharing k-NN windows are not independent, and the same cells are re-scored
by every model and at every threshold. A cell-level test across conditions would
therefore have a wildly inflated n. Every **primary** test is instead **paired on
the experimental unit**:

* **regions** for border location, composition and enrichment (n = 64 intestine,
  31 mouse brain, 21 melanoma),
* **neighborhoods** for the assigned-probability panel (n = 20 intestine, 16 melanoma,
  4 mouse brain). Note these rows are recorded in `statistics.csv` with
  `unit="region"`: `stats.paired_condition_tests` hardcodes that label. The pairing
  is per neighborhood.

With ≥ 3 conditions: Friedman omnibus + Kendall's W, then Wilcoxon signed-rank
against the reference (the GMM in Figure 1, t=0.25 in Figure 2), Holm-corrected,
with the matched-pairs rank-biserial correlation as effect size. Composition
distances get a percentile bootstrap over regions. Cell-level Mann-Whitney /
Kolmogorov-Smirnov / chi-square results are still written out, but every row
carries `unit="cell"` and a note that it is descriptive only.

Everything printed to the terminal is also written to `statistics.csv`, one tidy
row per test with `p_value`, `p_adjusted`, `effect_name`, `effect_size`, `n`,
`unit` and `note`.

---

## Running on the lab data

Paths are read-only; all outputs stay under `border_figures/outputs/`.

```bash
# Figure 1 — panels a-c on one melanoma border, panel d on all three datasets
python -m border_figures.figure1_emission_models \
    --data melanoma=/path/melanoma_all_information.csv \
    --data intestine=/path/05_25_HuBMAP_tunit.csv \
    --data spatial=/path/mousebrain.csv \
    --pair "Inflamed Tumor|Productive T cell & Tumor"
```

```bash
# Figure 2 — null + three intestine levels + melanoma + mouse brain
python -m border_figures.figure2_threshold_effects \
    --data intestine=/path/05_25_HuBMAP_tunit.csv \
    --data spatial=/path/mousebrain.csv \
    --levels spatial=tissue
```

Useful switches:

| flag | why |
|---|---|
| `--pair "CN1\|CN2"` (fig 1), `--pair COND="CN1\|CN2"` (fig 2) | choose the focus border; otherwise the manuscript default, else the pair sharing the most border cells |
| `--subsample-frac`, `--max-cells` | region-stratified thinning for a smoke test (it changes the windows, so not for final numbers) |
| `--melanoma-max-cells` | fig 1 caps the 5.0M-cell melanoma file at 1.5M by default; `0` disables |
| `--models` | restrict Figure 1's models (the Dirichlet-multinomial MLE is the slow one) |
| `--thresholds` | default `0.01 0.05 0.1 0.15 0.2 0.25 0.33 0.4 0.49`; above 0.5 no cell can have two positive memberships |
| `--highlight COND` | which condition's enrichment heat map goes in Figure 2 panel f |
| `--min-border-cells` | a region needs this many border cells to enter the paired statistics (default 20) |
| `--null-data PATH` | use your own intestine null file instead of permuting |

Posteriors are cached under `outputs/_cache/` keyed by dataset, level, model and
cell count, so re-plotting does not re-fit — worth having, since the k=300
tissue-unit windows on 2.5M cells are the expensive step. `--no-cache` disables it.

**Scale.** Intestine (2.5M) and mouse brain (0.38M) run at full size. Melanoma is
5.0M cells with a 1.5M-cell largest region; Figure 1 caps it by default. If memory
is tight, add `--subsample-frac 0.3` for a first pass.

---

## Local validation (no lab data)

```bash
python -m border_figures.figure1_emission_models --synthetic --focus-level tissue_unit
python -m border_figures.figure2_threshold_effects --synthetic
```

`loading.synthetic_tissue` builds a layered tissue that mimics the intestine —
four tissue units split into communities and then neighborhoods, with genuine
mixing zones where the bands meet — so every code path runs, including the null
and all three hierarchy levels.

```bash
python -m pytest border_figures/tests -q      # 40 tests
```

Unit tests pin the border definitions, the Methods 4.5 enrichment formula, the
interface coordinate (sign convention, scale-invariance, absent-unit handling)
and every statistic against hand-computed values. Three end-to-end tests run both
drivers and assert the invariants that matter — border counts never rise with the
threshold, each level uses its own k, and the null's border cells really are
further from the interface than every real condition's.

---

## Files

| file | contents |
|---|---|
| `config.py` | dataset / hierarchy-level / focus-border presets, model names and colors |
| `loading.py` | reading, validation, region-stratified subsampling, null construction, cached posteriors, the synthetic tissue |
| `pair_borders.py` | pair-specific border masks, interface coordinate, composition, Methods 4.5 enrichment, region-level aggregation |
| `stats.py` | Holm correction, paired region/neighborhood tests, effect sizes, JS divergence, bootstrap |
| `plotting.py` | style, cell-type palette (reuses `src/mingl/pl/cell_type_color_map.json`), stacked bars, legends |
| `figure1_emission_models.py` | Figure 1 driver |
| `figure2_threshold_effects.py` | Figure 2 driver |
| `figure3_network_graphs.py` | Figure 3 driver |
| `tests/` | unit + end-to-end tests |

### Outputs

Both drivers write every number behind their figure:

```
outputs/figure1_emission_models/
    figure1_emission_models.png / .pdf
    focus_border_summary.csv               n_border, distances, enrichment counts per model
    border_location_per_region.csv         regions x models (the paired-test input)
    border_composition_by_model.csv        models x cell types
    border_composition_jsd_vs_gmm.csv      JSD + bootstrap CI
    assigned_probability_per_neighborhood.csv
    border_enrichment_<model>.csv          Methods 4.5 enrichment per model
    statistics.csv

outputs/figure2_threshold_effects/
    figure2_threshold_effects.png / .pdf
    enrichment_heatmap_<condition>.png     one per condition
    conditions.csv                         what each condition actually was (level, k, pair, n)
    threshold_summary.csv                  per condition x threshold
    threshold_composition_enrichment_stability.csv
    composition_<condition>.csv  enrichment_<condition>.csv  location_per_region_<condition>.csv
    statistics.csv

outputs/figure3_network_graphs/
    network_<level>_<model>.png            one per level x emission model
    network_summary.csv                    nodes, edges and top edge per graph
```
