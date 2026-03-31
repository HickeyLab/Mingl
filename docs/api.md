# API

```{note}
This repository currently mixes `mingl`, `MINGL`, and `MINGLE` across the source tree, packaging metadata, tests, and tutorial notebooks. The source code audited for this page lives under `src/mingl`, while notebooks usually alias the package as `mg` and sometimes import advanced helpers from submodules such as `MINGL.tl.gmm_gpu`. This page documents the actual API surface discovered in the repository and uses `mg` only as a generic package alias.
```

## How The API Is Organized

MINGL follows the scverse-style split between preprocessing, tools, and plotting:

- `mg.pp` loads tabular or `AnnData` inputs.
- `mg.tl` builds local composition windows, computes centroid models, scores probabilistic memberships, analyzes borders and gradients, and compares local versus global organization.
- `mg.pl` visualizes assigned neighborhoods, overlapping memberships, gradient summaries, and region- or patient-level delta structure.

Across the tutorial notebooks, the most common workflow is:

1. Load a cell table with `mg.pp.read_file`.
2. Choose dataset-specific column names for coordinates, regions, cell types, and hierarchy labels.
3. Summarize local composition with `mg.tl.KNN2` and `mg.tl.centroid_Calculation`.
4. Score each cell against centroid profiles with `mg.tl.cpu_gmm_probability` or the GPU-only `mg.tl.gmm_gpu.gpu_gmm_probability`.
5. Count or compare positive memberships with `mg.tl.findPositives`, graph utilities, or enrichment plots.
6. For transition analyses, derive `Score`, `Probability_Level`, and `Probability_Bin_Cluster` with `mg.tl.mingl_neighborhoods_scverse`, then visualize them with `mg.tl.gb`, `mg.pl.plot_pooled_violin`, and `mg.pl.cell_type_distributions`.
7. For heterogeneity analyses, compute context- or region-specific deltas with `mg.tl.ccd`, `mg.tl.crd`, or `mg.tl.crd2`, then summarize them with `mg.pl.cnd`, `mg.pl.rnd`, `mg.pl.dpp`, `mg.pl.plot_global_vs_subset_horizontal_buckets`, and `mg.pl.plot_log2fc_vs_mean_abundance`.

```{important}
The tutorials consistently treat `k=10` as a neighborhood-scale window, `k=100` as a community-scale window, and `k=300` as a tissue-unit-scale window. Those are usage conventions rather than hard requirements, but they are the defaults around which most examples are written.
```

```{important}
Several functions read probabilities from different places:

- `cpu_gmm_probability` writes the probability matrix to `adata.obsm[prob_key]`.
- `gpu_gmm_probability` writes to `adata.obsm[prob_key]` and also expands probabilities into `adata.obs[...]`.
- Network helpers such as `build_neighborhood_pair_graph` expect explicit probability columns in `adata.obs`.
- Some plotting helpers read from `adata.obsm`, while others can reconstruct a probability table from `adata.obsm` or `adata.obs`.

If you are working with more than one hierarchy level in the same `AnnData`, set `prob_key` and `prob_variable_key` explicitly so the stored probability matrices do not overwrite each other.
```

## `mg.pp`

### `read_file(path)`

Load a `.csv` or `.h5ad` file into an `AnnData` object.

- Use this as the entry point when your spatial table is stored as a flat CSV rather than an existing `AnnData`.
- For `.csv` inputs, the entire table is copied into `adata.obs` and `adata.X` is created as an empty matrix with shape `(n_obs, 0)`.
- For `.h5ad` inputs, the file is returned through `anndata.read_h5ad` without further transformation.
- Raises `ValueError` for unsupported file extensions.

Typical next steps:

- `mg.tl.KNN2` if you want local composition windows.
- `mg.tl.centroid_Calculation` if you already have a hierarchy label such as `Neighborhood`, `Community`, or `Tissue Unit`.

## `mg.tl`

### Local Composition Windows

#### `KNN(adata, ..., ks=(5, 10, 20))`

Build per-cell local composition tables using k-nearest neighbors within each region.

- Reads coordinates from `adata.obs[x_key]` and `adata.obs[y_key]`.
- Keeps neighborhoods inside `adata.obs[region_key]`; cells are not allowed to mix across regions.
- Uses `adata.obs[cluster_col]` as the categorical feature space to count in each local window.
- Returns `dict[int, pandas.DataFrame]` keyed by `k`.
- Each returned table contains one row per cell and one count column per category in `cluster_col`.

`KNN` is the older interface. In this repository, the notebooks rely much more heavily on `KNN2`.

#### `KNN2(adata, ..., ks=(5, 10, 20, 100, 300), keep_obs_cols=None)`

Preferred local-composition builder.

- Reads the same core inputs as `KNN`.
- Dummy-encodes `adata.obs[cluster_col]`, then sums those one-hot columns across each k-nearest-neighbor window.
- Returns `dict[int, pandas.DataFrame]` keyed by `k`.
- If `keep_obs_cols` is provided, those metadata columns are prepended to every returned window table.
- Even when `keep_obs_cols=None`, the returned tables still preserve the coordinate, region, and cluster columns when they exist.
- Handles duplicated `adata.obs` column names by making them unique before processing.

Use `KNN2` when you need:

- reusable per-cell window features for downstream clustering,
- window tables that still carry metadata columns,
- the same workflow used throughout the tutorial notebooks.

#### `centroid_Calculation(adata, ..., k=10, cluster_col="cell_type", neighborhood_col="neighborhood", region_col="unique_region", store_key=None)`

Summarize the mean and standard deviation of local cell-type composition for each annotated organizational unit.

- Internally calls `KNN2` and uses the selected `k` window table.
- Requires `adata.obs[cluster_col]` and `adata.obs[neighborhood_col]`.
- Assumes the coordinate and region columns needed by `KNN2` are present in `adata.obs`.
- Computes one row per unique value in `adata.obs[neighborhood_col]`.
- Produces feature columns of the form `<cell_type>_mean` and `<cell_type>_std`.
- Returns a new `AnnData`:
  - `.obs` is indexed by neighborhood label.
  - `.var_names` are the mean and standard-deviation feature names.
  - `.X` stores the numeric centroid matrix.
- If `store_key` is provided, the returned centroid `AnnData` is also stored in `adata.uns[store_key]`.

Typical next step:

- `mg.tl.cpu_gmm_probability` or `mg.tl.gmm_gpu.gpu_gmm_probability`.

### Probability Scoring And Membership Calls

#### `cpu_gmm_probability(CELLS_ADATA, CENTROIDS_ADATA, ..., ks=(10, 20, 100, 300), k=10, prob_key="neighborhood_probabilities", prob_variable_key="neighborhood_probability_neighborhoods")`

Score each cell against every centroid profile using a Gaussian likelihood model on local-composition features.

- Requires `CELLS_ADATA.obs[cluster_col]` and `CELLS_ADATA.obs[neighborhood_col]`.
- Recomputes local windows with `KNN2`; the chosen `k` must be included in `ks`.
- Uses the centroid `AnnData` returned by `centroid_Calculation`.
- Runs on CPU and can parallelize across cells with `num_processes`.
- Returns the same `AnnData` object after mutating it in place.

Writes:

- `CELLS_ADATA.obsm[prob_key]`: dense per-cell probability matrix.
- `CELLS_ADATA.uns[prob_variable_key]`: ordered list of neighborhood names corresponding to the matrix columns.

Does not write probability columns into `CELLS_ADATA.obs` by default.

Use this when:

- you want the standard notebook workflow,
- you do not have CuPy or a GPU available,
- you want explicit control over output storage keys for multiple hierarchy levels.

#### `mergeGMM(GMM_adata, cell_adata, join="outer")`

Merge a GMM-result `AnnData` with the main cell-level `AnnData` along observations.

- Aligns on `.obs_names`.
- Concatenates along variables with `anndata.concat(..., axis=1)`.
- Returns a new merged `AnnData`; inputs are not modified.

Use this if you have computed probabilities or metadata in a separate `AnnData` and want to attach them back to the main cell object.

#### `findPositives(adata, prob_key="neighborhood_probabilities", threshold=0.25, result_key="Count_Above_Threshold")`

Count how many neighborhood probabilities exceed a threshold for each cell.

- Reads the probability matrix from `adata.obsm[prob_key]`.
- Accepts either a NumPy-like matrix or a `pandas.DataFrame`.
- Writes integer counts to `adata.obs[result_key]`.
- Returns the same `AnnData` object after mutation.

Typical uses:

- identify singly versus multiply positive cells,
- prepare input for `mg.pl.edges_positive_probability`,
- add a quick border-cell proxy before more detailed overlap analysis.

### Border, Pair, And Proportion Analysis

#### `build_neighborhood_pair_graph(adata, prob_cols, ..., threshold=0.25, region_key="unique_region", top_n=15, uns_key="neighborhood_pair_graph")`

Construct a graph of neighborhood pairs that co-occur above a probability threshold.

- Expects `prob_cols` to be explicit columns in `adata.obs`.
- Adds `adata.obs[count_key]` with the number of neighborhoods above threshold per cell.
- Keeps only cells with exactly two positive neighborhoods.
- Counts co-occurring pairs by region, then sums across regions.
- Stores the graph and supporting tables in `adata.uns[uns_key]`.
- Returns `(networkx.Graph, top_pairs_dataframe)`.

Stored in `adata.uns[uns_key]`:

- `"graph"`: weighted `networkx.Graph`.
- `"top_pairs"`: top ranked pair table.
- `"pair_counts"`: region-level counts.
- `"pair_counts_summed"`: counts summed across regions.

This helper is easiest to use after a workflow that has already placed one probability column per neighborhood into `adata.obs`, as the GPU path does by default.

#### `plot_neighborhood_pair_graph(adata, ..., uns_key="neighborhood_pair_graph")`

Visualize the weighted pair graph stored by `build_neighborhood_pair_graph`.

- Reads the graph from `adata.uns[uns_key]`.
- Supports spring and circular layouts.
- Scales edge widths by pair count.
- Optionally draws a custom edge-width legend through `edge_legend_values`.
- Produces a Matplotlib plot and does not return a structured result.

#### `compute_grouped_proportions(df_or_adata, n1, n2, ..., cell_type_col="Cell Type", threshold=0.25, prob_key="neighborhood_probabilities")`

Summarize cell-type proportions across three border subsets:

- `n1 only`
- `n1 + n2`
- `n2 only`

Behavior:

- Accepts either an `AnnData` or a `pandas.DataFrame`.
- When given an `AnnData`, it first reconstructs a probability table with `build_df_probs_from_adata`.
- Returns a tidy `pandas.DataFrame` with columns `[cell_type_col, "Subset", "Proportion"]`.

This is a data-preparation utility used by the border-enrichment plotting code and can also be useful on its own when you want publication-ready subset summaries without plotting.

#### `build_df_probs_from_adata(adata, prob_key="neighborhood_probabilities", cell_type_col="Cell Type")`

Advanced helper for reconstructing a probability table from an `AnnData`.

- First looks for `adata.obsm[prob_key]`.
- If that is missing, scans other `obsm` entries.
- If no probability-like `obsm` entry exists, scans `adata.obs` for probability-like columns.
- Uses `adata.uns["neighborhood_probability_neighborhoods"]` when available to recover matrix column names.
- Returns `(df_probs, numeric_df)`, where `df_probs` may also include the detected cell-type column.

Use this when you want plotting helpers to work even if probabilities were stored under slightly different keys or expanded into `adata.obs`.

### Gradient And Transition Analysis

#### `mingl_neighborhoods_scverse(adata, ..., tu1="Inner Follicle", tu2="Outer Follicle", k=20, target_neighborhoods=("Inner Follicle", "Outer Follicle"))`

High-level transition-gradient workflow for comparing two organizational units.

- Expects probability columns named `tu1` and `tu2` in `adata.obs`.
- Also expects the metadata listed in `extra_cols`; by default these include spatial coordinates, region, and hierarchy labels such as `Neighborhood`, `Cell Type`, `Community`, and `Tissue Unit`.
- Computes:
  - a probability ratio,
  - `log_ratio`,
  - `max_prob`,
  - a combined `Score`,
  - categorical `Probability_Level`.
- Builds windows over `Probability_Level`, clusters the requested `target_neighborhoods`, and assigns transition-cluster labels.
- Returns `(adata, df_sub, fc, windows_k, edges_used)` while also mutating `adata`.

Writes to `adata.obs`:

- `out_score_key` (default `Score`)
- `out_prob_level_key` (default `Probability_Level`)
- `out_neighborhood_key` (default `neighborhood{k}`)
- `out_prob_cluster_key` (default `Probability_Bin_Cluster`)

Writes to `adata.uns`:

- `"mingl_probability_edges"`
- `store_windows_key`
- `store_fc_key`

Typical next steps:

- `mg.tl.gb`
- `mg.pl.plot_pooled_violin`
- `mg.pl.cell_type_distributions`

#### `gb_prob_bin_cluster_plots(adata, ..., cluster_key="Probability_Bin_Cluster", score_key="Score", neighborhood_key="Neighborhood", out_prefix="pb")`

Create the first two score-driven summary plots used in the gradient notebooks.

- Reads cluster assignments, scores, and neighborhood labels from `adata.obs`.
- Ranks transition clusters by weighted score-bin composition.
- Produces:
  - a bar plot comparing the proportion of two neighborhoods within each ranked cluster,
  - a pooled violin plot of `Score` across those clusters.
- Stores ranking tables and metadata in `adata.uns`.
- Returns a dictionary containing the DataFrames, ordering information, metadata, and created figures.

Writes to `adata.uns`:

- `"{out_prefix}_rank_df"`
- `"{out_prefix}_agg_df"`
- `"{out_prefix}_cluster_order_global"`
- `"{out_prefix}_cluster_order_plot"`
- `"{out_prefix}_meta"`

#### `gb_local_score_gradients(adata, ..., region_key="unique_region", score_key="Score", out_prefix="grad")`

Estimate local spatial gradients of a scalar score within one region or across all cells.

- Requires coordinate columns in `adata.obs`.
- Reads the score either from `adata.obs[score_key]` or from a layer.
- Fits a local linear model against each cell's nearest neighbors.
- Writes per-cell gradient vectors and magnitudes back into `adata.obs`.
- Stores summary statistics and plotting parameters in `adata.uns`.
- Returns a dictionary with summary data, parameter metadata, and figures.

Writes to `adata.obs`:

- `"{out_prefix}_x"`
- `"{out_prefix}_y"`
- `"{out_prefix}_mag"`
- `"{out_prefix}_mag_norm"`
- `"{out_prefix}_valid"`

Writes to `adata.uns`:

- `"{out_prefix}_summary"`
- `"{out_prefix}_params"`

#### `gb(adata, ..., pb_prefix="pb", grad_prefix="grad")`

Convenience wrapper that runs both `gb_prob_bin_cluster_plots` and `gb_local_score_gradients`.

- This is the main entry point used in the transition-gradient notebooks.
- Returns `{"plot12": ..., "plot3": ...}`.
- Mutates `adata` through the two wrapped functions.

### Choosing A Neighborhood Count

#### `run_mingl_over_n_clusters(adata, knn_feature_cols, ..., n_range=range(1, 51), results_uns_key="mingl_n_clusters")`

Benchmark multiple cluster counts for a local-composition feature space.

- Expects `knn_feature_cols` to already exist in `adata.obs`.
- The figure 6 notebook constructs these columns from a `KNN2` window table and then wraps that table in a fresh `AnnData`.
- Standardizes the feature columns, runs `MiniBatchKMeans` for each `n`, computes centroid likelihoods, and records per-cell assigned probabilities.
- Returns either:
  - `(summary_df, per_cell_df)` if `return_per_cell=True`, or
  - `summary_df` otherwise.

Writes to `adata.obs` for each tested `n`:

- `Neighborhood_{n}`
- `log_likelihood_n{n}`
- `assigned_prob_n{n}`

Writes to `adata.uns[results_uns_key]`:

- `"summary_df"`
- `"knn_feature_cols"`

Typical next steps:

- `find_elbow_point`
- `find_best_unsupervised_plateau`
- `plot_stable_composite`

#### `find_elbow_point(y_values, ..., adata=None, uns_key=None, y_key=None, x_key=None)`

Detect an elbow or flattening point in a one-dimensional summary curve.

- Can be called on arrays directly.
- Can also read values out of `adata.uns[uns_key]`, including nested `{"summary_df": ...}` payloads.
- Returns `(elbow_idx, x_at_elbow, slope)`.

The notebooks use it separately on average log-likelihood and average assigned-probability curves.

#### `find_best_unsupervised_plateau(log_likelihoods, assigned_probs, ..., adata=None, uns_key=None, ll_key=None, prob_key=None, out_uns_key=None)`

Combine normalized log-likelihood and assignment-probability curves to choose a stable cluster-count plateau.

- Accepts arrays directly or pulls both curves from `adata.uns[uns_key]`.
- Supports harmonic or weighted composite scoring.
- Returns `(composite_df, best_n, ranked_plateaus)`.
- If `out_uns_key` is provided, stores those objects in `adata.uns[out_uns_key]`.

#### `plot_stable_composite(df, best_n, ll_n=None, prob_n=None)`

Plot the composite score, composite slope, and selected cluster count.

- Accepts the `composite_df` returned by `find_best_unsupervised_plateau`.
- Optionally annotates elbow points for the log-likelihood and assignment-probability curves.
- Returns `(figure, legend_figure)`.

### Context- And Region-Specific Delta Analysis

#### `ccd(cells_path, probs_paths, pp, ..., assigned_neigh_key="neigh_name", min_count=10)`

File-driven context-versus-combined delta pipeline.

- Loads cells through `pp.read_file`.
- Requires `probs_paths["combined"]`.
- Optionally consumes context-specific probability CSVs for tumor, normal, metaplasia, and dysplasia.
- Computes `context - combined` probability deltas.
- Keeps only the assigned neighborhood for each cell.
- Returns `(adata, combined_melted_filtered, combo_counts)`.

Writes into the returned `AnnData`:

- `adata.var_names`: neighborhood names.
- `adata.layers["delta_<Context>"]`: wide delta matrices, one layer per context.
- `adata.obs["region_full"]`: region parsed from `cellid`.

Use this when your inputs are still on disk as CSV tables and you want a compact `AnnData` container with per-context delta layers.

#### `crd(cells, windows2, probabilities_df, cell_type_features, ..., out_probs_path="local_probs.csv", out_delta_path="delta_probs.csv")`

Compare region-local centroid models against a global probability model.

- Expects:
  - a cell-level `AnnData`,
  - a selected `KNN2` window table,
  - a global probability table,
  - the list of cell-type feature columns used for scoring.
- Fits region-specific centroid profiles and recomputes local probabilities.
- Saves local probabilities and local-minus-global deltas to CSV.
- Returns `(final_probs_df, final_deltas_df)`.

This is the original region-comparison path. The `crd2` variant is the more robust notebook workflow in this repository.

#### `crd2(..., windows2, probabilities_df, assigned_df, neighborhoods, cell_type_features, copy_cells, min_count=10, min_floor_abs=0.05, use_region_frac=0.25, shrink_alpha=5.0)`

Stabilized region-versus-global comparison with more explicit controls for sparse neighborhoods.

- Works from DataFrames rather than directly from an `AnnData`.
- Shrinks region-level standard deviations toward region-wide variation.
- Excludes underpopulated neighborhoods with `min_count`.
- Saves local probabilities and deltas to CSV.
- Returns `(final_probs_df, final_deltas_df)`.

Typical next steps:

- `mg.pl.rnd` for region-by-neighborhood dot plots.
- `mg.pl.dpp` for patient-level divergence summaries.

## `mg.pl`

### Spatial Assignment And Border Plots

#### `spatial_neighborhood_plot(adata, *, desired_region, prob_key="neighborhood_probabilities", neighborhood_key="neighborhood", region_key="unique_region")`

Plot cells from one region colored by their assigned neighborhood.

- Requires coordinates in `adata.obs[x_key]` and `adata.obs[y_key]`.
- Requires assigned labels in `adata.obs[neighborhood_key]`.
- Requires a probability matrix in `adata.obsm[prob_key]`.
- Filters to `desired_region` and returns a Matplotlib `Axes`.
- Does not mutate `adata`.

Use this as the standard single-region assignment map after `cpu_gmm_probability`.

#### `edges_positive_probability(adata, ..., prob_key="neighborhood_probabilities", threshold=0.25)`

Plot the ranked positive neighborhood probabilities for cells that exceed a threshold.

- Calls `mg.tl.findPositives` internally.
- Mutates `adata.obs["Count_Above_Threshold"]`.
- Reconstructs a long table of `Prob1`, `Prob2`, and so on for each positive cell.
- Returns a Matplotlib `Axes`.

This is the plot used in the melanoma and figure 2 border workflows to compare cells with one, two, or more positive neighborhood calls.

#### `spatial_loc_region(adata, *, region, n1, n2, threshold=0.25, region_key="filename")`

Visualize where cells are positive for one, the other, or both of two neighborhoods.

- Reads probabilities from `adata.obsm["neighborhood_probabilities"]`.
- Reads neighborhood names from `adata.uns["neighborhood_probability_neighborhoods"]`.
- Uses `adata.obs[region_key]`, `adata.obs[x_col]`, and `adata.obs[y_col]` for subsetting and coordinates.
- Returns `(figure, axes, masks)` where `masks` contains boolean arrays for:
  - `other`
  - `only_1`
  - `only_2`
  - `both`

```{warning}
`spatial_loc_region` is less configurable than most other plotting helpers because it currently assumes the default probability keys shown above. If you stored probabilities under different keys, align them first or copy them into the expected locations.
```

#### `plot_border_enrichment(..., adata=None, df_probabilities=None, n1, n2, cell_type_col="Cell Type", pos_threshold=0.25, prob_key="neighborhood_probabilities")`

Generate the border-enrichment summary used throughout the melanoma and tissue-unit notebooks.

- Accepts either:
  - an `AnnData`, or
  - a prebuilt probability DataFrame.
- Computes grouped cell-type proportions for `n1 only`, `n1 + n2`, and `n2 only`.
- Also computes a per-cell count of how many neighborhoods exceed a separate threshold used for neighborhood-count analysis.
- Returns three figures:
  - the main enrichment scatter,
  - a size legend figure,
  - a color legend figure.

Use this when you want to identify cell types enriched in singly positive versus shared-positive border populations.

#### `spatial_probability_mapping(adata, centroids, cell_type_features, ..., k=300, desired_region="B008_Sigmoid")`

Notebook-oriented helper that recomputes local windows, evaluates centroid probabilities, and maps the assigned probability back into spatial coordinates for one region.

- Expects many hard-coded column names by default, including `Community`, `Tissue Unit`, `Neighborhood`, `Cell Type`, `x`, `y`, and `unique_region`.
- Treats `centroids` as a DataFrame-like object with columns such as `<cell_type>_mean`, `<cell_type>_std`, and `"Tissue Unit"`.
- Returns `(probabilities_df, visualization_df, filtered_region_df)`.
- Does not mutate `adata`.

This helper is useful for reproducing the figure 2 tissue-unit notebook but is less general than the other plotting functions. For a standard assignment map, prefer `spatial_neighborhood_plot`.

### Transition-Gradient Visualization

#### `plot_pooled_violin(adata, neighborhood_key, ..., neighborhoods_to_plot=("Inner Follicle", "Outer Follicle"), cluster_key="Probability_Bin_Cluster", score_key="Score")`

Plot pooled score distributions across transition clusters.

- Requires:
  - `adata.obs[cluster_key]`
  - `adata.obs[score_key]`
  - `adata.obs[neighborhood_key]`
- Ranks transition clusters using binned `Score`.
- Keeps only clusters with at least `min_cells` in every selected neighborhood.
- Returns `(ax, rank_df)`.

The returned `Axes` also carries convenience attributes such as `_mingl_rank_df`, `_mingl_bin_edges`, and `_mingl_cluster_order`.

#### `cell_type_distributions(adata, ..., neighborhoods_to_plot=("Inner Follicle", "Outer Follicle"), cluster_key="Probability_Bin_Cluster", score_key="Score", cell_type_key="Cell Type", neighborhood_key="Neighborhood")`

Plot stacked cell-type composition across ranked transition clusters.

- Requires the score, cluster, cell-type, and neighborhood columns in `adata.obs`.
- Filters to clusters with at least `min_cells` in each requested neighborhood.
- Computes cell-type percentages within the pooled subset of those neighborhoods.
- Returns:
  - `(ax, rank_df, combined_perc)` by default, or
  - `(fig, ax, rank_df, combined_perc)` when `return_fig=True`.
- Optionally stores summary metadata in `adata.uns[store_key]`.

This is the main composition summary paired with `gb` in the figure 4 notebooks.

### Region- And Patient-Level Delta Plots

#### `rnd(delta, ..., region_col="region", neighborhood_col="Neighborhood", delta_col="Delta")`

Plot region-by-neighborhood delta structure as a dot matrix with a context-colored side bar.

- Accepts either:
  - a long-form delta DataFrame,
  - a wide delta DataFrame with `*_delta` columns,
  - or a CSV path to one of those tables.
- Can derive region labels from `cellid` when needed.
- Filters to region-neighborhood combinations with at least `min_cells_per_region_neigh` cells.
- Returns `(fig, ax, fig_cb, plot_df)`.

Use this for region-level heterogeneity maps after `crd2` or `cnd`.

#### `dpp(delta, ..., neighborhood_col="Neighborhood", delta_col="Delta")`

Aggregate region-level deltas into patient-level divergence summaries.

- Accepts long- or wide-format delta tables or a CSV path.
- Computes region-level mean deltas, then sums their absolute values by patient.
- Produces:
  - a stacked total-divergence bar figure,
  - an average-divergence bar figure.
- Returns a dictionary containing figures, axes, ordering tables, and intermediate DataFrames.

This is the patient-level companion to `rnd`.

#### `cnd(..., cells_path, probs_paths, assigned_neigh_key="neigh_name", min_count=10, make_plot=True)`

High-level context-delta pipeline plus plot.

- Reads cells with `mg.pp.read_file`.
- Loads combined and context-specific probability CSVs.
- Computes assigned-neighborhood deltas by context.
- Melts and filters the result.
- When `make_plot=True`, draws the manuscript-style context delta summary.
- Returns a dictionary with:
  - `adata`
  - `delta_wide_by_context`
  - `combined_melted`
  - `combo_counts`
  - `fig`
  - `fig_cb`

Use `cnd` when you want the full file-driven context pipeline in one call rather than separately invoking `ccd` and plotting helpers.

### Composition Shift Plots For Specific Neighborhoods

#### `plot_global_vs_subset_horizontal_buckets(data, neighborhood, bucket_map, cell_type_color_map, ..., subset_region=None, subset_patient=None, subset_context=None)`

Compare global versus subset cell-type composition within one neighborhood using horizontal stacked bars.

- Accepts either an `AnnData` or a `pandas.DataFrame`.
- Filters to one neighborhood, then compares the global distribution against one requested subset:
  - a region,
  - a patient,
  - or a context.
- `bucket_map` groups cell types into user-defined categories such as epithelial, mesenchymal, and immune.
- Plots directly and does not return a structured object.

Tutorial note:

- Some notebook code accesses this helper through a submodule import such as `mg.pl.gvs.plot_global_vs_subset_horizontal_buckets`.
- The callable documented here is the function itself.

#### `auto_assign_buckets(unique_cts)`

Rule-based helper for splitting cell types into three coarse buckets:

- epithelial
- mesenchymal
- immune

Returns `(epithelial, mesenchymal, immune)` as three lists of matched labels.

Use this to bootstrap `bucket_map` for the horizontal-composition and log-fold-change plots.

#### `plot_log2fc_vs_mean_abundance(data, neighborhood, bucket_map, cell_type_color_map, ..., subset_region=None, subset_patient=None, subset_context=None)`

Plot log2 fold change versus mean abundance for cell types within one neighborhood.

- Accepts either an `AnnData` or a `pandas.DataFrame`.
- Compares one selected region, patient, or context against the global composition in the same neighborhood.
- Computes:
  - subset percentage,
  - global percentage,
  - mean percentage,
  - log2 fold change.
- Returns the plotting DataFrame used to draw the figure.

Tutorial note:

- Some notebook code accesses this helper through a submodule import such as `mg.pl.dv.plot_log2fc_vs_mean_abundance`.
- The documented callable is the underlying function itself.

## Advanced And Submodule-Specific APIs

### `mg.tl.gmm_gpu.gpu_gmm_probability`

GPU-accelerated probability scoring, imported explicitly from the `gmm_gpu` submodule in the tutorials.

- Requires CuPy and a working GPU environment.
- Recomputes windows with `KNN2`, just like the CPU path.
- Returns the same `AnnData` after mutation.

Writes:

- `cells.obsm[prob_key]`
- `cells.uns[prob_variable_key]`
- probability columns directly into `cells.obs[...]`

Important differences from the CPU path:

- The default `prob_key` is `"neighborhood_probability"` rather than `"neighborhood_probabilities"`.
- Because it expands probabilities into `adata.obs`, it plugs more directly into `build_neighborhood_pair_graph`.

### `Neighborhoods`

Lower-level window-construction class used internally by `mingl_neighborhoods_scverse`.

- Builds neighborhood windows from DataFrames rather than from an `AnnData`.
- Supports distance-based masking through `k_windows(distance_max=...)`.

Most users should prefer `KNN2` or `mingl_neighborhoods_scverse`, but this class is currently importable and can be useful for custom experiments.

### `assign_probability_level_with_edges(series, n_bins=5, labels=None, use_quantiles=True)`

Low-level helper that bins a continuous score into ordered probability levels.

- Returns `(categorical_series, edges)`.
- Used internally by `mingl_neighborhoods_scverse`.

### `find_prob_col(df, name)` and `make_celltype_palette_from_adata(cell_types)`

Helper functions used by the enrichment plotting workflow.

- `find_prob_col` fuzzy-matches a requested neighborhood name to a probability-like column.
- `make_celltype_palette_from_adata` builds a palette suitable for border-enrichment plots.

These are useful when extending the packaged notebook analyses, but they are not the main user-facing entry points.
