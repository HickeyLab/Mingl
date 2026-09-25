# Neighborhood-Number Benchmarking

This folder contains the analyses used to evaluate the number of cellular neighborhoods (`N`) identified from local cell-type composition. The workflow compares MINGL's biologically motivated resolution-selection approach with classical clustering metrics across healthy human intestine, human melanoma, and mouse brain spatial datasets.

## Files

### MINGL neighborhood-number analyses

- `intestine_n_neighborhoods.ipynb` — Runs the complete MINGL neighborhood-number workflow for the healthy human intestine dataset.
- `melanoma_n_neighborhoods.ipynb` — Runs the corresponding workflow for the human melanoma dataset.
- `mousebrain_n_neighborhoods.ipynb` — Runs the corresponding workflow for the mouse brain spatial-transcriptomics dataset.

These notebooks construct local cell-type-composition windows, cluster the cells across candidate values of `N`, calculate MINGL probability and log-likelihood metrics, identify elbows, select a stable intermediate resolution, and generate dataset-specific biological comparisons.

### Classical clustering benchmarks

- `silhouettescore_intestine.ipynb` — Compares MINGL resolution selection with classical clustering metrics for healthy intestine.
- `silhouettescore_melanoma.ipynb` — Performs the same benchmark for human melanoma.
- `silhouettescore_mousebrain.ipynb` — Performs the same benchmark for mouse brain.

These notebooks use a standardized, checkpointed workflow to calculate silhouette, Davies–Bouldin, Calinski–Harabasz, pairwise Jaccard, and Fowlkes–Mallows metrics across candidate neighborhood numbers.

## Datasets and required columns

| Dataset | Cell-type column | Region column | Reference annotation |
|---|---|---|---|
| Healthy intestine | `Cell Type` | `unique_region` | `Neighborhood` |
| Human melanoma | `Cell_Type` | `filename` | `Neighborhood` |
| Mouse brain | `cell_type` | `unique_region` | `tissue` |

All datasets must also contain numeric spatial-coordinate columns named `x` and `y`.

The notebooks currently use machine-specific input paths:

```text
Healthy intestine:
Z:\MINGLE\Data\Intestine\05_25_HuBMAP_tunit.csv

Human melanoma:
Z:\MINGLE\Data\Melanoma\23_10_11_Melanoma_Marker_Cell_Neighborhood.csv

Mouse brain:
Z:\MINGLE\Reviewer Response\Data\Spatial Transcriptomics\MouseBrainSpatialTranscriptomics_centroids.h5ad
```

Update these paths before running the notebooks on another computer.

## MINGL resolution-selection workflow

The three `*_n_neighborhoods.ipynb` notebooks follow the same primary workflow:

1. Read the spatial dataset and validate the required annotations.
2. Construct local cell-type-composition windows with `mg.tl.KNN2`.
3. Use a local neighborhood size of `k=10`.
4. Evaluate candidate neighborhood numbers from `N=1` through `N=50`.
5. Fit cluster assignments and calculate the average GMM log likelihood and average assigned probability for each value of `N`.
6. Identify elbows in the log-likelihood and probability curves.
7. Search for a stable intermediate plateau between the elbow-defined bounds.
8. Generate spatial maps, cell-type enrichment heatmaps, and probability and log-likelihood summaries at selected resolutions.
9. Compare selected clusters with expert-provided tissue annotations.
10. Evaluate stability under cell- or region-level subsampling.

The main MINGL functions used are:

```python
mg.tl.KNN2(...)
mg.tl.run_mingl_over_n_clusters(...)
mg.tl.find_elbow_point(...)
mg.tl.find_best_unsupervised_plateau(...)
mg.tl.plot_stable_composite(...)
```

The log-likelihood elbow identifies the point after which adding more neighborhoods provides diminishing improvement in compositional fit. The assigned-probability elbow identifies the point after which additional neighborhoods reduce assignment confidence. The plateau analysis combines these signals to select a stable intermediate resolution rather than relying on a single clustering index.

## Classical clustering benchmark

The three `silhouettescore_*.ipynb` notebooks use the same configuration:

```python
K = 10
N_RANGE = list(range(1, 51))
SAMPLE_SIZE = 5000
SEEDS = [42, 43, 44]
CPU_THREADS = 8
```

The metrics are interpreted as follows:

| Metric | Preferred value | Purpose |
|---|---|---|
| Silhouette score | Maximum | Measures within-cluster cohesion relative to separation |
| Davies–Bouldin index | Minimum | Measures similarity between each cluster and its most similar alternative |
| Calinski–Harabasz score | Maximum | Measures between-cluster dispersion relative to within-cluster dispersion |
| Pairwise Jaccard similarity | Maximum | Measures agreement with the original expert annotation |
| Fowlkes–Mallows score | Maximum | Measures pairwise agreement with the original expert annotation |

Silhouette, Davies–Bouldin, and Calinski–Harabasz are calculated from the local cell-type-composition feature space. Jaccard and Fowlkes–Mallows compare each candidate clustering with the original dataset-specific reference annotation.

Because silhouette calculations are computationally expensive for large spatial datasets, the benchmark uses a reproducible 5,000-cell sample with three random seeds. The same saved sample positions are reused for the other feature-space metrics.

## How to run

### Dataset-specific MINGL analysis

Open the appropriate `*_n_neighborhoods.ipynb` notebook and:

1. Update the input path.
2. Confirm the cell-type and region-column names.
3. Run the notebook from the beginning.
4. Review the two MINGL elbows and the plateau-selected value of `N`.
5. Update dataset-specific regions, cluster numbers, and biological annotations in the plotting sections.

### Standardized benchmark analysis

Open the matching `silhouettescore_*.ipynb` notebook and:

1. Update `ROOT` and the dataset input path.
2. Confirm that `DATASET` is set correctly:
   
   ```python
   DATASET = "intestine"
   DATASET = "melanoma"
   DATASET = "mousebrain"
   ```

3. Run the notebook in order.
4. Allow the notebook to reuse its saved checkpoints if they already exist.
5. Review the generated metric summaries and selected resolutions.

The notebooks save expensive intermediate calculations in:

```text
features_and_annotations.pkl
clustering_results.pkl
```

Only load these pickle checkpoints if they were generated locally or come from a trusted source.

## Main outputs

Each standardized benchmark creates a dataset-specific output directory containing files such as:

```text
features_and_annotations.pkl
clustering_results.pkl
mingl_summary.csv
composite_scores.csv
ranked_plateaus.csv
mingl_selections.json
silhouette_sample_*.csv
silhouette_*.csv
reference_agreement_metrics.csv
selected_cluster_numbers.csv
silhouette_repeat_selections.csv
best_resolution_by_metric.csv
```

The output tables contain:

- MINGL average log likelihood and assigned probability across values of `N`.
- Log-likelihood and probability elbow locations.
- The MINGL plateau-selected resolution.
- Repeated silhouette estimates.
- Davies–Bouldin and Calinski–Harabasz values.
- Jaccard and Fowlkes–Mallows agreement with expert annotations.
- The resolution selected by each metric.

## Dataset-specific biological analyses

The notebooks also contain figure-generation and biological-validation sections, including:

- Spatial maps at the MINGL elbows and selected intermediate resolution.
- Cell-type enrichment heatmaps.
- Average probability and log-likelihood summaries.
- Comparison with expert-annotated neighborhoods or tissue regions.
- Cell-type-composition comparisons for selected organizational structures.
- Jensen–Shannon similarity between selected clusters and expert annotations.
- Stability analyses based on cell or tissue-region subsampling.

These sections require dataset-specific neighborhood numbers, tissue regions, and cluster identities. Review these settings before rerunning them.

## Dependencies

The analyses require Python with:

- `mingl`
- `anndata`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scipy`
- `tqdm`
- `threadpoolctl`

GPU-enabled MINGL calculations may additionally require a compatible CuPy and CUDA installation.

## Important notes

- Run notebook cells in order because later sections reuse objects created earlier.
- The notebooks evaluate `N=1–50`, but most classical clustering metrics require at least two clusters and therefore begin at `N=2`.
- Preserve cell ordering between the KNN feature matrix, cluster assignments, and reference annotations.
- Some headings in the melanoma and mouse-brain notebooks still refer to the intestine because the notebooks were adapted from a shared template. The configured input path and column names determine the dataset actually analyzed.
- Some later plotting cells in the standardized benchmark notebooks contain hard-coded example values, such as `MINGL_SELECTED_N`, `SELECTED_N`, region names, or intestine-specific biological labels. Update these before using those cells for melanoma or mouse brain.
- Do not compare raw metric magnitudes across metrics because each metric has a different scale and optimization direction.
- Clear saved notebook outputs before committing updated notebooks if smaller GitHub files are desired. Clearing outputs does not modify the code or Markdown cells.
