# Mouse Brain Spatial-Transcriptomics Analyses

This folder contains the MINGL analyses used to characterize organizational borders, interaction networks, and continuous gradients in a mouse brain spatial-transcriptomics dataset.

The analyses use annotated anatomical regions as organizational units and local cell-type composition as the feature space for estimating per-cell compatibility with each region.

## Files

- `MouseBrain_MINGL_Border_Network_Results.ipynb` — Calculates mouse brain anatomical-region probabilities, identifies multi-region border cells, summarizes border-associated cell types, and constructs a network of anatomical-region relationships.
- `MouseBrain_MINGL_Gradient_Results.ipynb` — Measures the continuous transition between the corpus callosum and cortical layer VI, divides that transition into ordered probability-bin clusters, and calculates spatial, compositional, and steepness summaries.

## Input data

Both notebooks use the mouse brain spatial-transcriptomics dataset:

```text
Z:\MINGLE\Reviewer Response\Data\Spatial Transcriptomics\MouseBrainSpatialTranscriptomics.h5ad
```

A processed version containing MINGL centroids and probabilities is written to or loaded from:

```text
Z:\MINGLE\Reviewer Response\Data\Spatial Transcriptomics\20260813_mousebrain_centroids.h5ad
```

Update these paths before running the notebooks on another computer.

## Required observation metadata

The AnnData object must contain the following columns in `adata.obs`:

| Column | Meaning |
|---|---|
| `center_x`, `center_y` | Original spatial coordinates |
| `x`, `y` | Renamed coordinates used by MINGL |
| `cell_type` | Cell-type annotation used to construct local composition vectors |
| `tissue` | Anatomical-region annotation modeled by MINGL |
| `donor_id` | Mouse brain donor identifier |
| `slice` | Tissue-slice identifier |
| `fov` | Field-of-view identifier |

The preprocessing cells create:

```python
unique_region = donor_id + "_slice" + slice
```

This ensures that nearest-neighbor windows do not cross donor or tissue-slice boundaries.

The notebooks reset the observation index because the original cell identifiers are unusually large. Maintain the same row order across `cells.obs`, probability matrices, and derived dataframes.

## Anatomical regions

The analyses use the following eight anatomical-region annotations:

```text
olfactory region
corpus callosum
striatum
cortical layer VI
pia mater
brain ventricle
cortical layer V
cortical layer II/III
```

## Border and network analysis

`MouseBrain_MINGL_Border_Network_Results.ipynb` performs the following steps:

1. Loads and preprocesses the mouse brain AnnData object.
2. Converts the spatial coordinates to numeric values.
3. Creates a unique donor-and-slice region identifier.
4. Renames `center_x` and `center_y` to `x` and `y`.
5. Defines:
   
   ```python
   cluster_col = "cell_type"
   neighborhood_col = "tissue"
   region_key = "unique_region"
   ```

6. Calculates cell-type-composition centroids for each anatomical region using `k=10`.
7. Calculates per-cell anatomical-region probabilities using the GPU-accelerated GMM implementation.
8. Counts the number of anatomical regions with probability greater than `0.25` for each cell.
9. Classifies cells with substantial probability in multiple regions as border cells.
10. Summarizes border-cell abundance by cell type and original anatomical annotation.
11. Counts frequently co-occurring anatomical-region pairs.
12. Constructs a weighted anatomical-region interaction network.
13. Examines the detailed border between the corpus callosum and cortical layer VI.

The main probability calculation stores:

```python
cells.obsm["neighborhood_probability"]
cells.uns["neighborhood_probability_neighborhoods"]
```

The positive-membership count is stored as:

```python
cells.obs["Count_Above_Threshold"]
```

The default positive-membership threshold is:

```python
threshold = 0.25
```

### Border-cell interpretation

A cell is considered positive for an anatomical region when its probability for that region exceeds the selected threshold.

- `Count_Above_Threshold == 1` indicates compatibility with one region.
- `Count_Above_Threshold > 1` indicates compatibility with multiple regions and is used to identify organizational border cells.

The notebook generates:

- Probability-distribution summaries for cells positive in one, two, or three regions.
- Cell-type-specific border-cell proportions.
- Anatomical-region-specific border-cell proportions.
- Counts of co-positive anatomical-region pairs.
- Bar plots of the most frequent region pairs.
- Weighted anatomical-region interaction networks.
- Cell-type enrichment at the corpus callosum–cortical layer VI border.
- Spatial maps of border cells within selected donor/slice regions.

## Gradient analysis

`MouseBrain_MINGL_Gradient_Results.ipynb` characterizes the continuous transition between:

```python
tu1 = "cortical layer VI"
tu2 = "corpus callosum"
```

The notebook:

1. Loads the processed mouse brain AnnData object.
2. Calculates or retrieves per-cell anatomical-region probabilities.
3. Extracts the probabilities for cortical layer VI and corpus callosum.
4. Calculates a probability ratio:
   
   ```python
   ratio = (P_cortical_layer_VI + eps) / (P_corpus_callosum + eps)
   ```

5. Calculates the log ratio and weights it by the larger of the two probabilities:
   
   ```python
   Score = log_ratio * max_probability
   ```

6. Divides the continuous score into five equal-width probability levels.
7. Constructs local windows using `k=20`.
8. Clusters the probability-level composition vectors into five ordered gradient groups.
9. Calculates cell-type enrichment and fold changes along the transition.
10. Generates spatial maps of the gradient groups.
11. Runs `mg.tl.gb` to calculate ordered probability-bin summaries and local gradient magnitudes.
12. Generates stacked and cell-type-specific composition plots.
13. Calculates a global transition steepness score.

### Gradient-score interpretation

The score is defined using cortical layer VI in the numerator and corpus callosum in the denominator.

- Positive scores indicate stronger cortical layer VI compatibility.
- Negative scores indicate stronger corpus callosum compatibility.
- Scores near zero indicate mixed or intermediate organizational compatibility.
- Larger absolute changes between ordered groups indicate a sharper transition.

The detailed gradient analysis is currently focused on:

```text
MsBrainAgingSpatialDonor_12_slice0
```

Update `target_region` and `region_value_gb` to analyze another donor or slice.

## Main adjustable parameters

### Both notebooks

- `file_path` — Input dataset path.
- `cluster_col` — Local composition feature; currently `cell_type`.
- `neighborhood_col` — Anatomical label; currently `tissue`.
- `region_key` — Boundary-preserving sample identifier; currently `unique_region`.
- `k` — Number of local neighbors used for centroid and probability calculations.
- `batch_size` — GPU processing batch size.
- Probability storage keys in `adata.obsm` and `adata.uns`.

### Border analysis

- `threshold` — Positive-membership threshold; currently `0.25`.
- `N1` and `N2` — Anatomical-region pair used for detailed border analysis.
- `desired_region` — Donor/slice selected for spatial plotting.
- Number of top anatomical-region pairs retained in network plots.
- `min_count` — Minimum cell-type count used in enrichment plots.

### Gradient analysis

- `tu1` and `tu2` — Anatomical regions defining the transition.
- `n_bins` — Number of initial probability-score bins; currently `5`.
- `k` — Local window size for the gradient clustering; currently `20`.
- `clusters` — Number of final ordered gradient groups; currently `5`.
- `target_region` — Donor/slice used for spatial plotting.
- `k_neighbors` — Local neighborhood size used by `mg.tl.gb`; currently `20`.
- `normalize_by` — Gradient normalization method; currently `"iqr"`.

## How to run

### Border and network workflow

1. Update the input and output paths.
2. Run the preprocessing cells to construct `unique_region` and rename the coordinate columns.
3. Confirm that `cell_type`, `tissue`, `unique_region`, `x`, and `y` contain no missing values.
4. Calculate the anatomical-region centroids.
5. Run either the CPU or GPU probability implementation.
6. Run `findPositives` using the same probability key created by the probability calculation.
7. Save the processed AnnData object.
8. Run the downstream border, enrichment, spatial, and network plotting cells.

### Gradient workflow

1. Update the path to the processed mouse brain H5AD file.
2. Confirm that the anatomical-region probability matrix and names are present.
3. Select the two anatomical regions to compare.
4. Calculate the probability-ratio score.
5. Create the five probability levels.
6. Construct and cluster the local probability-level windows.
7. Generate the spatial, compositional, gradient-magnitude, and steepness summaries.

## Dependencies

The notebooks require Python with:

- `mingl`
- `anndata`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `networkx`

The GPU probability calculation additionally requires a compatible CuPy and CUDA installation.

## Important notes

- Some notebook headings and comments still refer to an intestine or Inner-to-Outer Follicle analysis because the notebooks were adapted from shared tutorial templates. In these files, the actual analysis concerns mouse brain anatomical regions.
- `MouseBrain_MINGL_Gradient_Results.ipynb` currently saves and reloads a file named `simfast_results.h5ad`. This is a leftover filename from the synthetic-gradient workflow. Rename it to a mouse-brain-specific filename or update the path before running.
- In the border notebook, probabilities are stored under:
  
  ```python
  cells.obsm["neighborhood_probability"]
  ```

  Some later plotting code refers to:

  ```python
  prob_key="neighborhood_probabilities"
  ```

  Change this to:

  ```python
  prob_key="neighborhood_probability"
  ```

  unless the plural key exists in the loaded AnnData object.

- The gradient notebook stores probabilities under the plural key:
  
  ```python
  cells.obsm["neighborhood_probabilities"]
  ```

  Keep the probability key consistent within each notebook.

- The border notebook contains both a save path named `20260813_mousebrain_centroids.h5ad` and a reload path named `MouseBrainSpatialTranscriptomics_centroids.h5ad`. Confirm which processed file is authoritative before continuing after the save/reload step.
- Verify that probability rows and observation rows remain aligned after resetting the original cell index.
- GPU probability calculations may take substantial time and memory for the complete dataset.
- Clear saved notebook outputs before committing updated notebooks if smaller GitHub files are desired. Clearing outputs does not change the code or Markdown cells.
