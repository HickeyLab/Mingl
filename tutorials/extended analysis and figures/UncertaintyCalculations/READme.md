# Alternative Uncertainty Calculations

This directory contains notebooks used to test whether MINGL’s biological conclusions are preserved when organizational membership uncertainty is calculated using different probability methods.

The notebooks compare:

- MINGL’s Gaussian mixture model (GMM)
- Softmax-transformed Euclidean distance to fixed centroids
- Fuzzy c-means membership calculated from fixed centroids

All three methods use the same local composition features and organizational centroids. The GMM additionally accounts for the variation of each compositional feature within an organizational unit, whereas the softmax and fuzzy methods depend only on distance to the fixed centroid.

## Files

### `otheruncertainties_neighborhood.ipynb`

Compares uncertainty calculations at the cellular-neighborhood level in the intestine dataset.

The analysis uses:

| Setting | Value |
|---|---|
| Local composition feature | `Cell Type` |
| Organizational label | `Neighborhood` |
| Region identifier | `unique_region` |
| Number of nearest neighbors | `k = 10` |
| Reference probability threshold | `0.25` |

The notebook:

- Loads the intestine spatial dataset.
- Calculates cell-type composition within each cell’s local neighborhood.
- Calculates a reference centroid for each annotated neighborhood.
- Generates GMM, softmax, and fuzzy neighborhood-membership probabilities.
- Compares maximum probabilities, assigned-neighborhood probabilities, and top-two probability margins.
- Measures agreement in the highest-probability neighborhood assignments.
- Determines where the original discrete neighborhood appears within each method’s ranked memberships.
- Maps probability and relative uncertainty across representative tissue regions.
- Identifies cells assigned to more than one neighborhood.
- Compares the spatial positions of border cells identified by different methods.
- Compares uncertainty with a spatial interface score based on neighboring cells carrying different discrete labels.
- Constructs neighborhood-interaction networks.
- Compares border prevalence, cell-type composition, and border enrichment between methods.
- Calculates matched thresholds so all methods identify approximately the same number of border cells.
- Calculates Spearman correlations between the cell-type enrichment results produced by the different methods.

The processed results are saved as:

```text
20260826_otheruncertainties_data.h5ad
```

### `otheruncertainties_communities.ipynb`

Repeats the uncertainty comparison at the broader community level in the intestine dataset.

The analysis uses:

| Setting | Value |
|---|---|
| Local composition feature | `Neighborhood` |
| Organizational label | `Community` |
| Region identifier | `unique_region` |
| Number of nearest neighbors | `k = 100` |
| Reference probability threshold | `0.25` |

The notebook follows the same general workflow as the neighborhood-level analysis but calculates probabilities across communities using local neighborhood composition.

It compares the three uncertainty methods using:

- Probability distributions
- Highest-probability community assignments
- Top-two probability margins
- Spatial uncertainty maps
- Border-cell classifications
- Community-interaction networks
- Border cell-type composition
- Border enrichment
- Fixed and matched probability thresholds
- Correlations in enrichment results between methods

The processed results are saved as:

```text
20260917_otheruncertainties_data_communities.h5ad
```

## Probability methods

### GMM

MINGL’s GMM method evaluates how compatible a cell’s local composition is with each organizational unit. It considers both the organizational centroid and the variation of individual compositional features within that unit.

The notebooks use MINGL’s GPU implementation:

```python
from mingl.tl.gmm_gpu import gpu_gmm_probability
```

A compatible CuPy and CUDA installation is therefore required.

### Centroid softmax

The softmax method calculates the Euclidean distance between each cell’s composition vector and every fixed organizational centroid. Negative distances are converted into probabilities using a softmax transformation.

This method does not refit or move the centroids.

### Fixed-centroid fuzzy membership

The fuzzy method applies the fuzzy c-means membership equation to the distances between cells and the same fixed organizational centroids.

The notebooks use a fuzzifier of:

```python
m = 2.0
```

This method calculates fuzzy memberships but does not perform fuzzy clustering or update the centroids.

## Border definition

A cell is considered a border cell when it has probabilities above the selected threshold for at least two organizational units:

```python
border = number_of_positive_memberships >= 2
```

The notebooks first compare all methods using the same threshold of `0.25`.

Because the three methods produce differently calibrated probability distributions, they also perform a matched-threshold analysis. In this analysis:

1. GMM uses the reference threshold of `0.25`.
2. The number of GMM border cells is calculated.
3. A separate softmax and fuzzy threshold is selected to reproduce approximately the same number of border cells.
4. Biological outputs are then compared using these matched border counts.

This separates differences caused by probability calibration from differences in which cells and biological relationships each method identifies.

## Required input data

Both notebooks use the intestine dataset:

```text
05_25_HuBMAP_tunit.csv
```

The required annotations include:

| Column | Meaning |
|---|---|
| `x`, `y` | Spatial cell coordinates |
| `Cell Type` | Cell-type annotation |
| `Neighborhood` | Cellular-neighborhood annotation |
| `Community` | Broader community annotation |
| `unique_region` | Tissue-region identifier |

## Stored probability results

The neighborhood-level notebook stores probability matrices using keys such as:

```text
neighborhood_probability
neighborhood_probability_softmax
neighborhood_probability_fuzzy
```

The community-level notebook stores probability matrices using:

```text
community_probability
community_probability_softmax
community_probability_fuzzy
```

The corresponding organizational names are stored in `adata.uns`. The probability matrices are stored in `adata.obsm`.

## Main matched-threshold outputs

The final matched-threshold sections create output directories containing files such as:

```text
matched_threshold_summary.csv
border_proportion_by_cell_type.csv
network_pair_counts.csv
pair_composition_GMM.csv
pair_composition_Softmax.csv
pair_composition_Fuzzy.csv
pair_group_counts_GMM.csv
pair_group_counts_Softmax.csv
pair_group_counts_Fuzzy.csv
border_enrichment_combined_values.csv
border_enrichment_spearman_combined.csv
```

Figures are generally exported in both PNG and PDF formats.

## How to use these notebooks

1. Update the dataset and output paths. The notebooks currently contain machine-specific Windows paths.
2. Confirm that all required metadata columns are present.
3. Run the notebook cells in order to calculate the centroids and probability matrices.
4. Save the resulting AnnData object before starting the downstream comparisons.
5. Run the fixed-threshold analysis to compare the raw probability calibration of the methods.
6. Run the matched-threshold section to compare the biological results after controlling for the total number of border cells.
7. Update the selected tissue region and organizational pairs if applying the notebooks to another dataset.

The notebooks contain example regions and organizational pairs selected for the intestine analysis. These values should be changed when analyzing another dataset.

## Software requirements

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
- `tqdm`

The GMM calculations use MINGL’s GPU implementation and require a compatible CuPy and CUDA installation.

## Important notes

- The methods use the same composition features and fixed organizational centroids.
- Softmax and fuzzy membership are distance-based alternatives, while GMM also incorporates feature variation.
- Probability values are not calibrated identically across the three methods.
- Use the matched-threshold analysis when comparing biological outputs across methods.
- The neighborhood notebook uses `k = 10`; the community notebook uses `k = 100`.
- Verify that each `prob_key` and its corresponding name key match the analysis level before running downstream cells.
- Some earlier cells use neighborhood-oriented variable names even in the community notebook. The final matched-threshold section uses the community-specific probability keys.
- Absolute Windows paths must be updated before running the notebooks on another computer.
- The saved H5AD files and generated figures are not included in this directory unless added separately.
