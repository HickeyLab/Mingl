# Distribution Model Comparison

This folder contains the analysis used to test whether MINGL's biological conclusions are robust to the probability distribution used to model local cell composition.

## File

- `distributions.ipynb` — Fits and compares five emission models for MINGL neighborhood probabilities, matches their border-cell frequencies, and compares the resulting border assignments, cell-type compositions, spatial maps, and organizational networks.

## Models compared

The notebook evaluates:

1. Diagonal Gaussian mixture model (the MINGL baseline)
2. Full-covariance Gaussian mixture model
3. Multinomial model
4. Dirichlet-multinomial model
5. Logistic-normal model

All models use the same local composition features and organizational labels. The diagonal Gaussian model uses a reference border threshold of `0.25`. For each alternative model, the notebook finds the threshold that most closely matches the baseline number of cells with substantial probability in at least two organizational units. This controls for differences in probability scale before comparing biological results.

## Analyses performed

The notebook:

- Fits all five models and stores their per-cell probability matrices.
- Validates probability ranges, row sums, unit names, and cell alignment.
- Compares border-cell abundance at a fixed threshold and at matched thresholds.
- Measures agreement with the baseline using Jaccard similarity and baseline-border recovery.
- Compares border-cell frequencies across individual tissue regions.
- Compares the cell-type composition of a selected melanoma border.
- Generates spatial border maps for each model.
- Reconstructs intestine neighborhood- and tissue-unit-level interaction networks.
- Saves summary tables and figures for downstream manuscript assembly.

## Input data

Two datasets are used.

### Melanoma

The raw melanoma table must contain:

| Column | Meaning |
|---|---|
| `Cell_Type` | Cell-type annotation used to construct local composition features |
| `Neighborhood` | Original neighborhood label |
| `filename` | Tissue-region identifier |
| `x`, `y` | Spatial coordinates |

The notebook currently reads:

```text
Z:\MINGLE\Data\Melanoma\23_10_11_Melanoma_Marker_Cell_Neighborhood.csv
```

Update `file_path` before running the notebook on another computer.

### Intestine

The intestine AnnData object must contain:

| Column | Meaning |
|---|---|
| `cell_type` | Cell-type annotation |
| `Neighborhood` | Neighborhood label |
| `Community` | Community annotation used for tissue-unit-scale composition |
| `Tissue Unit` | Tissue-unit label |
| `unique_region` | Tissue-region identifier |
| `x`, `y` | Spatial coordinates |

The notebook expects the original intestine AnnData object to be available as `intestine`, or a previously fitted file at:

```text
intestine_emission_results/intestine_emission_probabilities.h5ad
```

The raw melanoma and intestine datasets are not stored in this folder.

## How to run

### Option 1: Fit models from the original data

1. Update the melanoma input and output paths in the first section.
2. Run the melanoma fitting cells to create `cells_mingl_all_emission_probabilities.h5ad`.
3. Load the original intestine AnnData object as `intestine`.
4. Run the **Intestine Analysis** section.
5. Set `REFIT=True` only when probabilities must be recalculated. Otherwise, existing results are reused.
6. Check the dataset-specific column names, regions, neighborhood pairs, and color dictionaries before generating figures.

The primary fitting settings are:

| Analysis level | Composition feature | Organizational label | `k` |
|---|---|---|---:|
| Melanoma neighborhood | `Cell_Type` | `Neighborhood` | 10 |
| Intestine neighborhood | `cell_type` | `Neighborhood` | 10 |
| Intestine tissue unit | `Community` | `Tissue Unit` | 300 |

### Option 2: Regenerate downstream results from saved probabilities

Start at the **Recovered Plotting/Analysis with saved data** section when fitted H5AD files already exist.

Update:

```python
MELANOMA_FILE = Path(
    "path/to/cells_mingl_all_emission_probabilities.h5ad"
)

INTESTINE_FILE = Path(
    "path/to/intestine_emission_probabilities.h5ad"
)
```

This route reloads the saved probability matrices, recalculates matched thresholds and comparison summaries, and regenerates the figures without refitting the five models.

## Important adjustable settings

- `REFERENCE_THRESHOLD` — Baseline diagonal-Gaussian threshold; currently `0.25`.
- `N1` and `N2` — Melanoma neighborhood pair used for detailed border analysis.
- `REGION` or `MELANOMA_REGION` — Tissue region used for spatial maps.
- `REFIT` — Whether to recompute intestine model probabilities.
- `SAVE_FIGURES` and `SAVE_MELANOMA_MAPS` — Whether figures are written to disk.
- `TOP_EDGES` — Maximum number of network edges plotted at each intestine hierarchy level.
- Palette dictionaries — Must contain colors for every plotted cell type or organizational unit.

## Outputs

Depending on the sections run, the notebook creates:

- H5AD files containing all five fitted probability matrices and matched thresholds.
- Model-specific threshold summaries.
- Border-cell overlap and regional-frequency tables.
- Selected-border cell counts and cell-type compositions.
- Network edge tables for intestine neighborhoods and tissue units.
- PNG and PDF figures showing model comparisons, spatial border maps, compositions, and networks.

Outputs are written to directories including:

```text
intestine_emission_results/
recovered_downstream_results/
melanoma_border_maps/
```

## Dependencies

The analysis requires Python with:

- `mingl`
- `anndata`
- `numpy`
- `pandas`
- `matplotlib`
- `networkx`

The installed MINGL version must provide `mg.tl.attach_all_model_probabilities` and the five model implementations used by that function.

## Important notes

- Run cells in order within the selected workflow because later cells reuse variables created earlier.
- Preserve the full-precision matched thresholds for classification. Rounded values should be used only for display.
- Verify that probability rows remain aligned with the AnnData observation names before interpreting model differences.
- The notebook contains saved high-resolution figures and is therefore large. Clear the notebook outputs before committing a new version to GitHub if a smaller repository file is desired. Clearing outputs does not alter the code or Markdown cells.
