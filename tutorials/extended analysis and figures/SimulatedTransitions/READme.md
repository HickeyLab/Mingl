# Simulated Ground-Truth Tissue Data

This directory contains code and outputs used to generate and analyze synthetic spatial tissues with known organizations, gradients, and borders. These simulations provide ground truth for testing whether MINGL recovers continuous transitions and compositional borders between organizations `A` and `B`.

## Workflow

### Gradient simulations

1. `FunctionsSimulator.ipynb` generates spatial coordinates, assigns ground-truth organizations and cell types, recovers discrete neighborhoods, and exports the fast-, medium-, and slow-transition CSV files.
2. `SyntheticGradientTestFast(2).ipynb`, `SyntheticGradientTestMedium.ipynb`, and `SyntheticGradientTestSlow.ipynb` run the MINGL probability and gradient analyses for the corresponding CSV files.
3. The three `sim*_results.h5ad` files store the processed AnnData objects produced by the gradient notebooks.
4. `SimulatedParameterSweep.ipynb` tests gradient-analysis settings across all three transition widths.

### Border simulation

1. `syntheticdatasetwithborders_v2.ipynb` generates a tissue containing two organizations separated by a known compositional mixing zone.
2. It assigns known border-enriched and border-depleted cell types.
3. It recovers discrete neighborhoods from local cell-type composition.
4. It calculates MINGL membership probabilities and identifies cells positive for both organizations.
5. It compares MINGL-derived borders with the planted geometric and operational border definitions.
6. It evaluates border recovery, enrichment recovery, and probability calibration.

## Gradient simulation design

`FunctionsSimulator.ipynb` creates approximately 3,340 non-overlapping cells in a 1,000 × 1,000 coordinate field using a scrambled Sobol sequence. Each cell receives one of 12 synthetic cell types and a known organization label. Organization membership changes across a vertical logistic boundary centered at `x = 500`.

The final datasets use:

| Dataset | `organization_mixing_width` | Interpretation |
|---|---:|---|
| Fast | 40 | Abrupt transition |
| Medium | 120 | Intermediate transition |
| Slow | 250 | Gradual transition |

The other final simulation settings are:

```text
organization_difference = 0.30
local_weight = 0.0
n_passes = 2
seed = 0
```

The notebook applies the neighborhood finder and stores its recovered label in `Neighborhood`. The known simulated label remains in `true_organization` and is the ground truth used for validation.

## Border simulation design

`syntheticdatasetwithborders_v2.ipynb` creates up to 6,000 non-overlapping cells in a 1,500 × 1,500 coordinate field. Organizations `A` and `B` are separated by an undulating spatial interface.

Unlike a simulation that treats the border as a third organization, this notebook defines the border as a mixing zone between `A` and `B`. Organization identity changes deterministically across the interface, while cell-type composition transitions smoothly between the two organizational templates.

The validated simulation uses:

| Parameter | Value |
|---|---:|
| Number of cells | 6,000 |
| Number of cell types | 8 |
| Neighborhood window size | `k = 20` |
| Mixing-zone half-width | 2 neighborhood-window radii |
| Identity contrast | 0.90 |
| Marker base fraction | 0.40 |
| Border enrichment | 3.0 |
| Border depletion | 0.4 |
| Random seed | 42 |

Cell types have separate roles:

- Organization-identity cell types distinguish `A` from `B`.
- Marker cell types have the same baseline abundance in both organizations.
- Half of the marker types are enriched at the border.
- Half of the marker types are depleted at the border.
- Any remaining cell types provide background composition.

This separation ensures that the border signal does not create a third discrete organization.

The notebook records two border definitions:

- `true_border` — The geometric border containing cells within the planted mixing-zone width.
- `true_border_operational` — Cells whose local `k = 20` window contains at least 25% cells from each organization.

The operational border represents the interface that a window-based method can detect.

## File descriptions

### Simulation and analysis notebooks

- `FunctionsSimulator.ipynb` — Main gradient-simulation notebook. Defines the tissue simulator and neighborhood-recovery functions, evaluates recovery against known labels, explores simulation parameters, creates diagnostic plots, and exports the final gradient CSV files.
- `SimulatedParameterSweep.ipynb` — Sweeps gradient-analysis parameters, including probability-bin count, neighborhood-window size, and cluster count, across the fast, medium, and slow datasets.
- `SyntheticGradientTestFast(2).ipynb` — Full MINGL gradient workflow for `synthetic_tissue_fast.csv`.
- `SyntheticGradientTestMedium.ipynb` — Full MINGL gradient workflow for `synthetic_tissue_medium.csv`.
- `SyntheticGradientTestSlow.ipynb` — Full MINGL gradient workflow for `synthetic_tissue_slow.csv`.
- `syntheticdatasetwithborders_v2.ipynb` — Creates and evaluates a synthetic tissue containing a planted compositional border between organizations `A` and `B`.

The three gradient notebooks calculate neighborhood centroids, estimate per-cell A/B membership probabilities using MINGL’s GMM implementation, compute an A-to-B score, divide it into five equal-width probability levels, summarize those levels in local windows, and cluster the windows into five ordered gradient groups. They also create spatial, composition, and steepness plots and compare recovered neighborhood composition with the known ground truth.

The border notebook:

- Generates the synthetic tissue and planted border.
- Recovers organizations using a CellHier-style neighborhood finder.
- Calculates MINGL centroids and membership probabilities using `k = 20`.
- Produces spatial and compositional diagnostic plots.
- Compares true organizations with recovered neighborhoods.
- Evaluates MINGL border precision, recall, F1 score, intersection over union, and AUC.
- Tests whether planted border-enriched and border-depleted cell types are recovered.
- Performs a diagnostic temperature sweep to distinguish probability-calibration effects from ranking performance.

The temperature sweep is a diagnostic analysis and does not modify the released MINGL method.

### Input and intermediate CSV files

- `synthetic_tissue.csv` — Earlier/base simulated tissue containing coordinates, cell types, region, and ground-truth organization, but no recovered `Neighborhood` column.
- `synthetic_tissue_neighborhoods.csv` — Neighborhood-annotated version of the base simulation. It contains the same core fields plus the recovered `Neighborhood` label and omits `cell_id`.
- `synthetic_tissue_fast.csv` — Final fast-transition ground-truth dataset.
- `synthetic_tissue_medium.csv` — Final intermediate-transition ground-truth dataset.
- `synthetic_tissue_slow.csv` — Final slow-transition ground-truth dataset.
- `synthetic_tissue_fast_withborder.csv` — Synthetic tissue containing the planted organizations, border annotations, cell-type markers, and recovered neighborhood labels. This file is generated by `syntheticdatasetwithborders_v2.ipynb`.

The gradient CSV files contain:

| Column | Meaning |
|---|---|
| `cell_id` | Simulated cell identifier; absent from `synthetic_tissue_neighborhoods.csv` |
| `x`, `y` | Simulated spatial coordinates |
| `cell_type` | Synthetic cell-type label |
| `unique_region` | Sample or region identifier |
| `true_organization` | Known simulated organization (`A` or `B`) |
| `Neighborhood` | Organization recovered by the neighborhood-finding procedure |

The border simulation additionally contains:

| Column | Meaning |
|---|---|
| `true_border` | Geometric planted-border label |
| `true_border_operational` | Whether the local window contains substantial contributions from both organizations |
| `signed_distance` | Signed distance from the planted interface |
| `distance_from_border` | Absolute distance from the planted interface |
| `window_frac_B` | Fraction of organization-B cells in the local window |

The border notebook also writes `planted_truth.json`, which preserves simulation parameters and cell-type roles stored in the DataFrame attributes.

### Processed results

- `simfast_results.h5ad` — Processed fast-transition AnnData object.
- `simmedium_results.h5ad` — Processed medium-transition AnnData object.
- `simslow_results.h5ad` — Processed slow-transition AnnData object.
- `simulation_withborder_results.h5ad` — Processed AnnData object produced by the border-simulation notebook.

The gradient H5AD files preserve cell metadata and MINGL results, including neighborhood centroids, the per-cell `neighborhood_probabilities` matrix, and the corresponding neighborhood names.

The border H5AD file stores the simulated cell metadata, recovered neighborhood labels, neighborhood centroids, and MINGL organization-membership probabilities.

The border notebook may also export:

- `border_recovery_benchmark.csv` — Cell-type enrichment results for the planted border, released MINGL probabilities, and temperature-adjusted diagnostic probabilities.
- `temperature_sweep.csv` — Border precision, recall, F1, intersection-over-union, and AUC across diagnostic temperature values.

## How to reproduce the gradient analysis

1. Open `FunctionsSimulator.ipynb`.
2. Update the machine-specific input and output paths.
3. Run the notebook from top to bottom to regenerate the gradient CSV files.
4. Open the matching `SyntheticGradientTest*.ipynb`.
5. Update its input and output paths and run the cells in order.
6. Use `SimulatedParameterSweep.ipynb` only if you want to repeat the parameter-sensitivity analysis.

## How to reproduce the border analysis

1. Open `syntheticdatasetwithborders_v2.ipynb`.
2. Update `SHARE_DIR` or allow the notebook to use the local `R1.5_output` fallback directory.
3. Update or remove the optional CellHier import path if the helper package is installed elsewhere.
4. Run the simulation and neighborhood-recovery cells in order.
5. Confirm that the recovered `Neighborhood` labels correspond to organizations `A` and `B`.
6. Run the MINGL centroid and probability calculations using `K_WINDOW = 20`.
7. Run the border-comparison and enrichment sections.
8. Run the final probability-calibration section only if you want to reproduce the diagnostic temperature sweep.

The same neighborhood-window size must be used to define the simulated mixing-zone scale and to calculate the MINGL probabilities. Changing one without changing the other creates a mismatch between the planted spatial scale and the scale evaluated by MINGL.

## Software requirements

The notebooks require Python with:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `anndata`
- `tqdm`
- MINGL

Some simulation steps also use local CellHier neighborhood-recovery functions. Update the machine-specific CellHier path or replace those imports with equivalent installed functions.

The gradient probability notebooks may use MINGL’s GPU GMM implementation and therefore require a compatible CuPy and CUDA installation. The border notebook uses the CPU GMM implementation by default.

## Important notes

- Treat `true_organization`, not `Neighborhood`, as the organization ground-truth label.
- In the border dataset, treat `true_border` and `true_border_operational` as the border ground truths.
- `true_border_operational` is generally the more appropriate comparison for a local-window method because it identifies windows that actually contain cells from both organizations.
- The border is a mixing zone between `A` and `B`, not a third organizational class.
- The fast, medium, and slow gradient CSV files were generated with the same random seed and settings except for transition width, enabling direct comparison.
- Some markdown in the gradient notebooks still refers to an “Inner-to-Outer Follicle” intestine example. In these simulations, those units correspond to organizations `A` and `B`.
- Several paths are absolute Windows or macOS paths and must be changed for another computer or directory structure.
- Some generated outputs are written to an external results directory and will not appear in this folder unless they are copied here.
