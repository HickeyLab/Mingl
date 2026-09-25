# Simulated Ground-Truth Tissue Data

This directory contains the code and outputs used to generate and analyze synthetic spatial tissues with two known organizations (`A` and `B`). The simulations differ only in the width of the transition between organizations, providing fast, medium, and slow ground-truth gradients for testing MINGL.

## Workflow

1. `FunctionsSimulator.ipynb` generates spatial coordinates, assigns ground-truth organizations and cell types, recovers discrete neighborhoods, and exports the fast, medium, and slow CSV files.
2. `SyntheticGradientTestFast(2).ipynb`, `SyntheticGradientTestMedium.ipynb`, and `SyntheticGradientTestSlow.ipynb` run the MINGL probability and gradient analyses for the corresponding CSV files.
3. The three `sim*_results.h5ad` files store the processed AnnData objects produced by those notebooks.
4. `SimulatedParameterSweep.ipynb` tests analysis settings across all three gradients.

## Simulation design

`FunctionsSimulator.ipynb` creates approximately 3,340 non-overlapping cells in a 1,000 × 1,000 coordinate field using a scrambled Sobol sequence. Each cell receives one of 12 synthetic cell types and a known organization label. Organization membership changes across a vertical logistic boundary centered at `x = 500`.

The final datasets use:

| Dataset | `organization_mixing_width` | Interpretation |
|---|---:|---|
| Fast | 40 | Abrupt transition |
| Medium | 120 | Intermediate transition |
| Slow | 250 | Gradual transition |

The other final simulation settings are `organization_difference=0.30`, `local_weight=0.0`, `n_passes=2`, and `seed=0`. The notebook then applies the neighborhood finder and stores its recovered label in `Neighborhood`. The known simulated label remains in `true_organization` and is the ground truth used for validation.

## File descriptions

### Code

- `FunctionsSimulator.ipynb` — Main simulation-development notebook. Defines the tissue simulator and neighborhood-recovery functions, evaluates recovery against the known labels, explores simulation parameters, produces diagnostic plots, and exports the three final gradient CSV files.
- `SimulatedParameterSweep.ipynb` — Sweeps gradient-analysis parameters, including probability-bin count, neighborhood window size, and cluster count, for the fast, medium, and slow datasets. It is used to assess how these choices affect the estimated transition steepness.
- `SyntheticGradientTestFast(2).ipynb` — Full MINGL workflow for `synthetic_tissue_fast.csv`.
- `SyntheticGradientTestMedium.ipynb` — Full MINGL workflow for `synthetic_tissue_medium.csv`.
- `SyntheticGradientTestSlow.ipynb` — Full MINGL workflow for `synthetic_tissue_slow.csv`.
- `SyntheticGradientTestMedium(2).py` — Python export of a gradient-analysis notebook. Despite its filename, the current script references the **fast** CSV and H5AD files; update those paths before using it for the medium dataset.

The three gradient notebooks calculate neighborhood centroids, estimate per-cell A/B membership probabilities with the GPU GMM implementation (`k=10`), compute an A-to-B score, divide it into five equal-width probability levels, summarize those levels in 20-cell windows, and cluster the windows into five ordered gradient groups. They also create spatial, composition, and steepness plots and compare recovered neighborhood composition with the known ground truth.

### Input and intermediate CSV files

- `synthetic_tissue.csv` — Earlier/base simulated tissue containing coordinates, cell types, region, and ground-truth organization, but no recovered `Neighborhood` column.
- `synthetic_tissue_neighborhoods.csv` — Neighborhood-annotated version of the base simulation. It contains the same core fields plus the recovered `Neighborhood` label and omits `cell_id`.
- `synthetic_tissue_fast.csv` — Final fast-transition ground-truth dataset.
- `synthetic_tissue_medium.csv` — Final intermediate-transition ground-truth dataset.
- `synthetic_tissue_slow.csv` — Final slow-transition ground-truth dataset.

CSV columns:

| Column | Meaning |
|---|---|
| `cell_id` | Simulated cell identifier; absent from `synthetic_tissue_neighborhoods.csv` |
| `x`, `y` | Simulated spatial coordinates |
| `cell_type` | Synthetic cell-type label (`1`–`12`) |
| `unique_region` | Sample/region identifier; all current simulations use region `1` |
| `true_organization` | Known simulated organization (`A` or `B`); use as ground truth |
| `Neighborhood` | Organization recovered by the neighborhood-finding procedure (`A` or `B`) |

### Processed results

- `simfast_results.h5ad` — Processed fast-transition AnnData object.
- `simmedium_results(1).h5ad` — Processed medium-transition AnnData object.
- `simslow_results(1).h5ad` — Processed slow-transition AnnData object.

These files preserve the cell metadata and MINGL results, including neighborhood centroids, the per-cell `neighborhood_probabilities` matrix, and the corresponding neighborhood names. They can be loaded directly to skip the centroid and GMM probability calculations.

## How to reproduce the analysis

1. Open `FunctionsSimulator.ipynb` and run it from top to bottom to regenerate the CSV datasets. Change the Windows output paths before exporting.
2. Open the matching `SyntheticGradientTest*.ipynb`, update its input/output paths, and run the cells in order. The saved H5AD filenames in the notebooks do not include the attachment suffix `(1)`.
3. Use `SimulatedParameterSweep.ipynb` only if you want to repeat the parameter-sensitivity analysis.

The code requires Python with `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `anndata`, `tqdm`, and MINGL. The probability notebooks use MINGL's GPU GMM implementation and therefore require a compatible CuPy/CUDA installation. `FunctionsSimulator.ipynb` also imports local CellHier helper modules through a machine-specific path; update that path or replace it with the equivalent installed functions before rerunning.

## Important notes

- Treat `true_organization`, not `Neighborhood`, as the ground-truth label.
- The fast, medium, and slow CSV files were generated with the same random seed and settings except for transition width, enabling direct comparison.
- Some markdown in the gradient notebooks still refers to an “Inner-to-Outer Follicle” intestine example. In these simulations, those units correspond to synthetic organizations `A` and `B`.
- Several paths are absolute Windows paths and must be changed for another computer or directory structure.
