# Reviewer response: emission-model assumptions for MINGL

> **Reviewer.** The Gaussian mixture model assumptions require stronger
> justification. The features used by MINGL are nearest-neighbor cell-type or
> lower-level label counts, which are compositional, discrete, sparse, bounded by
> k, and correlated because the counts sum to a fixed window size. Modeling each
> feature as an independent univariate Gaussian with diagonal covariance may be a
> poor approximation. The authors should discuss this limitation and compare it
> with models more appropriate for compositional count data, such as multinomial,
> Dirichlet-multinomial, logistic-normal, or transformed compositional features.

We agree, and we have implemented and compared the requested alternatives. This
document summarizes the additions; all code is additive (the shipped
diagonal-covariance path is unchanged, so every previously reported figure remains
reproducible).

## 1. What the current model is

MINGL fixes the neighborhood definition upstream (a discrete assignment per cell)
and then scores each cell's k-NN cell-type **count window** against each
neighborhood. The shipped scorer
(`mingl.tl.gmm.cpu_gmm_probability`) models each of the `d` count features as an
**independent univariate Gaussian** and multiplies them — i.e. a Gaussian with
**diagonal covariance** — then normalizes across neighborhoods with a uniform
prior. Because the window is a sum of `k` one-hot neighbor labels, the features are
non-negative integers that sum to `k`, are sparse, and are negatively correlated by
construction. The diagonal-Gaussian assumption ignores that correlation and the
count/compositional support, which is exactly the reviewer's concern.

## 2. Models implemented

All models keep the neighborhood definition fixed and change only the per-window
emission density. They live in `mingl.tl.emission_models` and share one entry
point, `mingl_membership_probabilities(adata, model=...)`, which writes the same
`obsm["neighborhood_probabilities"]` object the rest of MINGL consumes (so they are
drop-in for the border / gradient / network tools).

| model | assumption | reviewer item |
|---|---|---|
| `diagonal_gaussian` | independent univariate Gaussians (current MINGL) | baseline |
| `full_gaussian` | multivariate Gaussian, **full covariance** | **Task 1** |
| `multinomial` | `Multinomial(n=k, p_c)`; one composition per neighborhood | Task 2 |
| `dirichlet_multinomial` | over-dispersed multinomial (variability between instances) | Task 2 |
| `logistic_normal` | additive-log-ratio transform + full-cov Gaussian | Task 2 |

Numerical notes: the full-covariance and logistic-normal models regularize each
covariance (`+reg·I`, jitter-escalated Cholesky); the multinomial uses Laplace
smoothing; the Dirichlet-multinomial fits `alpha_c` by Minka's fixed-point MLE; the
logistic-normal uses the globally most-abundant cell type as the ALR reference.

**A note on the fixed-sum singularity.** Because every window sums to `k`, each
component covariance is singular along the all-ones direction. That direction is
identical for every neighborhood *and* orthogonal to every centered window (a
window and its component mean both sum to `k`), so it contributes nothing to the
Mahalanobis distance and cancels in the posterior; the regularizer only conditions
the remaining directions. The full-covariance model is therefore well behaved on
compositional counts despite the exact linear dependence.

## 3. How the models are compared

Raw log-likelihood / BIC is **not** comparable across a continuous density
(Gaussian) and a discrete pmf (multinomial / Dirichlet-multinomial): they are
defined on different base measures. We therefore compare on a quantity that is
identical in kind for every model — the **posterior over the shared neighborhood
label set** — evaluated on a held-out split:

* **held-out neighborhood log-loss / accuracy** — how well the emission posterior
  recovers the held-out neighborhood label (comparable across all five models);
* **membership entropy** — how sharp/confident the soft assignment is;
* **border-cell fraction** — the downstream biological consequence;
* **within-family BIC** — used only to compare `diagonal_gaussian` vs
  `full_gaussian` (same Gaussian base measure, so directly comparable).

`mingl.tl.compare_emission_models` produces this table;
`mingl.tl.attach_all_model_probabilities` writes every model's posterior for
downstream border comparison.

## 4. Local validation (synthetic, reproducible without lab data)

On a synthetic 4-neighborhood compositional tissue (4,800 cells, 6 cell types;
`python tools/reviewer_emission_models.py --synthetic`):

| model | test log-loss ↓ | test acc ↑ | entropy | border frac | BIC (Gaussian only) |
|---|---|---|---|---|---|
| dirichlet_multinomial | **0.186** | 0.934 | 0.19 | 0.088 | — |
| multinomial | 0.186 | 0.934 | 0.15 | 0.078 | — |
| full_gaussian | 0.206 | 0.935 | 0.11 | 0.059 | **13,175** |
| logistic_normal | 0.232 | 0.922 | 0.21 | 0.107 | — |
| diagonal_gaussian | 0.236 | 0.935 | 0.077 | 0.038 | 56,076 |

Two points the reviewer asked us to make quantitatively:

1. **Full covariance is a much better Gaussian.** Within the Gaussian family (a
   fair, same-base-measure comparison), full covariance improves BIC by ~43,000
   over the diagonal model — the diagonal assumption is a poor approximation.
2. **The emission model materially changes the biology.** At a fixed threshold
   0.25 on the full dataset, the number of MINGL border cells ranges from 191
   (diagonal) to 520 (logistic-normal) — a 2.7× swing — and the diagonal model is
   the most overconfident (lowest entropy, fewest border cells). Model choice is
   not cosmetic.

The correctness of the implementation is pinned by tests (`tests/test_emission_models.py`),
including an exact check that `diagonal_gaussian` reproduces the shipped scorer
(argmax identical, max probability difference < 1e-4) and a check that the
full-covariance model uses feature correlation the diagonal model cannot.

## 5. Threshold sensitivity of border cells (Task 3)

The border definition (a cell with ≥2 neighborhood probabilities above a threshold)
depends on that threshold. `mingl.tl.threshold_sensitivity_analysis` sweeps
`(0.01, 0.1, 0.25, 0.4, 0.49)` and reports the four requested outputs — number of
border cells, border location, border composition, and border cell-type
enrichment — plus stability (`tools/reviewer_threshold_sensitivity.py`). On the
same synthetic tissue (regular MINGL / diagonal model):

| threshold | n border | border frac | centroid (x, y) | Jaccard vs previous |
|---|---|---|---|---|
| 0.01 | 870 | 0.181 | (180.5, 49.0) | — |
| 0.10 | 420 | 0.088 | (182.5, 50.8) | 0.48 |
| 0.25 | 191 | 0.040 | (178.8, 51.7) | 0.45 |
| 0.40 | 65 | 0.014 | (147.4, 50.0) | 0.34 |
| 0.49 | 7 | 0.001 | (173.9, 60.2) | 0.11 |

The count is strictly monotone (a hard invariant, tested), the border-cell set stays
moderately stable (Jaccard) across the usable range and only destabilizes near
0.49, and enrichment vectors correlate strongly across neighboring thresholds
(Spearman ≥ 0.77 except at the extreme step). A permutation **null**
(`--null`) confirms border cells are spatially clustered — not random — at every
threshold with enough border cells (empirical p ≈ 0.01), losing significance only at
0.49 where just 7 cells remain.

## 6. Reproduction

```bash
# Tasks 1 & 2 — emission-model comparison
python tools/reviewer_emission_models.py --synthetic                     # local demo
python tools/reviewer_emission_models.py --dataset intestine --data intestine_results.h5ad
python tools/reviewer_emission_models.py --dataset melanoma  --data melanoma_all_information.csv
python tools/reviewer_emission_models.py --dataset spatial   --data <spatial>.h5ad
#   large datasets: add --subsample-frac 0.3 ; restrict models with --models full_gaussian multinomial

# Task 3 — threshold sensitivity (regular MINGL = diagonal model)
python tools/reviewer_threshold_sensitivity.py --synthetic --null        # local demo
python tools/reviewer_threshold_sensitivity.py --dataset intestine --data intestine_results.h5ad --null
python tools/reviewer_threshold_sensitivity.py --dataset spatial   --data <spatial>.h5ad
```

Library API:

```python
import mingl as mg
mg.tl.mingl_membership_probabilities(adata, model="full_gaussian",
    cluster_col="Cell Type", neighborhood_col="Neighborhood", region_key="unique_region")
cmp = mg.tl.compare_emission_models(adata, cluster_col="Cell Type",
    neighborhood_col="Neighborhood", region_key="unique_region")
res = mg.tl.threshold_sensitivity_analysis(adata, cell_type_col="Cell Type")
```

Column-name presets in the drivers are defaults; override with `--cluster-col`,
`--neighborhood-col`, `--region-key`, `--x-key`, `--y-key` as needed.

## 7. Results tables to complete from the lab-server runs

Fill these from the CSVs written to `tools/reviewer_emission_models_outputs/` and
`tools/reviewer_threshold_sensitivity_outputs/` on each real dataset.

**Emission-model comparison (per dataset).**

| dataset | model | test log-loss | test acc | entropy | border frac | BIC (Gaussian) |
|---|---|---|---|---|---|---|
| intestine | … | | | | | |
| melanoma | … | | | | | |
| spatial | … | | | | | |

**Border threshold sensitivity (per dataset).**

| dataset | threshold | n border | border frac | top enriched cell types |
|---|---|---|---|---|
| intestine | 0.01 … 0.49 | | | |
| melanoma | 0.01 … 0.49 | | | |
| spatial | 0.01 … 0.49 | | | |
