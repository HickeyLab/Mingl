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

# intestine, three hierarchy levels (each writes to its own tagged subfolder):
#   neighborhood level uses the posterior already stored in the h5ad;
#   community / tissue-unit levels are re-scored against that column.
python tools/reviewer_threshold_sensitivity.py --dataset intestine --data intestine_results.h5ad --null
python tools/reviewer_threshold_sensitivity.py --dataset intestine --data intestine_results.h5ad \
    --recompute --neighborhood-col "Community" --null
python tools/reviewer_threshold_sensitivity.py --dataset intestine --data intestine_results.h5ad \
    --recompute --neighborhood-col "Tissue Unit" --null

# intestine null condition (point --data at your null h5ad):
python tools/reviewer_threshold_sensitivity.py --dataset intestine --data intestine_null.h5ad --null

# spatial transcriptomics (Barrett's esophagus):
python tools/reviewer_threshold_sensitivity.py --dataset spatial   --data <esophagus>.h5ad --null
```

Notes for Task 3:
* The neighborhood-level run reads the posterior already in the intestine h5ad
  (the regular MINGL result); the community / tissue-unit runs re-score with the
  diagonal model against those columns (`--recompute --neighborhood-col ...`),
  which reproduces the regular MINGL pipeline at that hierarchy level.
* Outputs are written to `--out-dir/<dataset>__<level>/`, so the three levels do
  not overwrite each other.
* If a real file has no stored posterior and no neighborhood column under the
  expected name, pass `--neighborhood-col` (and `--cluster-col` etc.) explicitly.

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

### Verified dataset schemas (lab data, read-only inspection)

The driver presets were checked against the real files and match exactly:

| dataset | rows | cluster col | neighborhood col | region col | hierarchy levels | min cells/region |
|---|---|---|---|---|---|---|
| intestine (`05_25_HuBMAP_tunit.csv`) | 2,512,002 | `Cell Type` (25) | `Neighborhood` (20) | `unique_region` (64) | Neighborhood, Community (10), Tissue Unit (4) | 5,829 |
| melanoma (`melanoma_all_information.csv`) | 5,019,159 | `Cell_Type` (39) | `Neighborhood` (16) | `filename` (21) | Neighborhood | 24,849 |
| esophagus (`all_regions_from_h5mu.csv`) | 645,661 | `Cell Type` (45) | `neigh_name` (24) | `region` (28) | neigh_name, community (10, lowercase) | 5,157 |

Consequences for the runs:

* All three presets are correct as shipped; no overrides needed for the default
  (neighborhood-level) runs. No NaNs in any key column; every region has far more
  than `k=10` cells.
* Intestine has all three hierarchy levels — the three-level Task 3 run works with
  `--neighborhood-col "Community"` and `--neighborhood-col "Tissue Unit"`.
  Esophagus has a lowercase `community` level (no Tissue Unit); use
  `--recompute --neighborhood-col "community"`.
* **Melanoma is 5.0M cells** (largest region 1.5M). Recomputing windows and fitting
  every emission model at that scale is heavy, and the Dirichlet-multinomial MLE is
  O(cells × iterations). For the melanoma comparison, use
  `--subsample-frac 0.1` (or `0.2`) and/or restrict with
  `--models diagonal_gaussian full_gaussian multinomial`. Intestine (2.5M) and
  esophagus (0.65M) are fine at full size, though a first pass with
  `--subsample-frac 0.3` is a good smoke test.

## 7. Results on the lab datasets

Produced by the drivers on the real data (intestine 2.51M cells, melanoma 5.02M,
esophagus 0.65M); melanoma emission was fit on a 30% subsample. CSVs/figures live
in `tools/reviewer_emission_models_outputs/<dataset>/` and
`tools/reviewer_threshold_sensitivity_outputs/<dataset>__<level>/`.

### 7a. Emission-model comparison (held-out neighborhood recovery)

Lower log-loss / higher accuracy is better. The **count-based models win decisively
on every dataset**, and the full-covariance Gaussian — best in-sample BIC — is the
**worst** out-of-sample (it over-fits its 7k–26k parameters), which is exactly why
raw likelihood/BIC is the wrong yardstick and the held-out metric is used.

| dataset | model | test log-loss ↓ | test acc ↑ | entropy | border frac | n_params |
|---|---|---|---|---|---|---|
| **intestine** | multinomial | **0.611** | **0.817** | 0.51 | 0.217 | 480 |
| (20 nbhd, 25 ct) | dirichlet_multinomial | 0.658 | 0.790 | 0.75 | 0.288 | 500 |
| | logistic_normal | 2.725 | 0.640 | 0.24 | 0.138 | 6,480 |
| | diagonal_gaussian (current) | 2.847 | 0.665 | 0.22 | 0.115 | 1,000 |
| | full_gaussian | 2.955 | 0.640 | 0.25 | 0.124 | 7,000 |
| **melanoma** | dirichlet_multinomial | **1.399** | 0.542 | 0.99 | 0.311 | 624 |
| (16 nbhd, 39 ct) | multinomial | 1.712 | **0.554** | 0.65 | 0.257 | 608 |
| | logistic_normal | 4.748 | 0.487 | 0.36 | 0.200 | 12,464 |
| | diagonal_gaussian (current) | 5.283 | 0.494 | 0.28 | 0.158 | 1,248 |
| | full_gaussian | 5.658 | 0.473 | 0.28 | 0.162 | 13,104 |
| **esophagus** | multinomial | **0.586** | **0.821** | 0.38 | 0.174 | 1,056 |
| (24 nbhd, 45 ct) | dirichlet_multinomial | 0.663 | 0.778 | 0.62 | 0.241 | 1,080 |
| | diagonal_gaussian (current) | 4.907 | 0.624 | 0.11 | 0.064 | 2,160 |
| | logistic_normal | 7.838 | 0.559 | 0.06 | 0.037 | 24,816 |
| | full_gaussian | 12.662 | 0.452 | 0.05 | 0.027 | 25,920 |

Border-cell counts at threshold 0.25 vary 2–9× across emission models (e.g.
esophagus: full-Gaussian 17.1k vs Dirichlet-multinomial 154.9k), so the emission
assumption is not cosmetic — it changes which cells are called interfaces.

### 7b. Border threshold sensitivity (regular MINGL / diagonal model, full data)

Number of border cells (strictly monotone in every case):

| threshold | intestine · nbhd | intestine · community | intestine · tissue-unit | melanoma · nbhd | esophagus · neigh |
|---|---|---|---|---|---|
| 0.01 | 1,096,727 | 1,050,271 | 403,946 | 1,849,774 | 171,638 |
| 0.10 | 603,690 | 569,686 | 203,421 | 948,447 | 85,323 |
| 0.25 | 291,543 | 279,554 | 106,940 | 464,229 | 41,652 |
| 0.40 | 77,712 | 80,026 | 37,964 | 165,108 | 12,610 |
| 0.49 | 3,360 | 3,800 | 4,204 | 10,556 | 898 |

* **Stability.** Border-cell-type enrichment is highly robust to the threshold:
  on intestine the per-type log2-enrichment correlates Pearson ≥ 0.99 (Spearman ≥
  0.99) across the 0.01→0.25 steps, dropping only at the extreme 0.40→0.49 step.
  The border-cell *set* Jaccard is 0.55 → 0.48 → 0.27 → 0.04 as the threshold rises,
  i.e. location is stable through the usable range and only collapses near 0.49.
* **Top border-enriched cell types (@0.25)** are biologically coherent and match the
  paper's themes: intestine — Lymphatic, ICC, DC, CD4+ T, M2 Macrophage; melanoma —
  Tumor subsets + CD163+CD206+ Macrophage + DC (tumor–immune interfaces); esophagus
  — CD4+ T, CD4+ Treg, M1/M2 Macrophage, CD8+ T (immune-enriched interfaces).
* **Null condition.** The spatial permutation null (esophagus) shows border cells are
  significantly clustered — observed nearest-neighbor distance is well below the null
  at every threshold (e.g. @0.25: 42.1 vs 57.7 ± 0.2), empirical p = 0.01 (floor for
  100 permutations) — so border *location* is real spatial structure, not chance.
