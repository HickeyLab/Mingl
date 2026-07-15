# Spec: Alternative emission models & threshold sensitivity for MINGL

Addresses the reviewer comment on the Gaussian-mixture assumption:

> The features used by MINGL are nearest-neighbor cell-type or lower-level label
> counts, which are compositional, discrete, sparse, bounded by k, and correlated
> because the counts sum to a fixed window size. Modeling each feature as an
> independent univariate Gaussian with diagonal covariance may be a poor
> approximation. […] compare it with models more appropriate for compositional
> count data, such as multinomial, Dirichlet-multinomial, logistic-normal, or
> transformed compositional features.

## 0. Ground truth about the current model

* Window features come from `KNN2`: for each cell, the count of each cell type
  among its `k` nearest spatial neighbors. Counts are integers `>= 0`, sum to
  `k` (a fixed window size), sparse (many zeros), and negatively correlated by
  construction (one type up ⇒ another down).
* `centroid_Calculation` stores, per neighborhood, the **mean and std of each
  feature** (sample std, `ddof=1`).
* `cpu_gmm_probability` scores each cell as the **product over features** of a
  univariate Gaussian pdf (⇒ diagonal covariance), then normalizes across
  neighborhoods with a **uniform prior**. The result is a posterior
  responsibility written to `obsm["neighborhood_probabilities"]` /
  `uns["neighborhood_probability_neighborhoods"]`.
* Neighborhood assignments (the mixture components) are produced **upstream**
  and live in `obs[neighborhood_col]`. The emission model is only the per-cell
  scoring step. So every alternative below keeps the neighborhood definition
  fixed and only changes how a window is scored against each neighborhood.

## 1. Constraints (from the user)

1. Work only on branch `gradient-transition-clustering-improvements`.
2. **Do not modify existing shipped logic** (`gmm.py`, `centroids.py`, `edges.py`,
   `grad.py`, `gb.py`, plotting, etc.). New capability is **additive**: new
   modules + purely additive export lines in `tl/__init__.py`. The existing
   diagonal-Gaussian path stays byte-for-byte unchanged so already-generated
   figures remain reproducible.
3. Cannot read lab data. Library + driver scripts are validated locally on the
   existing synthetic generator (`mingl.tl.simulate`); the driver scripts are
   run by the user on the lab server against the real intestine / melanoma /
   spatial-transcriptomics `.h5ad` files.
4. Iterate to production quality (tests, determinism, numerical safety).

## 2. Deliverables

### 2a. `src/mingl/tl/emission_models.py`  (Tasks 1 & 2)

A single scoring entry point plus a registry of emission models. Every model
implements `fit(X, labels)`, `log_likelihood(X) -> (n_cells, n_components)`, and
`n_parameters()`. Membership probabilities = softmax over
`log_prior + log_likelihood` via `scipy.special.logsumexp` (log-space, no
underflow).

Models:

| key                     | family                          | reviewer item |
|-------------------------|---------------------------------|---------------|
| `diagonal_gaussian`     | product of univariate Gaussians | current MINGL (baseline / equivalence check) |
| `full_gaussian`         | multivariate Gaussian, full cov | **Task 1 (first priority)** |
| `multinomial`           | Multinomial(n=k, p_c)           | Task 2 |
| `dirichlet_multinomial` | Dirichlet–Multinomial(α_c)      | Task 2 |
| `logistic_normal`       | ALR transform + full-cov Gaussian | Task 2 |

Numerical design:

* **Full Gaussian.** μ_c = mean, Σ_c = sample covariance `+ reg_covar·I`.
  Counts summing to `k` make Σ singular along the all-ones direction; `reg_covar`
  (default `1e-6`, scaled by mean feature variance) restores invertibility, and
  that singular direction is identical for every component so it cancels in the
  posterior. Log-density via Cholesky (`cho_factor`/`cho_solve`, `slogdet`).
  Components with `< 2` cells fall back to `reg_covar·I`.
* **Multinomial.** p_c = pooled counts normalized with Laplace smoothing `α`
  (default `0.5`) so no `log(0)`. Per-cell loglik `Σ_j x_ij log p_cj`
  (+ multinomial coefficient, constant across components).
* **Dirichlet–Multinomial.** α_c fit by Minka's fixed-point MLE (moment init,
  capped iterations, positivity floor) → captures overdispersion / biological
  variability between instances of a neighborhood. Loglik from the exact
  DirMult log-pmf (`gammaln`).
* **Logistic-normal.** π = (x + pseudocount)/(n + pseudocount·d); additive
  log-ratio (ALR) with reference = globally most abundant type ⇒ `d-1` dims
  (avoids the CLR singularity); full-cov Gaussian in ALR space. The transform
  Jacobian is constant across components ⇒ cancels in the posterior.

Public API (mirrors `cpu_gmm_probability`'s I/O contract exactly):

```python
mingl_membership_probabilities(
    cells, *, model="full_gaussian",
    cluster_col="cell_type", neighborhood_col="neighborhood",
    region_key="unique_region", x_key="x", y_key="y",
    ks=(10, 20, 100, 300), k=10, prior="uniform",
    reg_covar=1e-6, smoothing=0.5, pseudocount=0.5,
    prob_key="neighborhood_probabilities",
    prob_variable_key="neighborhood_probability_neighborhoods",
    random_state=0,
) -> AnnData   # writes obsm[prob_key], uns[prob_variable_key], uns["mingl_emission_<model>"]
```

Design guarantees:
* `model="diagonal_gaussian"` reproduces `cpu_gmm_probability` (argmax identical,
  probabilities close) — proven by a test.
* Row-order invariant, seed-deterministic.
* Feature order = sorted unique cell types (internal consistency; posterior is
  order-independent anyway).

### 2b. `src/mingl/tl/model_comparison.py`  (Tasks 1 & 2 — the "compare" the reviewer asked for)

Comparing a continuous density (Gaussian) to a count pmf (multinomial) by raw
log-likelihood/BIC is a base-measure mismatch. The fair, biologically meaningful
comparison uses the **posterior over the shared neighborhood label set** on a
held-out split:

`compare_emission_models(...) -> DataFrame` with, per model:
* `test_logloss`, `test_accuracy` — how well the emission posterior recovers the
  held-out neighborhood label (measure-agnostic, comparable across all models);
* `mean_entropy` — sharpness of memberships;
* `frac_multimember`, `frac_border` at a threshold — downstream impact;
* `train_loglik`, `test_loglik`, `n_params`, `bic` — within-family fit (flagged
  as only comparable inside the same likelihood base);
* `runtime_s`.

Windows are computed once on the full spatial data; only the emission
**parameters** are fit on the train rows (no label leakage), test rows evaluated.
`attach_all_model_probabilities(...)` writes each model's posterior to a distinct
`obsm` key for downstream border comparison.

### 2c. `src/mingl/tl/threshold_sensitivity.py`  (Task 3)

Model-agnostic; operates on any `obsm` posterior. A "border cell" = a cell with
`>= 2` neighborhood probabilities strictly `> threshold` (matches the tutorials'
`Count_Above_Threshold in {2,3}` definition).

* `border_metrics_at_threshold(...)` → for one threshold: number of border
  cells, border fraction, **border composition** (per-cell-type counts &
  proportions among border cells), **border cell-type enrichment**
  (log2 (proportion among border cells / proportion among all cells)), and
  **border location** (per-region border fraction; spatial centroid + spread of
  border cells).
* `threshold_sensitivity_analysis(..., thresholds=(0.01, 0.1, 0.25, 0.4, 0.49))`
  → tidy long table across thresholds + **stability** metrics: monotone border
  count (hard invariant: strictly non-increasing in threshold), Jaccard overlap
  of border-cell sets between consecutive thresholds (location stability), and
  Spearman/Pearson correlation of enrichment vectors across thresholds.
* Optional permutation **null** (`n_null_permutations`): shuffle posterior rows
  within region to get a null border-count distribution — supports the
  "intestine null condition."

### 2d. Driver scripts in `tools/`

* `tools/reviewer_emission_models.py` — runs `compare_emission_models` (+ per
  model border summary) and writes CSV + comparison figures. `--dataset
  {intestine,melanoma,spatial}` sets column-name presets; `--data PATH` points at
  the user's `.h5ad`; `--synthetic` validates locally with no lab data.
* `tools/reviewer_threshold_sensitivity.py` — runs the threshold sweep +
  null; writes CSV + figures (border count vs threshold, enrichment heatmap,
  Jaccard stability, null comparison). Works on any h5ad already carrying a
  posterior, and can first compute one via `emission_models` for a chosen model.

Outputs land in `tools/reviewer_emission_models_outputs/` and
`tools/reviewer_threshold_sensitivity_outputs/`, matching the existing
`tools/reviewer_r1_5_*` convention.

### 2e. Tests (`tests/test_emission_models.py`, `tests/test_threshold_sensitivity.py`)

* Hand-computed log-likelihood for each model on a tiny fixture.
* Every model: rows are valid probabilities (finite, `>= 0`, sum to 1).
* `diagonal_gaussian` ≈ `cpu_gmm_probability` (argmax identical).
* Full Gaussian recovers correlations a diagonal model cannot (correlated fixture).
* Determinism + row-order invariance.
* Threshold sweep: border count strictly non-increasing in threshold; enrichment
  finite; null runs.

### 2f. `reviewer_response_emission_models.md`

Reviewer-facing writeup: the limitation, the five models, the comparison
methodology, reproduction commands, the local synthetic-validation results, and a
results table the user fills from the lab-server runs.

## 3. Order of work

1. `emission_models.py` (+ additive export) → tests → local synthetic run.
2. `model_comparison.py` → tests → local synthetic run.
3. `threshold_sensitivity.py` → tests → local synthetic run.
4. Driver scripts → run on synthetic locally.
5. Critical review pass; fix; re-run full suite.
6. `reviewer_response_emission_models.md`; commit (authored by user only).
