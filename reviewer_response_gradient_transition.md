# Findings: Gradient / Transition Clustering (Reviewer R1.5 & R1.9)

**To:** Kyra
**From:** James
**Branch:** `gradient-transition-clustering-improvements`
**Scope:** Only the gradient/transition-clustering comments (R1.5, and the R1.9
ordering clarification). Reproducibility of the downstream plots/conclusions was
**investigated, not modified**. No other reviewer comments were touched.

---

## TL;DR

- **The pipeline is fully reproducible.** With the seed that the notebooks and
  `grad.py` already set (`random_state=0`), the Score, the `Probability_Bin_Cluster`
  labels, and the steepness value are **bit-identical run-to-run**.
- **The steepness metric recovers ground truth.** On synthetic tissues with known
  transitions, recovered steepness increases from gradual → medium → sharp.
- **The manuscript's actual conclusion holds.** The *sharp-vs-gradual distinction*
  ("distinct regimes of sharp and gradual transitions") survives changes to neighbor
  count, bin count, cluster count, and sampling density in **17 of 18** parameter
  settings tested.
- **But the reviewer's concern is legitimate:** the *absolute* steepness magnitude
  is highly parameter-dependent, and *intermediate* (medium) transitions are noisy.
  Steepness should be interpreted **ordinally at fixed parameters**, not as a
  transferable absolute number.
- **Two real inconsistencies were found and left in place** (out of scope): mixed
  equal-width vs. quantile binning, and two different cluster-ordering rules. These
  don't break reproducibility but should be unified before final submission.

---

## 1. What was added (code, additive only)

No existing numeric behavior was changed. New, tested, documented pieces:

| File | What |
|------|------|
| `src/mingl/tl/simulate.py` | `simulate_transition_tissue(...)` — synthetic tissues with a **known** sharp/medium/gradual transition; output matches exactly what `mingl_neighborhoods_scverse` consumes. |
| `src/mingl/tl/gb.py` | `steepness_score(...)` and `order_transition_clusters(...)` — the manuscript's notebook steepness metric and cluster-ordering rule, extracted into reproducible, testable library functions. Plus a module docstring documenting the full methodology (R1.9). |
| `src/mingl/tl/grad.py` | Module docstring giving the step-by-step Score → binning → neighbor-composition → clustering methodology, and explicitly noting the quantile-vs-equal-width binning distinction (R1.5). |
| `src/mingl/tl/sensitivity.py` | `run_gradient_transition`, `gradient_sensitivity_analysis`, `validate_ground_truth_recovery` — the R1.5 ground-truth and robustness harness. |
| `tests/test_gradient_transition.py` | 6 tests: generator schema, steepness keys/ordering, determinism, ground-truth recovery, sensitivity smoke. All pass. |
| `tools/reviewer_r1_5_gradient_validation.py` | One-command reproduction of every figure/table below. |

---

## 2. Reproducibility of downstream plotting & conclusions (investigation only)

**Determinism:** running the full pipeline twice on the same input gives identical
`Score`, identical `Probability_Bin_Cluster` labels, and identical steepness. The
downstream plotters (`cell_type_distributions`, `plot_pooled_violin`, `gb`) take
those columns and re-derive the cluster ordering deterministically, so their outputs
are reproducible **given the same upstream inputs and seed**.

**Caveat — this reproducibility is conditional on the seed.** `MiniBatchKMeans` is
mini-batch/order sensitive; it is deterministic *only* because `random_state=0` is
set everywhere it's called. If a user omits the seed, the transition clusters (and
therefore every downstream panel) will vary run-to-run. Worth stating explicitly in
Methods.

**Two inconsistencies found (left unchanged, per scope):**

1. **Binning is not consistent across the codebase.** `grad.py` bins the Score by
   **quantile**; the fig4 tutorials bin by **equal width** with percentile-style
   labels; and `gb.py` / `violin.py` / `cell_composition.py` **re-bin the Score
   again** by quantile. The same cell can land in different bins in different panels.
   This is exactly the "alternates between equal-width bins and percentile-like
   labels" issue the reviewer flagged.
2. **Two ordering rules coexist.** `grad.py` orders fold-change columns by a
   hand-coded label score (`Very High = 10000 …`); everything else orders clusters
   by `Σ prop·2ⁱ`. They agree in practice but are documented as if one rule.

Neither breaks reproducibility, but both should be unified (single binning function,
single ordering function) before submission. I've documented the intended rule in
the new `gb.py` module docstring so the team can decide.

---

## 3. Ground-truth recovery (R1.5: "show the metric recovers ground truth")

Synthetic tissues, 3000 cells, 5 repeats/level, default params (k=20, 5 bins,
5 clusters). Ground-truth sharpness = 1 / (transition width fraction).

| Transition | Ground-truth sharpness | Recovered steepness (mean ± sd) |
|------------|-----------------------:|--------------------------------:|
| gradual | 1.43 | 0.213 ± 0.036 |
| medium  | 4.00 | 0.469 ± 0.517 |
| sharp   | 20.0 | 0.734 ± 0.160 |

Steepness increases monotonically with true sharpness. Note the **large spread on
the medium case** — the metric separates the extremes cleanly but resolves
intermediate gradations only on average, not per-region. (Figure:
`tools/reviewer_r1_5_outputs/ground_truth_recovery.png`.)

---

## 4. Sensitivity analysis (R1.5: robustness to bins / clusters / neighbors / density)

One-at-a-time sweeps around the defaults, steepness per transition:

| Param | Value | sharp | medium | gradual |
|-------|------:|------:|-------:|--------:|
| k | 10 | 0.657 | 0.186 | 0.265 |
| k | 20 | 0.751 | 0.213 | 0.262 |
| k | 30 | 1.579 | 0.223 | 0.246 |
| k | 50 | 2.115 | 0.200 | 0.251 |
| n_bins | 3 | 0.461 | 0.163 | 0.293 |
| n_bins | 4 | 0.537 | **3.020** | 0.637 |
| n_bins | 5 | 0.751 | 0.213 | 0.262 |
| n_bins | 6 | 1.773 | 1.015 | 0.437 |
| n_bins | 7 | 1.425 | 1.121 | 0.296 |
| n_clusters | 3 | 0.751 | 0.145 | 0.306 |
| n_clusters | 4 | 0.895 | 0.896 | 0.551 |
| n_clusters | 5 | 0.751 | 0.213 | 0.262 |
| n_clusters | 6 | 0.730 | 0.617 | 0.696 |
| n_clusters | 8 | 0.564 | 1.583 | 0.144 |
| subsample | 0.25 | 1.970 | 0.661 | 0.262 |
| subsample | 0.50 | 1.544 | 0.289 | 0.255 |
| subsample | 0.75 | 1.559 | 0.114 | 0.250 |
| subsample | 1.00 | 0.751 | 0.213 | 0.262 |

**Key reads:**

- **Sharp > gradual holds in 17/18 settings.** The single failure is `n_bins=4`,
  where the *medium* run is an outlier (3.02) and gradual (0.637) edges out sharp
  (0.537). The core binary conclusion is robust; the metric is weakest at even bin
  counts and at the intermediate transition.
- **Absolute magnitude is not portable.** Sharp steepness ranges 0.46 → 2.12 across
  settings — confirming the reviewer's point. Steepness must be compared only at
  fixed parameters.
- **Cluster count matters most for the intermediate case**; sharp/gradual are stable.

(Figure: `tools/reviewer_r1_5_outputs/sensitivity_analysis.png`.)

---

## 5. Recommendations for the reviewer response

1. **Report determinism + ground-truth recovery** — directly answers "show the
   metric recovers ground truth." (Deterministic given seed; sharp > medium >
   gradual.)
2. **Report the sensitivity sweep and state the interpretation rule**: steepness is
   an **ordinal** measure to be compared **at fixed parameters**, not an absolute
   transferable number.
3. **Stabilize the metric** for intermediate cases: average steepness over multiple
   regions/realizations, and/or avoid even bin counts. Consider whether more
   transition clusters (smoother mean-Score curve) reduce second-derivative noise.
4. **Unify binning and ordering** (separate small PR): one binning function
   (state quantile vs equal-width in Methods) and one ordering function. Removes the
   "alternates between equal-width and percentile-like labels" criticism at the
   source.
5. **State the seed dependence** of `MiniBatchKMeans` in Methods.

---

## 6. Reproduce everything

```bash
# from repo root, on branch gradient-transition-clustering-improvements
python tools/reviewer_r1_5_gradient_validation.py --n-cells 3000 --repeats 5
python -m pytest tests/test_gradient_transition.py -q
```

Outputs land in `tools/reviewer_r1_5_outputs/` (CSV tables + PNG figures).
