# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

## [0.1.0] - 2026-09-30

### Added

- Alternative emission models for MINGL membership probabilities (`mg.tl.build_emission_model`, `mg.tl.mingl_membership_probabilities`; diagonal/full Gaussian, multinomial, Dirichlet-multinomial, logistic-normal) and cross-model comparison (`mg.tl.compare_emission_models`, `mg.tl.attach_all_model_probabilities`).
- Border threshold sensitivity tools (`mg.tl.threshold_sensitivity_analysis`, `mg.tl.border_metrics_at_threshold`, `mg.tl.spatial_border_clustering_null`).
- Gradient/transition steepness validation (`mg.tl.order_transition_clusters`, `mg.tl.steepness_score`, `mg.tl.gradient_sensitivity_analysis`, `mg.tl.validate_ground_truth_recovery`) and a synthetic transition-tissue simulator (`mg.tl.simulate_transition_tissue`).

### Changed

- `mg.tl.run_mingl_over_n_clusters` now fits MiniBatchKMeans in a canonical original-cell order (`cluster_row_order="original"`), so the selected neighborhood count no longer depends on input row order. Pass `cluster_row_order="as_given"` for the previous behavior, or `order_key` to pin the order explicitly.
- `mg.tl.KNN2` window counts are now computed and returned as float32 (previously float16).

### Fixed

- Reworked `mg.tl.cpu_gmm_probability` to score cells in batched NumPy blocks instead of per-cell multiprocessing tasks, which reduces Windows stalls caused by repeated process spawning and object serialization.

## [0.0.1] - 2026-04-06

### Added

- Basic tool, preprocessing and plotting functions
