"""3D-capable MINGL tools.

Drop-in versions of every coordinate-dependent tool in :mod:`mingl.tl`, each
taking an optional ``z_key`` (``Z`` in the gradient pipeline). ``z_key=None`` is
the 2D behavior of the original. Nothing in :mod:`mingl.tl` is modified;
coordinate-free tools (edges, network graphs, proportions, ``crd``, ``ccd``,
elbow/plateau selection, steepness/ordering) are used from :mod:`mingl.tl`
directly.

Coordinates must be isotropic physical units (µm); use
:func:`cells_from_label_volume` to build a 3D cell table from a segmentation mask.

Usage::

    from mingl import tl3d
    windows = tl3d.KNN2(adata, z_key="z", cluster_col="cell_type")
"""

from .crd import crd2
from .datasets import KUETT2022_MODELS, KUETT2022_VOXEL_SIZE, load_kuett2022_model
from .emission import (
    attach_all_model_probabilities,
    compare_emission_models,
    mingl_membership_probabilities,
    mingl_window_matrix,
)
from .gb import gb, gb_local_score_gradients, local_linear_gradients
from .gmm import centroid_Calculation, cpu_gmm_probability, gpu_gmm_probability
from .grad import Neighborhoods, mingl_neighborhoods_scverse
from .ingest import cells_from_label_volume, label_centroids, read_label_volume
from .knn import KNN, KNN2
from .n_neighbors import fit_neighborhood_labels, run_mingl_over_n_clusters, score_partition
from .sensitivity import gradient_sensitivity_analysis, run_gradient_transition, validate_ground_truth_recovery
from .simulate import SHARPNESS_PRESETS, simulate_transition_tissue
from .threshold import border_metrics_at_threshold, spatial_border_clustering_null, threshold_sensitivity_analysis
from ._spatial import get_coords, spatial_keys

__all__ = [
    "KNN",
    "KUETT2022_MODELS",
    "KUETT2022_VOXEL_SIZE",
    "KNN2",
    "Neighborhoods",
    "SHARPNESS_PRESETS",
    "attach_all_model_probabilities",
    "border_metrics_at_threshold",
    "cells_from_label_volume",
    "centroid_Calculation",
    "compare_emission_models",
    "cpu_gmm_probability",
    "crd2",
    "fit_neighborhood_labels",
    "gb",
    "gb_local_score_gradients",
    "get_coords",
    "gpu_gmm_probability",
    "gradient_sensitivity_analysis",
    "label_centroids",
    "load_kuett2022_model",
    "local_linear_gradients",
    "mingl_membership_probabilities",
    "mingl_neighborhoods_scverse",
    "mingl_window_matrix",
    "read_label_volume",
    "run_gradient_transition",
    "run_mingl_over_n_clusters",
    "score_partition",
    "simulate_transition_tissue",
    "spatial_border_clustering_null",
    "spatial_keys",
    "threshold_sensitivity_analysis",
    "validate_ground_truth_recovery",
]
