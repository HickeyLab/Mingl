"""3D-aware neighborhood-count sweep with a choice of clusterer.

:func:`run_mingl_over_n_clusters` is :func:`mingl.tl.n_neighbors.run_mingl_over_n_clusters`
plus two keywords:

``z_key``
    Validated and copied into the per-cell output (the sweep clusters k-NN
    *window features*, never coordinates, so the math does not change).
``clusterer``
    ``"minibatch"`` (default) is the original ``MiniBatchKMeans(random_state=0)``
    path and, with ``random_state=0``, delegates to :mod:`mingl.tl` so published
    2D results stay bit-identical. ``"kmeans"`` fits full Lloyd k-means with
    ``n_init`` restarts instead. On the Kuett 2022 Main HER2 3D windows the
    mini-batch partition changes substantially with the seed (seed-to-seed ARI
    about 0.4-0.75 for n = 4-15) while full k-means with 10 restarts is stable
    (ARI about 0.8-1.0) at well under a second per fit for 1e5 cells; see
    :func:`fit_neighborhood_labels`.

Both paths score every partition with the same diagonal-Gaussian model as the
original (per-neighborhood mean/std of the raw window counts, std floor 1e-2):
:func:`score_partition` reproduces the original's per-cell ``log_likelihood_n{n}``
and ``assigned_prob_n{n}`` exactly for a given labelling.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ..tl.n_neighbors import _clustering_order
from ..tl.n_neighbors import run_mingl_over_n_clusters as _run_mingl_over_n_clusters_2d
from ._spatial import require_columns, spatial_keys

__all__ = ["CLUSTERERS", "fit_neighborhood_labels", "run_mingl_over_n_clusters", "score_partition"]

CLUSTERERS = ("minibatch", "kmeans")


def fit_neighborhood_labels(
    X: np.ndarray,
    n_clusters: int,
    *,
    clusterer: str = "minibatch",
    random_state: int = 0,
    n_init: int = 10,
) -> np.ndarray:
    """Cluster standardized window features into ``n_clusters`` neighborhoods.

    ``clusterer="minibatch"`` is exactly what :mod:`mingl.tl` does
    (``MiniBatchKMeans(n_clusters, random_state)``); ``"kmeans"`` is
    ``KMeans(n_clusters, n_init=n_init, random_state)``. Returns integer labels
    in row order of ``X``.

    The k-means fit is run single-threaded. Multithreaded Lloyd iterations
    reduce per-thread centroid sums in scheduling-dependent order, so the same
    seed can give slightly different labels from one run to the next (ARI
    0.9999 on 1e5 cells); one thread makes the result bitwise reproducible at
    a cost of well under a second per fit at that size.
    """
    if clusterer == "minibatch":
        return MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(X)
    if clusterer == "kmeans":
        with threadpool_limits(limits=1):
            return KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state).fit_predict(X)
    raise ValueError(f"clusterer must be one of {CLUSTERERS}, got {clusterer!r}")


def score_partition(
    features: pd.DataFrame,
    labels: np.ndarray,
    n_clusters: int,
    *,
    batch_size: int = 20000,
) -> tuple[list[float], list[float]]:
    """Per-cell log-likelihood and assigned probability of a neighborhood partition.

    Identical arithmetic to the loop body of
    :func:`mingl.tl.n_neighbors.run_mingl_over_n_clusters`: ``features`` are the
    raw (unscaled) float32 window counts, each neighborhood is a diagonal
    Gaussian with the mean/std of its members (std floored at 1e-2), the
    log-likelihood is that of the cell's own neighborhood, and the assigned
    probability is the cell's normalized density under its own neighborhood
    relative to all neighborhoods. Labels are ``"0"``..``"n-1"`` strings or ints.
    """
    cols = list(features.columns)
    df = features.astype(np.float32).copy()
    lab = pd.Series(np.asarray(labels).astype(str), index=df.index)

    centroids = []
    for label in map(str, range(n_clusters)):
        group = df[lab.to_numpy() == label][cols]
        centroids.append(
            {
                "Neighborhood": label,
                **{f"{c}_mean": group[c].mean() for c in cols},
                **{f"{c}_std": group[c].std(ddof=0) for c in cols},
            }
        )
    df_centroids = pd.DataFrame(centroids).set_index("Neighborhood")
    means = np.array(df_centroids[[f"{c}_mean" for c in cols]].values, dtype=np.float32)
    stds = np.array(df_centroids[[f"{c}_std" for c in cols]].values, dtype=np.float32)
    stds = np.where(stds < 1e-2, 1e-2, stds)

    label_map = {name: i for i, name in enumerate(df_centroids.index)}
    assigned_indices = [label_map[label] for label in lab.to_numpy()]

    num_cells = len(df)
    num_batches = (num_cells + batch_size - 1) // batch_size
    log_likelihoods: list[float] = []
    assigned_probs: list[float] = []
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, num_cells)
        batch_df = df.iloc[start:end]

        batch_data = np.array(batch_df[cols].values, dtype=np.float32)
        batch_indices = np.array(assigned_indices[start:end], dtype=np.int32)

        batch_means = means[batch_indices]
        batch_stds = stds[batch_indices]

        log_coeff = -np.log(batch_stds) - 0.5 * np.log(2 * np.pi)
        squared_diff = ((batch_data - batch_means) / batch_stds) ** 2
        log_pdf_vals = log_coeff - 0.5 * squared_diff
        log_probs = np.sum(log_pdf_vals, axis=1)
        log_likelihoods.extend(log_probs.tolist())

        coeffs = 1.0 / (stds[np.newaxis, :, :] * np.sqrt(2 * np.pi))
        exponents = (
            -0.5
            * ((batch_data[:, np.newaxis, :] - means[np.newaxis, :, :]) / stds[np.newaxis, :, :])
            ** 2
        )
        pdf_vals = coeffs * np.exp(exponents)

        total_probs = np.prod(pdf_vals, axis=2)
        total_probs = np.nan_to_num(total_probs, nan=1e-300, posinf=1e-300, neginf=1e-300)

        prob_sums = np.sum(total_probs, axis=1, keepdims=True)
        prob_sums = np.where(prob_sums == 0, 1e-300, prob_sums)

        normalized_probs = total_probs / prob_sums
        normalized_probs = np.nan_to_num(normalized_probs, nan=0.0, posinf=0.0, neginf=0.0)

        row_indices = np.arange(end - start)
        batch_probs = normalized_probs[row_indices, batch_indices]
        assigned_probs.extend(batch_probs.tolist())

    return log_likelihoods, assigned_probs


def _sweep(
    adata,
    knn_feature_cols,
    n_range,
    output_col_template,
    return_per_cell,
    *,
    coord_keys,
    region_key,
    results_uns_key,
    cluster_row_order,
    order_key,
    clusterer,
    random_state,
    n_init,
):
    """The original sweep loop with a pluggable clusterer (see module docstring)."""
    df = adata.obs.copy()
    df[knn_feature_cols] = df[knn_feature_cols].astype(np.float32)
    per_cell_cols: dict = {}  # joined onto the coordinate frame at the end (avoids per-column inserts)
    summary_rows = []

    X_raw = df[knn_feature_cols].to_numpy(dtype=np.float32)
    X_scaled_all = StandardScaler().fit_transform(X_raw).astype(np.float32)

    order = _clustering_order(adata, len(df), how=cluster_row_order, order_key=order_key)
    inv = np.argsort(order, kind="stable")

    for n in n_range:
        labels = fit_neighborhood_labels(
            X_scaled_all[order], n, clusterer=clusterer, random_state=random_state, n_init=n_init
        )[inv]
        cluster_col = output_col_template.format(n)
        log_likelihoods, assigned_probs = score_partition(df[knn_feature_cols], labels, n)
        summary_rows.append(
            {
                "n_clusters": n,
                "avg_log_likelihood": float(np.mean(log_likelihoods)),
                "avg_assigned_probability": float(np.mean(assigned_probs)),
            }
        )
        adata.obs[cluster_col] = pd.Categorical(labels.astype(str))
        adata.obs[f"log_likelihood_n{n}"] = np.asarray(log_likelihoods, dtype=np.float32)
        adata.obs[f"assigned_prob_n{n}"] = np.asarray(assigned_probs, dtype=np.float32)
        if return_per_cell:
            per_cell_cols[f"log_likelihood_n{n}"] = log_likelihoods
            per_cell_cols[f"assigned_prob_n{n}"] = assigned_probs
            per_cell_cols[cluster_col] = labels

    summary_df = pd.DataFrame(summary_rows)
    per_cell_df = None
    if return_per_cell:
        per_cell_df = pd.concat(
            [df[[*coord_keys, region_key]], pd.DataFrame(per_cell_cols, index=df.index)], axis=1
        )
    adata.uns[results_uns_key] = {
        "summary_df": summary_df,
        "knn_feature_cols": list(knn_feature_cols),
        "clusterer": clusterer,
        "random_state": random_state,
        "n_init": n_init if clusterer == "kmeans" else None,
    }
    return (summary_df, per_cell_df) if return_per_cell else summary_df


def run_mingl_over_n_clusters(
    adata,
    knn_feature_cols,
    n_range=range(1, 51),
    output_col_template="Neighborhood_{}",
    return_per_cell=True,
    plot_summary=True,
    show: bool = True,
    *,
    x_key="x",
    y_key="y",
    z_key: Optional[str] = None,
    region_key="unique_region",
    results_uns_key="mingl_n_clusters",
    cluster_row_order="original",
    order_key=None,
    clusterer: str = "minibatch",
    random_state: int = 0,
    n_init: int = 10,
):
    """Same as :func:`mingl.tl.n_neighbors.run_mingl_over_n_clusters`, plus ``z_key`` and ``clusterer``.

    ``z_key``, when given, must exist in ``adata.obs`` and is inserted after
    ``y_key`` in the per-cell frame. With the default ``clusterer="minibatch"``
    and ``random_state=0`` the call is forwarded to the 2D function, so results
    are identical to it. ``clusterer="kmeans"`` (full k-means, ``n_init``
    restarts) or a non-zero ``random_state`` run the same sweep and scoring
    here; ``plot_summary`` is ignored on that path.
    """
    if clusterer not in CLUSTERERS:
        raise ValueError(f"clusterer must be one of {CLUSTERERS}, got {clusterer!r}")
    coord_keys = spatial_keys(x_key, y_key, z_key)
    require_columns(adata.obs, [*coord_keys, region_key])

    if clusterer == "minibatch" and random_state == 0:
        out = _run_mingl_over_n_clusters_2d(
            adata,
            knn_feature_cols,
            n_range=n_range,
            output_col_template=output_col_template,
            return_per_cell=return_per_cell,
            plot_summary=plot_summary,
            show=show,
            x_key=x_key,
            y_key=y_key,
            region_key=region_key,
            results_uns_key=results_uns_key,
            cluster_row_order=cluster_row_order,
            order_key=order_key,
        )
        if not return_per_cell or z_key is None:
            return out
        summary_df, per_cell_df = out
        per_cell_df.insert(
            per_cell_df.columns.get_loc(y_key) + 1, z_key, adata.obs.loc[per_cell_df.index, z_key].values
        )
        return summary_df, per_cell_df

    return _sweep(
        adata,
        knn_feature_cols,
        n_range,
        output_col_template,
        return_per_cell,
        coord_keys=coord_keys,
        region_key=region_key,
        results_uns_key=results_uns_key,
        cluster_row_order=cluster_row_order,
        order_key=order_key,
        clusterer=clusterer,
        random_state=random_state,
        n_init=n_init,
    )
