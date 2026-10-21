"""3D-capable probability-ratio Score and transition-cluster construction.

Mirrors :mod:`mingl.tl.grad`. :class:`Neighborhoods` subclasses the 2D class and
only overrides the two methods that read coordinates; the binning, fold-change
and ordering helpers are imported unchanged. See the 2D module docstring for the
step-by-step algorithm.

In 3D, ``distance_max`` is a *sphere* radius in the coordinate units (µm).
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from ..tl.grad import Neighborhoods as _Neighborhoods2D
from ..tl.grad import _col_score, assign_probability_level_with_edges
from ._spatial import spatial_keys

__all__ = ["Neighborhoods", "mingl_neighborhoods_scverse"]


class Neighborhoods(_Neighborhoods2D):
    """:class:`mingl.tl.grad.Neighborhoods` with an optional third axis ``Z``."""

    def __init__(self, cells, ks, cluster_col, sum_cols, keep_cols, neigh,
                 X="X:X", Y="Y:Y", reg="Exp", add_dummies=True, Z=None):
        super().__init__(cells, ks, cluster_col, sum_cols, keep_cols, neigh,
                         X=X, Y=Y, reg=reg, add_dummies=add_dummies)
        self.Z = Z
        self.coord_cols = spatial_keys(X, Y, Z)

    def get_tissue_chunks(self):
        self.tissue_group = self.cells[[*self.coord_cols, self.reg]].groupby(self.reg)
        return [
            (time.time(), self.exps.index(t), t, a)
            for t, indices in self.tissue_group.groups.items()
            for a in np.array_split(indices, 1)
        ]

    def make_windows(self, job):
        start_time, idx, tissue_name, indices = job
        job_start = time.time()
        print("Starting:", str(idx + 1) + "/" + str(len(self.exps)), ": " + self.exps[idx])

        tissue = self.tissue_group.get_group(tissue_name)
        to_fit = tissue.loc[indices][self.coord_cols].values

        fit = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(tissue[self.coord_cols].values)
        m = fit.kneighbors(to_fit)
        # don't include the index cell in its own window
        m = m[0][:, 1:], m[1][:, 1:]

        args = m[0].argsort(axis=1)
        add = np.arange(m[1].shape[0]) * m[1].shape[1]
        sorted_indices = m[1].flatten()[args + add[:, None]]
        sorted_dists = m[0].flatten()[args + add[:, None]]
        neighbors = tissue.index.values[sorted_indices].astype(np.int32)

        end_time = time.time()
        print("Finishing:", str(idx + 1) + "/" + str(len(self.exps)), ": " + self.exps[idx],
              end_time - job_start, end_time - start_time)
        return neighbors, sorted_dists


def mingl_neighborhoods_scverse(
    adata: ad.AnnData,
    *,
    tu1="Inner Follicle",
    tu2="Outer Follicle",
    extra_cols: Optional[Sequence[str]] = None,
    ks=(10, 20, 30, 50, 100),
    k=20,
    distance_max=1000,
    X="x",
    Y="y",
    Z=None,
    reg="unique_region",
    neigh="Neighborhood",
    cluster_col="Probability_Level",
    celltype="Cell Type",
    tiss="Tissue Unit",
    comm="Community",
    n_bins=5,
    bin_labels=("Very Low", "Low", "Medium", "High", "Very High"),
    use_quantiles=True,
    n_clusters=5,
    target_neighborhoods=("Inner Follicle", "Outer Follicle"),
    random_state=0,
    out_score_key="Score",
    out_prob_level_key="Probability_Level",
    out_neighborhood_key=None,
    out_prob_cluster_key="Probability_Bin_Cluster",
    store_windows_key="mingl_windows",
    store_fc_key="mingl_fc",
):
    """3D-capable :func:`mingl.tl.grad.mingl_neighborhoods_scverse`.

    ``Z=None`` reproduces the 2D function. ``extra_cols=None`` builds the
    required ``obs`` columns from the key arguments
    (``X, Y[, Z], reg, neigh, celltype, tiss, comm``), which for default keys is
    the 2D function's default tuple.

    Writes / stores / returns exactly what the 2D function does.
    """
    del cluster_col  # accepted for signature parity; the 2D function ignores it too

    if out_neighborhood_key is None:
        out_neighborhood_key = f"neighborhood{k}"
    coord_cols = spatial_keys(X, Y, Z)
    if extra_cols is None:
        extra_cols = (*coord_cols, reg, neigh, celltype, tiss, comm)
    else:
        extra_cols = tuple(extra_cols) + tuple(c for c in coord_cols if c not in extra_cols)

    obs = adata.obs.copy()
    missing = [c for c in [tu1, tu2, *extra_cols] if c not in obs.columns]
    if missing:
        raise KeyError(f"Missing required adata.obs columns: {missing}")

    df2 = obs[[tu1, tu2, *extra_cols]].copy()

    eps = 1e-9
    df2["ratio"] = (df2[tu1].astype(float) + eps) / (df2[tu2].astype(float) + eps)
    df2["log_ratio"] = np.log(df2["ratio"])
    df2["max_prob"] = df2[[tu1, tu2]].astype(float).max(axis=1)
    df2[out_score_key] = df2["log_ratio"] * df2["max_prob"]

    df2[out_prob_level_key], edges_used = assign_probability_level_with_edges(
        df2[out_score_key], n_bins=n_bins, labels=list(bin_labels), use_quantiles=use_quantiles
    )

    df = df2.copy().reset_index(drop=True)
    sum_cols = list(df[out_prob_level_key].unique())
    keep_cols = [*coord_cols, reg, neigh, celltype, comm, tiss, out_prob_level_key]

    Neigh = Neighborhoods(
        df,
        ks=list(ks),
        cluster_col=out_prob_level_key,
        sum_cols=sum_cols,
        keep_cols=keep_cols,
        neigh=neigh,
        X=X, Y=Y, Z=Z, reg=reg,
        add_dummies=True,
    )
    cluster_name_windows = Neigh.k_windows(distance_max=distance_max)
    windows_k = cluster_name_windows[k].copy()

    df = pd.concat([df, pd.get_dummies(df[out_prob_level_key], dtype=int)], axis=1)
    sum_cols2 = pd.get_dummies(df[out_prob_level_key], dtype=int).columns.tolist()

    if not windows_k.index.equals(df.index):
        if len(windows_k) == len(df):
            windows_k = windows_k.reset_index(drop=True)
            df = df.reset_index(drop=True)
        else:
            raise ValueError("windows_k and df indices differ and lengths differ; align them before filtering.")

    mask = df[neigh].isin(list(target_neighborhoods))
    windows_sub = windows_k.loc[mask].copy()
    df_sub = df.loc[mask].copy()

    km = MiniBatchKMeans(n_clusters=int(n_clusters), random_state=random_state)
    if set(sum_cols2).issubset(windows_sub.columns):
        feature_matrix = windows_sub[sum_cols2].values
    else:
        feature_matrix = df_sub[sum_cols2].values
    labels_sub = km.fit_predict(feature_matrix)

    df[out_neighborhood_key] = -1
    df.loc[mask, out_neighborhood_key] = labels_sub
    df_sub[out_neighborhood_key] = labels_sub

    niche_clusters = km.cluster_centers_
    tissue_avgs = df_sub[sum_cols2].values.mean(axis=0)
    eps_fc = 1e-12
    tissue_row = tissue_avgs.reshape(1, -1)
    niche_plus = niche_clusters + tissue_row
    normed = niche_plus / (niche_plus.sum(axis=1, keepdims=True) + eps_fc)
    fc = pd.DataFrame(np.log2((normed + eps_fc) / (tissue_row + eps_fc)), columns=sum_cols2)

    desired_col_order = sorted(list(fc.columns), key=lambda c: -_col_score(c))
    if len(set(_col_score(c) for c in fc.columns)) == 1:
        desired_col_order = list(sum_cols2[::-1])
    high_bin = desired_col_order[0]
    ordered_rows = fc[high_bin].sort_values(ascending=False).index.tolist()
    fc = fc.reindex(index=ordered_rows, columns=desired_col_order)

    conv = {i: str(i) for i in range(int(n_clusters))}
    df_sub[out_prob_cluster_key] = df_sub[out_neighborhood_key].map(conv)

    adata.obs[out_score_key] = df2[out_score_key].values
    adata.obs[out_prob_level_key] = df2[out_prob_level_key].values
    adata.obs[out_neighborhood_key] = df[out_neighborhood_key].values

    full_prob_cluster = pd.Series([np.nan] * len(df), index=df.index, dtype="object")
    full_prob_cluster.loc[mask] = df_sub[out_prob_cluster_key].values
    adata.obs[out_prob_cluster_key] = full_prob_cluster.values

    adata.uns["mingl_probability_edges"] = edges_used
    adata.uns.setdefault(store_windows_key, {})
    adata.uns[store_windows_key][int(k)] = windows_k
    adata.uns[store_fc_key] = fc

    return adata, df_sub, fc, windows_k, edges_used
