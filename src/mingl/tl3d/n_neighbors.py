"""3D-aware wrapper around :func:`mingl.tl.n_neighbors.run_mingl_over_n_clusters`.

The neighborhood-count sweep clusters k-NN *window features*, never coordinates,
so the math is unchanged. Coordinates are only validated and copied into the
per-cell output; this wrapper adds ``z_key`` to both.
"""

from __future__ import annotations

from typing import Optional

from ..tl.n_neighbors import run_mingl_over_n_clusters as _run_mingl_over_n_clusters_2d
from ._spatial import require_columns, spatial_keys

__all__ = ["run_mingl_over_n_clusters"]


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
):
    """Same as :func:`mingl.tl.n_neighbors.run_mingl_over_n_clusters`, plus ``z_key``.

    When ``z_key`` is given it must exist in ``adata.obs`` and is inserted after
    ``y_key`` in the returned per-cell frame. Clustering results are identical to
    the 2D function (they do not depend on coordinates).
    """
    require_columns(adata.obs, [*spatial_keys(x_key, y_key, z_key), region_key])

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
    per_cell_df.insert(per_cell_df.columns.get_loc(y_key) + 1, z_key, adata.obs.loc[per_cell_df.index, z_key].values)
    return summary_df, per_cell_df
