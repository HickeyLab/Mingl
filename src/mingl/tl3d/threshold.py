"""Strict-coordinate wrappers around :mod:`mingl.tl.threshold_sensitivity`.

The 2D border tools already accept arbitrary ``coord_keys``, so 3D works by
passing ``("x", "y", "z")``. The one gap is that they silently drop requested
coordinate columns that are missing -- a missing ``z`` would quietly produce a 2D
result. These wrappers raise instead, then delegate unchanged.

Note: nearest-neighbor distances (``spatial_border_clustering_null``) are not
comparable between a 2D run and a 3D run on the same tissue.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import anndata as ad
import pandas as pd

from ..tl import threshold_sensitivity as _ts
from ._spatial import require_columns

__all__ = [
    "border_metrics_at_threshold",
    "threshold_sensitivity_analysis",
    "spatial_border_clustering_null",
]

_COORDS_3D = ("x", "y", "z")


def border_metrics_at_threshold(
    cells: ad.AnnData,
    *,
    threshold: float,
    coord_keys: Sequence[str] = _COORDS_3D,
    **kwargs,
) -> Dict[str, object]:
    """:func:`mingl.tl.threshold_sensitivity.border_metrics_at_threshold`, 3D by default, strict keys."""
    require_columns(cells.obs, coord_keys)
    return _ts.border_metrics_at_threshold(cells, threshold=threshold, coord_keys=coord_keys, **kwargs)


def threshold_sensitivity_analysis(
    cells: ad.AnnData,
    *,
    coord_keys: Sequence[str] = _COORDS_3D,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """:func:`mingl.tl.threshold_sensitivity.threshold_sensitivity_analysis`, 3D by default, strict keys."""
    require_columns(cells.obs, coord_keys)
    return _ts.threshold_sensitivity_analysis(cells, coord_keys=coord_keys, **kwargs)


def spatial_border_clustering_null(
    cells: ad.AnnData,
    *,
    coord_keys: Sequence[str] = _COORDS_3D,
    region_key: Optional[str] = "unique_region",
    **kwargs,
) -> Dict[str, object]:
    """:func:`mingl.tl.threshold_sensitivity.spatial_border_clustering_null`, 3D by default, strict keys."""
    require_columns(cells.obs, coord_keys)
    return _ts.spatial_border_clustering_null(cells, coord_keys=coord_keys, region_key=region_key, **kwargs)
