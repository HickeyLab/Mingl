"""Coordinate handling shared by every 3D-capable tool.

All 2D/3D branching lives here: ``z_key=None`` means the classic 2D ``(x, y)``
path, ``z_key="z"`` adds a third Euclidean axis. Coordinates are assumed to be in
isotropic physical units (e.g. µm) -- voxel anisotropy is corrected at ingest
(:mod:`mingl.tl3d.ingest`), not here.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


def spatial_keys(x_key: str, y_key: str, z_key: Optional[str] = None) -> List[str]:
    """Ordered coordinate column names: ``[x, y]`` or ``[x, y, z]``."""
    return [x_key, y_key] if z_key is None else [x_key, y_key, z_key]


def n_spatial_dims(x_key: str, y_key: str, z_key: Optional[str] = None) -> int:
    return len(spatial_keys(x_key, y_key, z_key))


def require_columns(df: pd.DataFrame, cols: Iterable[str], *, what: str = "adata.obs") -> None:
    """Raise ``KeyError`` naming every requested column that is missing.

    Unlike some 2D tools, which silently drop absent coordinate columns, this
    never lets a requested ``z`` quietly degrade a run to 2D.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s) in {what}: {missing}")


def get_coords(
    df: pd.DataFrame,
    x_key: str = "x",
    y_key: str = "y",
    z_key: Optional[str] = None,
    *,
    dtype=None,
) -> np.ndarray:
    """Return an ``(n, 2)`` or ``(n, 3)`` coordinate array from ``df``.

    With ``dtype=None`` the values come back exactly as pandas' ``.values`` would
    give them, so 2D callers reproduce the original tools bit-for-bit.
    """
    cols = spatial_keys(x_key, y_key, z_key)
    require_columns(df, cols)
    values = df[cols].values
    return values if dtype is None else values.astype(dtype, copy=False)
