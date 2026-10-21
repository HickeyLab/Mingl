"""Build 3D cell tables from volumetric segmentation masks.

Voxel anisotropy is corrected here, once: every centroid leaves this module in
physical units (e.g. µm), which is what every tool in :mod:`mingl.tl3d` assumes.

Axis convention: label volumes are ``(z, y, x)`` arrays and ``voxel_size`` is
``(z, y, x)`` in physical units per voxel. Kuett et al. 2022 3D IMC is
``voxel_size=(2.0, 1.0, 1.0)`` (2 µm sections, 1 µm pixels).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd

__all__ = ["read_label_volume", "label_centroids", "cells_from_label_volume"]


def read_label_volume(path: Union[str, Path], *, memmap: bool = True) -> np.ndarray:
    """Read a ``(z, y, x)`` label TIFF, memory-mapped when the file layout allows it.

    Requires ``tifffile`` (``pip install tifffile``).
    """
    try:
        import tifffile
    except ImportError as exc:  # pragma: no cover - dependency message
        raise ImportError("read_label_volume needs tifffile: pip install tifffile") from exc

    if memmap:
        try:
            return tifffile.memmap(str(path), mode="r")
        except ValueError:
            pass  # compressed / non-contiguous TIFF: fall back to a full read
    return tifffile.imread(str(path))


def label_centroids(
    labels: np.ndarray,
    *,
    voxel_size: Sequence[float] = (1.0, 1.0, 1.0),
    background: int = 0,
) -> pd.DataFrame:
    """Centroid and voxel count of every label in a ``(z, y, x)`` label volume.

    Accumulates per-slice ``bincount`` sums, so memory stays at one z-slice
    (works on memory-mapped volumes of any size).

    Returns a DataFrame indexed by label (background and empty labels dropped)
    with columns ``n_voxels, x, y, z`` -- coordinates in physical units
    (voxel index x ``voxel_size``, measured at voxel centers).
    """
    if labels.ndim != 3:
        raise ValueError(f"Expected a 3D (z, y, x) label volume, got shape {labels.shape}.")
    if len(voxel_size) != 3:
        raise ValueError("voxel_size must be (z, y, x).")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"Label volume must be integer-typed, got {labels.dtype}.")

    n_z, n_y, n_x = labels.shape
    n_labels = int(max(int(labels[z].max()) for z in range(n_z))) + 1

    count = np.zeros(n_labels, dtype=np.int64)
    sum_z = np.zeros(n_labels, dtype=np.float64)
    sum_y = np.zeros(n_labels, dtype=np.float64)
    sum_x = np.zeros(n_labels, dtype=np.float64)

    yy, xx = np.indices((n_y, n_x), dtype=np.float64)
    yy, xx = yy.ravel(), xx.ravel()
    for z in range(n_z):
        flat = np.asarray(labels[z]).ravel().astype(np.int64, copy=False)
        c = np.bincount(flat, minlength=n_labels)
        count += c
        sum_z += c * float(z)
        sum_y += np.bincount(flat, weights=yy, minlength=n_labels)
        sum_x += np.bincount(flat, weights=xx, minlength=n_labels)

    present = count > 0
    present[background] = False
    ids = np.flatnonzero(present)
    n = count[ids].astype(np.float64)
    sz, sy, sx = (float(v) for v in voxel_size)

    return pd.DataFrame(
        {
            "n_voxels": count[ids],
            "x": sum_x[ids] / n * sx,
            "y": sum_y[ids] / n * sy,
            "z": sum_z[ids] / n * sz,
        },
        index=pd.Index(ids, name="label"),
    )


def cells_from_label_volume(
    labels: np.ndarray,
    intensities: Optional[pd.DataFrame] = None,
    *,
    voxel_size: Sequence[float] = (1.0, 1.0, 1.0),
    region: Optional[str] = None,
    region_key: str = "unique_region",
    background: int = 0,
) -> ad.AnnData:
    """AnnData of 3D cells from a label volume plus an optional per-label intensity table.

    Parameters
    ----------
    labels
        ``(z, y, x)`` integer label volume (0 = background).
    intensities
        Per-cell marker table **indexed by label**. Cells are the inner join of
        labels present in the volume and rows of this table; the unmatched counts
        are recorded in ``uns["tl3d_ingest"]``.
    voxel_size
        ``(z, y, x)`` physical size of one voxel.
    region
        Value for ``obs[region_key]`` (one contiguous volume = one region).

    Returns
    -------
    AnnData
        ``obs``: ``label, x, y, z, n_voxels[, region_key]`` with integer-castable
        ``obs_names`` ("0".."n-1", as :func:`mingl.tl3d.KNN2` requires);
        ``X``: intensities as float32 (``n x 0`` when ``intensities`` is None).
    """
    cent = label_centroids(labels, voxel_size=voxel_size, background=background)
    info = {"voxel_size_zyx": [float(v) for v in voxel_size], "n_labels_in_volume": int(len(cent))}

    if intensities is not None:
        if not intensities.index.is_unique:
            raise ValueError("intensities must be indexed by unique labels.")
        shared = cent.index.intersection(intensities.index)
        info["n_labels_without_intensities"] = int(len(cent.index.difference(intensities.index)))
        info["n_intensity_rows_without_label"] = int(len(intensities.index.difference(cent.index)))
        cent = cent.loc[shared]
        X = intensities.loc[shared].to_numpy(dtype=np.float32)
        var = pd.DataFrame(index=[str(c) for c in intensities.columns])
    else:
        X = np.zeros((len(cent), 0), dtype=np.float32)
        var = None

    obs = cent.rename_axis("label").reset_index()[["label", "x", "y", "z", "n_voxels"]]
    if region is not None:
        obs[region_key] = region
    obs.index = obs.index.astype(str)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    info["n_cells"] = int(adata.n_obs)
    adata.uns["tl3d_ingest"] = info
    return adata
