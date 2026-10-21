"""Loaders for public 3D spatial datasets.

Kuett et al. 2022, Nature Cancer -- 3D imaging mass cytometry of breast cancer
(Zenodo 10.5281/zenodo.4752030, CC-BY). Each deposited model folder holds:

* ``measured_mask_*.tif``   -- (z, y, x) uint32 3D cell labels for the measured crop
* ``*labels_area.csv``      -- ``label, area`` (area = voxel count in the measured mask)
* ``*mean_intensities.csv`` -- per-cell mean marker intensities, **no label column**;
  rows are in ``labels_area.csv`` order (verified on the LVI lymph model: recomputed
  channel means correlate at r >= 0.99 row-aligned vs ~0.13 when shifted by one row)
* ``full_mask_*.tif``       -- labels for the whole acquired volume (not used here)

Voxels are 1 x 1 µm in-plane and 2 µm in z.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import anndata as ad
import pandas as pd

from .ingest import cells_from_label_volume, read_label_volume

__all__ = ["KUETT2022_VOXEL_SIZE", "KUETT2022_MODELS", "load_kuett2022_model"]

KUETT2022_VOXEL_SIZE = (2.0, 1.0, 1.0)  # (z, y, x) µm
KUETT2022_MODELS = (
    "MainHer2BreastCancerModel",
    "SecondHer2BreastCancerModel",
    "LVILymphBreastCancerModel",
    "LVIBloodBreastCancerModel",
)


def _one(model_dir: Path, pattern: str) -> Path:
    hits = sorted(model_dir.glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected exactly one {pattern!r} in {model_dir}, found {[h.name for h in hits]}")
    return hits[0]


def load_kuett2022_model(model_dir: Union[str, Path], *, region: str | None = None) -> ad.AnnData:
    """Build a 3D cell AnnData for one extracted Kuett 2022 model folder.

    Returns an AnnData with ``obs`` = ``label, x, y, z, n_voxels,
    unique_region`` (coordinates in µm, measured-crop frame) and ``X`` = the
    deposited mean intensities (``var_names`` = marker names). Raises if the
    deposited voxel counts disagree with the mask, which would mean the
    label/row correspondence is broken.
    """
    model_dir = Path(model_dir)
    region = region or model_dir.name

    labels_area = pd.read_csv(_one(model_dir, "*labels_area.csv"))
    intensities = pd.read_csv(_one(model_dir, "*mean_intensities.csv"))
    if len(labels_area) != len(intensities):
        raise ValueError(
            f"labels_area has {len(labels_area)} rows but mean_intensities has {len(intensities)}; "
            "cannot align cells to labels."
        )
    intensities.index = pd.Index(labels_area["label"].to_numpy(), name="label")

    mask = read_label_volume(_one(model_dir, "measured_mask_*.tif"))
    adata = cells_from_label_volume(mask, intensities, voxel_size=KUETT2022_VOXEL_SIZE, region=region)

    area = labels_area.set_index("label")["area"].reindex(adata.obs["label"].to_numpy()).to_numpy()
    if not (area == adata.obs["n_voxels"].to_numpy()).all():
        raise ValueError("Deposited label areas do not match voxel counts in the measured mask.")
    if adata.n_obs != len(labels_area):
        raise ValueError(f"{len(labels_area) - adata.n_obs} deposited labels are missing from the measured mask.")

    adata.uns["dataset"] = {
        "name": "Kuett2022_3D_IMC",
        "model": model_dir.name,
        "doi": "10.5281/zenodo.4752030",
        "voxel_size_zyx_um": list(KUETT2022_VOXEL_SIZE),
        "mask_shape_zyx": list(mask.shape),
        "coordinate_frame": "measured_mask crop, µm",
    }
    return adata
