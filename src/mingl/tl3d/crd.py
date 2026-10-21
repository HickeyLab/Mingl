"""3D pass-through for the region-local recompute (:func:`mingl.tl.crd.crd2`).

``crd2`` never uses coordinates in its math; it only copies ``x``/``y`` into its
output tables. This wrapper runs it unchanged, then carries ``z`` (looked up by
``cellid`` from ``windows2``) into both outputs and rewrites the CSVs.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ..tl.crd import crd2 as _crd2_2d

__all__ = ["crd2"]


def crd2(
    *,
    windows2: pd.DataFrame,
    out_probs_path: str = "local_probs.csv",
    out_delta_path: str = "delta_probs.csv",
    z_key: Optional[str] = "z",
    **kwargs,
):
    """:func:`mingl.tl.crd.crd2` that also carries ``z`` into its outputs.

    ``z_key`` names the z column in ``windows2``; if it is ``None`` or absent,
    the result is exactly the 2D function's. All other arguments are forwarded.
    """
    final_probs_df, final_deltas_df = _crd2_2d(
        windows2=windows2, out_probs_path=out_probs_path, out_delta_path=out_delta_path, **kwargs
    )
    if z_key is None or z_key not in windows2.columns:
        return final_probs_df, final_deltas_df

    z_by_cell = pd.Series(windows2[z_key].to_numpy(), index=windows2["cellid"].to_numpy())
    if not z_by_cell.index.is_unique:
        raise ValueError("windows2['cellid'] must be unique to carry z into crd2 outputs.")

    for df, path in ((final_probs_df, out_probs_path), (final_deltas_df, out_delta_path)):
        at = df.columns.get_loc("y") + 1 if "y" in df.columns else len(df.columns)
        df.insert(at, "z", df["cellid"].map(z_by_cell).to_numpy())
        df.to_csv(path, index=False)

    return final_probs_df, final_deltas_df
