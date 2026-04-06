import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData


def spatial_loc_region(
    adata: AnnData,
    *,
    region: str,
    n1: str,
    n2: str,
    threshold: float = 0.25,
    region_key: str = "filename",
    x_col: str = "x",
    y_col: str = "y",
    figsize=(20, 20),
    dpi: int = 50,
    s_other: float = 1.0,
    s_single: float = 1.0,
    s_both: float = 3.0,
    alpha_other: float = 0.5,
    alpha_single: float = 0.5,
    alpha_both: float = 0.9,
    colors: dict | None = None,
    ax=None,
):
    """
    scverse-compatible spatial overlap plot for two neighborhoods.
    """
    probabilities_df = pd.DataFrame(
        adata.obsm["neighborhood_probabilities"],
        index=adata.obs_names,  # restores cell index
        columns=adata.uns["neighborhood_probability_neighborhoods"]
    )
    if colors is None:
        colors = {"other": "lightgray", "only_1": "plum", "only_2": "blue", "both": "red"}

    region_mask = adata.obs[region_key].astype(str).eq(region)
    #region_mask = (adata.obs[region_key].astype(str).values == str(region)).to_numpy()

    x = adata.obs.loc[region_mask, x_col].to_numpy()
    y = adata.obs.loc[region_mask, y_col].to_numpy()

    p1 = probabilities_df.loc[region_mask, n1].to_numpy()
    p2 = probabilities_df.loc[region_mask, n2].to_numpy()

    # ---- masks ----
    pos_1 = p1 > threshold
    pos_2 = p2 > threshold

    mask_only_1 = pos_1 & ~pos_2
    mask_only_2 = ~pos_1 & pos_2
    mask_both = pos_1 & pos_2
    mask_other = ~(mask_only_1 | mask_only_2 | mask_both)

    border_count = int(mask_both.sum())

    # ---- plotting ----
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    ax.scatter(x[mask_other], y[mask_other], s=s_other, c=colors["other"], alpha=alpha_other, label="Other cells")
    ax.scatter(x[mask_only_1], y[mask_only_1], s=s_single, c=colors["only_1"], alpha=alpha_single, label=f"{n1} only")
    ax.scatter(x[mask_only_2], y[mask_only_2], s=s_single, c=colors["only_2"], alpha=alpha_single, label=f"{n2} only")
    ax.scatter(
        x[mask_both],
        y[mask_both],
        s=s_both,
        c=colors["both"],
        alpha=alpha_both,
        label=f"{n1} + {n2} (n={border_count})",
    )

    ax.set_title(f"Spatial location in {region}\n{n1} vs {n2}", fontsize=22)
    ax.axis("off")
    ax.legend(markerscale=6, fontsize=14, loc="upper right")
    fig.tight_layout()

    masks = {"other": mask_other, "only_1": mask_only_1, "only_2": mask_only_2, "both": mask_both}
    plt.show()
    return fig, ax, masks

