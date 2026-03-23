from __future__ import annotations
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData

from ._utils import save_figure  # same helper used elsewhere


def spatial_neighborhood_plot(
    adata: AnnData,
    *,
    desired_region: str,
    prob_key: str = "neighborhood_probabilities",
    x_key: str = "x",
    y_key: str = "y",
    neighborhood_key: str = "neighborhood",      # matches centroid_Calculation
    region_key: str = "unique_region",           # matches KNN / centroid_Calculation
    figsize: Union[float, tuple[float, float]] = 30,
    dpi: int = 300,
    s: float = 1.0,
    palette: str = "tab20",
    legend: bool = True,
    legend_markerscale: float = 15.0,
    legend_title_fontsize: float = 35.0,
    legend_fontsize: float = 35.0,
    title_fontsize: float = 35.0,
    invert_y: bool = False,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    ax: Optional[plt.Axes] = None,
):
    """
    Spatial scatter plot of cells colored by assigned neighborhood for a single region.

    This is the AnnData-based, scverse-style version of your original:

        df -> filtered_cells -> probabilities_df -> visualization_df -> catplot2(...)

    but:
      - uses plt.scatter only
      - uses the same key names as the KNN / centroid_Calculation code.

    Parameters
    ----------
    adata
        AnnData with:
          - adata.obs[x_key], adata.obs[y_key]       (spatial coords)
          - adata.obs[neighborhood_key]              (assigned neighborhood)
          - adata.obs[region_key]                    (region / unique_region)
          - adata.obsm[prob_key]                     (neighborhood probabilities)
    desired_region
        Region / unique_region to plot (e.g. '14_06_23_reg002.tsv').
    prob_key
        Key in adata.obsm where the neighborhood probability matrix is stored.
        Expected: shape (n_cells, n_neighborhoods), columns matching neighborhood labels.
    x_key, y_key
        Column names in adata.obs for x/y coordinates.
    neighborhood_key
        Column in adata.obs with the assigned neighborhood per cell.
    region_key
        Column in adata.obs specifying the region / unique_region.
    figsize
        Figure size. If scalar, interpreted as (figsize, figsize).
    dpi
        Figure DPI.
    s
        Marker size for points.
    palette
        Matplotlib categorical colormap name for neighborhoods (e.g. 'tab20').
    legend
        Whether to draw a legend.
    legend_markerscale, legend_title_fontsize, legend_fontsize
        Legend aesthetics.
    title_fontsize
        Title font size.
    invert_y
        If True, invert y-axis (useful for image-like coordinates).
    show
        Whether to show the figure. If None, show only if not saving.
    save
        If True, save via save_figure; if str, treated as suffix/filename.
    ax
        Optional Matplotlib Axes to draw into.

    Returns
    -------
    ax
        Matplotlib Axes with the plot.
    """
    obs = adata.obs

    if prob_key not in adata.obsm:
        raise KeyError(f"{prob_key!r} not found in adata.obsm")

    # --- 0. Build probabilities_df aligned with obs ---
    prob_raw = adata.obsm[prob_key]
    if isinstance(prob_raw, pd.DataFrame):
        probabilities_df = prob_raw.reindex(obs.index)
    else:
        # if it's an array-like, we assume columns are neighborhoods in same order
        probabilities_df = pd.DataFrame(prob_raw, index=obs.index)

    # --- 1. Filter for the region you want to plot (like your filtered_cells) ---
    filtered_cells = obs[obs[region_key] == desired_region].copy()
    if filtered_cells.empty:
        raise ValueError(
            f"No cells found for region {desired_region!r} in {region_key!r}."
        )

    filtered_probabilities_df = probabilities_df.loc[filtered_cells.index]

    # --- 2. Assigned neighborhoods + probabilities (same logic as your snippet) ---
    assigned_neighborhoods = filtered_cells[neighborhood_key]

    assigned_probabilities = filtered_probabilities_df.reindex(
        filtered_cells.index
    ).apply(
        lambda row: row[filtered_cells.loc[row.name, neighborhood_key]],
        axis=1,
    )

    # --- 3. visualization_df, mirroring your original DataFrame ---
    visualization_df = pd.DataFrame({
        "x": filtered_cells[x_key],
        "y": filtered_cells[y_key],
        "Assigned Neighborhood": assigned_neighborhoods,
        "Assigned Probability": assigned_probabilities,
        "unique_region": filtered_cells[region_key],
    })

    # --- 4. Plotting via plt.scatter (no catplot2) ---
    # handle figsize as scalar or tuple
    if isinstance(figsize, (int, float)):
        fig_size = (figsize, figsize)
    else:
        fig_size = figsize

    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    else:
        fig = ax.figure

    # Color by Assigned Neighborhood (equivalent to catplot2 hue=...)
    neighborhoods = visualization_df["Assigned Neighborhood"].astype("category")
    categories = neighborhoods.cat.categories

    cmap = plt.get_cmap(palette)
    colors = cmap(np.linspace(0, 1, len(categories)))

    for color, cat in zip(colors, categories):
        sub = visualization_df[neighborhoods == cat]
        ax.scatter(
            sub["x"],
            sub["y"],
            s=s,
            c=[color],
            label=str(cat),
        )

    title = f"Assigned Neighborhoods\n({desired_region})"
    ax.set_title(title, fontsize=title_fontsize)

    if invert_y:
        ax.invert_yaxis()

    # legend aesthetics (like your rcParams + catplot2)
    if legend:
        ax.legend(
            title="Assigned Neighborhood",
            markerscale=legend_markerscale,
            fontsize=legend_fontsize,
            title_fontsize=legend_title_fontsize,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )

    # remove axes & tidy
    ax.set_axis_off()
    fig.tight_layout()

    # show/save logic
    if show is None:
        show = not bool(save)

    if save:
        base = f"spatial_neighborhood_{desired_region}"
        save_figure(fig, base=base, save=save)

    if show:
        plt.show()

    return ax
