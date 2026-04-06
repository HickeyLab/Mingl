from __future__ import annotations
from typing import Optional, Union, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData

from ._utils import save_figure
from ..tl.edges import findPositives

def edges_positive_probability(
    adata: AnnData,
    *,
    prob_key: str = "neighborhood_probabilities",
    neighborhoods_to_loop: Optional[Sequence[str]] = None,
    threshold: float = 0.25,
    figsize: Union[float, tuple[float, float]] = 15,
    dpi: int = 300,
    palette: str = "tab20",
    legend: bool = True,
    legend_markerscale: float = 10.0,
    legend_title_fontsize: float = 25.0,
    legend_fontsize: float = 25.0,
    title_fontsize: float = 35.0,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    ax: Optional[plt.Axes] = None,
):
    """
    Plot distributions of neighborhood probabilities for neighborhoods that are
    above `threshold` per-cell.

    Parameters
    ----------
    adata
        AnnData with neighborhood probability matrix stored under adata.obsm[prob_key].
        The obs index (adata.obs_names) must align with the rows of the probability matrix.
    prob_key
        Key in adata.obsm where the probability matrix is stored (array-like or DataFrame).
    neighborhoods_to_loop
        Optional list of neighborhood column names to restrict the analysis to a subset.
        If None, all columns present in the probability matrix are used.
    threshold
        Probability threshold used to determine a "positive" neighborhood for a cell.
    figsize
        Figure size. If scalar, interpreted as (figsize, figsize).
    dpi
        Figure DPI.
    palette
        Matplotlib/seaborn palette name for categorical coloring (e.g. 'tab20').
    legend, legend_markerscale, legend_title_fontsize, legend_fontsize
        Legend display and aesthetics.
    title_fontsize
        Title font size.
    show
        Whether to call plt.show(). If None, figure is shown only if not saving.
    save
        If True, save via save_figure using a default base name.
        If str, treated as filename suffix and passed to save_figure.
    ax
        Optional Matplotlib Axes to draw into.

    Returns
    -------
    ax
        Matplotlib Axes containing the plot.
    """
    obs = adata.obs

    # --- validate presence of probability data ---
    if prob_key not in adata.obsm:
        raise KeyError(f"{prob_key!r} not found in adata.obsm")

    # Ensure Count_Above_Threshold exists (adds it in-place to adata.obs)
    adata = findPositives(adata, prob_key=prob_key, threshold=threshold)
    if "Count_Above_Threshold" not in adata.obs:
        raise RuntimeError("findPositives did not populate adata.obs['Count_Above_Threshold']")

    # --- Build probabilities DataFrame aligned to adata.obs index ---
    prob_raw = adata.obsm[prob_key]
    if isinstance(prob_raw, pd.DataFrame):
        probabilities_df = prob_raw.reindex(obs.index).copy()
    else:
        arr = np.asarray(prob_raw)
        if arr.shape[0] != adata.n_obs:
            raise ValueError(
                f"adata.obsm[{prob_key!r}] has {arr.shape[0]} rows but adata.n_obs is {adata.n_obs}"
            )
        cols = [f"N{i}" for i in range(arr.shape[1])]
        probabilities_df = pd.DataFrame(arr, index=adata.obs_names, columns=cols)

    # If user supplied neighborhood subset, restrict columns (but keep Count col separately)
    if neighborhoods_to_loop is not None:
        # accept Sequence[str]
        cols_to_keep = [c for c in neighborhoods_to_loop if c in probabilities_df.columns]
        if not cols_to_keep:
            raise ValueError("neighborhoods_to_loop did not match any probability columns")
        probabilities_df = probabilities_df.loc[:, cols_to_keep]

    # attach Count_Above_Threshold
    probabilities_df["Count_Above_Threshold"] = adata.obs["Count_Above_Threshold"].reindex(probabilities_df.index).astype(int)

    # --- For each cell, produce a sorted list of positive neighborhoods (> threshold) ---
    prob_cols = [c for c in probabilities_df.columns if c != "Count_Above_Threshold"]
    sorted_neighs = probabilities_df[prob_cols].apply(
        lambda row: row[row > threshold].sort_values(ascending=False).index.tolist(), axis=1
    )

    max_n = int(sorted_neighs.apply(len).max() if not sorted_neighs.empty else 0)
    if max_n == 0:
        raise ValueError("No neighborhoods exceed the threshold for any cell. Nothing to plot.")

    # Create Neighborhood1..N and Prob1..N columns
    for i in range(max_n):
        neigh_col = f"Neighborhood{i+1}"
        prob_col = f"Prob{i+1}"
        probabilities_df[neigh_col] = sorted_neighs.apply(lambda x, i=i: x[i] if len(x) > i else None)
        # assign probability values (None -> NaN)
        probabilities_df[prob_col] = probabilities_df.apply(
            lambda row, neigh_col=neigh_col: (row[neigh_col] and row.get(row[neigh_col])) if pd.notna(row[neigh_col]) else None,
            axis=1,
        )
        probabilities_df[prob_col] = pd.to_numeric(probabilities_df[prob_col], errors="coerce")

    # Subset to cells with at least one positive neighborhood
    subset = probabilities_df[probabilities_df["Count_Above_Threshold"] > 0].copy()
    if subset.empty:
        raise ValueError("No cells with Count_Above_Threshold > 0 after processing.")

    # Melt into long form for plotting
    prob_col_names = [f"Prob{i+1}" for i in range(max_n)]
    long_df = pd.melt(
        subset,
        id_vars="Count_Above_Threshold",
        value_vars=prob_col_names,
        var_name="NeighborhoodOrder",
        value_name="Probability",
    ).dropna(subset=["Probability"])

    # Ensure order of NeighborhoodOrder categories (Prob1, Prob2, ...)
    long_df["NeighborhoodOrder"] = pd.Categorical(long_df["NeighborhoodOrder"], categories=prob_col_names, ordered=True)

    # Prepare xtick labels with counts
    xtick_categories = sorted(long_df["Count_Above_Threshold"].unique())
    cell_counts = subset["Count_Above_Threshold"].value_counts().sort_index()
    xtick_labels = [f"{val}\n(N = {cell_counts.get(val, 0)})" for val in xtick_categories]

    # --- Plotting ---
    sns.set(style="whitegrid")

    if isinstance(figsize, (int, float)):
        fig_size = (figsize, figsize)
    else:
        fig_size = figsize

    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    else:
        fig = ax.figure

    # Stripplot (jittered points)
    sns.stripplot(
        data=long_df,
        x="Count_Above_Threshold",
        y="Probability",
        hue="NeighborhoodOrder",
        dodge=True,
        jitter=0.3,
        alpha=0.7,
        size=3,
        palette=palette,
        ax=ax,
        zorder=1,
    )

    # Boxplot overlay (transparent boxes, black edges)
    sns.boxplot(
        data=long_df,
        x="Count_Above_Threshold",
        y="Probability",
        hue="NeighborhoodOrder",
        dodge=True,
        showcaps=True,
        boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 1.2},
        medianprops={"color": "black", "linewidth": 1.2},
        showfliers=False,
        palette=palette,
        ax=ax,
        zorder=2,
    )

    # Format x-axis ticks and labels
    ax.set_xticks(range(len(xtick_categories)))
    ax.set_xticklabels(xtick_labels, fontsize=18)

    # Y ticks
    ax.tick_params(axis="y", labelsize=18)

    ax.set_title("Positive Neighborhood\nProbability Distributions", fontsize=title_fontsize)
    ax.set_xlabel("Number of Neighborhoods > Threshold", fontsize=20)
    ax.set_ylabel("Neighborhood Probability", fontsize=20)

    # Deduplicate legend entries (stripplot+boxplot create duplicates)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if legend:
        ax.legend(
            by_label.values(),
            by_label.keys(),
            title="Positive Neighborhood",
            title_fontsize=legend_title_fontsize,
            fontsize=legend_fontsize,
            markerscale=legend_markerscale,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
    else:
        ax.get_legend().remove()

    fig.tight_layout()

    # show/save logic consistent with project convention
    if show is None:
        show = not bool(save)

    if save:
        base = "edges_positive_probability"
        save_figure(fig, base=base, save=save)

    if show:
        plt.show()

    return ax
