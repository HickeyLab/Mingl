import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from anndata import AnnData


def build_neighborhood_pair_graph(
    adata: AnnData,
    prob_cols: list[str],
    *,
    threshold: float = 0.25,
    region_key: str = "unique_region",
    top_n: int = 15,
    count_key: str = "Count_Above_Threshold",
    uns_key: str = "neighborhood_pair_graph",
    display_sep: str = " ⟷ ",
) -> tuple[nx.Graph, pd.DataFrame]:

    # Validate inputs
    missing = [c for c in prob_cols + [region_key] if c not in adata.obs.columns]
    if missing:
        raise KeyError(f"Missing columns in adata.obs: {missing}")

    probs = adata.obs[prob_cols].apply(pd.to_numeric, errors="coerce")

    counts = (probs > threshold).sum(axis=1)
    adata.obs[count_key] = counts.values

    mask = adata.obs[count_key] == 2
    probs2 = probs.loc[mask]
    regions = adata.obs.loc[mask, region_key]

    above = probs2.gt(threshold)
    pair_list = []
    region_list = []

    for idx in above.index:
        cols = list(above.columns[above.loc[idx].values])
        if len(cols) == 2:
            pair_list.append((cols[0], cols[1]))
            region_list.append(regions.loc[idx])

    pairs_df = pd.DataFrame(pair_list, columns=["Neighborhood1", "Neighborhood2"])
    pairs_df[region_key] = region_list

    pair_counts = (
        pairs_df.groupby([region_key, "Neighborhood1", "Neighborhood2"])
        .size()
        .reset_index(name="count")
    )

    pair_counts_summed = (
        pair_counts.groupby(["Neighborhood1", "Neighborhood2"], as_index=False)["count"].sum()
    )

    pair_counts_summed = pair_counts_summed.sort_values("count", ascending=False)
    top_pairs = pair_counts_summed.head(top_n).reset_index(drop=True)

    top_pairs["Neighborhood Pair"] = (
        top_pairs["Neighborhood1"] + display_sep + top_pairs["Neighborhood2"]
    )

    G = nx.Graph()
    for _, row in top_pairs.iterrows():
        G.add_edge(row["Neighborhood1"], row["Neighborhood2"], weight=int(row["count"]))

    adata.uns[uns_key] = {
        "graph": G,
        "top_pairs": top_pairs,
        "pair_counts": pair_counts,
        "pair_counts_summed": pair_counts_summed,
    }

    return G, top_pairs


def plot_neighborhood_pair_graph(
    adata: AnnData,
    *,
    uns_key: str = "neighborhood_pair_graph",
    layout: str = "spring",
    layout_k: float = 10.0,
    seed: int = 42,
    figsize: tuple = (12, 12),
    dpi: int = 300,
    node_size: int = 4000,
    font_size: int = 14,
    title: str = "Neighborhood Network",
    title_fontsize: int = 20,
    edge_color: str = "gray",
    min_edge_width: float = 1.0,
    max_edge_width: float = 10.0,

    # 🔥 NEW: legend controls
    edge_legend_values: list[float] | None = None,
    edge_legend_title: str = "Number of Cells",
    edge_legend_fontsize: int = 16,
    edge_legend_title_fontsize: int = 18,
    edge_legend_loc: str = "lower right",
    edge_legend_bbox: tuple = (0.98, 0.02),
):
    if uns_key not in adata.uns:
        raise KeyError(f"{uns_key} not found in adata.uns")

    G: nx.Graph = adata.uns[uns_key]["graph"]

    if G.number_of_edges() == 0:
        raise ValueError("Graph has no edges.")

    # --- Layout ---
    if layout == "spring":
        pos = nx.spring_layout(G, k=layout_k, seed=seed)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        raise ValueError("layout must be 'spring' or 'circular'")

    # --- Edge weights ---
    weights = np.array([G[u][v]["weight"] for u, v in G.edges()], dtype=float)

    w_min, w_max = float(weights.min()), float(weights.max())
    if np.isclose(w_min, w_max):
        w_max = w_min + max(1.0, abs(w_min) * 0.1)

    normalized_weights = [
        min_edge_width + (w - w_min) / (w_max - w_min) * (max_edge_width - min_edge_width)
        for w in weights
    ]

    # --- Node colors ---
    cmap = plt.get_cmap("tab20")
    node_colors = [cmap(i % 20) for i, _ in enumerate(G.nodes())]

    # --- Plot ---
    plt.figure(figsize=figsize, dpi=dpi)

    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=normalized_weights)

    for node, (x, y) in pos.items():
        plt.text(x, y, node, fontsize=font_size, ha="center", va="center")

    # =========================
    # 🔥 EDGE LEGEND (YOUR STYLE)
    # =========================
    if edge_legend_values is not None:

        def to_width(w):
            return min_edge_width + (w - w_min) / (w_max - w_min) * (max_edge_width - min_edge_width)

        legend_vals = np.clip(np.array(edge_legend_values), w_min, w_max)
        legend_widths = [to_width(w) for w in legend_vals]

        from matplotlib.lines import Line2D

        legend_handles = [
            Line2D(
                [],
                [],
                color=edge_color,
                linewidth=lw,
                solid_capstyle="round",
                label=f"{int(val):,}",
            )
            for val, lw in zip(edge_legend_values, legend_widths)
        ]

        plt.legend(
            handles=legend_handles,
            title=edge_legend_title,
            fontsize=edge_legend_fontsize,
            title_fontsize=edge_legend_title_fontsize,
            loc=edge_legend_loc,
            bbox_to_anchor=edge_legend_bbox,
            frameon=False,
        )

    plt.title(title, fontsize=title_fontsize)
    plt.axis("off")
    plt.tight_layout()
    plt.show()