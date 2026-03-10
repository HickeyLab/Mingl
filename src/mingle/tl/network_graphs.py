import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
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
    # Use a separator ONLY for display (not parsing)
    display_sep: str = " ⟷ ",
) -> tuple[nx.Graph, pd.DataFrame]:
    """
    Build a neighborhood-pair graph from `adata.obs[prob_cols]` (numeric probabilities).

    Logic preserved:
    - Count #neighborhood probs > threshold per cell
    - Keep only cells where count == 2
    - For each such cell, record the 2 neighborhoods and its region
    - Count pairs per region, then sum across regions
    - Take top N, build undirected graph with edge weight=count
    """

    # Validate inputs
    missing = [c for c in prob_cols + [region_key] if c not in adata.obs.columns]
    if missing:
        raise KeyError(f"Missing columns in adata.obs: {missing}")

    # Probabilities -> numeric
    probs = adata.obs[prob_cols].apply(pd.to_numeric, errors="coerce")

    # Count > threshold per cell
    counts = (probs > threshold).sum(axis=1)
    adata.obs[count_key] = counts.values

    # Subset: exactly 2 above threshold
    mask = adata.obs[count_key] == 2
    probs2 = probs.loc[mask]
    regions = adata.obs.loc[mask, region_key]

    # Identify the two neighborhoods per row
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

    # Count occurrences of each pair in each region (same grouping)
    pair_counts = (
        pairs_df.groupby([region_key, "Neighborhood1", "Neighborhood2"])
        .size()
        .reset_index(name="count")
    )

    # Sum counts across regions, but KEEP Neighborhood1/2 columns (no ambiguous string parsing)
    pair_counts_summed = (
        pair_counts.groupby(["Neighborhood1", "Neighborhood2"], as_index=False)["count"].sum()
    )

    # Filter and sort
    pair_counts_summed_filtered = pair_counts_summed[pair_counts_summed["count"] > 0]
    pair_counts_summed_sorted = pair_counts_summed_filtered.sort_values("count", ascending=False)

    # Top N
    top_pairs = pair_counts_summed_sorted.head(top_n).reset_index(drop=True)

    # Add display label (safe; never parsed)
    top_pairs["Neighborhood Pair"] = (
        top_pairs["Neighborhood1"] + display_sep + top_pairs["Neighborhood2"]
    )

    # Build graph
    G = nx.Graph()
    for _, row in top_pairs.iterrows():
        n1 = row["Neighborhood1"]
        n2 = row["Neighborhood2"]
        G.add_edge(n1, n2, weight=int(row["count"]))

    # Store in adata.uns
    adata.uns[uns_key] = {
        "threshold": threshold,
        "prob_cols": list(prob_cols),
        "region_key": region_key,
        "top_n": top_n,
        "display_sep": display_sep,
        "pair_counts_by_region": pair_counts,
        "pair_counts_summed": pair_counts_summed,
        "top_pairs": top_pairs,
        "graph": G,
    }

    return G, top_pairs


def plot_neighborhood_pair_graph(
    adata: AnnData,
    *,
    uns_key: str = "neighborhood_pair_graph",
    layout_k: float = 10.0,
    seed: int = 42,
    figsize: tuple[float, float] = (15, 10),
    dpi: int = 100,
    node_size: int = 1200,
    font_size: int = 12,
    title: str = "Top Neighborhood Pairs Network",
    title_fontsize: int = 26,
    edge_color: str = "gray",
    # edge width styling
    min_edge_width: float = 1.0,
    max_edge_width: float = 8.0,
    # node palette styling
    palette_mode: str = "tab20",   # "tab20" or "strict"
    strict_palette_names: tuple[str, ...] = ("tab20", "Set3", "Set2", "Paired", "Dark2", "Accent"),
    default_node_color: str = "lightgray",
) -> None:
    """
    Plot style updated to match your example:
    - node colors come from a categorical palette (tab20 cycling or strict combined palettes)
    - edges are gray, widths represent connection strength (normalized weights)
    - labels drawn via plt.text for styling consistency
    """

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.cm import get_cmap

    if uns_key not in adata.uns:
        raise KeyError(f"adata.uns does not contain '{uns_key}'. Run build_* first.")

    payload = adata.uns[uns_key]
    G: nx.Graph = payload["graph"]

    if G.number_of_edges() == 0:
        raise ValueError("Graph has no edges to plot (no qualifying pairs found).")

    # --- Layout (same as before) ---
    pos = nx.spring_layout(G, k=layout_k, seed=seed)

    # --- Edge widths represent connection strength (weight) ---
    weights = np.array([G[u][v].get("weight", 1.0) for u, v in G.edges()], dtype=float)
    if weights.size == 0:
        normalized_weights = []
    else:
        wmin, wmax = float(weights.min()), float(weights.max())
        if np.isclose(wmin, wmax):
            normalized_weights = np.full_like(weights, (min_edge_width + max_edge_width) / 2.0, dtype=float)
        else:
            normalized_weights = min_edge_width + (weights - wmin) / (wmax - wmin) * (max_edge_width - min_edge_width)

    # --- Node colors (two options) ---
    unique_nodes = list(G.nodes())

    if palette_mode == "tab20":
        tab20 = get_cmap("tab20")
        node_colors_dict = {node: tab20(i % 20) for i, node in enumerate(unique_nodes)}
        node_colors = [node_colors_dict[node] for node in G.nodes()]

    elif palette_mode == "strict":
        # Make a long, non-repeating-ish palette by combining qualitative palettes
        combined_colors = []
        for name in strict_palette_names:
            combined_colors.extend(sns.color_palette(name))

        if len(combined_colors) < len(unique_nodes):
            raise ValueError(
                f"Not enough distinct colors for {len(unique_nodes)} nodes. "
                f"Max supported is {len(combined_colors)}. Add more palettes or reduce nodes."
            )

        node_colors_dict = dict(zip(sorted(unique_nodes), combined_colors[: len(unique_nodes)]))
        node_colors = [node_colors_dict.get(node, default_node_color) for node in G.nodes()]

    else:
        raise ValueError("palette_mode must be 'tab20' or 'strict'.")

    # --- Plot (matches your style) ---
    plt.figure(figsize=figsize, dpi=dpi)

    # Nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors)

    # Edges (gray, width by normalized weights)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=normalized_weights)

    # Labels via plt.text
    _texts = [
        plt.text(x, y, node, fontsize=font_size, ha="center", va="center")
        for node, (x, y) in pos.items()
    ]

    plt.title(title, fontsize=title_fontsize)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

