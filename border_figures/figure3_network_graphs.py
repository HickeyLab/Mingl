"""Figure 3 -- MINGL interaction network graphs, one per emission distribution.

A network graph is the most direct check of whether a distribution preserves
known tissue organization: nodes are organizational units, edges are the number
of cells co-positive for that pair, so units that are anatomically adjacent
should be connected and distant ones should not. If an alternative emission model
still recovers e.g. Mucosa-Muscularis mucosa as a dominant edge, it has not
broken the tissue's known architecture.

Both graphs are built and drawn by the shipped MINGL tools --
:func:`mingl.tl.build_neighborhood_pair_graph` and
:func:`mingl.tl.plot_neighborhood_pair_graph`. This driver only swaps the
posterior underneath them, once per emission model.

Note the shipped builder reads the posterior from ``obs`` columns (one per unit)
rather than ``obsm``, and counts a pair only when *exactly two* memberships clear
the threshold -- the manuscript's Figure 3a definition. Both are honoured here.

    python -m border_figures.figure3_network_graphs \
        --data intestine=/path/05_25_HuBMAP_tunit.csv \
        --levels neighborhood tissue_unit
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mingl.tl as mtl
from .config import MODEL_LABELS, MODELS, REFERENCE_THRESHOLD, get_level, resolve_dataset
from .loading import compute_posterior, load_dataset, slim_obs
from .plotting import save_figure, set_style


#: Edges kept per level. The cap binds very differently by scale: the tissue-unit
#: level has only 4 units (6 possible pairs) so a small cap isolates the dominant
#: anatomy, while the neighborhood level has 20 units (190 possible pairs) and
#: needs a larger cap before edge counts can differ between models at all.
TOP_N_BY_LEVEL = {"tissue_unit": 5, "neighborhood": 25}

#: Layout per level. With only four tissue units a circular layout places them
#: evenly and makes which pairs are connected immediately legible; the spring
#: layout is better at the neighborhood scale where structure, not symmetry,
#: carries the information. ``layout_k`` applies to the spring layout only.
LAYOUT_BY_LEVEL = {"tissue_unit": "circular", "neighborhood": "spring"}


def network_graphs_for_level(adata, *, dataset, level_name, models, threshold,
                             cache_dir, cache_tag, out_dir, top_n, layout_k, layout):
    """One MINGL network graph per emission model at one hierarchy level."""
    spec, level = resolve_dataset(dataset), get_level(dataset, level_name)
    written, summaries = [], []

    for model in models:
        probs = compute_posterior(adata, model=model, spec=spec, level=level,
                                  cache_dir=cache_dir, cache_tag=cache_tag)
        # The shipped builder reads obs columns, so write the posterior there.
        prob_cols = [f"_ng_{c}" for c in probs.columns]
        for col, unit in zip(prob_cols, probs.columns):
            adata.obs[col] = probs[unit].to_numpy()

        uns_key = "neighborhood_pair_graph"
        G, pair_df = mtl.build_neighborhood_pair_graph(
            adata, prob_cols, threshold=threshold,
            region_key=spec.region_key, top_n=top_n, uns_key=uns_key,
        )
        # Strip the "_ng_" prefix so nodes carry the real unit names, in place in
        # uns -- the shipped plotter reads the graph from adata.uns, not an
        # argument, so the rename has to happen there.
        import networkx as nx
        G = nx.relabel_nodes(G, {n: str(n).removeprefix("_ng_") for n in G.nodes()})
        adata.uns[uns_key]["graph"] = G

        if G.number_of_edges() == 0:
            print(f"  ! {model}: no pair edges at t={threshold:g}; skipped")
            for col in prob_cols:
                del adata.obs[col]
            continue

        fig = mtl.plot_neighborhood_pair_graph(
            adata, uns_key=uns_key, layout=layout, layout_k=layout_k,
            title=f"{MODEL_LABELS[model].replace(chr(10), ' ')}\n"
                  f"{spec.label} · {level.label} · t={threshold:g}",
            return_fig=True,
        )
        if fig is None:
            fig = plt.gcf()
        written += save_figure(fig, out_dir, f"network_{level_name}_{model}", pdf=False)

        summaries.append({
            "model": model, "level": level_name,
            "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
            "top_edge": max(G.edges(data=True), key=lambda e: e[2].get("weight", 0))[:2]
            if G.number_of_edges() else None,
        })
        print(f"  [{model}] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
              f"(top_n={top_n}, layout={layout})")

        for col in prob_cols:
            del adata.obs[col]
        del probs
        gc.collect()
    return written, summaries


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", required=True, metavar="DATASET=PATH")
    ap.add_argument("--levels", nargs="+", default=["neighborhood", "tissue_unit"])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--threshold", type=float, default=REFERENCE_THRESHOLD)
    ap.add_argument("--top-n", action="append", metavar="LEVEL=N", default=None,
                    help="Edges kept, per level (e.g. --top-n tissue_unit=5 --top-n neighborhood=25). "
                         f"Defaults: {TOP_N_BY_LEVEL}.")
    ap.add_argument("--layout", action="append", metavar="LEVEL=spring|circular", default=None,
                    help=f"Layout per level. Defaults: {LAYOUT_BY_LEVEL}.")
    ap.add_argument("--layout-k", type=float, default=30.0,
                    help="Spring-layout spacing; larger spreads nodes further apart (shipped default 10).")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs" / "figure3_network_graphs"))
    ap.add_argument("--cache-dir", default=str(Path(__file__).resolve().parent / "outputs" / "_cache"))
    args = ap.parse_args(argv)

    top_n_by_level = dict(TOP_N_BY_LEVEL)
    for item in args.top_n or []:
        lv, n = item.split("=", 1)
        top_n_by_level[lv.strip()] = int(n)

    layout_by_level = dict(LAYOUT_BY_LEVEL)
    for item in args.layout or []:
        lv, v = item.split("=", 1)
        layout_by_level[lv.strip()] = v.strip()

    set_style()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for item in args.data:
        name, path = item.split("=", 1)
        paths[resolve_dataset(name).name] = path

    import pandas as pd
    all_summaries, written = [], []
    for ds, path in paths.items():
        spec = resolve_dataset(ds)
        levels = [lv for lv in args.levels if lv in spec.levels]
        deepest = max((get_level(ds, lv) for lv in levels), key=lambda l: l.k)
        extra = tuple({get_level(ds, lv).unit_col for lv in levels}
                      | {get_level(ds, lv).feature_col for lv in levels})
        print(f"Loading {ds} from {path} (levels: {levels})")
        adata = load_dataset(path, ds, deepest, required_extra=extra)
        adata = slim_obs(adata, [spec.cell_type_col, spec.region_key,
                                 spec.x_key, spec.y_key, *extra])
        print(f"  {ds}: {adata.n_obs:,} cells")

        for lv in levels:
            print(f"\n== {ds} / {lv} ==")
            w, s = network_graphs_for_level(
                adata, dataset=ds, level_name=lv, models=tuple(args.models),
                threshold=args.threshold, cache_dir=Path(args.cache_dir),
                cache_tag="ng", out_dir=out_dir,
                top_n=top_n_by_level.get(lv, 15), layout_k=args.layout_k,
                layout=layout_by_level.get(lv, 'spring'),
            )
            written += w; all_summaries += s

    pd.DataFrame(all_summaries).to_csv(out_dir / "network_summary.csv", index=False)
    print(f"\nWrote {len(written)} graphs to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
