"""Cell-typing summary figures for the Kuett 2022 Main HER2 volume.

Two annotated heatmaps of marker expression (mean z-score of arcsinh intensity,
the same values ``01_celltype_main_her2.py`` clustered on):

* ``fig_leiden_clusters_marker_expression``: one row per Leiden cluster, with
  the cell type each cluster was assigned to and its cell count on the right.
* ``fig_cell_types_marker_expression``: one row per final cell type, with the
  Leiden clusters that were merged into it.

Rows are grouped by lineage, columns by marker family; the state markers
(right-most block) were not used for clustering. Numbers in the cells are the
unclipped means; color saturates at +/-2.

Input:  results/main_her2/main_her2_celltyped.h5ad (from 01_celltype_main_her2.py)
Output: results/main_her2/figures/
Run:    .venv/bin/python analysis_3d/04_celltype_figures_main_her2.py
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle

DATA = Path.home() / "Desktop/Mingle_data/kuett2022_3dimc"
IN_H5AD = DATA / "results/main_her2/main_her2_celltyped.h5ad"
OUT = DATA / "results/main_her2/figures"

# Lineage groups (row blocks), in display order, each with its cell types in display order.
LINEAGE: dict[str, list[str]] = {
    "Tumor (epithelial)": [
        "Tumor HER2-high", "Tumor HER2+ CK7-", "Tumor CK7+ CK19+", "Tumor CK14+",
        "Tumor marker-low", "Tumor SMA+ CK5+",
    ],
    "Myoepithelial": ["Myoepithelial"],
    "Stroma / vascular": [
        "Fibroblasts (Vimentin+)", "Fibroblasts (Collagen+)", "Stroma (other)", "Endothelial / vascular",
    ],
    "Immune": ["T cells", "B cells", "Macrophages", "Plasma cells"],
    "Ambiguous": ["Mixed epithelial/immune (ambiguous)"],
    "Unassigned": ["Unassigned (low signal / artifact)"],
}
LINEAGE_COLOR = {
    "Tumor (epithelial)": "#eb6834",
    "Myoepithelial": "#6250d6",
    "Stroma / vascular": "#1baf7a",
    "Immune": "#008300",
    "Ambiguous": "#eda100",
    "Unassigned": "#8c8b86",
}

# Marker families (column blocks), in display order.
MARKER_GROUPS: dict[str, list[str]] = {
    "Epithelial / tumor": ["panCK", "CK5", "CK7", "CK8/18", "CK14", "CK19", "E/P-Cadherin", "HER2 (bis)"],
    "Immune": ["CD45", "CD3", "CD8a", "CD20", "CD68", "CD138"],
    "Stroma / vascular": ["SMA", "Vimentin", "Collagen I", "vWF+ CD31"],
    "State (not clustered on)": ["Ki-67", "phospho-H3", "pS6", "cPARP+cCasp3", "CD44"],
}
DISPLAY_NAME = {"HER2 (bis)": "HER2", "vWF+ CD31": "vWF / CD31", "cPARP+cCasp3": "cPARP / cCasp3"}

VMAX = 2.0
CMAP = plt.get_cmap("RdBu_r")
ROW_GAP, COL_GAP = 0.45, 0.45  # gap between blocks, in cell units
INK, INK_2 = "#1a1a1a", "#52514e"


def _positions(blocks: list[list], gap: float) -> tuple[dict, list[tuple[float, float]], float]:
    """Start coordinate of every item and the (start, end) span of every block."""
    pos, spans, cur = {}, [], 0.0
    for i, items in enumerate(blocks):
        start = cur
        for it in items:
            pos[it] = cur
            cur += 1
        spans.append((start, cur))
        if i < len(blocks) - 1:
            cur += gap
    return pos, spans, cur


def annotated_heatmap(
    means: pd.DataFrame,
    row_blocks: list[tuple[str, list]],
    *,
    left_labels: dict,
    right_labels: dict,
    right_colors: dict,
    title: str,
    subtitle: str,
    left_axis_name: str,
    right_axis_name: str,
    pct: pd.Series,
    out_stem: Path,
    cell_in: float = 0.42,
):
    col_blocks = list(MARKER_GROUPS.items())
    rpos, rspans, H = _positions([r for _, r in row_blocks], ROW_GAP)
    cpos, cspans, W = _positions([c for _, c in col_blocks], COL_GAP)
    norm = Normalize(-VMAX, VMAX)

    fig_w = cell_in * W + 9.5
    fig_h = cell_in * H + 3.2
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([1.6 / fig_w, 1.9 / fig_h, cell_in * W / fig_w, cell_in * H / fig_h])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)

    # cells + numbers
    for row in rpos:
        for col in cpos:
            v = float(means.loc[row, col])
            x, y = cpos[col], rpos[row]
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=CMAP(norm(v)), edgecolor="white", linewidth=0.6))
            ax.text(x + 0.5, y + 0.5, f"{v:.1f}", ha="center", va="center", fontsize=6.3,
                    color="white" if abs(v) > 1.25 else INK)

    # left: row ids
    ax.set_yticks([rpos[r] + 0.5 for r in rpos])
    ax.set_yticklabels([left_labels[r] for r in rpos], fontsize=9, color=INK)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.text(-0.3, -0.35, left_axis_name, ha="right", va="bottom", fontsize=9, fontweight="bold", color=INK)

    # bottom: marker names + family headers on top
    ax.set_xticks([cpos[c] + 0.5 for c in cpos])
    ax.set_xticklabels([DISPLAY_NAME.get(c, c) for c in cpos], rotation=60, ha="right", rotation_mode="anchor",
                       fontsize=9, color=INK)
    ax.tick_params(axis="x", length=0, pad=4)
    for (name, _), (x0, x1) in zip(col_blocks, cspans):
        ax.plot([x0 + 0.08, x1 - 0.08], [-0.35, -0.35], color=INK_2, lw=1, clip_on=False, solid_capstyle="butt")
        ax.text((x0 + x1) / 2, -0.55, name, ha="center", va="bottom", fontsize=9, color=INK_2, clip_on=False)

    # right: lineage strip, assigned label, share-of-cells bar
    x_strip = W + 0.25
    x_text = W + 0.75
    # bar column starts after the longest right-hand label (~0.15 cell units per character at 9 pt)
    x_bar = x_text + 0.15 * max(len(str(v)) for v in right_labels.values()) + 1.2
    bar_w = 4.0
    ax.text(x_text, -0.35, right_axis_name, ha="left", va="bottom", fontsize=9, fontweight="bold", color=INK,
            clip_on=False)
    ax.text(x_bar, -0.35, "% of all cells", ha="left", va="bottom", fontsize=9, fontweight="bold", color=INK,
            clip_on=False)
    pmax = float(pct.max())
    for row in rpos:
        y = rpos[row]
        ax.add_patch(Rectangle((x_strip, y + 0.1), 0.3, 0.8, facecolor=right_colors[row], edgecolor="none",
                               clip_on=False))
        ax.text(x_text, y + 0.5, right_labels[row], ha="left", va="center", fontsize=9, color=INK, clip_on=False)
        w = bar_w * float(pct[row]) / pmax
        ax.add_patch(Rectangle((x_bar, y + 0.22), w, 0.56, facecolor="#9ec5f4", edgecolor="none", clip_on=False))
        ax.text(x_bar + w + 0.15, y + 0.5, f"{pct[row]:.1f}", ha="left", va="center", fontsize=8, color=INK_2,
                clip_on=False)

    # colorbar + lineage legend: bottom right, in the band occupied by the rotated marker names
    x_frac = 1.0 + (x_text - W) / W
    cax = ax.inset_axes([x_frac, -1.4 / H, 5.5 / W, 0.35 / H])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation="horizontal")
    cb.set_ticks([-2, -1, 0, 1, 2])
    cb.ax.tick_params(labelsize=8, length=2, color=INK_2, labelcolor=INK_2)
    cb.outline.set_visible(False)
    cax.set_title("Mean z-score of arcsinh intensity\n(color clipped at ±2; printed numbers are unclipped)",
                  fontsize=8.5, color=INK_2, loc="left", pad=4)
    handles = [Patch(facecolor=LINEAGE_COLOR[g], label=g) for g, _ in row_blocks]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(x_frac, -3.0 / H), ncol=3, frameon=False,
              fontsize=8.5, handlelength=1.0, handleheight=1.0, columnspacing=1.2, borderaxespad=0,
              title="Lineage (color strip)", title_fontsize=8.5, alignment="left")

    # title + two-line subtitle
    ax.text(0, -2.7, title, transform=ax.transData, ha="left", va="bottom", fontsize=13, fontweight="bold",
            color=INK, clip_on=False)
    ax.text(0, -1.5, subtitle, transform=ax.transData, ha="left", va="bottom", fontsize=9, color=INK_2,
            clip_on=False, linespacing=1.4)

    fig.savefig(out_stem.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(IN_H5AD)
    Z = pd.DataFrame(np.asarray(adata.layers["arcsinh_z"], dtype=np.float64), columns=adata.var_names)
    leiden = adata.obs["leiden"].astype(str).to_numpy()
    ctype = adata.obs["cell_type"].astype(str).to_numpy()
    n_total = adata.n_obs

    markers = [m for ms in MARKER_GROUPS.values() for m in ms]
    missing = set(markers) - set(Z.columns)
    if missing:
        raise KeyError(f"markers not in var_names: {sorted(missing)}")

    # cluster -> cell type -> lineage table
    tab = (pd.DataFrame({"leiden": leiden, "cell_type": ctype}).value_counts().rename("n_cells").reset_index())
    assert tab.groupby("leiden").size().max() == 1, "a Leiden cluster maps to more than one cell type"
    type_to_lineage = {t: g for g, ts in LINEAGE.items() for t in ts}
    unknown = set(tab["cell_type"]) - set(type_to_lineage)
    if unknown:
        raise KeyError(f"cell types without a lineage group: {sorted(unknown)}")
    tab["lineage"] = tab["cell_type"].map(type_to_lineage)
    tab["pct_cells"] = 100 * tab["n_cells"] / n_total
    type_rank = {t: i for i, t in enumerate(t for ts in LINEAGE.values() for t in ts)}
    tab = tab.sort_values(["cell_type", "n_cells"], key=lambda s: s.map(type_rank) if s.name == "cell_type" else -s)
    tab = tab.reset_index(drop=True)
    tab.round(2).to_csv(OUT / "leiden_to_cell_type.csv", index=False)

    subtitle = (f"{n_total:,} cells (Kuett et al. 2022, 3D IMC). Leiden clustering (resolution 1.0) on a 15-NN graph "
                "of the 18 lineage markers;\nintensities arcsinh-transformed, clipped at the 99.9th percentile and "
                "z-scored per marker. State markers and DNA channels were not used for clustering.")

    # ---- Figure 1: Leiden clusters ----
    cl_means = Z.groupby(leiden).mean().loc[tab["leiden"], markers]
    cl_pct = tab.set_index("leiden")["pct_cells"]
    cl_type = tab.set_index("leiden")["cell_type"]
    cl_n = tab.set_index("leiden")["n_cells"]
    row_blocks = [(g, tab.loc[tab["lineage"] == g, "leiden"].tolist()) for g in LINEAGE if (tab["lineage"] == g).any()]
    annotated_heatmap(
        cl_means, row_blocks,
        left_labels={c: f"Leiden {c}" for c in cl_means.index},
        right_labels={c: f"{cl_type[c]}   (n = {cl_n[c]:,})" for c in cl_means.index},
        right_colors={c: LINEAGE_COLOR[type_to_lineage[cl_type[c]]] for c in cl_means.index},
        title="Main HER2 volume: Leiden clusters, marker expression and assigned cell type",
        subtitle=subtitle,
        left_axis_name="Leiden cluster", right_axis_name="Assigned cell type",
        pct=cl_pct, out_stem=OUT / "fig_leiden_clusters_marker_expression",
    )

    # ---- Figure 2: cell types ----
    ct_tab = (tab.groupby("cell_type").agg(n_cells=("n_cells", "sum"), pct_cells=("pct_cells", "sum"),
                                           lineage=("lineage", "first"),
                                           clusters=("leiden", lambda s: ", ".join(sorted(s, key=int))))
              .reset_index())
    ct_tab = ct_tab.sort_values("cell_type", key=lambda s: s.map(type_rank)).reset_index(drop=True)
    ct_means = Z.groupby(ctype).mean().loc[ct_tab["cell_type"], markers]
    ct = ct_tab.set_index("cell_type")
    row_blocks = [(g, ct_tab.loc[ct_tab["lineage"] == g, "cell_type"].tolist()) for g in LINEAGE
                  if (ct_tab["lineage"] == g).any()]
    annotated_heatmap(
        ct_means, row_blocks,
        left_labels={t: t for t in ct_means.index},
        right_labels={t: f"n = {ct.loc[t, 'n_cells']:,}   (Leiden {ct.loc[t, 'clusters']})" for t in ct_means.index},
        right_colors={t: LINEAGE_COLOR[ct.loc[t, 'lineage']] for t in ct_means.index},
        title="Main HER2 volume: cell types and their marker expression",
        subtitle=subtitle,
        left_axis_name="Cell type", right_axis_name="Cells   (Leiden clusters merged)",
        pct=ct["pct_cells"], out_stem=OUT / "fig_cell_types_marker_expression",
    )
    ct_tab.round(2).to_csv(OUT / "cell_type_summary.csv", index=False)
    print(tab.to_string(index=False))
    print(f"\nwrote figures to {OUT}")


if __name__ == "__main__":
    main()
