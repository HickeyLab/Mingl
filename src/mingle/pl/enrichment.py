# plot_border_enrichment_labeled.py
"""
Pasteable, AnnData-compatible plotting function (notebook-style fuzzy column
finding) with optional labeling of plotted dots.

New: computes per-row counts of how many neighborhoods exceed a separate threshold
(default 0.15). This count is stored in df_probabilities['Count_Above_Neigh_Thresh']
and is computed over either user-specified neighborhood columns or auto-detected
probability-like columns. It can optionally use tqdm to show progress.

Usage:
    figs = plot_border_enrichment(..., label_dots=True)
"""
from typing import Optional, Tuple, Sequence, Dict, List, Tuple as Tup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from anndata import AnnData
from typing import Any

# tqdm is optional; used only if use_tqdm_for_count=True
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore

from MINGLE.tl.utils_adata import build_df_probs_from_adata
from MINGLE.tl.compute_proportions import compute_grouped_proportions


# ---------------- DEFAULT CONFIG ----------------
MIN_COUNT = 5
POS_THRESHOLD = 0.25   # used for plotting / border masks
EPS = 1e-9
LOG_BASE = 2

# separate threshold for neighborhood-count analysis (default 0.15)
NEIGHBORHOOD_COUNT_DEFAULT_THRESHOLD = 0.15

MIN_AREA = 50.0
MAX_AREA = 800.0
LEGEND_COUNTS = np.array([2000, 5000, 20000], dtype=int)
# ------------------------------------------------




def find_prob_col(df: pd.DataFrame, name: str) -> Optional[str]:
    """Notebook-style fuzzy finder for a probability column name."""
    if name in df.columns:
        return name
    ln = name.lower().replace(" ", "_")
    for col in df.columns:
        if ln in col.lower().replace(" ", "_"):
            return col
    for col in df.columns:
        cl = col.lower()
        if ("prob" in cl or "probability" in cl or "%" in cl) and any(tok in cl for tok in name.lower().split()):
            return col
    return None




def _count_to_area_linear(k: int, max_count: int, min_area: float = MIN_AREA, max_area: float = MAX_AREA) -> float:
    if max_count <= 0:
        return float(min_area)
    return float(min_area + (k / float(max_count)) * (max_area - min_area))




def make_celltype_palette_from_adata(cell_types: Sequence[str]) -> Dict[str, Tup[float, float, float]]:
    import seaborn as sns
    palette_names = ["tab20", "Set3", "Set2", "Paired", "Dark2", "Accent"]
    colors = []
    for name in palette_names:
        try:
            colors.extend(sns.color_palette(name))
        except Exception:
            cmap = plt.cm.get_cmap("tab20")
            colors.extend([tuple(c[:3]) for c in cmap(np.linspace(0, 1, 20))])
    if len(colors) < len(cell_types):
        extra = plt.cm.viridis(np.linspace(0, 1, len(cell_types) - len(colors)))
        colors.extend([tuple(c[:3]) for c in extra])
    final = colors[: len(cell_types)]
    return dict(zip(cell_types, final))




def _normalize_subset_labels(df_plot: pd.DataFrame, n1: str, n2: str) -> pd.DataFrame:
    """
    Normalize the 'Subset' values to canonical forms:
      - exact n1 or n2 -> unchanged
      - '{n1} only' -> n1
      - '{n1} + {n2}' or '{n1} +\n{n2}' -> joint_name
    Returns a copy of df_plot with Subset normalized.
    """
    joint_name = f"{n1} +\n{n2}"


    def map_label(s: str) -> str:
        if not isinstance(s, str):
            s = str(s)
        s_strip = s.strip()
        # direct matches
        if s_strip == n1 or s_strip == n2 or s_strip == joint_name:
            return s_strip
        low = s_strip.lower()
        # patterns: "X only"
        if low == f"{n1.lower()} only":
            return n1
        if low == f"{n2.lower()} only":
            return n2
        # patterns: "X + Y" (various spacing/newline)
        combo1 = f"{n1.lower()} + {n2.lower()}"
        combo2 = f"{n1.lower()} +\n{n2.lower()}"
        combo3 = f"{n1.lower()}+{n2.lower()}"
        if low in (combo1, combo2, combo3):
            return joint_name
        # fallback: contains first token of n1 / n2
        token1 = n1.split()[0].lower()
        token2 = n2.split()[0].lower()
        if token1 in low and "only" in low:
            return n1
        if token2 in low and "only" in low:
            return n2
        if token1 in low and token2 in low:
            return joint_name
        # last attempt: if the label equals n1/n2 ignoring whitespace/newline differences
        s_clean = s_strip.replace("\n", " ").replace("\r", " ")
        if s_clean.lower() == n1.lower():
            return n1
        if s_clean.lower() == n2.lower():
            return n2
        return s_strip


    out = df_plot.copy()
    out["Subset"] = out["Subset"].astype(str).apply(map_label)
    return out




def plot_border_enrichment(
    *,
    adata: Optional[AnnData] = None,
    df_probabilities: Optional[pd.DataFrame] = None,
    n1: str,
    n2: str,
    cell_type_col: str = "Cell Type",
    pos_threshold: float = POS_THRESHOLD,
    min_count: int = MIN_COUNT,
    prob_key: str = "neighborhood_probabilities",
    color_dict: Optional[Dict[str, Tup[float, float, float]]] = None,
    legend_counts: Optional[Sequence[int]] = None,
    show: bool = True,
    dpi: int = 300,
    # --- labeling parameters ---
    label_dots: bool = False,
    label_cts: Optional[Sequence[str]] = None,  # if provided, only label these cell types (must be subset of kept_cts)
    label_fontsize: int = 8,
    label_offset_frac: float = 0.02,
    label_color: str = "black",
    label_bbox: Optional[Dict[str, Any]] = None,
    neighborhoods_to_count: Optional[Sequence[str]] = None,
    neighborhood_count_threshold: float = NEIGHBORHOOD_COUNT_DEFAULT_THRESHOLD,
    use_tqdm_for_count: bool = False,
) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:

    if legend_counts is None:
        legend_counts = LEGEND_COUNTS
    legend_counts = np.array(list(legend_counts), dtype=int)


    # build df_probabilities if necessary
    if df_probabilities is None:
        if adata is None:
            raise ValueError("Please pass either `adata` or `df_probabilities`.")
        df_probabilities, _ = build_df_probs_from_adata(adata, prob_key=prob_key, cell_type_col=cell_type_col)


    df_probabilities = df_probabilities.copy()


    # ensure cell-type column present
    if cell_type_col not in df_probabilities.columns:
        ct_candidates = [c for c in df_probabilities.columns if c.lower().replace(" ", "") in ("celltype", "celllabel", "cell_label", "cell_types")]
        if ct_candidates:
            df_probabilities[cell_type_col] = df_probabilities[ct_candidates[0]].astype(str)
        elif adata is not None and cell_type_col in adata.obs.columns:
            df_probabilities[cell_type_col] = adata.obs[cell_type_col].reindex(df_probabilities.index).astype(str)
        else:
            raise KeyError(f"No cell-type column found. Need '{cell_type_col}' or an alternative.")


    # --- NEW: compute per-row count of neighborhoods above neighborhood_count_threshold ---
    # If neighborhoods_to_count is provided, use those columns (must exist in df_probabilities).
    # Otherwise auto-detect probability-like numeric columns.
    try:
        # decide which columns to evaluate
        if neighborhoods_to_count is not None:
            # use provided list (filter to those actually present)
            neigh_cols = [c for c in neighborhoods_to_count if c in df_probabilities.columns]
            if not neigh_cols:
                # if none match, fall back to auto-detect
                neigh_cols = None
        else:
            neigh_cols = None

        # auto-detect numeric prob-like columns if needed
        if neigh_cols is None:
            candidate_cols = [c for c in df_probabilities.columns if c != cell_type_col]
            numeric_candidates = df_probabilities[candidate_cols].apply(pd.to_numeric, errors="coerce")
            # heuristic: prefer columns that either have "prob" in name OR values within [0,1] (allow small epsilon)
            prob_like = []
            for c in numeric_candidates.columns:
                col = numeric_candidates[c]
                if "prob" in str(c).lower() or "probability" in str(c).lower() or "%" in str(c):
                    prob_like.append(c)
                    continue
                try:
                    mn = float(col.min(skipna=True))
                    mx = float(col.max(skipna=True))
                    if mn >= -1e-6 and mx <= 1.0001:
                        prob_like.append(c)
                        continue
                    # if looks like 0-100 percentages, we will include too
                    if mx > 1 and mx <= 100.0:
                        prob_like.append(c)
                        continue
                except Exception:
                    continue
            if prob_like:
                neigh_cols = prob_like
            else:
                # fallback: all numeric candidate columns
                numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df_probabilities[c])]
                neigh_cols = numeric_cols

        # build dataframe to evaluate
        if not neigh_cols:
            # no numeric columns found: set zeros and skip
            df_probabilities["Count_Above_Neigh_Thresh"] = 0
        else:
            df_neigh = df_probabilities[neigh_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

            # normalize percent-style columns if needed (detect columns whose max >1 and <=100)
            cols_to_norm = [c for c in df_neigh.columns if df_neigh[c].max() > 1.001 and df_neigh[c].max() <= 100.0]
            if cols_to_norm:
                df_neigh.loc[:, cols_to_norm] = df_neigh.loc[:, cols_to_norm] / 100.0

            # now compute counts: either vectorized or tqdm loop
            if use_tqdm_for_count and tqdm is not None:
                counts = []
                with tqdm(total=df_neigh.shape[0]) as pbar:
                    for _, row in df_neigh.iterrows():
                        counts.append(int((row > float(neighborhood_count_threshold)).sum()))
                        pbar.update(1)
                df_probabilities["Count_Above_Neigh_Thresh"] = counts
            else:
                # fast vectorized
                df_probabilities["Count_Above_Neigh_Thresh"] = (df_neigh > float(neighborhood_count_threshold)).sum(axis=1).astype(int)
    except Exception:
        # conservative fallback to avoid breaking existing logic
        df_probabilities["Count_Above_Neigh_Thresh"] = 0
    # --- end NEW block ---


    # compute grouped proportions (returns Subset labels that may be 'X only', 'X + Y', etc.)
    df_plot = compute_grouped_proportions(df_probabilities, n1, n2, cell_type_col=cell_type_col, threshold=pos_threshold, prob_key=prob_key)


    # Normalize Subset labels to canonical forms
    df_plot = _normalize_subset_labels(df_plot, n1, n2)


    # pivot & ensure Average row
    subset_order = [n1, f"{n1} +\n{n2}", "Average", n2]
    pivot = df_plot.pivot(index="Subset", columns=cell_type_col, values="Proportion").fillna(0)


    joint_name = f"{n1} +\n{n2}"


    # If intended rows are missing, try fuzzy contains mapping:
    def _ensure_row_exists(pivot_df: pd.DataFrame, desired: str) -> Optional[str]:
        if desired in pivot_df.index:
            return desired
        alt = desired.replace("\n", " ")
        if alt in pivot_df.index:
            return alt
        if f"{desired} only" in pivot_df.index:
            return f"{desired} only"
        first_tok = desired.split()[0].lower()
        matches = [idx for idx in pivot_df.index if first_tok in str(idx).lower()]
        if len(matches) == 1:
            return matches[0]
        return None


    r1 = _ensure_row_exists(pivot, n1)
    r2 = _ensure_row_exists(pivot, n2)
    rjoint = _ensure_row_exists(pivot, joint_name) or _ensure_row_exists(pivot, f"{n1} + {n2}")


    missing = [name for name, found in ((n1, r1), (n2, r2), (joint_name, rjoint)) if found is None]
    if missing:
        raise ValueError(
            "Could not locate required pivot rows for: "
            f"{missing}. Available pivot index rows: {list(pivot.index)[:50]} "
        )


    # compute Average row safely
    map_name_to_index = {n1: r1, n2: r2, joint_name: rjoint}
    try:
        pivot.loc["Average"] = (pivot.loc[map_name_to_index[n1]] + pivot.loc[map_name_to_index[n2]]) / 2
    except Exception:
        pivot.loc["Average"] = 0.0


    # reindex pivot into canonical ordering (only where present)
    pivot = pivot.reindex([s for s in [n1, joint_name, "Average", n2] if s in pivot.index])


    # find probability columns for masks
    col_n1 = find_prob_col(df_probabilities, n1)
    col_n2 = find_prob_col(df_probabilities, n2)
    label_cols = [c for c in df_probabilities.columns if c.lower() in ("subset", "neighborhood", "label", "assigned", "cn", "neighborhood_label")]


    if col_n1 and col_n2:
        prob_n1 = pd.to_numeric(df_probabilities[col_n1], errors="coerce").fillna(0.0).astype(float)
        prob_n2 = pd.to_numeric(df_probabilities[col_n2], errors="coerce").fillna(0.0).astype(float)
        pos_n1 = prob_n1 > float(pos_threshold)
        pos_n2 = prob_n2 > float(pos_threshold)
        mask_n1_only = pos_n1 & ~pos_n2
        mask_n2_only = pos_n2 & ~pos_n1
        mask_border = pos_n1 & pos_n2
    else:
        if not label_cols:
            raise RuntimeError("No probability columns for n1/n2 found and no label column present to fallback on.")
        label_col = label_cols[0]
        mask_n1_only = df_probabilities[label_col].astype(str) == n1
        mask_n2_only = df_probabilities[label_col].astype(str) == n2
        mask_border = df_probabilities[label_col].astype(str) == joint_name


    # counts by cell type
    all_types = sorted(df_probabilities[cell_type_col].astype(str).unique())
    counts_n1 = df_probabilities.loc[mask_n1_only].groupby(cell_type_col).size().reindex(all_types).fillna(0).astype(int)
    counts_n2 = df_probabilities.loc[mask_n2_only].groupby(cell_type_col).size().reindex(all_types).fillna(0).astype(int)
    counts_border = df_probabilities.loc[mask_border].groupby(cell_type_col).size().reindex(all_types).fillna(0).astype(int)


    # align pivot columns
    common_cts = [ct for ct in pivot.columns if ct in all_types]
    if len(common_cts) == 0:
        raise ValueError("No matching cell types between pivot and df_probabilities.")


    def _safe_row(arr_pivot, label):
        try:
            return arr_pivot.loc[label, common_cts].astype(float).values
        except Exception:
            return np.zeros(len(common_cts), dtype=float)


    p1 = _safe_row(pivot, n1)
    p2 = _safe_row(pivot, n2)
    try:
        pb = pivot.loc[joint_name, common_cts].astype(float).values
    except Exception:
        pb = np.zeros(len(common_cts), dtype=float)


    c1 = counts_n1.reindex(common_cts).values.astype(int)
    c2 = counts_n2.reindex(common_cts).values.astype(int)
    cb = counts_border.reindex(common_cts).values.astype(int)


    mask_ok = (c1 >= min_count) & (c2 >= min_count) & (cb >= min_count)
    kept_cts = [ct for ct, ok in zip(common_cts, mask_ok) if ok]
    if len(kept_cts) == 0:
        raise ValueError("No cell types pass the min_count filter. Lower min_count or inspect counts.")


    p1_f = np.array([pivot.loc[n1, ct] for ct in kept_cts], float)
    p2_f = np.array([pivot.loc[n2, ct] for ct in kept_cts], float)
    pb_f = np.array([pivot.loc[joint_name, ct] for ct in kept_cts], float)
    cb_f = counts_border.reindex(kept_cts).values.astype(int)


    # log ratios
    logfn = np.log2 if LOG_BASE == 2 else np.log
    x_vals = logfn((pb_f + EPS) / (p1_f + EPS))
    y_vals = logfn((pb_f + EPS) / (p2_f + EPS))


    # color dict
    if color_dict is None:
        try:
            if adata is not None and cell_type_col in adata.obs.columns:
                palette = make_celltype_palette_from_adata(sorted(adata.obs[cell_type_col].astype(str).unique()))
                color_dict_local = {ct: palette.get(ct, "#cccccc") for ct in kept_cts}
            else:
                palette = make_celltype_palette_from_adata(kept_cts)
                color_dict_local = {ct: palette.get(ct, "#cccccc") for ct in kept_cts}
        except Exception:
            color_dict_local = {ct: "#cccccc" for ct in kept_cts}
    else:
        color_dict_local = {ct: color_dict.get(ct, "#cccccc") for ct in kept_cts}


    colors = [color_dict_local.get(ct, "#cccccc") for ct in kept_cts]


    max_count = int(max(int(cb_f.max()) if len(cb_f) > 0 else 0, int(legend_counts.max())))
    areas = np.array([_count_to_area_linear(int(c), max_count) for c in cb_f], float)
    legend_areas = np.array([_count_to_area_linear(int(c), max_count) for c in legend_counts], float)


    # FIGURE 1
    fig_scatter, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals) & (areas > 0)
    if not np.any(finite_mask):
        raise ValueError("No finite points to plot after filtering NaNs/infs.")
    plotted_colors = [colors[i] for i, ok in enumerate(finite_mask) if ok]
    ax.scatter(x_vals[finite_mask], y_vals[finite_mask], s=areas[finite_mask], c=plotted_colors, edgecolor="k", alpha=0.95, linewidth=0.35)


    ax.axhline(0, color="gray", linestyle="--")
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal", "box")
    ax.grid(False)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.tick_params(axis="both", labelsize=12)
    plt.tight_layout()


    # ----- labeling (if requested) -----
    if label_dots:
        # compute offset in data coordinates using axis range
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        x_offset = label_offset_frac * (x1 - x0)
        y_offset = label_offset_frac * (y1 - y0)


        # keep only those to label (default -> all kept plotted cts)
        finite_kept_cts = [ct for ct, ok in zip(kept_cts, finite_mask) if ok]
        finite_x = x_vals[finite_mask]
        finite_y = y_vals[finite_mask]


        # determine which labels to show
        if label_cts is None:
            to_label = set(finite_kept_cts)
        else:
            # normalize provided list to strings and intersect with kept
            requested = set([str(x) for x in label_cts])
            to_label = set(finite_kept_cts).intersection(requested)


        # label points
        for ct, x, y in zip(finite_kept_cts, finite_x, finite_y):
            if ct not in to_label:
                continue
            # attempt to place text to the upper-right by default; small logic to reduce overlap:
            ha = "left"
            va = "bottom"
            # if near right edge, place left
            if (x + x_offset) > (x1 - 0.05 * (x1 - x0)):
                ha = "right"
            if (y + y_offset) > (y1 - 0.05 * (y1 - y0)):
                va = "top"
            txt = ax.text(
                x + x_offset,
                y + y_offset,
                ct,
                fontsize=label_fontsize,
                ha=ha,
                va=va,
                color=label_color,
                bbox=label_bbox,
            )


    # FIGURE 2: size legend
    fig_leg_counts, ax_leg = plt.subplots(figsize=(4, 6), dpi=dpi)
    handles = [plt.scatter([], [], s=float(ar), color="gray", edgecolors="none") for ar in legend_areas]
    labels = [f"{int(c)} cells" for c in legend_counts]
    ax_leg.legend(handles=handles, labels=labels, title="Border Cell Count", loc="center", frameon=False)
    ax_leg.axis("off")
    plt.tight_layout()


    # FIGURE 3: color legend
    fig_leg_colors, ax_leg_colors = plt.subplots(figsize=(4, 8), dpi=dpi)
    color_patches = [Patch(facecolor=color_dict_local.get(ct, "#cccccc"), edgecolor="none", label=ct) for ct in kept_cts]
    leg_colors = ax_leg_colors.legend(handles=color_patches, title="Cell Type", loc="center", frameon=False, ncol=1)
    if leg_colors is not None:
        leg_colors.set_frame_on(False)
    ax_leg_colors.axis("off")
    plt.tight_layout()


    if show:
        plt.show()


    return fig_scatter, fig_leg_counts, fig_leg_colors