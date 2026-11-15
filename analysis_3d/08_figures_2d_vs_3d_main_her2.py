"""Figures: the Main HER2 volume in 3D, and 2D vs 3D MINGL on the same cells.

Fig 1  The volume and its 3D neighborhoods: 3D render, orthogonal slabs,
       neighborhood composition along depth.
Fig 2  Interfaces between neighborhoods in 3D: where they are, how they are
       oriented relative to the section plane, and how often a section
       registers them as a boundary.
Fig 3  Border cells in 3D vs 2D: exploded section stack, one section as a
       three-class difference map, the overlap of the border sets, and
       detection in 2D by border orientation.
Fig 4  How much 2D and 3D agree, next to the references (seed only, k only,
       chance): matched section maps, neighborhood confusion matrix,
       border-score density scatter, window differences, threshold sweep, own
       vs fixed neighborhoods, and the agreement dot plot.
Fig 5  Supporting checks: block-face effects and centroid re-estimation.

Everything is read from the outputs of 02, 03, 05, 06 and 07; nothing is
refit. Border classes use the 3D k=30 and 2D k=10 runs (matched window radius)
at the n used throughout, and the default threshold. Layouts are set in mm.

Output: results/main_her2/figures_2d_vs_3d/  (PDF and PNG per figure)
Run:    .venv/bin/python analysis_3d/08_figures_2d_vs_3d_main_her2.py
"""

from __future__ import annotations

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from mingl import tl3d
from border_metrics import border_score, fractions
from common import CT, OUT, REG, THRESHOLD
import figstyle as fs

IN_3D = OUT / "main_her2_mingl3d.h5ad"
IN_2D = OUT / "2d_vs_3d" / "sections_2d_mingl.h5ad"
OUT5, OUT6, OUT7 = OUT / "boundary_and_borders", OUT / "fixed_neighborhoods", OUT / "interfaces_3d"
FIG = OUT / "figures_2d_vs_3d"

K3, K2 = 30, 10  # matched window radius
SLAB = 4.0  # µm half-thickness of the slabs drawn as "sections"
STACK_Z = (50.0, 100.0, 150.0, 200.0, 250.0)  # sections drawn in the exploded stack
MAP_Z = 150.0  # section used for the 2D maps
ORIENT_NEIGHBORS = 60  # cells used to estimate the local normal of a border-cell sheet
ORIENT_MIN_SIDE = 5  # of which at least this many on each side
IN_PLANE = np.cos(np.radians(30))  # tilt above this: interface within 30° of the section plane
W = 183.0  # figure width, mm

TILT_CMAP = LinearSegmentedColormap.from_list("tilt", ["#dedede", "#f2b48c", fs.VERMILLION, "#6e2f00"])
RATIO_CMAP = LinearSegmentedColormap.from_list("ratio", [fs.BLUE, "#9cc3dc", "#eeeeee", "#f0b089", fs.VERMILLION])


def _n_from_3d_run(k: int = 30) -> int:
    rs = pd.read_csv(OUT / "run_summary.csv")
    return int(rs.loc[rs["k"] == k, "prob_elbow_n"].iloc[0])


N = _n_from_3d_run()


def new_figure(height_mm: float):
    fig = plt.figure(figsize=(W * fs.MM, height_mm * fs.MM))
    fig._h_mm = height_mm
    return fig


def place(fig, x, y, w, h, **kwargs):
    """Axes at (x, y) mm from the top-left corner of the figure, w x h mm."""
    H = fig._h_mm
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kwargs)


def letter(fig, x, y, s):
    fig.text(x / W, 1 - y / fig._h_mm, s, fontsize=8, fontweight="bold", va="top", ha="left")


def note(fig, x, y, s, ha="left", va="top", color=fs.MID, size=6):
    fig.text(x / W, 1 - y / fig._h_mm, s, fontsize=size, color=color, ha=ha, va=va)


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def load():
    cells = ad.read_h5ad(IN_3D)
    cells.obs[CT] = cells.obs[CT].astype(str)
    sec = ad.read_h5ad(IN_2D)
    sec.obs[CT] = sec.obs[CT].astype(str)
    sec.obs[REG] = sec.obs[REG].astype(str)
    pos = cells.obs_names.get_indexer(sec.obs["cell_id"].astype(str).to_numpy())

    d = {"cells": cells, "sec": sec, "pos": pos, "xyz": cells.obs[["x", "y", "z"]].to_numpy(float)}
    d["lo"], d["hi"] = d["xyz"].min(axis=0), d["xyz"].max(axis=0)
    d["in_section"] = np.zeros(cells.n_obs, bool)
    d["in_section"][pos] = True
    d["section_z"] = np.full(cells.n_obs, np.nan)
    d["section_z"][pos] = sec.obs["section_z"].to_numpy(float)

    for k in (30, 50):
        d[f"p3_{k}"] = np.asarray(cells.obsm[f"neighborhood_probabilities_k{k}_n{N}"], float)
        d[f"s3_{k}"] = border_score(d[f"p3_{k}"])
        d[f"lab3_{k}"] = cells.obs[f"neighborhood_k{k}_n{N}"].astype(str).to_numpy()
    for k in (10, 15):
        s = np.full(cells.n_obs, np.nan)
        s[pos] = border_score(sec.obsm[f"neighborhood_probabilities_2d_k{k}_n{N}"])
        d[f"s2_{k}"] = s
        lab = np.full(cells.n_obs, "", dtype=object)
        lab[pos] = sec.obs[f"neighborhood_2d_k{k}_n{N}"].astype(str).to_numpy()
        d[f"lab2_{k}"] = lab
    full = cells.obs.groupby(f"neighborhood_k{K3}_n{N}", observed=True)[f"neighborhood_k{K3}_n{N}_name"].first()
    d["nb_name"] = {str(k): str(v).split(": ", 1)[1] for k, v in full.items()}

    b3, b2 = d[f"s3_{K3}"] > THRESHOLD, d[f"s2_{K2}"] > THRESHOLD  # NaN > x is False
    cls = np.full(cells.n_obs, "", dtype=object)
    cls[d["in_section"] & b3 & ~b2] = "3D only"
    cls[d["in_section"] & b3 & b2] = "2D and 3D"
    cls[d["in_section"] & ~b3 & b2] = "2D only"
    d["b3"], d["b2"], d["border_class"] = b3, b2, cls

    sizes = pd.Series(d[f"lab3_{K3}"]).value_counts()
    d["top"] = sizes.index[:len(fs.NEIGHBORHOOD_COLORS)].tolist()
    d["pal"] = dict(zip(d["top"], fs.NEIGHBORHOOD_COLORS))
    d["nb_rgba"] = np.array([to_rgba(d["pal"].get(l, fs.CONTEXT)) for l in d[f"lab3_{K3}"]])
    return d


def short(name: str, n: int = 2) -> str:
    return " + ".join(part.split(" (")[0] for part in name.split(" + ")[:n])


def neighborhood_legend(fig, d, x, y, ncol=4):
    handles = [Line2D([], [], marker="o", ls="", ms=3.2, color=d["pal"][t], label=f"N{t}  {short(d['nb_name'][t])}")
               for t in d["top"]]
    handles.append(Line2D([], [], marker="o", ls="", ms=3.2, color=fs.CONTEXT,
                          label=f"other {N - len(d['top'])} neighborhoods"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x / W, 1 - y / fig._h_mm), ncol=ncol, fontsize=5.8,
               columnspacing=1.2, handletextpad=0.2)


def border_normals(d, k: int = K3):
    """Tilt of the sheet each 3D border cell sits on: |cos| between its normal and z.

    The normal is the direction between the centroids of the nearby cells
    labelled with the cell's most probable and second most probable
    neighborhood. NaN where either side has too few cells.
    """
    p, lab, xyz = d[f"p3_{k}"], d[f"lab3_{k}"], d["xyz"]
    names = np.asarray(d["cells"].uns[f"neighborhood_probability_neighborhoods_k{k}_n{N}"]).astype(str)
    idx_b = np.flatnonzero(d[f"s3_{k}"] > THRESHOLD)
    order = np.argsort(p[idx_b], axis=1)
    top1, top2 = names[order[:, -1]], names[order[:, -2]]
    _, nb = cKDTree(xyz).query(xyz[idx_b], k=ORIENT_NEIGHBORS)
    a, b = lab[nb] == top1[:, None], lab[nb] == top2[:, None]
    ok = (a.sum(axis=1) >= ORIENT_MIN_SIDE) & (b.sum(axis=1) >= ORIENT_MIN_SIDE)
    ca = (xyz[nb] * a[:, :, None]).sum(axis=1) / np.maximum(a.sum(axis=1), 1)[:, None]
    cb = (xyz[nb] * b[:, :, None]).sum(axis=1) / np.maximum(b.sum(axis=1), 1)[:, None]
    n = cb - ca
    with np.errstate(invalid="ignore"):
        cosz = np.abs(n[:, 2]) / np.linalg.norm(n, axis=1)
    out = np.full(len(xyz), np.nan)
    out[idx_b[ok]] = cosz[ok]
    return out


def map2d(ax, d, xy, colors, sizes, i=0, j=1, flip=True):
    order = np.argsort(sizes) if np.ndim(sizes) else slice(None)
    ax.scatter(xy[order, 0], xy[order, 1], c=colors[order], s=sizes[order] if np.ndim(sizes) else sizes,
               linewidths=0, rasterized=True)
    ax.set_xlim(d["lo"][i], d["hi"][i])
    ax.set_ylim((d["hi"][j], d["lo"][j]) if flip else (d["lo"][j], d["hi"][j]))
    fs.bare(ax)


# ----------------------------------------------------------------------------
# Fig 1: the volume
# ----------------------------------------------------------------------------

def fig1_volume(d):
    xyz, lo, hi, colors = d["xyz"], d["lo"], d["hi"], d["nb_rgba"]
    mid, ext = (lo + hi) / 2, hi - lo
    fig = new_figure(90)

    ax = place(fig, -4, 0, 84, 74, projection="3d")
    ax.scatter(*xyz.T, c=colors, s=0.3, linewidths=0, depthshade=False, rasterized=True)
    fs.box_edges(ax, lo, hi)
    fs.bare3d(ax, lo, hi)
    ax.set_box_aspect(tuple(ext), zoom=1.0)
    ax.text(mid[0], lo[1] - 45, lo[2], f"{ext[0]:.0f} µm", ha="center", va="top", fontsize=6)
    ax.text(hi[0] + 45, mid[1], lo[2], f"{ext[1]:.0f} µm", ha="left", va="top", fontsize=6)
    ax.text2D(0.062, 0.46, f"{ext[2]:.0f} µm", transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=6)
    letter(fig, 1, 2, "a")

    s = 50 / ext[0]  # mm per µm in the slab panels
    x0, y0, gap = 79, 9, 2
    slabs = ((place(fig, x0, y0, ext[0] * s, ext[1] * s), 0, 1, 2, True),
             (place(fig, x0, y0 + ext[1] * s + gap, ext[0] * s, ext[2] * s), 0, 2, 1, True),
             (place(fig, x0 + ext[0] * s + gap, y0, ext[2] * s, ext[1] * s), 2, 1, 0, True))
    for ax, i, j, cut, flip in slabs:
        m = np.abs(xyz[:, cut] - mid[cut]) <= SLAB
        map2d(ax, d, xyz[m][:, [i, j]], colors[m], 1.5, i, j, flip)
        for other, line in ((i, ax.axvline), (j, ax.axhline)):
            if other != cut:
                line(mid[other], color=fs.INK, lw=0.3, alpha=0.55)
    fs.scale_bar(slabs[1][0], 100, loc=(0.0, -0.16))
    letter(fig, x0 - 4, 2, "b")
    note(fig, x0, y0 - 1, f"xy slab, z = {mid[2]:.0f} ± {SLAB:.0f} µm", va="bottom")
    note(fig, x0 + ext[0] * s + gap, y0 - 1, "zy slab", va="bottom")
    note(fig, x0 + ext[0] * s + gap, y0 + ext[1] * s + gap + 1, "xz slab (left).\nLines mark the\nother two cuts.")

    ax = place(fig, 166, y0, 14, ext[1] * s + gap + ext[2] * s)
    edges = np.arange(0, hi[2] + 10, 10.0)
    zbin = np.clip(np.digitize(xyz[:, 2], edges) - 1, 0, len(edges) - 2)
    tab = pd.crosstab(zbin, d[f"lab3_{K3}"], normalize="index")
    centers = ((edges[:-1] + edges[1:]) / 2)[tab.index]
    left = np.zeros(len(tab))
    for c in d["top"] + [c for c in tab.columns if c not in d["top"]]:
        ax.fill_betweenx(centers, left, left + tab[c].to_numpy(), color=d["pal"].get(c, fs.CONTEXT), lw=0)
        left += tab[c].to_numpy()
    ax.set_xlim(0, 1)
    ax.set_ylim(hi[2], lo[2])
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_yticks([0, 100, 200, 300])
    ax.set_xlabel("Fraction of cells")
    ax.set_ylabel("Depth z (µm)")
    letter(fig, 153, 2, "c")
    neighborhood_legend(fig, d, 2, 80)
    fs.save(fig, FIG / "fig1_volume_3d_neighborhoods")


# ----------------------------------------------------------------------------
# Fig 2: interfaces between neighborhoods in 3D
# ----------------------------------------------------------------------------

def fig2_interfaces(d):
    pts = pd.read_csv(OUT7 / f"interface_points_k{K3}.csv.gz")
    hist = pd.read_csv(OUT7 / "interface_tilt_histogram.csv")
    det = pd.read_csv(OUT7 / "interface_detection_by_tilt.csv")
    xyz, lo, hi = d["xyz"], d["lo"], d["hi"]
    ext = hi - lo
    tilt = pts["tilt"].to_numpy()
    fig = new_figure(142)

    # a: interface points in 3D, coloured by tilt
    ax = place(fig, -6, 0, 100, 76, projection="3d")
    flat = tilt > IN_PLANE
    colors = np.where(flat[:, None], to_rgba(fs.VERMILLION), to_rgba(fs.CONTEXT, 0.35))
    ax.scatter(pts["x"], pts["y"], pts["z"], c=colors, s=np.where(flat, 0.8, 0.08), linewidths=0, depthshade=False,
               rasterized=True)
    fs.box_edges(ax, lo, hi)
    fs.bare3d(ax, lo, hi)
    ax.set_box_aspect(tuple(ext), zoom=1.0)
    letter(fig, 1, 2, "a")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.2, color=fs.VERMILLION,
                               label=f"interface within 30° of the section plane ({100 * flat.mean():.1f}%)"),
                        Line2D([], [], marker="o", ls="", ms=3.2, color=fs.CONTEXT, label="other interface points")],
               loc="upper left", bbox_to_anchor=(8 / W, 1 - 3 / fig._h_mm), fontsize=6)
    cax = place(fig, 12, 77, 26, 1.5)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=TILT_CMAP, norm=plt.Normalize(0, 1)), cax=cax, orientation="horizontal",
                      ticks=[0, 0.5, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1.5, labelsize=5.5)
    note(fig, 41, 76.5, "Interface tilt, |cos| of the normal to z:\n0 = cut across, 1 = in the section plane", va="top")

    # b: pole figure of interface normals (equal-area)
    ax = place(fig, 104, 16, 38, 38, projection="polar")
    nr, nphi = 8, 36
    r = np.sqrt(2) * np.sin(np.arccos(np.clip(tilt, 0, 1)) / 2)
    phi = np.arctan2(pts["ny"], pts["nx"]) % (2 * np.pi)
    r_edges, p_edges = np.sqrt(np.linspace(0, 1, nr + 1)), np.linspace(0, 2 * np.pi, nphi + 1)
    counts, _, _ = np.histogram2d(r, phi, bins=[r_edges, p_edges])
    ratio = counts / (len(pts) / (nr * nphi))
    mesh = ax.pcolormesh(p_edges, r_edges, ratio, cmap=RATIO_CMAP, norm=TwoSlopeNorm(1.0, 0.5, 1.5), rasterized=True)
    ax.set_ylim(0, 1)
    ax.set_yticks([np.sqrt(2) * np.sin(np.radians(a) / 2) for a in (30, 60)], ["", ""])
    ax.set_xticks(np.radians([0, 90, 180, 270]), ["x", "y", "", ""], fontsize=6)
    ax.tick_params(pad=-3)
    ax.grid(color="white", lw=0.4)
    ax.spines["polar"].set_linewidth(0.5)
    cax = place(fig, 146, 20, 1.6, 30)
    cb = fig.colorbar(mesh, cax=cax, ticks=[0.5, 1, 1.5])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1.5, labelsize=5.5)
    cb.set_label("Density relative to\nrandom orientation", fontsize=6)
    letter(fig, 98, 2, "b")
    note(fig, 104, 3, "Directions of interface normals (equal-area).\nCentre: normal along z, interface in the section\nplane. Rings at 30° and 60°; rim: cut across.")

    # c: tilt distribution
    ax = place(fig, 168, 20, 13, 30)
    for run, color in ((f"3D k={K3}", fs.INK), ("3D k=50", fs.MID)):
        sub = hist[hist["run"] == run]
        ax.stairs(sub["density"], np.r_[sub["tilt_from"], 1.0], color=color, lw=0.9, label=run.replace("3D k=", "k = "))
    ax.axhline(1, color=fs.MID, lw=0.5, ls=(0, (1, 1.5)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.3)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Tilt")
    ax.set_ylabel("Density")
    ax.legend(loc="lower left", fontsize=5.5, handlelength=1)
    letter(fig, 158, 2, "c")

    # d: vertical reslice with the interface points in it
    s = 112 / ext[0]
    ax = place(fig, 12, 85, 112, ext[2] * s)
    y0 = (lo[1] + hi[1]) / 2
    m = np.abs(xyz[:, 1] - y0) <= 5.0
    ax.scatter(xyz[m, 0], xyz[m, 2], c=fs.CONTEXT, s=0.8, linewidths=0, rasterized=True)
    p = pts[np.abs(pts["y"] - y0) <= 2.5].sort_values("tilt")
    ax.scatter(p["x"], p["z"], c=TILT_CMAP(p["tilt"].to_numpy()), s=np.where(p["tilt"] > IN_PLANE, 3.2, 1.0),
               linewidths=0, rasterized=True)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(hi[2], lo[2])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([0, 100, 200, 300])
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("Depth z (µm)")
    fs.scale_bar(ax, 100, loc=(0.0, -0.09))
    letter(fig, 1, 70, "d")
    note(fig, 12, 71, "Vertical cut (xz) through the block centre: cells in grey, interface points coloured by tilt")

    # e: what a section sees
    ax = place(fig, 142, 85, 38, 40)
    centers = None
    for pair, color, off in ((f"3D k={K3}, 2D k={K2}", fs.INK, -0.012), ("3D k=50, 2D k=15", fs.MID, 0.012)):
        sub = det[(det["pair"] == pair) & (det["labels"] == "2D neighborhoods")]
        near = sub[sub["group"] == "near a 3D interface"]
        centers = ((near["tilt_from"] + near["tilt_to"]) / 2).to_numpy()
        ax.errorbar(centers + off, 100 * near["boundary_rate"], yerr=196 * near["se"], color=color, marker="o", ms=3,
                    lw=0.8, label=pair.replace("=", " = "))
        base = 100 * sub.loc[sub["group"] == "not near a 3D interface", "boundary_rate"].iloc[0]
        ax.plot([0, 1], [base, base], color=color, lw=0.6, ls=(0, (1, 1.5)))
    ax.text(0.5, 21.5, "section cells away from any 3D interface", fontsize=5.3, color=fs.MID, ha="center", va="bottom")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 55)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_xlabel("Tilt of the nearby 3D interface\n(0 = cut across, 1 = in the section plane)")
    ax.set_ylabel("Section cells on a 2D\nneighborhood boundary (%)")
    ax.legend(loc="upper right", fontsize=5.5, handlelength=1.2)
    letter(fig, 128, 70, "e")
    fs.save(fig, FIG / "fig2_interfaces_3d")


# ----------------------------------------------------------------------------
# Fig 3: border cells in 3D vs 2D
# ----------------------------------------------------------------------------

def class_colors(cls):
    c = np.array([to_rgba(fs.BORDER_CLASS.get(x, fs.CONTEXT)) for x in cls])
    c[cls == "", 3] = 0.6
    return c


def upset(fig, d, x, y, w, h_bar, h_mat):
    sets = {"3D k = 30": d["s3_30"] > THRESHOLD, "3D k = 50": d["s3_50"] > THRESHOLD,
            "2D k = 10": d["s2_10"] > THRESHOLD, "2D k = 15": d["s2_15"] > THRESHOLD}
    member = np.column_stack([v[d["in_section"]] for v in sets.values()])
    codes = member @ (1 << np.arange(member.shape[1]))
    counts = pd.Series(codes[codes > 0]).value_counts()
    xs = np.arange(len(counts))
    bits = np.array([[(code >> j) & 1 for j in range(member.shape[1])] for code in counts.index])
    only3, only2 = bits[:, 2:].sum(axis=1) == 0, bits[:, :2].sum(axis=1) == 0
    colors = np.where(only3, fs.DIM["3D"], np.where(only2, fs.DIM["2D"], fs.MID))
    axb, axm = place(fig, x, y, w, h_bar), place(fig, x, y + h_bar + 1.5, w, h_mat)
    axb.bar(xs, counts.to_numpy(), width=0.62, color=colors, lw=0)
    axb.set_xlim(-0.6, len(xs) - 0.4)
    axb.set_xticks([])
    axb.set_yticks([0, 2000, 4000, 6000], ["0", "2,000", "4,000", "6,000"])
    axb.spines["bottom"].set_visible(False)
    axb.set_ylabel("Border cells")
    for j in range(len(sets)):
        axm.plot(xs, [j] * len(xs), "o", ms=2.8, color=fs.HAIR, zorder=1)
        on = bits[:, j] == 1
        axm.plot(xs[on], [j] * on.sum(), "o", ms=2.8, color=fs.INK, zorder=3)
    for xx, b in zip(xs, bits):
        on = np.flatnonzero(b)
        axm.plot([xx, xx], [on.min(), on.max()], color=fs.INK, lw=0.7, zorder=2)
    axm.set_xlim(-0.6, len(xs) - 0.4)
    axm.set_ylim(len(sets) - 0.5, -0.5)
    axm.set_yticks(range(len(sets)), [f"{name}  ({int(member[:, j].sum()):,})" for j, name in enumerate(sets)])
    axm.set_xticks([])
    axm.tick_params(axis="y", length=0)
    for s in axm.spines.values():
        s.set_visible(False)
    axb.legend(handles=[Rectangle((0, 0), 1, 1, color=fs.DIM["3D"], label="3D runs only"),
                        Rectangle((0, 0), 1, 1, color=fs.DIM["2D"], label="2D runs only"),
                        Rectangle((0, 0), 1, 1, color=fs.MID, label="found in 2D and 3D")],
               loc="upper right", fontsize=5.8, handlelength=1)


def fig3_borders(d, cosz):
    xyz, lo, hi, cls = d["xyz"], d["lo"], d["hi"], d["border_class"]
    ext = hi - lo
    fig = new_figure(162)

    # a: exploded stack of sections
    ax = place(fig, -2, 2, 100, 96, projection="3d")
    m = np.isin(d["section_z"], STACK_Z)
    sizes = np.where(cls[m] == "", 0.4, 2.0)
    ax.scatter(xyz[m, 0], xyz[m, 1], d["section_z"][m], c=class_colors(cls[m]), s=sizes, linewidths=0,
               depthshade=False, rasterized=True)
    for z in STACK_Z:
        ax.plot([lo[0], hi[0], hi[0], lo[0], lo[0]], [lo[1], lo[1], hi[1], hi[1], lo[1]], [z] * 5, color=fs.MID, lw=0.3)
        ax.text(lo[0] - 15, lo[1], z, f"{z:.0f} µm", fontsize=5.5, color=fs.MID, va="center", ha="right")
    ax.set_proj_type("ortho")
    ax.view_init(elev=24, azim=-62)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(STACK_Z[-1] + 5, STACK_Z[0] - 5)
    ax.set_box_aspect((ext[0], ext[1], 2.6 * (STACK_Z[-1] - STACK_Z[0])), zoom=0.98)
    ax.set_axis_off()
    letter(fig, 1, 2, "a")
    handles = [Line2D([], [], marker="o", ls="", ms=3.2, color=c, label=f"border in {k}") for k, c in fs.BORDER_CLASS.items()]
    handles.append(Line2D([], [], marker="o", ls="", ms=2.6, color=fs.CONTEXT, label="other cells"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(4 / W, 1 - 99 / fig._h_mm), ncol=4, fontsize=6,
               columnspacing=1.0, handletextpad=0.2)
    note(fig, 9, 3, "Five of the 29 simulated sections (depth z at left); spacing in z stretched 2.6-fold")

    # b: one section, three views
    m = d["section_z"] == MAP_Z
    xy = xyz[m][:, :2]
    s = 38 / ext[0]
    for x, title, mask, color in ((104, "Borders from 2D analysis", d["b2"][m], fs.DIM["2D"]),
                                  (144, "Borders from 3D analysis", d["b3"][m], fs.DIM["3D"])):
        ax = place(fig, x, 9, ext[0] * s, ext[1] * s)
        c = np.array([to_rgba(color if b else fs.CONTEXT) for b in mask])
        c[~mask, 3] = 0.6
        map2d(ax, d, xy, c, np.where(mask, 1.6, 0.35))
        note(fig, x, 8, title, va="bottom")
    s2 = 78 / ext[0]
    ax = place(fig, 104, 44, ext[0] * s2, ext[1] * s2)
    map2d(ax, d, xy, class_colors(cls[m]), np.where(cls[m] == "", 0.6, 3.6))
    fs.scale_bar(ax, 100, loc=(0.0, -0.09))
    letter(fig, 99, 2, "b")
    note(fig, 104, 43, f"Same section (z = {MAP_Z:.0f} µm), both analyses", va="bottom")

    # c: overlap of the two border sets
    ax = place(fig, 4, 120, 44, 31.6)
    n3, n2, both = (cls == "3D only").sum(), (cls == "2D only").sum(), (cls == "2D and 3D").sum()
    total = int(d["in_section"].sum())
    expected = (n3 + both) * (n2 + both) / total
    ax.add_patch(Circle((-0.42, 0), 1, fc=to_rgba(fs.DIM["3D"], 0.25), ec=fs.DIM["3D"], lw=0.6))
    ax.add_patch(Circle((0.42, 0), 1, fc=to_rgba(fs.DIM["2D"], 0.25), ec=fs.DIM["2D"], lw=0.6))
    for x, val, lab_ in ((-0.98, n3, "3D only"), (0.0, both, "both"), (0.98, n2, "2D only")):
        ax.text(x, 0.1, f"{val:,}", ha="center", va="center", fontsize=7)
        ax.text(x, -0.2, lab_, ha="center", va="center", fontsize=6, color=fs.MID)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.15, 1.15)
    fs.bare(ax)
    letter(fig, 1, 114, "c")
    note(fig, 26, 153, f"{both:,} shared; {expected:,.0f} expected if independent", ha="center")

    # d: all four runs
    upset(fig, d, 74, 119, 60, 22, 13)
    letter(fig, 54, 114, "d")

    # e: detection in 2D by orientation of the border sheet
    ax = place(fig, 153, 121, 27, 30)
    edges = np.linspace(0, 1, 6)
    centers = (edges[:-1] + edges[1:]) / 2
    for k3, k2, color, off in ((30, 10, fs.INK, -0.015), (50, 15, fs.MID, 0.015)):
        cz = cosz if k3 == K3 else border_normals(d, k3)
        m = d["in_section"] & ~np.isnan(cz)
        found = (d[f"s2_{k2}"] > THRESHOLD)[m]
        b = np.clip(np.digitize(cz[m], edges) - 1, 0, len(edges) - 2)
        rate = np.array([found[b == i].mean() for i in range(len(centers))])
        n = np.array([(b == i).sum() for i in range(len(centers))])
        ax.errorbar(centers + off, 100 * rate, yerr=196 * np.sqrt(rate * (1 - rate) / n), color=color, marker="o",
                    ms=2.8, lw=0.8, label=f"k = {k3} (3D), {k2} (2D)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 30)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_xlabel("Tilt of the border sheet")
    ax.set_ylabel("3D border cells that are\nalso a 2D border (%)")
    ax.legend(loc="upper right", fontsize=5.5, handlelength=1.2)
    letter(fig, 139, 114, "e")
    fs.save(fig, FIG / "fig3_border_cells_3d_vs_2d")


# ----------------------------------------------------------------------------
# Fig 4: agreement between 2D and 3D, next to the references
# ----------------------------------------------------------------------------

COMPARISON_LABEL = {
    "3D k=30, seed 0 vs 1": "3D, seed only (k = 30)",
    "3D k=50, seed 0 vs 1": "3D, seed only (k = 50)",
    "3D k=30 vs k=50": "3D, k = 30 vs 50",
    "2D k=10 vs k=15": "2D, k = 10 vs 15",
    "2D k=10 vs 3D k=30": "2D vs 3D (k = 10, 30)",
    "2D k=15 vs 3D k=50": "2D vs 3D (k = 15, 50)",
}


def window_tvds(d):
    cells, sec, pos = d["cells"], d["sec"], d["pos"]
    cell_types = sorted(cells.obs[CT].unique())
    w3 = tl3d.KNN2(cells, z_key="z", cluster_col=CT, region_key=REG, ks=(30, 50))
    w2 = tl3d.KNN2(sec, z_key=None, cluster_col=CT, region_key=REG, ks=(10, 15))
    f = {("3D", k): fractions(w3[k].iloc[pos], cell_types) for k in (30, 50)}
    f.update({("2D", k): fractions(w2[k], cell_types) for k in (10, 15)})

    def tvd(a, b):
        return 0.5 * np.abs(f[a] - f[b]).sum(axis=1)

    return {"2D vs 3D (k = 10, 30)": (tvd(("2D", 10), ("3D", 30)), "2D vs 3D"),
            "3D, k = 30 vs 50": (tvd(("3D", 30), ("3D", 50)), "k only"),
            "2D, k = 10 vs 15": (tvd(("2D", 10), ("2D", 15)), "k only")}


def fig4_agreement(d):
    table = pd.read_csv(OUT5 / "border_comparisons.csv").query("cells == 'all'").set_index("comparison")
    sweep = pd.read_csv(OUT5 / "border_threshold_sweep.csv").query("cells == 'all' and sweep == 'threshold'")
    fixed = pd.read_csv(OUT6 / "fixed_neighborhood_comparisons.csv")
    kinds = sweep.drop_duplicates("comparison").set_index("comparison")["kind"]
    sec_mask, xyz, lo, hi = d["in_section"], d["xyz"], d["lo"], d["hi"]
    ext = hi - lo
    lab3, lab2 = d[f"lab3_{K3}"][sec_mask], d[f"lab2_{K2}"][sec_mask]
    fig = new_figure(169)

    # a: one section, 3D neighborhoods vs 2D neighborhoods in matched colours
    ct = pd.crosstab(lab3, lab2)
    r, c = linear_sum_assignment(-ct.to_numpy())
    match = {ct.columns[j]: ct.index[i] for i, j in zip(r, c)}
    m = d["section_z"] == MAP_Z
    xy = xyz[m][:, :2]
    s = 38 / ext[0]
    c2 = np.array([to_rgba(d["pal"].get(match.get(l), fs.CONTEXT)) for l in d[f"lab2_{K2}"][m]])
    for y, title, colors in ((9, f"3D neighborhoods (k = {K3})", d["nb_rgba"][m]), (43, f"2D neighborhoods (k = {K2})", c2)):
        ax = place(fig, 4, y, ext[0] * s, ext[1] * s)
        map2d(ax, d, xy, colors, 1.6)
        note(fig, 4, y - 1, title, va="bottom")
    fs.scale_bar(ax, 100, loc=(0.0, -0.14))
    letter(fig, 1, 2, "a")
    note(fig, 4, 80, f"Section z = {MAP_Z:.0f} µm. 2D neighborhoods take the\ncolour of their best-matching 3D neighborhood.")

    # b: confusion matrix
    ax = place(fig, 62, 10, 50, 50)
    order = np.argsort(-ct.to_numpy()[r, c])
    rows, cols = ct.index[r][order], ct.columns[c][order]
    pct = ct.loc[rows, cols].div(ct.loc[rows].sum(axis=1), axis=0) * 100
    im = ax.imshow(pct.to_numpy(), cmap="Blues", vmin=0, vmax=100, aspect="equal")
    ax.set_xticks(range(len(cols)), [str(x) for x in cols], fontsize=5)
    ax.set_yticks(range(len(rows)), [str(x) for x in rows], fontsize=5)
    ax.tick_params(length=0, pad=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel(f"2D neighborhood (k = {K2})")
    ax.set_ylabel(f"3D neighborhood (k = {K3})")
    cax = place(fig, 114.5, 20, 1.6, 30)
    cb = fig.colorbar(im, cax=cax, ticks=[0, 50, 100])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1.5, labelsize=5.5)
    cb.set_label("% of the 3D neighborhood's cells", fontsize=6)
    letter(fig, 52, 2, "b")
    note(fig, 62, 9, f"Best-matching pairs on the diagonal; ARI = {table.loc['2D k=10 vs 3D k=30', 'neighborhood_ari']:.2f}",
         va="bottom")

    # c: border score, 2D vs 3D
    ax = place(fig, 142, 10, 36, 36)
    x, y = d[f"s3_{K3}"][sec_mask], d[f"s2_{K2}"][sec_mask]
    hb = ax.hexbin(x, y, gridsize=44, extent=(0, 0.5, 0, 0.5), cmap="Greys", norm=LogNorm(vmin=1), linewidths=0,
                   rasterized=True)
    ax.axvline(THRESHOLD, color=fs.INK, lw=0.5)
    ax.axhline(THRESHOLD, color=fs.INK, lw=0.5)
    ax.plot([0, 0.5], [0, 0.5], color=fs.MID, lw=0.4)
    n = len(x)
    for qx, qy, mask, name, col, va in ((0.375, 0.485, (x > THRESHOLD) & (y > THRESHOLD), "both", fs.BORDER_CLASS["2D and 3D"], "top"),
                                        (0.375, 0.015, (x > THRESHOLD) & (y <= THRESHOLD), "3D only", fs.BORDER_CLASS["3D only"], "bottom"),
                                        (0.125, 0.485, (x <= THRESHOLD) & (y > THRESHOLD), "2D only", fs.BORDER_CLASS["2D only"], "top")):
        ax.text(qx, qy, f"{name}\n{100 * mask.sum() / n:.1f}%", ha="center", va=va, fontsize=6, color=fs.INK)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.5)
    ax.set_xticks([0, 0.25, 0.5], ["0", "0.25", "0.5"])
    ax.set_yticks([0, 0.25, 0.5], ["0", "0.25", "0.5"])
    ax.set_xlabel(f"Border score in 3D (k = {K3})")
    ax.set_ylabel(f"Border score in 2D (k = {K2})")
    cax = place(fig, 142, 55, 36, 1.5)
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1.5, labelsize=5.5, which="both")
    cb.set_label("Cells per bin", fontsize=6, labelpad=1)
    letter(fig, 130, 2, "c")
    note(fig, 142, 9, f"Spearman ρ = {table.loc['2D k=10 vs 3D k=30', 'p2_spearman']:.2f}", va="bottom")

    # d: window composition differences
    ax = place(fig, 14, 94, 42, 32)
    for (name, (v, kind)), ls in zip(window_tvds(d).items(), ("-", "-", (0, (4, 1.5)))):
        xs = np.sort(v)
        ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=fs.KIND[kind], ls=ls, label=name)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("Difference in window cell-type mix\n(total variation distance, same cell)")
    ax.set_ylabel("Cumulative fraction of cells")
    ax.legend(loc="lower right", fontsize=5.6, handlelength=1.6)
    letter(fig, 1, 88, "d")

    # e: threshold sweep
    ax = place(fig, 76, 94, 42, 32)
    seen = set()
    for comp in COMPARISON_LABEL:
        sub = sweep[sweep["comparison"] == comp]
        kind = kinds[comp]
        ax.plot(sub["value"], sub["jaccard"], color=fs.KIND[kind], ls="-" if kind not in seen else (0, (4, 1.5)),
                marker="o", ms=1.8, label=kind if kind not in seen else None)
        seen.add(kind)
    chance = sweep.groupby("value")["chance"].mean()
    ax.plot(chance.index, chance.to_numpy(), color=fs.INK, lw=0.6, ls=(0, (1, 1.5)), label="chance")
    ax.axvline(THRESHOLD, color=fs.HAIR, lw=0.8, zorder=0)
    ax.set_xlim(0.04, 0.46)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0.05, 0.15, 0.25, 0.35, 0.45])
    ax.set_xlabel("Membership threshold that defines a border cell")
    ax.set_ylabel("Border Jaccard")
    ax.legend(loc="center right", fontsize=5.6, bbox_to_anchor=(1.0, 0.6))
    letter(fig, 64, 88, "e")

    # f: own vs fixed neighborhoods
    ax = place(fig, 140, 94, 20, 32)
    ends = {}
    for comp in ("3D k=30 vs k=50", "2D k=10 vs k=15", "2D k=10 vs 3D k=30", "2D k=15 vs 3D k=50"):
        sub = fixed[fixed["comparison"] == comp]
        own = sub.loc[sub["neighborhoods"] == "each run's own", "border_jaccard"].iloc[0]
        fx = sub.loc[sub["neighborhoods"] != "each run's own", "border_jaccard"]
        color = fs.KIND[sub["kind"].iloc[0]]
        ax.plot([0, 1], [own, fx.mean()], color=color, lw=0.9)
        ax.plot([0, 1], [own, fx.mean()], "o", ms=3, color=color)
        ax.plot([1.0] * len(fx), fx, "o", ms=1.8, color=color, alpha=0.4, mew=0)
        ends[comp] = fx.mean()
    label_y = -np.inf  # direct labels, pushed apart where the line ends are close
    for comp, y_end in sorted(ends.items(), key=lambda kv: kv[1]):
        label_y = max(y_end, label_y + 0.022)
        ax.text(1.12, label_y, COMPARISON_LABEL[comp], fontsize=5.3, va="center")
    chance = fixed["border_jaccard_chance"].mean()
    ax.axhline(chance, color=fs.INK, lw=0.6, ls=(0, (1, 1.5)))
    ax.text(-0.25, chance - 0.006, "chance", fontsize=5.5, va="top")
    ax.set_xlim(-0.3, 1.1)
    ax.set_ylim(0, 0.3)
    ax.set_xticks([0, 1], ["own", "fixed"])
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_xlabel("Neighborhoods")
    ax.set_ylabel("Border Jaccard")
    letter(fig, 128, 88, "f")

    # g: agreement dot plot
    metrics = (("neighborhood_ari", "Neighborhood ARI"), ("border_jaccard", "Border Jaccard"),
               ("p2_spearman", "Border-score Spearman ρ"), ("celltype_enrichment_r", "Border cell-type enrichment r"))
    order = list(COMPARISON_LABEL)
    ypos = np.arange(len(order))[::-1]
    for i, (col, label) in enumerate(metrics):
        ax = place(fig, 36 + i * 37, 142, 32, 19)
        for yy, comp in zip(ypos, order):
            ax.plot([0, table.loc[comp, col]], [yy, yy], color=fs.HAIR, lw=0.6, zorder=1)
            ax.plot(table.loc[comp, col], yy, "o", ms=3.2, color=fs.KIND[kinds[comp]], zorder=3)
            if col == "border_jaccard":
                ax.plot(table.loc[comp, "border_jaccard_chance"], yy, marker="|", ms=5, mew=0.8, color=fs.INK, zorder=2)
        ax.set_xlim(0, 1.04)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.set_xlabel(label)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_yticks(ypos, [COMPARISON_LABEL[c] for c in order] if i == 0 else [""] * len(order))
        if col == "border_jaccard":
            ax.text(1.0, 1.03, "|  expected by chance", transform=ax.transAxes, fontsize=5.3, ha="right", va="bottom")
    letter(fig, 1, 137, "g")
    fs.save(fig, FIG / "fig4_agreement_2d_vs_3d")


# ----------------------------------------------------------------------------
# Fig 5: supporting checks
# ----------------------------------------------------------------------------

def fig5_checks(d):
    depth = pd.read_csv(OUT5 / "boundary_depth_profile.csv")
    ref = pd.read_csv(OUT6 / "centroid_subsample_reference.csv")
    fig = new_figure(56)
    labels = [f"{a:.0f}–{b:.0f}" for a, b in depth.query("k == 30")[["d_z_face_from", "d_z_face_to"]].to_numpy()]
    for i, (col, ylabel, ylim) in enumerate((("median_radius_um", "Median window radius (µm)", (0, 27)),
                                             (f"pct_border_n{N}", "Border cells (%)", (0, 14)))):
        ax = place(fig, 14 + i * 60, 7, 40, 30)
        for k, color in ((30, fs.INK), (50, fs.MID)):
            sub = depth[depth["k"] == k]
            ax.plot(range(len(sub)), sub[col], marker="o", ms=2.3, color=color, label=f"k = {k}")
        ax.set_xticks(range(len(labels))[::2], labels[::2], rotation=45, ha="right")
        ax.set_ylim(*ylim)
        ax.set_xlabel("Distance to top or bottom face (µm)")
        ax.set_ylabel(ylabel)
        if i == 0:
            ax.legend(loc="lower right")
        letter(fig, 2 + i * 60, 2, "ab"[i])

    ax = place(fig, 136, 7, 40, 30)
    for run, color, off in (("3D k=30", fs.INK, -0.012), ("3D k=50", fs.MID, 0.012)):
        sub = ref[ref["run"] == run]
        ax.plot(sub["centroid_cells_fraction"] + off, sub["border_jaccard"], "o", ms=2.2, color=color, alpha=0.45, mew=0)
        mean = sub.groupby("centroid_cells_fraction")["border_jaccard"].mean()
        ax.plot(mean.index + off, mean.to_numpy(), "_", ms=8, mew=1.1, color=color, label=run.replace("=", " = "))
    ax.set_xlim(0.42, 0.98)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0.5, 0.75, 0.9], ["50", "75", "90"])
    ax.set_xlabel("Cells used to estimate centroids (%)")
    ax.set_ylabel("Border Jaccard with the\nall-cell result")
    ax.legend(loc="lower right")
    letter(fig, 122, 2, "c")
    fs.save(fig, FIG / "fig5_supporting_checks")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fs.use_style()
    d = load()
    fig1_volume(d)
    fig2_interfaces(d)
    fig3_borders(d, border_normals(d))
    fig4_agreement(d)
    fig5_checks(d)
    print(f"Figures written to {FIG}")


if __name__ == "__main__":
    main()
