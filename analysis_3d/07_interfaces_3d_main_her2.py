"""Interfaces between 3D neighborhoods, their orientation, and what a 2D section sees of them.

05 and 06 showed that border *cells* (membership-probability borders) are too
unstable to compare cell by cell. The neighborhood labels themselves are
stable (seed-to-seed ARI 0.998), so this script works with the geometric
interfaces between neighborhoods instead: the places where cells of two
different 3D neighborhoods touch.

Interface points. Cells are joined by the 3D Delaunay triangulation of their
centroids; edges longer than MAX_EDGE are dropped (they span gaps, not
contacts). Every remaining edge whose two cells carry different 3D
neighborhood labels marks a contact between two neighborhoods, and its
midpoint is an interface point.

Orientation. One cell-to-cell edge is a poor estimate of the interface normal,
so the normal at an interface point is the mean direction (always pointing
from the lower-numbered to the higher-numbered neighborhood) of all interface
edges of the same neighborhood pair within SMOOTH_RADIUS. The tilt is
|cos| of the angle between that normal and the z axis: 0 when the sections cut
straight across the interface, 1 when the interface lies in the section plane.
For randomly oriented interfaces |cos| is uniform on [0, 1].

What a section sees. A simulated section can only register an interface as a
boundary between neighborhoods within its own plane. For every section cell
within NEAR of an interface point, the script asks whether the 2D analysis
puts that cell on a 2D neighborhood boundary (at least half of its
IN_PLANE_NEIGHBORS nearest neighbors in the same section carry a different 2D
label), and reports that rate by interface tilt, next to the rate for section
cells that are not near any 3D interface. The same is done with the 3D labels
read within the section plane, which isolates the geometry from the 2D
re-clustering.

Inputs: results/main_her2/main_her2_mingl3d.h5ad            (02)
        results/main_her2/2d_vs_3d/sections_2d_mingl.h5ad   (03)
Output: results/main_her2/interfaces_3d/
Run:    .venv/bin/python analysis_3d/07_interfaces_3d_main_her2.py
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, cKDTree

from common import OUT

IN_3D = OUT / "main_her2_mingl3d.h5ad"
IN_2D = OUT / "2d_vs_3d" / "sections_2d_mingl.h5ad"
OUT7 = OUT / "interfaces_3d"

PAIRS = ((30, 10), (50, 15))  # (3D k, 2D k) with matched window radius, as in 03
MAX_EDGE = 20.0  # µm; 95% of Delaunay edges are shorter
SMOOTH_RADIUS = 15.0  # µm; interface edges averaged for the normal
MIN_EDGES = 8  # interface edges required within SMOOTH_RADIUS
NEAR = 6.0  # µm; a section cell this close to an interface point is "at" the interface
IN_PLANE_NEIGHBORS = 6
TILT_BINS = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0])


def _n_from_3d_run(k: int = 30) -> int:
    rs = pd.read_csv(OUT / "run_summary.csv")
    return int(rs.loc[rs["k"] == k, "prob_elbow_n"].iloc[0])


N = _n_from_3d_run()


def delaunay_edges(xyz: np.ndarray) -> np.ndarray:
    """Unique cell pairs joined by the Delaunay triangulation, shorter than MAX_EDGE."""
    s = Delaunay(xyz).simplices
    e = np.vstack([s[:, [i, j]] for i in range(4) for j in range(i + 1, 4)])
    e.sort(axis=1)
    e = np.unique(e, axis=0)
    return e[np.linalg.norm(xyz[e[:, 0]] - xyz[e[:, 1]], axis=1) < MAX_EDGE]


def interface_points(xyz: np.ndarray, labels: np.ndarray, edges: np.ndarray) -> pd.DataFrame:
    """Interface points between neighborhoods with a smoothed unit normal and tilt."""
    e = edges[labels[edges[:, 0]] != labels[edges[:, 1]]].copy()
    swap = labels[e[:, 0]] > labels[e[:, 1]]
    e[swap] = e[swap][:, ::-1]  # every edge points from the lower to the higher label
    a, b = labels[e[:, 0]], labels[e[:, 1]]
    mid = (xyz[e[:, 0]] + xyz[e[:, 1]]) / 2
    u = xyz[e[:, 1]] - xyz[e[:, 0]]
    u /= np.linalg.norm(u, axis=1, keepdims=True)

    normal, count = np.zeros_like(u), np.zeros(len(u), int)
    pair = a * 1000 + b
    for p in np.unique(pair):
        m = np.flatnonzero(pair == p)
        for i, idx in zip(m, cKDTree(mid[m]).query_ball_point(mid[m], SMOOTH_RADIUS)):
            normal[i] = u[m[idx]].sum(axis=0)
            count[i] = len(idx)
    length = np.linalg.norm(normal, axis=1)
    ok = (count >= MIN_EDGES) & (length > 0)
    normal = normal[ok] / length[ok, None]
    normal[normal[:, 2] < 0] *= -1  # upper hemisphere
    return pd.DataFrame({
        "x": mid[ok, 0], "y": mid[ok, 1], "z": mid[ok, 2],
        "nb_a": a[ok], "nb_b": b[ok],
        "nx": normal[:, 0], "ny": normal[:, 1], "nz": normal[:, 2],
        "tilt": np.abs(normal[:, 2]),
        "coherence": length[ok] / count[ok],  # 1 = all edges parallel, 0 = no common direction
    })


def in_plane_boundary(xy: np.ndarray, labels: np.ndarray, section: np.ndarray) -> np.ndarray:
    """Section cells with at least half of their nearest in-section neighbors in another neighborhood."""
    out = np.zeros(len(labels), bool)
    for s in np.unique(section):
        m = np.flatnonzero(section == s)
        _, nb = cKDTree(xy[m]).query(xy[m], k=IN_PLANE_NEIGHBORS + 1)
        out[m] = (labels[m][nb[:, 1:]] != labels[m][:, None]).mean(axis=1) >= 0.5
    return out


def detection_by_tilt(points: pd.DataFrame, xyz_sec, at_2d, at_3d_in_plane, pair_name: str) -> pd.DataFrame:
    """Rate of in-plane boundaries among section cells near a 3D interface, by interface tilt."""
    dist, j = cKDTree(points[["x", "y", "z"]].to_numpy()).query(xyz_sec, k=1)
    near = dist < NEAR
    tilt = points["tilt"].to_numpy()[j]
    rows = []

    def add(label, lo, hi, mask):
        n = int(mask.sum())
        for name, flag in (("2D neighborhoods", at_2d), ("3D neighborhoods read in the section plane", at_3d_in_plane)):
            rate = float(flag[mask].mean())
            rows.append({"pair": pair_name, "labels": name, "group": label, "tilt_from": lo, "tilt_to": hi, "n_cells": n,
                         "boundary_rate": rate, "se": float(np.sqrt(rate * (1 - rate) / n))})

    for lo, hi in zip(TILT_BINS[:-1], TILT_BINS[1:]):
        add("near a 3D interface", lo, hi, near & (tilt >= lo) & ((tilt < hi) | (hi == 1.0)))
    add("near a 3D interface, any tilt", np.nan, np.nan, near)
    add("not near a 3D interface", np.nan, np.nan, ~near)
    return pd.DataFrame(rows)


def main() -> None:
    OUT7.mkdir(parents=True, exist_ok=True)
    cells = ad.read_h5ad(IN_3D)
    sec = ad.read_h5ad(IN_2D)
    pos = cells.obs_names.get_indexer(sec.obs["cell_id"].astype(str).to_numpy())
    xyz = cells.obs[["x", "y", "z"]].to_numpy(float)
    section = sec.obs["section_z"].to_numpy(float)
    edges = delaunay_edges(xyz)
    print(f"{len(edges):,} cell contacts (Delaunay edges under {MAX_EDGE:.0f} µm)")

    summary, detection, hist, pairs = [], [], [], []
    for k3, k2 in PAIRS:
        lab3 = cells.obs[f"neighborhood_k{k3}_n{N}"].astype(int).to_numpy()
        lab2 = sec.obs[f"neighborhood_2d_k{k2}_n{N}"].astype(int).to_numpy()
        pts = interface_points(xyz, lab3, edges)
        pts.round(3).to_csv(OUT7 / f"interface_points_k{k3}.csv.gz", index=False)

        tilt = pts["tilt"].to_numpy()
        summary.append({
            "run": f"3D k={k3}",
            "contacts_between_neighborhoods_pct": round(100 * float((lab3[edges[:, 0]] != lab3[edges[:, 1]]).mean()), 1),
            "interface_points": len(pts),
            "median_coherence": round(float(pts["coherence"].median()), 2),
            "mean_tilt": round(float(tilt.mean()), 3),  # 0.5 if random
            "pct_within_30deg_of_section_plane": round(100 * float((tilt > np.cos(np.radians(30))).mean()), 1),  # 13.4 if random
            "pct_within_30deg_of_vertical": round(100 * float((tilt < np.sin(np.radians(30))).mean()), 1),  # 50.0 if random
        })
        h, e = np.histogram(tilt, np.linspace(0, 1, 11), density=True)
        hist.append(pd.DataFrame({"run": f"3D k={k3}", "tilt_from": e[:-1], "tilt_to": e[1:], "density": h}))
        by_pair = (pts.groupby(["nb_a", "nb_b"]).agg(interface_points=("tilt", "size"), mean_tilt=("tilt", "mean"),
                                                     pct_within_30deg_of_section_plane=("tilt", lambda t: 100 * (t > np.cos(np.radians(30))).mean()))
                   .sort_values("interface_points", ascending=False).reset_index())
        by_pair.insert(0, "run", f"3D k={k3}")
        pairs.append(by_pair)

        at_2d = in_plane_boundary(xyz[pos][:, :2], lab2, section)
        at_3d = in_plane_boundary(xyz[pos][:, :2], lab3[pos], section)
        detection.append(detection_by_tilt(pts, xyz[pos], at_2d, at_3d, f"3D k={k3}, 2D k={k2}"))

    summary = pd.DataFrame(summary)
    detection = pd.concat(detection, ignore_index=True)
    summary.to_csv(OUT7 / "interface_summary.csv", index=False)
    pd.concat(hist, ignore_index=True).round(4).to_csv(OUT7 / "interface_tilt_histogram.csv", index=False)
    pd.concat(pairs, ignore_index=True).round(3).to_csv(OUT7 / "interface_tilt_by_pair.csv", index=False)
    detection.round(4).to_csv(OUT7 / "interface_detection_by_tilt.csv", index=False)

    print("\nInterfaces between 3D neighborhoods (random orientation: mean tilt 0.5, 13.4% within 30° of the section plane)\n"
          + summary.to_string(index=False))
    show = detection.assign(pct=(100 * detection["boundary_rate"]).round(1))
    show["tilt"] = np.where(show["tilt_from"].isna(), show["group"],
                            show["tilt_from"].map("{:.1f}".format) + "–" + show["tilt_to"].map("{:.1f}".format))
    print("\nSection cells on an in-plane neighborhood boundary (%), by tilt of the nearby 3D interface\n"
          "(tilt 0 = interface cut straight across, 1 = interface lies in the section plane)\n"
          + show.pivot_table(index="tilt", columns=["pair", "labels"], values="pct", sort=False).to_string())


if __name__ == "__main__":
    main()
