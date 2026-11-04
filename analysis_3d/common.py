"""Shared steps for the Main HER2 analyses (3D run and the 2D-vs-3D comparison).

Each step works in 2D or 3D through ``z_key`` (None = the 2D MINGL code path,
which tests/test_tl3d.py checks is identical to ``mingl.tl``).
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Optional

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mingl.tl as tl
from mingl import tl3d

DATA = Path.home() / "Desktop/Mingle_data/kuett2022_3dimc"
OUT = DATA / "results/main_her2"

N_RANGE = range(1, 41)
THRESHOLD = 0.25
CT, REG = "cell_type", "unique_region"

# Neighborhood clusterer for the n sweep. MINGL's default MiniBatchKMeans is
# seed-unstable on these windows (seed-to-seed ARI ~0.5 at n=11, border sets
# ~10% Jaccard), which made neighborhood- and border-level 2D-vs-3D
# comparisons uninterpretable. Full k-means with restarts is stable; see
# mingl.tl3d.n_neighbors.
CLUSTERER = "kmeans"
N_INIT = 10


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def select_n_neighborhoods(
    cells: ad.AnnData,
    k: int,
    cell_types: list[str],
    out: Path,
    *,
    z_key: Optional[str] = "z",
    prefix: Optional[str] = None,
    region_key: str = REG,
    clusterer: str = CLUSTERER,
    n_init: int = N_INIT,
    random_state: int = 0,
):
    """k-NN windows, n = 1..40 sweep (``clusterer``), elbow + plateau selection.

    ``cells.obs_names`` must be "0".."n-1" (KNN2 indexes neighbors positionally).
    Returns ``(wa, windows, best_n, ll_elbow_n, prob_elbow_n)``; ``wa.obs`` holds
    the ``Neighborhood_{n}`` labels for every n in the sweep.
    """
    prefix = prefix or f"k{k}"
    windows = tl3d.KNN2(cells, z_key=z_key, cluster_col=CT, region_key=region_key, ks=(k,))[k]
    wa = ad.AnnData(obs=windows.copy())
    wa.obs_names = wa.obs.index.astype(str)

    summary, _ = quiet(
        tl3d.run_mingl_over_n_clusters, wa, cell_types, n_range=N_RANGE,
        plot_summary=False, z_key=z_key, region_key=region_key,
        clusterer=clusterer, n_init=n_init, random_state=random_state,
    )
    _, ll_n, _ = tl.find_elbow_point(None, adata=wa, uns_key="mingl_n_clusters",
                                     y_key="avg_log_likelihood", x_key="n_clusters")
    _, prob_n, _ = tl.find_elbow_point(None, adata=wa, uns_key="mingl_n_clusters",
                                       y_key="avg_assigned_probability", x_key="n_clusters")
    composite_df, best_n, _ = quiet(
        tl.find_best_unsupervised_plateau, None, None,
        elbow_min=min(ll_n, prob_n), elbow_max=max(ll_n, prob_n),
        adata=wa, uns_key="mingl_n_clusters",
        ll_key="avg_log_likelihood", prob_key="avg_assigned_probability",
    )
    fig, legend_fig = tl.plot_stable_composite(composite_df, best_n, int(ll_n), int(prob_n), show=False)
    fig.savefig(out / f"{prefix}_n_selection.png", dpi=150, bbox_inches="tight")
    legend_fig.savefig(out / f"{prefix}_n_selection_legend.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    summary.to_csv(out / f"{prefix}_n_sweep.csv", index=False)

    return wa, windows, int(best_n), int(ll_n), int(prob_n)


def name_neighborhoods(windows: pd.DataFrame, labels: np.ndarray, cell_types: list[str]):
    """Composition (mean window fraction) per neighborhood + a descriptive name.

    Names list the cell types most enriched over the volume average (log2 > 0.5
    and >= 5% of the window), up to three, ordered by share of the window.
    """
    frac = windows[cell_types].div(windows[cell_types].sum(axis=1), axis=0)
    comp = frac.groupby(labels).mean()
    enrich = np.log2((comp + 1e-3) / (frac.mean() + 1e-3))
    names = {}
    for nb in comp.index:
        keep = [c for c in comp.columns if enrich.loc[nb, c] > 0.5 and comp.loc[nb, c] >= 0.05]
        keep = sorted(keep, key=lambda c: -comp.loc[nb, c])[:3] or [comp.loc[nb].idxmax()]
        names[nb] = " + ".join(keep)
    # disambiguate identical names
    seen: dict[str, int] = {}
    for nb in sorted(names, key=int):
        n = names[nb]
        seen[n] = seen.get(n, 0) + 1
        if list(names.values()).count(n) > 1:
            names[nb] = f"{n} ({seen[n]})"
    return comp, enrich, names


def membership_and_borders(
    cells: ad.AnnData,
    nb_col: str,
    k: int,
    tag: str,
    *,
    z_key: Optional[str] = "z",
    region_key: str = REG,
) -> pd.Series:
    """Centroids -> GMM membership probabilities -> border counts for ``obs[nb_col]``.

    Writes ``obsm[f"neighborhood_probabilities_{tag}"]`` and
    ``obs[f"n_positive_{tag}"]`` (neighborhoods above THRESHOLD); returns the latter.
    """
    centroids = tl3d.centroid_Calculation(
        cells, k=k, ks=(k,), cluster_col=CT, neighborhood_col=nb_col, region_col=region_key, z_key=z_key,
    )
    prob_key = f"neighborhood_probabilities_{tag}"
    tl3d.cpu_gmm_probability(
        cells, centroids, cluster_col=CT, neighborhood_col=nb_col, region_key=region_key,
        z_key=z_key, ks=(k,), k=k, prob_key=prob_key,
        prob_variable_key=f"neighborhood_probability_neighborhoods_{tag}",
    )
    tl.findPositives(cells, prob_key=prob_key, threshold=THRESHOLD, result_key=f"n_positive_{tag}")
    return cells.obs[f"n_positive_{tag}"]
