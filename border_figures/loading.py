"""Loading, subsampling, null construction and posterior computation (with cache).

Nothing here touches the lab files: every function takes a path the caller
supplies and only ever reads it. All outputs (caches, CSVs, figures) go to local
paths under this folder.

Posteriors are always produced through
:func:`mingl.tl.emission_models.mingl_membership_probabilities`, including for the
baseline ``diagonal_gaussian`` -- that model is a verified re-implementation of the
shipped :func:`mingl.tl.gmm.cpu_gmm_probability` (``tests/test_emission_models.py``
pins argmax equality and max probability difference < 1e-4), so using one code
path for all five models keeps the comparison apples-to-apples.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import time
from pathlib import Path
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd

from .config import DatasetSpec, LevelSpec, resolve_dataset

__all__ = [
    "load_dataset",
    "slim_obs",
    "subsample_regions",
    "make_null",
    "compute_posterior",
    "synthetic_tissue",
    "NULL_MODES",
]

NULL_MODES = ("celltype", "coordinates", "units", "none")


def _quiet(func, *args, **kwargs):
    """Run ``func`` with stdout suppressed (KNN2 and friends are chatty)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


# ---------------------------------------------------------------------------
# Loading / subsampling
# ---------------------------------------------------------------------------
def _read_columns(path: str, columns: Sequence[str]) -> ad.AnnData | None:
    """Read only ``columns`` from a CSV into an AnnData, or ``None`` if not applicable.

    :func:`mingl.pp.read_file` loads every column, which is the right default but
    not survivable here: the melanoma table is 5M rows x 88 columns (6.3 GB on
    disk) and these figures need at most seven of those columns. Reading the
    needed subset keeps all three datasets in memory at once.

    Returns ``None`` for non-CSV inputs or when any requested column is absent,
    so the caller falls back to the package reader (and gets its error message).
    This is purely an I/O optimization -- the resulting AnnData is identical in
    structure to what :func:`mingl.pp.read_file` produces for a CSV.
    """
    if not str(path).lower().endswith(".csv"):
        return None
    header = pd.read_csv(path, nrows=0)
    missing = [c for c in columns if c not in header.columns]
    if missing:
        return None
    obs = pd.read_csv(path, usecols=list(columns))[list(columns)]
    obs.index = obs.index.astype(str)
    print(f"  read {obs.shape[0]:,} rows x {obs.shape[1]} of {header.shape[1]} columns")
    return ad.AnnData(X=np.zeros((obs.shape[0], 0), dtype=np.float32), obs=obs)


def load_dataset(
    path: str | os.PathLike,
    dataset: str,
    level: LevelSpec,
    *,
    required_extra: tuple[str, ...] = (),
) -> ad.AnnData:
    """Read a ``.csv``/``.h5ad`` and validate the columns this level needs."""
    import mingl as mg

    spec = resolve_dataset(dataset)
    path = str(path)

    needed_cols = list(
        dict.fromkeys(
            [level.unit_col, level.feature_col, spec.cell_type_col, spec.region_key,
             spec.x_key, spec.y_key, *required_extra]
        )
    )
    adata = _read_columns(path, needed_cols) or mg.pp.read_file(path)

    needed = {
        "unit label": level.unit_col,
        "window feature": level.feature_col,
        "cell type": spec.cell_type_col,
        "region": spec.region_key,
        "x": spec.x_key,
        "y": spec.y_key,
        **{f"extra:{c}": c for c in required_extra},
    }
    missing = {role: col for role, col in needed.items() if col not in adata.obs.columns}
    if missing:
        raise KeyError(
            "Missing required obs columns for "
            f"{spec.name}/{level.name}: {missing}. Present: {list(adata.obs.columns)[:25]}"
        )

    # Drop rows that cannot be scored rather than letting NaNs propagate silently.
    key_cols = [level.unit_col, level.feature_col, spec.cell_type_col, spec.region_key,
                spec.x_key, spec.y_key]
    complete = adata.obs.loc[:, key_cols].notna().all(axis=1).to_numpy()
    if not complete.all():
        print(f"  dropping {int((~complete).sum()):,} cells with missing key columns")
        adata = adata[complete].copy()

    # KNN2 needs at least k cells in every region.
    counts = adata.obs[spec.region_key].astype(str).value_counts()
    too_small = counts[counts < level.k]
    if len(too_small):
        print(
            f"  dropping {len(too_small)} region(s) with < k={level.k} cells "
            f"({int(too_small.sum()):,} cells)"
        )
        keep = adata.obs[spec.region_key].astype(str).isin(counts[counts >= level.k].index).to_numpy()
        adata = adata[keep].copy()

    return adata


def slim_obs(adata: ad.AnnData, columns: Sequence[str]) -> ad.AnnData:
    """Copy of ``adata`` keeping only ``columns`` in ``obs`` (and no X/layers).

    The lab CSVs carry dozens of marker columns; the border figures need at most
    seven. Slimming right after loading keeps the intestine (2.5M cells) and its
    permuted null copy comfortably in memory.
    """
    keep = [c for c in dict.fromkeys(columns) if c in adata.obs.columns]
    obs = adata.obs.loc[:, keep].copy()
    obs.index = obs.index.astype(str)  # AnnData warns otherwise
    return ad.AnnData(X=np.zeros((obs.shape[0], 0), dtype=np.float32), obs=obs)


def subsample_regions(
    adata: ad.AnnData,
    *,
    region_key: str,
    frac: float = 1.0,
    max_cells: int | None = None,
    min_per_region: int = 0,
    seed: int = 0,
) -> ad.AnnData:
    """Region-stratified subsample, preserving obs order.

    Subsampling thins the k-NN windows, so it changes the posterior slightly --
    use it for smoke tests and for melanoma-scale inputs, not for final numbers.
    ``min_per_region`` (set it to ``k``) keeps every region large enough for the
    window computation.
    """
    n = adata.n_obs
    target_frac = float(frac)
    if max_cells is not None and n > max_cells:
        target_frac = min(target_frac, max_cells / n)
    if target_frac >= 1.0:
        return adata

    rng = np.random.default_rng(seed)
    regions = adata.obs[region_key].astype(str).to_numpy()
    keep = np.zeros(n, dtype=bool)
    for region in pd.unique(regions):
        idx = np.where(regions == region)[0]
        n_keep = max(int(round(idx.size * target_frac)), min(min_per_region, idx.size))
        if n_keep >= idx.size:
            keep[idx] = True
        else:
            keep[rng.choice(idx, size=n_keep, replace=False)] = True
    print(f"  subsampled {int(keep.sum()):,} / {n:,} cells ({target_frac:.0%}, region-stratified)")
    return adata[keep].copy()


# ---------------------------------------------------------------------------
# Null condition
# ---------------------------------------------------------------------------
def make_null(
    adata: ad.AnnData,
    *,
    mode: str,
    spec: DatasetSpec,
    level: LevelSpec,
    seed: int = 0,
) -> ad.AnnData:
    """Build a spatially randomized negative control from a real dataset.

    The point of the null condition is to answer "what do these four border
    read-outs look like when there is no real border?", so the randomization
    must destroy the local compositional structure MINGL scores while keeping
    everything the analysis conditions on:

    ``celltype`` (default)
        Permute each cell's identity labels (the window ``feature_col``, and the
        cell-type column when it differs) **within each region**. Coordinates,
        region membership, organizational-unit labels and all label abundances
        are preserved, so the CN1|CN2 interface stays exactly where it was and
        "are border cells still at the interface?" remains a meaningful question.
        Cell identity is permuted as a block, so a cell keeps a coherent
        (cell type, lower-level label) pairing.
    ``coordinates``
        Permute x/y within region instead -- destroys spatial structure entirely,
        including the interface, so location panels are uninformative. Provided
        for completeness.
    ``units``
        Permute the organizational-unit labels within region: the mixture
        components no longer correspond to tissue anatomy.
    ``none``
        Return the data unchanged (used when ``--null-data`` supplies a real null
        file the user generated separately).
    """
    if mode not in NULL_MODES:
        raise ValueError(f"Unknown null mode {mode!r}; choose from {NULL_MODES}.")
    if mode == "none":
        return adata

    out = adata.copy()
    rng = np.random.default_rng(seed)
    regions = out.obs[spec.region_key].astype(str).to_numpy()

    if mode == "celltype":
        cols = [level.feature_col]
        if spec.cell_type_col != level.feature_col:
            cols.append(spec.cell_type_col)
    elif mode == "coordinates":
        cols = [spec.x_key, spec.y_key]
    else:  # "units"
        cols = [level.unit_col]

    block = out.obs.loc[:, cols].to_numpy(copy=True)
    for region in pd.unique(regions):
        idx = np.where(regions == region)[0]
        if idx.size < 2:
            continue
        block[idx] = block[idx][rng.permutation(idx.size)]
    for j, col in enumerate(cols):
        out.obs[col] = block[:, j]

    print(f"  null condition: permuted {cols} within each of {pd.unique(regions).size} regions (mode={mode})")
    return out


# ---------------------------------------------------------------------------
# Posteriors (with an on-disk cache)
# ---------------------------------------------------------------------------
def _cache_key(**parts) -> str:
    payload = "|".join(f"{k}={parts[k]}" for k in sorted(parts))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def compute_posterior(
    adata: ad.AnnData,
    *,
    model: str,
    spec: DatasetSpec,
    level: LevelSpec,
    cache_dir: str | os.PathLike | None = None,
    cache_tag: str = "",
    verbose: bool = True,
) -> pd.DataFrame:
    """MINGL membership posterior for one emission model at one hierarchy level.

    Returns a (cells x units) DataFrame indexed like ``adata.obs``. Results are
    cached under ``cache_dir`` keyed by dataset/level/model/cell count, so a
    figure can be re-plotted without re-fitting (the k=300 tissue-unit windows on
    2.5M cells are the expensive part of both figures).
    """
    import mingl.tl as tl

    key = _cache_key(
        dataset=spec.name, level=level.name, model=model, k=level.k,
        unit=level.unit_col, feature=level.feature_col, n=adata.n_obs,
        first=str(adata.obs_names[0]), last=str(adata.obs_names[-1]), tag=cache_tag,
    )
    cache_path = Path(cache_dir) / f"posterior_{spec.name}_{level.name}_{model}_{key}.npz" if cache_dir else None
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=True) as z:
            probs = pd.DataFrame(z["probs"], index=adata.obs_names, columns=[str(c) for c in z["units"]])
        if verbose:
            print(f"  [{model}] loaded cached posterior {tuple(probs.shape)} from {cache_path.name}")
        return probs

    t0 = time.perf_counter()
    scored = _quiet(
        tl.mingl_membership_probabilities,
        adata,
        model=model,
        cluster_col=level.feature_col,
        neighborhood_col=level.unit_col,
        region_key=spec.region_key,
        x_key=spec.x_key,
        y_key=spec.y_key,
        ks=(level.k,),
        k=level.k,
        prob_key="_bf_probs",
        prob_variable_key="_bf_units",
    )
    probs = pd.DataFrame(
        np.asarray(scored.obsm["_bf_probs"], dtype=float),
        index=adata.obs_names,
        columns=[str(c) for c in scored.uns["_bf_units"]],
    )
    # Keep the AnnData clean for the next model.
    del scored.obsm["_bf_probs"]
    scored.uns.pop("_bf_units", None)
    scored.uns.pop(f"mingl_emission_{model}", None)

    if verbose:
        print(
            f"  [{model}] scored {probs.shape[0]:,} cells x {probs.shape[1]} units "
            f"(k={level.k}, features={level.feature_col!r}) in {time.perf_counter() - t0:.1f}s"
        )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, probs=probs.to_numpy(), units=np.array(probs.columns, dtype=object))
    return probs


# ---------------------------------------------------------------------------
# Synthetic tissue (local validation, no lab data)
# ---------------------------------------------------------------------------
def synthetic_tissue(
    *,
    n_regions: int = 12,
    n_cells_per_region: int = 6000,
    width: float = 400.0,
    height: float = 400.0,
    mixing_width: float = 12.0,
    seed: int = 0,
) -> ad.AnnData:
    """A layered tissue with a real hierarchy and real interfaces.

    Mimics the intestine's layering so every code path in both figures can be
    exercised without lab data: horizontal bands give four tissue units, each
    split into communities and then neighborhoods, with cell types drawn from a
    neighborhood-specific composition. Labels near a band boundary are drawn from
    the two adjacent bands with a probability that ramps over ``mixing_width``,
    so genuine border cells exist exactly where the bands meet -- which is what
    the border-location panels measure.

    Returns an AnnData with ``obs`` columns ``x``, ``y``, ``unique_region``,
    ``Cell Type``, ``Neighborhood``, ``Community`` and ``Tissue Unit``.
    """
    rng = np.random.default_rng(seed)

    # Tissue unit bands (fractions of height), each with its communities, each
    # with its neighborhoods.
    hierarchy = [
        ("Mucosa", 0.00, 0.42, {
            "Secretory Epithelial": ["Mature Epithelial", "Secretory Epithelial"],
            "Plasma Cell Enriched": ["Plasma Cell Enriched", "Adaptive Immune Enriched"],
        }),
        ("Muscularis Mucosa", 0.42, 0.58, {
            "Smooth Muscle": ["Smooth Muscle", "Innervated Smooth Muscle"],
        }),
        ("Submucosa", 0.58, 0.80, {
            "Stroma": ["Stroma", "Microvasculature"],
        }),
        ("Muscularis Externa", 0.80, 1.00, {
            "Smooth Muscle Externa": ["Smooth Muscle Externa", "Innervated Stroma"],
        }),
    ]

    cell_types = [
        "Enterocyte", "Goblet", "TA", "Cycling TA", "Paneth", "Plasma", "B",
        "CD4+ T cell", "CD8+ T", "M1 Macrophage", "M2 Macrophage", "DC",
        "Smooth muscle", "Stroma", "Endothelial", "Nerve", "Neuroendocrine", "ICC",
    ]
    n_types = len(cell_types)

    # Each neighborhood gets a sparse, distinctive composition over cell types.
    neighborhoods: list[str] = []
    for _unit, _lo, _hi, comms in hierarchy:
        for nbs in comms.values():
            neighborhoods.extend(nbs)
    comp_rng = np.random.default_rng(seed + 1)
    compositions: dict[str, np.ndarray] = {}
    for i, nb in enumerate(neighborhoods):
        base = comp_rng.dirichlet(np.full(n_types, 0.35))
        anchor = i % n_types  # a dominant "anchor" cell type per neighborhood
        base[anchor] += 1.2
        compositions[nb] = base / base.sum()

    # Flatten the hierarchy into ordered neighborhood bands with their parents.
    bands: list[tuple[float, float, str, str, str]] = []
    for unit, lo, hi, comms in hierarchy:
        comm_names = list(comms)
        comm_edges = np.linspace(lo, hi, len(comm_names) + 1)
        for ci, comm in enumerate(comm_names):
            nbs = comms[comm]
            nb_edges = np.linspace(comm_edges[ci], comm_edges[ci + 1], len(nbs) + 1)
            for ni, nb in enumerate(nbs):
                bands.append((nb_edges[ni] * height, nb_edges[ni + 1] * height, nb, comm, unit))

    parent_of = {nb: (comm, unit) for _lo, _hi, nb, comm, unit in bands}
    frames = []
    for r in range(n_regions):
        x = rng.uniform(0.0, width, n_cells_per_region)
        y = rng.uniform(0.0, height, n_cells_per_region)

        # Assign each cell to a band, with stochastic mixing near the edges.
        edges = np.array([b[0] for b in bands] + [bands[-1][1]])
        band_idx = np.clip(np.searchsorted(edges, y, side="right") - 1, 0, len(bands) - 1)
        lower_edge = edges[band_idx]
        upper_edge = edges[band_idx + 1]
        # Probability of taking the neighbouring band's label, ramping to 0.5 at
        # the boundary itself.
        p_down = 0.5 * np.clip(1.0 - (y - lower_edge) / mixing_width, 0.0, 1.0)
        p_up = 0.5 * np.clip(1.0 - (upper_edge - y) / mixing_width, 0.0, 1.0)
        u = rng.random(n_cells_per_region)
        shifted = band_idx.copy()
        shifted = np.where((u < p_down) & (band_idx > 0), band_idx - 1, shifted)
        shifted = np.where(
            (u > 1.0 - p_up) & (band_idx < len(bands) - 1), band_idx + 1, shifted
        )

        nb_labels = np.array([bands[i][2] for i in shifted], dtype=object)
        ct = np.empty(n_cells_per_region, dtype=object)
        for nb in pd.unique(nb_labels):
            m = nb_labels == nb
            ct[m] = rng.choice(cell_types, size=int(m.sum()), p=compositions[nb])

        frames.append(
            pd.DataFrame(
                {
                    "x": x,
                    "y": y,
                    "unique_region": f"R{r + 1:02d}",
                    "Cell Type": ct.astype(str),
                    "Neighborhood": nb_labels.astype(str),
                    "Community": [parent_of[nb][0] for nb in nb_labels],
                    "Tissue Unit": [parent_of[nb][1] for nb in nb_labels],
                }
            )
        )

    obs = pd.concat(frames, ignore_index=True)
    obs.index = obs.index.astype(str)
    return ad.AnnData(X=np.zeros((obs.shape[0], 0), dtype=np.float32), obs=obs)
