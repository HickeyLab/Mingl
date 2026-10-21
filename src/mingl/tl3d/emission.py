"""3D-capable emission-model scoring and comparison.

Mirrors :mod:`mingl.tl.emission_models` (``mingl_window_matrix``,
``mingl_membership_probabilities``) and :mod:`mingl.tl.model_comparison`
(``compare_emission_models``, ``attach_all_model_probabilities``). Only the
window step changes (``z_key``); the emission models, priors, splits and metrics
are imported unchanged from the 2D modules.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from ..tl.emission_models import (
    EMISSION_MODELS,
    _encode_components,
    _log_prior,
    build_emission_model,
    responsibilities_from_loglik,
)
from ..tl.model_comparison import _DEFAULT_MODELS, _evidence, _mean_finite, _stratified_split
from ._spatial import require_columns, spatial_keys
from .knn import KNN2

__all__ = [
    "mingl_window_matrix",
    "mingl_membership_probabilities",
    "compare_emission_models",
    "attach_all_model_probabilities",
]


def mingl_window_matrix(
    cells: ad.AnnData,
    *,
    cluster_col: str = "cell_type",
    region_key: str = "unique_region",
    x_key: str = "x",
    y_key: str = "y",
    z_key: Optional[str] = None,
    ks: Sequence[int] = (10, 20, 100, 300),
    k: int = 10,
) -> Tuple[np.ndarray, List[str]]:
    """Return the k-NN cell-type count matrix (n_cells x n_cell_types), obs-ordered.

    See :func:`mingl.tl.emission_models.mingl_window_matrix`; ``z_key`` switches
    the neighbor search to 3D. Works for any ``obs_names``.
    """
    del ks  # only the k-nearest window is used

    coord_cols = spatial_keys(x_key, y_key, z_key)
    require_columns(cells.obs, [*coord_cols, region_key, cluster_col], what="obs")

    # KNN2 needs a contiguous 0..n-1 integer index; run it on a minimal copy.
    obs_min = cells.obs.loc[:, [*coord_cols, region_key, cluster_col]].copy()
    obs_min.index = np.arange(obs_min.shape[0]).astype(str)
    mini = ad.AnnData(X=np.zeros((obs_min.shape[0], 0), dtype=np.float32), obs=obs_min)

    windows = KNN2(
        mini,
        x_key=x_key,
        y_key=y_key,
        z_key=z_key,
        region_key=region_key,
        cluster_col=cluster_col,
        ks=(int(k),),
    )
    windows_k = windows[k]
    features = sorted(pd.Index(cells.obs[cluster_col].astype(str)).unique().tolist())

    missing = [f for f in features if f not in windows_k.columns]
    if missing:
        raise ValueError(f"Missing KNN window columns for cell types: {', '.join(map(str, missing[:10]))}")

    X = np.ascontiguousarray(windows_k.loc[:, features].to_numpy(dtype=np.float64))
    return X, features


def mingl_membership_probabilities(
    cells: ad.AnnData,
    *,
    model: str = "full_gaussian",
    cluster_col: str = "cell_type",
    neighborhood_col: str = "neighborhood",
    region_key: str = "unique_region",
    x_key: str = "x",
    y_key: str = "y",
    z_key: Optional[str] = None,
    ks: Sequence[int] = (10, 20, 100, 300),
    k: int = 10,
    prior: str = "uniform",
    reg_covar: float = 1e-6,
    smoothing: float = 0.5,
    pseudocount: float = 0.5,
    prob_key: str = "neighborhood_probabilities",
    prob_variable_key: str = "neighborhood_probability_neighborhoods",
    random_state: int = 0,
) -> ad.AnnData:
    """Score every cell's neighborhood membership under a chosen emission model.

    See :func:`mingl.tl.emission_models.mingl_membership_probabilities`. Writes
    ``obsm[prob_key]``, ``uns[prob_variable_key]`` and
    ``uns[f"mingl_emission_{model}"]`` (which also records ``z_key``).
    """
    del random_state  # models are deterministic; kept for API symmetry

    if neighborhood_col not in cells.obs or cluster_col not in cells.obs:
        raise KeyError(f"One or more required columns ({neighborhood_col}, {cluster_col}) are missing in obs.")

    X, features = mingl_window_matrix(
        cells, cluster_col=cluster_col, region_key=region_key,
        x_key=x_key, y_key=y_key, z_key=z_key, ks=ks, k=k,
    )
    labels = cells.obs[neighborhood_col].astype(str).to_numpy()
    comp, comp_names = _encode_components(labels)
    n_components = len(comp_names)

    est = build_emission_model(model, reg_covar=reg_covar, smoothing=smoothing, pseudocount=pseudocount)
    est.fit(X, comp, n_components)
    loglik = est.log_likelihood(X)
    log_prior = _log_prior(prior, comp, n_components)
    resp = responsibilities_from_loglik(loglik, log_prior)

    logpost = loglik if log_prior is None else loglik + log_prior[None, :]
    finite = np.isfinite(logpost).any(axis=1)
    log_evidence = np.full(logpost.shape[0], -np.inf)
    if finite.any():
        log_evidence[finite] = logsumexp(logpost[finite], axis=1)
    mean_log_evidence = float(np.mean(log_evidence[np.isfinite(log_evidence)])) if finite.any() else float("nan")
    n_eval = int(np.isfinite(log_evidence).sum())
    total_log_evidence = float(np.sum(log_evidence[np.isfinite(log_evidence)])) if n_eval else float("nan")
    n_params = est.n_parameters()
    bic = float(-2.0 * total_log_evidence + n_params * np.log(max(n_eval, 1))) if n_eval else float("nan")

    cells.obsm[prob_key] = resp
    cells.uns[prob_variable_key] = comp_names
    cells.uns[f"mingl_emission_{model}"] = {
        "model": model,
        "prior": prior,
        "k": int(k),
        "z_key": z_key,
        "n_parameters": int(n_params),
        "mean_log_evidence": mean_log_evidence,
        "total_log_evidence": total_log_evidence,
        "bic": bic,
        "features": list(features),
        "components": list(comp_names),
    }
    return cells


def compare_emission_models(
    cells: ad.AnnData,
    *,
    models: Sequence[str] = _DEFAULT_MODELS,
    cluster_col: str = "cell_type",
    neighborhood_col: str = "neighborhood",
    region_key: str = "unique_region",
    x_key: str = "x",
    y_key: str = "y",
    z_key: Optional[str] = None,
    ks: Sequence[int] = (10, 20, 100, 300),
    k: int = 10,
    prior: str = "uniform",
    threshold: float = 0.25,
    test_size: float = 0.3,
    seed: int = 0,
    reg_covar: float = 1e-6,
    smoothing: float = 0.5,
    pseudocount: float = 0.5,
) -> pd.DataFrame:
    """Fit every emission model on a train split and compare held-out metrics.

    See :func:`mingl.tl.model_comparison.compare_emission_models`.
    """
    unknown = [m for m in models if m not in EMISSION_MODELS]
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}. Choose from {sorted(EMISSION_MODELS)}.")

    X, features = mingl_window_matrix(
        cells, cluster_col=cluster_col, region_key=region_key,
        x_key=x_key, y_key=y_key, z_key=z_key, ks=ks, k=k,
    )
    labels = cells.obs[neighborhood_col].astype(str).to_numpy()
    comp, comp_names = _encode_components(labels)
    n_components = len(comp_names)

    rng = np.random.default_rng(seed)
    train_mask, test_mask = _stratified_split(comp, n_components, test_size, rng)
    X_tr, comp_tr = X[train_mask], comp[train_mask]
    X_te, comp_te = X[test_mask], comp[test_mask]
    if X_te.shape[0] == 0:
        raise ValueError("Test split is empty; provide more cells or a larger test_size.")

    log_prior = _log_prior(prior, comp_tr, n_components)
    eps = 1e-12
    rows: List[dict] = []

    for model in models:
        t0 = time.perf_counter()
        est = build_emission_model(model, reg_covar=reg_covar, smoothing=smoothing, pseudocount=pseudocount)
        est.fit(X_tr, comp_tr, n_components)
        ll_tr = est.log_likelihood(X_tr)
        ll_te = est.log_likelihood(X_te)
        runtime = time.perf_counter() - t0

        resp_te = responsibilities_from_loglik(ll_te, log_prior)
        p_true = np.clip(resp_te[np.arange(X_te.shape[0]), comp_te], eps, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.sum(np.where(resp_te > 0, resp_te * np.log(resp_te), 0.0), axis=1)
        pos_counts = (resp_te > threshold).sum(axis=1)

        ev_tr = _evidence(ll_tr, log_prior)
        ev_te = _evidence(ll_te, log_prior)
        n_tr = int(np.isfinite(ev_tr).sum())
        n_params = est.n_parameters()
        bic = (
            float(-2.0 * np.sum(ev_tr[np.isfinite(ev_tr)]) + n_params * np.log(max(n_tr, 1)))
            if n_tr
            else float("nan")
        )

        rows.append(
            {
                "model": model,
                "test_logloss": float(-np.mean(np.log(p_true))),
                "test_accuracy": float(np.mean(resp_te.argmax(axis=1) == comp_te)),
                "mean_entropy": float(np.mean(ent)),
                "frac_border": float(np.mean(pos_counts >= 2)),
                "n_border": int((pos_counts >= 2).sum()),
                "train_loglik": _mean_finite(ev_tr),
                "test_loglik": _mean_finite(ev_te),
                "n_params": int(n_params),
                "bic": bic,
                "n_train": int(X_tr.shape[0]),
                "n_test": int(X_te.shape[0]),
                "n_components": int(n_components),
                "n_features": int(len(features)),
                "runtime_s": float(runtime),
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values("test_logloss", ascending=True).reset_index(drop=True)


def attach_all_model_probabilities(
    cells: ad.AnnData,
    *,
    models: Sequence[str] = _DEFAULT_MODELS,
    cluster_col: str = "cell_type",
    neighborhood_col: str = "neighborhood",
    region_key: str = "unique_region",
    x_key: str = "x",
    y_key: str = "y",
    z_key: Optional[str] = None,
    ks: Sequence[int] = (10, 20, 100, 300),
    k: int = 10,
    prior: str = "uniform",
    reg_covar: float = 1e-6,
    smoothing: float = 0.5,
    pseudocount: float = 0.5,
    key_prefix: str = "mingl_prob_",
) -> ad.AnnData:
    """Fit each model on all cells and store its posterior in ``obsm[key_prefix+model]``.

    See :func:`mingl.tl.model_comparison.attach_all_model_probabilities`.
    """
    X, features = mingl_window_matrix(
        cells, cluster_col=cluster_col, region_key=region_key,
        x_key=x_key, y_key=y_key, z_key=z_key, ks=ks, k=k,
    )
    labels = cells.obs[neighborhood_col].astype(str).to_numpy()
    comp, comp_names = _encode_components(labels)
    n_components = len(comp_names)
    log_prior = _log_prior(prior, comp, n_components)

    for model in models:
        est = build_emission_model(model, reg_covar=reg_covar, smoothing=smoothing, pseudocount=pseudocount)
        est.fit(X, comp, n_components)
        cells.obsm[f"{key_prefix}{model}"] = responsibilities_from_loglik(est.log_likelihood(X), log_prior)
        cells.uns[f"{key_prefix}{model}_neighborhoods"] = list(comp_names)
    cells.uns[f"{key_prefix}features"] = list(features)
    return cells
