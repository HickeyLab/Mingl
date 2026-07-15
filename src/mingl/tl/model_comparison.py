"""Compare MINGL emission models (Tasks 1 & 2).

Raw log-likelihood / BIC cannot be compared *across* a continuous density
(Gaussian) and a discrete pmf (multinomial / Dirichlet-multinomial): they live on
different base measures. The fair, biologically meaningful comparison evaluates the
**posterior over the shared neighborhood label set** on a held-out split -- every
model produces a categorical distribution over the same neighborhoods, so its
held-out log-loss and accuracy at recovering the neighborhood label are directly
comparable.

:func:`compare_emission_models` returns one tidy row per model with:

* ``test_logloss`` / ``test_accuracy`` -- held-out neighborhood-label recovery
  (comparable across all models),
* ``mean_entropy`` -- average membership entropy (sharpness of the soft
  assignment),
* ``frac_border`` / ``n_border`` -- fraction/number of held-out cells that are
  multi-membership (``>= 2`` probabilities ``> threshold``): the downstream impact,
* ``train_loglik`` / ``test_loglik`` / ``n_params`` / ``bic`` -- within-family fit
  (only comparable inside one likelihood base; flagged accordingly),
* ``runtime_s``.

The k-NN windows are computed once on the full spatial data; only the emission
**parameters** are fit on the train rows (no label leakage), and the held-out rows
are scored. :func:`attach_all_model_probabilities` fits each model on all cells and
writes its posterior to a distinct ``obsm`` key for downstream border comparison.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .emission_models import (
    EMISSION_MODELS,
    _encode_components,
    _log_prior,
    build_emission_model,
    mingl_window_matrix,
    responsibilities_from_loglik,
)

__all__ = ["compare_emission_models", "attach_all_model_probabilities"]

_DEFAULT_MODELS = (
    "diagonal_gaussian",
    "full_gaussian",
    "multinomial",
    "dirichlet_multinomial",
    "logistic_normal",
)


def _stratified_split(
    comp: np.ndarray, n_components: int, test_size: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-component (stratified) train/test masks; components of size 1 stay in train."""
    test_mask = np.zeros(comp.shape[0], dtype=bool)
    for c in range(n_components):
        idx = np.where(comp == c)[0]
        if idx.shape[0] <= 1:
            continue
        rng.shuffle(idx)
        n_test = int(round(idx.shape[0] * test_size))
        n_test = min(max(n_test, 1), idx.shape[0] - 1)  # keep >=1 train and >=1 test
        test_mask[idx[:n_test]] = True
    return ~test_mask, test_mask


def _evidence(loglik: np.ndarray, log_prior: Optional[np.ndarray]) -> np.ndarray:
    logpost = loglik if log_prior is None else loglik + log_prior[None, :]
    out = np.full(logpost.shape[0], -np.inf)
    finite = np.isfinite(logpost).any(axis=1)
    if finite.any():
        out[finite] = logsumexp(logpost[finite], axis=1)
    return out


def _mean_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def compare_emission_models(
    cells: ad.AnnData,
    *,
    models: Sequence[str] = _DEFAULT_MODELS,
    cluster_col: str = "cell_type",
    neighborhood_col: str = "neighborhood",
    region_key: str = "unique_region",
    x_key: str = "x",
    y_key: str = "y",
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

    Returns a DataFrame with one row per model (see module docstring for columns),
    sorted by ``test_logloss`` (lower is better).
    """
    unknown = [m for m in models if m not in EMISSION_MODELS]
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}. Choose from {sorted(EMISSION_MODELS)}.")

    X, features = mingl_window_matrix(
        cells,
        cluster_col=cluster_col,
        region_key=region_key,
        x_key=x_key,
        y_key=y_key,
        ks=ks,
        k=k,
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
        est = build_emission_model(
            model, reg_covar=reg_covar, smoothing=smoothing, pseudocount=pseudocount
        )
        est.fit(X_tr, comp_tr, n_components)

        ll_tr = est.log_likelihood(X_tr)
        ll_te = est.log_likelihood(X_te)
        runtime = time.perf_counter() - t0

        resp_te = responsibilities_from_loglik(ll_te, log_prior)
        p_true = np.clip(resp_te[np.arange(X_te.shape[0]), comp_te], eps, 1.0)
        test_logloss = float(-np.mean(np.log(p_true)))
        test_accuracy = float(np.mean(resp_te.argmax(axis=1) == comp_te))

        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.sum(np.where(resp_te > 0, resp_te * np.log(resp_te), 0.0), axis=1)
        mean_entropy = float(np.mean(ent))

        pos_counts = (resp_te > threshold).sum(axis=1)
        n_border = int((pos_counts >= 2).sum())
        frac_border = float(np.mean(pos_counts >= 2))

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
                "test_logloss": test_logloss,
                "test_accuracy": test_accuracy,
                "mean_entropy": mean_entropy,
                "frac_border": frac_border,
                "n_border": n_border,
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
    ks: Sequence[int] = (10, 20, 100, 300),
    k: int = 10,
    prior: str = "uniform",
    reg_covar: float = 1e-6,
    smoothing: float = 0.5,
    pseudocount: float = 0.5,
    key_prefix: str = "mingl_prob_",
) -> ad.AnnData:
    """Fit each model on all cells and store its posterior in ``obsm[key_prefix+model]``.

    Also records the neighborhood name order under
    ``uns[key_prefix+model+"_neighborhoods"]``. Enables comparing downstream border
    calls across emission models on identical inputs.
    """
    X, features = mingl_window_matrix(
        cells,
        cluster_col=cluster_col,
        region_key=region_key,
        x_key=x_key,
        y_key=y_key,
        ks=ks,
        k=k,
    )
    labels = cells.obs[neighborhood_col].astype(str).to_numpy()
    comp, comp_names = _encode_components(labels)
    n_components = len(comp_names)
    log_prior = _log_prior(prior, comp, n_components)

    for model in models:
        est = build_emission_model(
            model, reg_covar=reg_covar, smoothing=smoothing, pseudocount=pseudocount
        )
        est.fit(X, comp, n_components)
        resp = responsibilities_from_loglik(est.log_likelihood(X), log_prior)
        cells.obsm[f"{key_prefix}{model}"] = resp
        cells.uns[f"{key_prefix}{model}_neighborhoods"] = list(comp_names)
    cells.uns[f"{key_prefix}features"] = list(features)
    return cells
