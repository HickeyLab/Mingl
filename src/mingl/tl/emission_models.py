"""Alternative emission models for MINGL neighborhood-membership scoring.

Addresses the reviewer comment that MINGL's per-cell features -- nearest-neighbor
cell-type counts from :func:`mingl.tl.knn2.KNN2` -- are *compositional count* data
(non-negative integers that sum to a fixed window size ``k``, sparse, and
negatively correlated), so scoring them with a product of independent univariate
Gaussians (diagonal covariance) may be a poor fit.

The shipped path (:func:`mingl.tl.gmm.cpu_gmm_probability`) is left untouched. This
module is *additive*: it keeps the neighborhood definition fixed (the discrete
assignment already in ``obs[neighborhood_col]`` is the set of mixture components)
and only swaps the **emission model** used to turn a cell's window into a posterior
over neighborhoods. Every model writes the exact same objects the rest of the
pipeline consumes (``obsm[prob_key]`` and ``uns[prob_variable_key]``), so it is a
drop-in for the border / gradient / network tools.

Models
------
``diagonal_gaussian``
    Product of univariate Gaussians -- reproduces the current MINGL scoring
    (baseline / equivalence check).
``full_gaussian``
    Multivariate Gaussian with a full covariance matrix. Most similar to the
    current model but captures the co-variance of features (Task 1).
``multinomial``
    ``Multinomial(n=k, p_c)`` per neighborhood: treats the window as counts with a
    fixed size and a single composition per neighborhood.
``dirichlet_multinomial``
    ``Dirichlet-Multinomial(alpha_c)`` per neighborhood: a multinomial that allows
    over-dispersion, i.e. biological variability between instances of a
    neighborhood.
``logistic_normal``
    Additive log-ratio (ALR) transform of the composition followed by a full-cov
    Gaussian: models the non-independence induced by the fixed window size in an
    unconstrained space.

The primary entry point is :func:`mingl_membership_probabilities`. Lower-level
helpers (:func:`mingl_window_matrix`, :func:`build_emission_model`,
:func:`responsibilities_from_loglik`) are exposed for reuse by
:mod:`mingl.tl.model_comparison`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.special import digamma, gammaln, logsumexp

__all__ = [
    "EMISSION_MODELS",
    "mingl_window_matrix",
    "build_emission_model",
    "responsibilities_from_loglik",
    "mingl_membership_probabilities",
    "DiagonalGaussian",
    "FullGaussian",
    "MultinomialEmission",
    "DirichletMultinomialEmission",
    "LogisticNormalEmission",
]

_LOG_2PI = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def mingl_window_matrix(
    cells: ad.AnnData,
    *,
    cluster_col: str = "cell_type",
    region_key: str = "unique_region",
    x_key: str = "x",
    y_key: str = "y",
    ks: Sequence[int] = (10, 20, 100, 300),
    k: int = 10,
) -> Tuple[np.ndarray, List[str]]:
    """Return the k-NN cell-type count matrix (n_cells x n_cell_types), obs-ordered.

    Thin wrapper over :func:`mingl.tl.knn2.KNN2` that selects the requested ``k``
    and returns a dense float64 array plus the ordered feature (cell-type) names.
    Feature order is the sorted unique cell types for determinism; the resulting
    posterior is invariant to feature order, so this does not change any result.

    Only the ``k``-nearest window is needed, so ``KNN2`` is called with ``ks=(k,)``
    -- the ``k``-nearest set is identical regardless of how many neighbors are
    requested, and this keeps the per-region size requirement at ``k`` rather than
    ``max(ks)``. ``ks`` is accepted for signature compatibility with
    :func:`mingl.tl.gmm.cpu_gmm_probability` but does not otherwise affect the result.
    """
    from .knn2 import KNN2

    del ks  # only the k-nearest window is used; see note above

    for required in (x_key, y_key, region_key, cluster_col):
        if required not in cells.obs:
            raise KeyError(f"Required column {required!r} missing from obs.")

    # KNN2 maps neighbors by obs-index *label* into a positional array, so it
    # requires a contiguous 0..n-1 integer index. Run it on a minimal view with a
    # reset positional index (row order preserved) so this works for any obs_names
    # and does not copy X / layers.
    obs_min = cells.obs.loc[:, [x_key, y_key, region_key, cluster_col]].copy()
    obs_min.index = np.arange(obs_min.shape[0]).astype(str)
    mini = ad.AnnData(X=np.zeros((obs_min.shape[0], 0), dtype=np.float32), obs=obs_min)

    windows = KNN2(
        mini,
        x_key=x_key,
        y_key=y_key,
        region_key=region_key,
        cluster_col=cluster_col,
        ks=(int(k),),
    )
    if k not in windows:
        raise ValueError(f"k={k} not in available ks from KNN2: {list(windows.keys())}")

    windows_k = windows[k]
    features = sorted(pd.Index(cells.obs[cluster_col].astype(str)).unique().tolist())

    missing = [f for f in features if f not in windows_k.columns]
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        raise ValueError(f"Missing KNN window columns for cell types: {preview}")

    # KNN2 already returns rows in adata.obs order; select feature columns positionally.
    X = np.ascontiguousarray(windows_k.loc[:, features].to_numpy(dtype=np.float64))
    return X, features


def _encode_components(labels: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Map string labels to 0..C-1 integer component ids with sorted, stable names."""
    names = sorted(pd.Index(labels).unique().tolist())
    index = {name: i for i, name in enumerate(names)}
    comp = np.fromiter((index[str(v)] for v in labels), dtype=np.int64, count=len(labels))
    return comp, names


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------
def _chol_and_logdet(cov: np.ndarray, reg: float, d: int) -> Tuple[tuple, float]:
    """Cholesky factor and log-determinant of ``cov + jitter*I`` with jitter escalation."""
    jitter = max(reg, 0.0)
    if jitter == 0.0:
        jitter = 1e-10
    eye = np.eye(d)
    for _ in range(10):
        try:
            cfac = cho_factor(cov + jitter * eye, lower=True, check_finite=False)
            logdet = 2.0 * float(np.sum(np.log(np.diag(cfac[0]))))
            return cfac, logdet
        except LinAlgError:
            jitter *= 10.0
    # Last resort: diagonal approximation (always PD).
    diagv = np.clip(np.diag(cov), 0.0, None) + jitter
    cfac = cho_factor(np.diag(diagv), lower=True, check_finite=False)
    return cfac, float(np.sum(np.log(diagv)))


def responsibilities_from_loglik(
    loglik: np.ndarray,
    log_prior: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Softmax over ``loglik + log_prior`` per row (log-space, underflow-safe).

    Rows whose every component log-posterior is ``-inf`` (no component can explain
    the window) are returned as all-zeros, matching the existing MINGL convention
    in :func:`mingl.tl.gmm.cpu_gmm_probability`.
    """
    logpost = np.asarray(loglik, dtype=np.float64)
    if log_prior is not None:
        logpost = logpost + np.asarray(log_prior, dtype=np.float64)[None, :]

    resp = np.zeros_like(logpost)
    finite_rows = np.isfinite(logpost).any(axis=1)
    if finite_rows.any():
        block = logpost[finite_rows]
        z = logsumexp(block, axis=1, keepdims=True)
        resp[finite_rows] = np.exp(block - z)
    return resp


def _log_prior(prior: str, comp: np.ndarray, n_components: int) -> Optional[np.ndarray]:
    if prior == "uniform":
        return None
    if prior == "empirical":
        counts = np.bincount(comp, minlength=n_components).astype(np.float64)
        counts = np.clip(counts, 1e-12, None)
        return np.log(counts / counts.sum())
    raise ValueError(f"Unknown prior {prior!r}; expected 'uniform' or 'empirical'.")


# ---------------------------------------------------------------------------
# Emission models
# ---------------------------------------------------------------------------
class _EmissionModel:
    """Interface: ``fit(X, comp, C)`` then ``log_likelihood(X) -> (n, C)``."""

    name = "base"

    def fit(self, X: np.ndarray, comp: np.ndarray, n_components: int) -> "_EmissionModel":
        raise NotImplementedError

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def n_parameters(self) -> int:
        raise NotImplementedError


class DiagonalGaussian(_EmissionModel):
    """Product of univariate Gaussians (diagonal covariance) -- the current MINGL model.

    Uses the sample standard deviation (``ddof=1``) to match
    :func:`mingl.tl.centroids.centroid_Calculation`. Zero-variance features
    contribute a degenerate density: log-density 0 where the value equals the mean
    and ``-inf`` otherwise, exactly as the shipped scorer treats ``std == 0``.
    """

    name = "diagonal_gaussian"

    def __init__(self, **_ignored):
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None

    def fit(self, X, comp, n_components):
        d = X.shape[1]
        self.means = np.zeros((n_components, d))
        self.stds = np.zeros((n_components, d))
        for c in range(n_components):
            Xc = X[comp == c]
            if Xc.shape[0] == 0:
                self.stds[c] = 1.0  # harmless placeholder for an empty component
                continue
            self.means[c] = Xc.mean(axis=0)
            if Xc.shape[0] > 1:
                self.stds[c] = Xc.std(axis=0, ddof=1)
        self._C = n_components
        self._d = d
        return self

    def log_likelihood(self, X):
        n = X.shape[0]
        ll = np.empty((n, self._C))
        for c in range(self._C):
            mu = self.means[c]
            sd = self.stds[c]
            zero = sd == 0.0
            safe = np.where(zero, 1.0, sd)
            z = (X - mu) / safe
            comp_ll = -0.5 * z * z - np.log(safe) - 0.5 * _LOG_2PI
            eq = X == mu
            comp_ll = np.where(zero & eq, 0.0, comp_ll)
            comp_ll = np.where(zero & ~eq, -np.inf, comp_ll)
            ll[:, c] = comp_ll.sum(axis=1)
        return ll

    def n_parameters(self):
        return int(self._C * 2 * self._d)  # mean + std per feature per component


class FullGaussian(_EmissionModel):
    """Multivariate Gaussian with full covariance (Task 1).

    ``reg_covar`` is scaled by the mean feature variance and added to each
    component covariance. Because windows sum to ``k``, every component
    covariance is singular along the all-ones direction; that direction is
    identical across components and orthogonal to every centered window, so it
    cancels in the posterior -- the regularizer only conditions the remaining
    directions.
    """

    name = "full_gaussian"

    def __init__(self, reg_covar: float = 1e-6, **_ignored):
        self.reg_covar = float(reg_covar)
        self.means: Optional[np.ndarray] = None
        self._chol: List[tuple] = []
        self._logdet: Optional[np.ndarray] = None

    def fit(self, X, comp, n_components):
        d = X.shape[1]
        self.means = np.zeros((n_components, d))
        self._chol = []
        self._logdet = np.zeros(n_components)
        global_var = float(X.var(axis=0).mean()) if X.shape[0] > 1 else 1.0
        reg = self.reg_covar * max(global_var, 1e-12)
        self._reg = reg
        for c in range(n_components):
            Xc = X[comp == c]
            if Xc.shape[0] >= 2:
                mu = Xc.mean(axis=0)
                cov = np.atleast_2d(np.cov(Xc, rowvar=False))
            elif Xc.shape[0] == 1:
                mu = Xc[0].astype(np.float64)
                cov = np.zeros((d, d))
            else:
                mu = np.zeros(d)
                cov = np.zeros((d, d))
            self.means[c] = mu
            cfac, logdet = _chol_and_logdet(cov, reg, d)
            self._chol.append(cfac)
            self._logdet[c] = logdet
        self._C = n_components
        self._d = d
        return self

    def log_likelihood(self, X):
        n = X.shape[0]
        ll = np.empty((n, self._C))
        const = -0.5 * self._d * _LOG_2PI
        for c in range(self._C):
            diff = X - self.means[c]
            sol = cho_solve(self._chol[c], diff.T, check_finite=False)  # (d, n)
            maha = np.einsum("ij,ji->i", diff, sol)
            ll[:, c] = const - 0.5 * self._logdet[c] - 0.5 * maha
        return ll

    def n_parameters(self):
        return int(self._C * (self._d + self._d * (self._d + 1) // 2))


class MultinomialEmission(_EmissionModel):
    """Multinomial(n=window size, p_c) with Laplace-smoothed pooled MLE for p_c."""

    name = "multinomial"

    def __init__(self, smoothing: float = 0.5, **_ignored):
        self.smoothing = float(smoothing)
        self.logp: Optional[np.ndarray] = None

    def fit(self, X, comp, n_components):
        d = X.shape[1]
        self.logp = np.zeros((n_components, d))
        for c in range(n_components):
            Xc = X[comp == c]
            counts = Xc.sum(axis=0) if Xc.shape[0] > 0 else np.zeros(d)
            p = (counts + self.smoothing) / (counts.sum() + self.smoothing * d)
            self.logp[c] = np.log(p)
        self._C = n_components
        self._d = d
        return self

    def log_likelihood(self, X):
        n_i = X.sum(axis=1)
        coeff = gammaln(n_i + 1.0) - gammaln(X + 1.0).sum(axis=1)
        ll = X @ self.logp.T
        return ll + coeff[:, None]

    def n_parameters(self):
        return int(self._C * (self._d - 1))  # each p_c lives on the simplex


class DirichletMultinomialEmission(_EmissionModel):
    """Dirichlet-Multinomial(alpha_c); alpha_c fit by Minka's fixed-point MLE.

    Over-dispersion relative to the multinomial captures biological variability
    between instances of the same neighborhood.
    """

    name = "dirichlet_multinomial"

    def __init__(self, max_iter: int = 200, tol: float = 1e-6, floor: float = 1e-6, **_ignored):
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.floor = float(floor)
        self.alpha: Optional[np.ndarray] = None

    def _fit_alpha(self, Xc: np.ndarray, d: int) -> np.ndarray:
        if Xc.shape[0] == 0:
            return np.ones(d)
        n_i = Xc.sum(axis=1)
        pooled = Xc.sum(axis=0) + self.floor
        p = pooled / pooled.sum()
        alpha = np.maximum(p * float(d), self.floor)  # moment-style init, precision ~ d
        for _ in range(self.max_iter):
            A = alpha.sum()
            num = (digamma(Xc + alpha) - digamma(alpha)).sum(axis=0)
            den = float((digamma(n_i + A) - digamma(A)).sum())
            if den <= 0.0 or not np.isfinite(den):
                break
            new = alpha * num / den
            new = np.maximum(new, self.floor)
            if np.max(np.abs(new - alpha)) < self.tol:
                alpha = new
                break
            alpha = new
        return alpha

    def fit(self, X, comp, n_components):
        d = X.shape[1]
        self.alpha = np.zeros((n_components, d))
        for c in range(n_components):
            self.alpha[c] = self._fit_alpha(X[comp == c], d)
        self._C = n_components
        self._d = d
        return self

    def log_likelihood(self, X):
        n = X.shape[0]
        n_i = X.sum(axis=1)
        coeff = gammaln(n_i + 1.0) - gammaln(X + 1.0).sum(axis=1)
        ll = np.empty((n, self._C))
        for c in range(self._C):
            a = self.alpha[c]
            A = float(a.sum())
            term = gammaln(A) - gammaln(n_i + A) + (gammaln(X + a) - gammaln(a)).sum(axis=1)
            ll[:, c] = coeff + term
        return ll

    def n_parameters(self):
        return int(self._C * self._d)


class LogisticNormalEmission(_EmissionModel):
    """Additive-log-ratio transform of the composition, then a full-cov Gaussian.

    The reference component (denominator of the log-ratio) is the globally most
    abundant cell type, which keeps the transform well conditioned. The transform
    Jacobian is identical across components, so it cancels in the posterior; the
    reported log-likelihood is therefore defined up to a per-cell additive
    constant (irrelevant to membership probabilities and to within-family BIC).
    """

    name = "logistic_normal"

    def __init__(self, reg_covar: float = 1e-6, pseudocount: float = 0.5, **_ignored):
        self.reg_covar = float(reg_covar)
        self.pseudocount = float(pseudocount)
        self.ref: Optional[int] = None
        self._keep: Optional[List[int]] = None
        self._gauss: Optional[FullGaussian] = None

    def _alr(self, X: np.ndarray) -> np.ndarray:
        pc = self.pseudocount
        n_i = X.sum(axis=1, keepdims=True)
        pi = (X + pc) / (n_i + pc * self._d)
        ref = pi[:, [self.ref]]
        return np.log(pi[:, self._keep] / ref)

    def fit(self, X, comp, n_components):
        self._d = X.shape[1]
        self.ref = int(np.argmax(X.sum(axis=0)))
        self._keep = [j for j in range(self._d) if j != self.ref]
        Y = self._alr(X)
        self._gauss = FullGaussian(reg_covar=self.reg_covar).fit(Y, comp, n_components)
        self._C = n_components
        return self

    def log_likelihood(self, X):
        return self._gauss.log_likelihood(self._alr(X))

    def n_parameters(self):
        return self._gauss.n_parameters()


EMISSION_MODELS: Dict[str, type] = {
    "diagonal_gaussian": DiagonalGaussian,
    "full_gaussian": FullGaussian,
    "multinomial": MultinomialEmission,
    "dirichlet_multinomial": DirichletMultinomialEmission,
    "logistic_normal": LogisticNormalEmission,
}


def build_emission_model(model: str, **kwargs) -> _EmissionModel:
    """Instantiate an emission model by key (see :data:`EMISSION_MODELS`)."""
    if model not in EMISSION_MODELS:
        raise ValueError(f"Unknown model {model!r}. Choose from {sorted(EMISSION_MODELS)}.")
    return EMISSION_MODELS[model](**kwargs)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def mingl_membership_probabilities(
    cells: ad.AnnData,
    *,
    model: str = "full_gaussian",
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
    prob_key: str = "neighborhood_probabilities",
    prob_variable_key: str = "neighborhood_probability_neighborhoods",
    random_state: int = 0,
) -> ad.AnnData:
    """Score every cell's neighborhood membership under a chosen emission model.

    Keeps the neighborhood definition fixed (the discrete labels in
    ``obs[neighborhood_col]`` are the mixture components) and swaps only the
    per-window emission model. Mutates ``cells`` in place and returns it, writing
    the same objects as :func:`mingl.tl.gmm.cpu_gmm_probability`:

    * ``cells.obsm[prob_key]`` -- (n_cells x n_neighborhoods) posterior matrix,
    * ``cells.uns[prob_variable_key]`` -- ordered neighborhood names,
    * ``cells.uns[f"mingl_emission_{model}"]`` -- diagnostics (parameter count,
      mean per-cell log evidence, BIC, feature/component names).

    Parameters
    ----------
    model
        One of ``diagonal_gaussian`` (reproduces current MINGL), ``full_gaussian``
        (Task 1), ``multinomial``, ``dirichlet_multinomial``, ``logistic_normal``.
    prior
        ``"uniform"`` (default, matches current MINGL) or ``"empirical"`` (mixing
        weights proportional to neighborhood size).
    reg_covar, smoothing, pseudocount
        Regularizers for the Gaussian covariance, multinomial Laplace smoothing,
        and logistic-normal pseudocount respectively.
    """
    del random_state  # models here are deterministic; kept for API symmetry

    if neighborhood_col not in cells.obs or cluster_col not in cells.obs:
        raise KeyError(
            f"One or more required columns ({neighborhood_col}, {cluster_col}) are missing in obs."
        )

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

    est = build_emission_model(
        model, reg_covar=reg_covar, smoothing=smoothing, pseudocount=pseudocount
    )
    est.fit(X, comp, n_components)
    loglik = est.log_likelihood(X)
    log_prior = _log_prior(prior, comp, n_components)
    resp = responsibilities_from_loglik(loglik, log_prior)

    # Per-cell log evidence (log marginal likelihood) for diagnostics / BIC.
    logpost = loglik if log_prior is None else loglik + log_prior[None, :]
    finite = np.isfinite(logpost).any(axis=1)
    log_evidence = np.full(logpost.shape[0], -np.inf)
    if finite.any():
        log_evidence[finite] = logsumexp(logpost[finite], axis=1)
    mean_log_evidence = float(np.mean(log_evidence[np.isfinite(log_evidence)])) if finite.any() else float("nan")
    n_eval = int(np.isfinite(log_evidence).sum())
    total_log_evidence = float(np.sum(log_evidence[np.isfinite(log_evidence)])) if n_eval else float("nan")
    n_params = est.n_parameters()
    bic = (
        float(-2.0 * total_log_evidence + n_params * np.log(max(n_eval, 1)))
        if n_eval
        else float("nan")
    )

    cells.obsm[prob_key] = resp
    cells.uns[prob_variable_key] = comp_names
    cells.uns[f"mingl_emission_{model}"] = {
        "model": model,
        "prior": prior,
        "k": int(k),
        "n_parameters": int(n_params),
        "mean_log_evidence": mean_log_evidence,
        "total_log_evidence": total_log_evidence,
        "bic": bic,
        "features": list(features),
        "components": list(comp_names),
    }
    return cells
