"""Tests for the alternative emission models (Tasks 1 & 2).

Covers hand-computed likelihoods, valid-probability guarantees, equivalence of the
diagonal model to the shipped scorer, the full-covariance model's ability to use
feature correlation, determinism, and row-order invariance.
"""

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mingl.tl as tl
from mingl.tl.emission_models import (
    DiagonalGaussian,
    FullGaussian,
    MultinomialEmission,
    build_emission_model,
    mingl_membership_probabilities,
    responsibilities_from_loglik,
)

pytestmark = pytest.mark.filterwarnings("ignore")

CT, NB, REG = "cell_type", "neighborhood", "unique_region"


def _make_adata(seed: int = 0, n_per: int = 400) -> ad.AnnData:
    """Three spatial bands, each a neighborhood with a distinct cell-type mix."""
    rng = np.random.default_rng(seed)
    types = ["A", "B", "C", "D"]
    comps = {
        "N0": [0.70, 0.10, 0.10, 0.10],
        "N1": [0.10, 0.70, 0.10, 0.10],
        "N2": [0.10, 0.10, 0.40, 0.40],
    }
    frames = []
    for i, (nb, p) in enumerate(comps.items()):
        x = rng.uniform(i * 100.0, i * 100.0 + 100.0, n_per)
        y = rng.uniform(0.0, 100.0, n_per)
        ct = rng.choice(types, size=n_per, p=p)
        frames.append(
            pd.DataFrame({"x": x, "y": y, REG: "R1", NB: nb, CT: ct})
        )
    obs = pd.concat(frames, ignore_index=True)
    obs.index = obs.index.astype(str)
    return ad.AnnData(X=np.zeros((len(obs), 0), dtype=np.float32), obs=obs)


# ---------------------------------------------------------------------------
# Hand-computed likelihood
# ---------------------------------------------------------------------------
def test_multinomial_loglik_matches_closed_form():
    X = np.array([[3.0, 1.0], [0.0, 4.0], [2.0, 2.0]])
    comp = np.array([0, 0, 1])
    m = MultinomialEmission(smoothing=0.5).fit(X, comp, 2)

    # Component 0 pooled counts = [3, 5] -> p = (3.5, 5.5)/9
    p0 = np.array([3.5, 5.5]) / 9.0
    # Component 1 pooled counts = [2, 2] -> p = (2.5, 2.5)/5 = 0.5, 0.5
    p1 = np.array([2.5, 2.5]) / 5.0
    from scipy.special import gammaln

    x = X[0]
    n = x.sum()
    coeff = gammaln(n + 1) - gammaln(x + 1).sum()
    expected0 = coeff + (x * np.log(p0)).sum()
    expected1 = coeff + (x * np.log(p1)).sum()

    ll = m.log_likelihood(X)
    assert ll[0, 0] == pytest.approx(expected0, rel=1e-10)
    assert ll[0, 1] == pytest.approx(expected1, rel=1e-10)


def test_responsibilities_all_neg_inf_row_is_zero():
    ll = np.array([[-np.inf, -np.inf], [0.0, np.log(3.0)]])
    resp = responsibilities_from_loglik(ll)
    assert np.allclose(resp[0], [0.0, 0.0])
    assert resp[1].sum() == pytest.approx(1.0)
    assert resp[1, 1] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Valid probabilities for every model
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model", sorted(tl.EMISSION_MODELS))
def test_model_produces_valid_probabilities(model):
    adata = _make_adata(seed=1)
    out = mingl_membership_probabilities(
        adata, model=model, cluster_col=CT, neighborhood_col=NB, region_key=REG, ks=(10, 20), k=10
    )
    P = out.obsm["neighborhood_probabilities"]
    assert P.shape == (adata.n_obs, 3)
    assert np.isfinite(P).all()
    assert (P >= -1e-12).all()
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-9)
    info = out.uns[f"mingl_emission_{model}"]
    assert info["n_parameters"] > 0
    assert len(info["components"]) == 3


# ---------------------------------------------------------------------------
# Diagonal model reproduces the shipped scorer
# ---------------------------------------------------------------------------
def test_diagonal_matches_shipped_cpu_gmm():
    import mingl as mg

    adata = _make_adata(seed=2)

    a = adata.copy()
    cents = mg.tl.centroid_Calculation(
        a, k=10, cluster_col=CT, neighborhood_col=NB, region_col=REG
    )
    mg.tl.cpu_gmm_probability(
        CELLS_ADATA=a, CENTROIDS_ADATA=cents, cluster_col=CT, neighborhood_col=NB,
        region_key=REG, ks=(10, 20, 100, 300), k=10,
    )
    ship = pd.DataFrame(
        a.obsm["neighborhood_probabilities"],
        columns=list(a.uns["neighborhood_probability_neighborhoods"]),
    ).sort_index(axis=1)

    b = adata.copy()
    mingl_membership_probabilities(
        b, model="diagonal_gaussian", cluster_col=CT, neighborhood_col=NB,
        region_key=REG, ks=(10, 20, 100, 300), k=10,
    )
    new = pd.DataFrame(
        b.obsm["neighborhood_probabilities"],
        columns=list(b.uns["neighborhood_probability_neighborhoods"]),
    ).sort_index(axis=1)

    assert (ship.values.argmax(1) == new.values.argmax(1)).all()
    assert np.allclose(ship.values, new.values, atol=1e-4)


# ---------------------------------------------------------------------------
# Full covariance uses correlation that the diagonal model cannot
# ---------------------------------------------------------------------------
def test_full_gaussian_uses_feature_correlation():
    rng = np.random.default_rng(0)
    n = 400
    # Two components with (nearly) identical marginals but opposite correlation.
    t = rng.normal(0, 1, n)
    c0 = np.column_stack([5 + t, 5 + t]) + rng.normal(0, 0.05, (n, 2))   # positively correlated
    c1 = np.column_stack([5 + t, 5 - t]) + rng.normal(0, 0.05, (n, 2))   # negatively correlated
    X = np.vstack([c0, c1])
    comp = np.array([0] * n + [1] * n)

    query = np.array([[7.0, 7.0]])  # lies on component 0's axis

    full = build_emission_model("full_gaussian").fit(X, comp, 2)
    diag = build_emission_model("diagonal_gaussian").fit(X, comp, 2)

    r_full = responsibilities_from_loglik(full.log_likelihood(query))[0]
    r_diag = responsibilities_from_loglik(diag.log_likelihood(query))[0]

    assert r_full[0] > 0.99           # full covariance is confident and correct
    assert abs(r_diag[0] - 0.5) < 0.1  # diagonal cannot tell the components apart


# ---------------------------------------------------------------------------
# Determinism and row-order invariance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model", ["full_gaussian", "multinomial", "dirichlet_multinomial", "logistic_normal"])
def test_determinism(model):
    adata = _make_adata(seed=4)
    p1 = mingl_membership_probabilities(
        adata.copy(), model=model, cluster_col=CT, neighborhood_col=NB, region_key=REG, ks=(10, 20), k=10
    ).obsm["neighborhood_probabilities"]
    p2 = mingl_membership_probabilities(
        adata.copy(), model=model, cluster_col=CT, neighborhood_col=NB, region_key=REG, ks=(10, 20), k=10
    ).obsm["neighborhood_probabilities"]
    assert np.array_equal(p1, p2)


def test_row_order_invariance_of_scoring():
    """The emission-model layer must be invariant to input row order.

    (End-to-end order invariance is limited only by KNN2's neighbor-tie behavior,
    which is upstream of these models; here we isolate the scoring layer.)
    """
    rng = np.random.default_rng(5)
    X = rng.poisson(3.0, size=(300, 5)).astype(float)
    comp = rng.integers(0, 3, size=300)

    for model in ["full_gaussian", "multinomial", "dirichlet_multinomial", "logistic_normal"]:
        est = build_emission_model(model).fit(X, comp, 3)
        base = responsibilities_from_loglik(est.log_likelihood(X))

        perm = rng.permutation(X.shape[0])
        est_p = build_emission_model(model).fit(X[perm], comp[perm], 3)
        shuffled = responsibilities_from_loglik(est_p.log_likelihood(X[perm]))

        # Undo the permutation and compare per-row.
        inv = np.empty_like(perm)
        inv[perm] = np.arange(perm.shape[0])
        assert np.allclose(shuffled[inv], base, atol=1e-9)


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
def test_compare_emission_models_shape_and_determinism():
    adata = _make_adata(seed=6)
    df1 = tl.compare_emission_models(
        adata, cluster_col=CT, neighborhood_col=NB, region_key=REG, ks=(10, 20), k=10, seed=0
    )
    assert len(df1) == len(tl.EMISSION_MODELS)
    for col in ["test_logloss", "test_accuracy", "mean_entropy", "frac_border", "n_params", "bic"]:
        assert col in df1.columns
    assert (df1["test_accuracy"].between(0, 1)).all()
    assert (df1["test_logloss"] >= 0).all()

    df2 = tl.compare_emission_models(
        adata, cluster_col=CT, neighborhood_col=NB, region_key=REG, ks=(10, 20), k=10, seed=0
    )
    # runtime_s is wall-clock and not expected to be bit-identical.
    drop = ["runtime_s"]
    pd.testing.assert_frame_equal(df1.drop(columns=drop), df2.drop(columns=drop))
