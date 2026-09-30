"""Row-order reproducibility of the neighborhood-count tool.

MiniBatchKMeans is sensitive to the order of its input rows, so the selected
neighborhood count used to depend on how ``adata.obs`` happened to be ordered.
``run_mingl_over_n_clusters`` now fits in a canonical original-cell order by
default, making the result invariant to caller row order.
"""
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mingl.tl import run_mingl_over_n_clusters


def _make_adata(seed=0, n=1500, n_feat=6):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 5, size=(3, n_feat))
    comp = rng.integers(0, 3, size=n)
    X = np.clip(centers[comp] + rng.normal(0, 1, size=(n, n_feat)), 0, None)
    cols = [f"ct{i}" for i in range(n_feat)]
    obs = pd.DataFrame(X.astype("float32"), columns=cols)
    obs["x"] = rng.uniform(0, 100, n)
    obs["y"] = rng.uniform(0, 100, n)
    obs["unique_region"] = "r"
    a = ad.AnnData(X=np.zeros((n, 1), dtype="float32"), obs=obs)
    a.obs_names = [str(i) for i in range(n)]
    return a, cols


def _run(a, cols, **kw):
    return run_mingl_over_n_clusters(
        a, cols, n_range=range(1, 11),
        return_per_cell=False, plot_summary=False, show=False, **kw,
    )


def _lexicographic_shuffle(a):
    """Physically reorder rows into lexicographic obs-name order ("0","1","10"...),
    the classic silent reshuffle from a DataFrame->AnnData str-index conversion."""
    perm = np.argsort([str(i) for i in range(a.n_obs)], kind="stable")
    return a[perm].copy()


def test_original_order_is_row_order_invariant():
    a, cols = _make_adata()
    base = _run(a.copy(), cols)
    shuffled = _run(_lexicographic_shuffle(a), cols)
    # canonical order recovers the same partition from obs_names -> identical stats
    np.testing.assert_allclose(
        base["avg_log_likelihood"].values, shuffled["avg_log_likelihood"].values,
        rtol=1e-5, atol=1e-5,
    )
    np.testing.assert_allclose(
        base["avg_assigned_probability"].values, shuffled["avg_assigned_probability"].values,
        rtol=1e-5, atol=1e-5,
    )


def test_order_key_pins_the_order():
    a, cols = _make_adata()
    a.obs["orig_pos"] = np.arange(a.n_obs)
    base = _run(a.copy(), cols, order_key="orig_pos")
    shuffled = _run(_lexicographic_shuffle(a), cols, order_key="orig_pos")
    np.testing.assert_allclose(
        base["avg_assigned_probability"].values, shuffled["avg_assigned_probability"].values,
        rtol=1e-5, atol=1e-5,
    )


def test_non_numeric_obs_names_warn_and_still_run():
    a, cols = _make_adata(n=600)
    a.obs_names = [f"cell_{i}" for i in range(a.n_obs)]
    with pytest.warns(UserWarning, match="canonical original cell order"):
        out = _run(a, cols)
    assert list(out["n_clusters"]) == list(range(1, 11))


def test_as_given_matches_original_when_already_in_order():
    # with rows already in original order, the new default must not change results
    a, cols = _make_adata()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no spurious warnings on the happy path
        default = _run(a.copy(), cols)
    as_given = _run(a.copy(), cols, cluster_row_order="as_given")
    np.testing.assert_allclose(
        default["avg_assigned_probability"].values, as_given["avg_assigned_probability"].values,
        rtol=1e-6, atol=1e-6,
    )
