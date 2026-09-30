"""Tests for the border threshold-sensitivity tooling (Task 3)."""

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mingl.tl as tl
from mingl.tl.threshold_sensitivity import (
    border_mask,
    border_metrics_at_threshold,
    spatial_border_clustering_null,
    threshold_sensitivity_analysis,
)

pytestmark = pytest.mark.filterwarnings("ignore")


def _adata_with_probs(seed: int = 0, n: int = 600):
    """AnnData carrying a hand-made posterior with a controllable border set."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 100, n)
    y = rng.uniform(0, 100, n)
    ct = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    obs = pd.DataFrame({"x": x, "y": y, "unique_region": "R1", "Cell Type": ct})
    obs.index = obs.index.astype(str)
    adata = ad.AnnData(X=np.zeros((n, 0), dtype=np.float32), obs=obs)

    # Three-neighborhood posterior; some cells are genuinely multi-membership.
    logits = rng.normal(0, 1.0, size=(n, 3))
    logits[: n // 5] += np.array([2.0, 2.0, -2.0])  # a block of two-way border cells
    P = np.exp(logits - logits.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    adata.obsm["neighborhood_probabilities"] = P
    adata.uns["neighborhood_probability_neighborhoods"] = ["N0", "N1", "N2"]
    return adata


def test_border_mask_definition():
    P = np.array([[0.6, 0.3, 0.1], [0.4, 0.4, 0.2], [0.9, 0.05, 0.05]])
    m = border_mask(P, threshold=0.25)
    # row 0: 0.6 and 0.3 both > 0.25 -> border; row 1: 0.4, 0.4 -> border; row 2: only 0.9 -> not
    assert list(m) == [True, True, False]


def test_border_count_monotone_non_increasing():
    adata = _adata_with_probs(seed=1)
    res = threshold_sensitivity_analysis(
        adata, thresholds=(0.01, 0.1, 0.25, 0.4, 0.49), cell_type_col="Cell Type"
    )
    nb = res["summary"]["n_border"].to_numpy()
    assert np.all(np.diff(nb) <= 0)
    assert set(res["summary"]["threshold"]) == {0.01, 0.1, 0.25, 0.4, 0.49}


def test_composition_and_enrichment_finite():
    adata = _adata_with_probs(seed=2)
    m = border_metrics_at_threshold(adata, threshold=0.25, cell_type_col="Cell Type")
    comp = m["composition"]
    # border counts sum to the number of border cells
    assert int(comp["n_border"].sum()) == m["n_border"]
    # composition shares sum to 1 when there are border cells
    if m["n_border"] > 0:
        assert comp["prop_of_border"].sum() == pytest.approx(1.0, abs=1e-9)
    assert np.isfinite(comp["log2_enrichment"]).all()


def test_stability_columns_present():
    adata = _adata_with_probs(seed=3)
    res = threshold_sensitivity_analysis(adata, cell_type_col="Cell Type")
    stab = res["stability"]
    assert {"threshold_low", "threshold_high", "jaccard_border_cells"}.issubset(stab.columns)
    assert (stab["jaccard_border_cells"].dropna().between(0, 1)).all()


def test_spatial_null_detects_clustering():
    # Place border cells in a tight spatial cluster; the null should flag it.
    n = 500
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 100, n)
    y = rng.uniform(0, 100, n)
    x[:50] = rng.uniform(0, 5, 50)  # border block in a corner
    y[:50] = rng.uniform(0, 5, 50)
    obs = pd.DataFrame({"x": x, "y": y, "unique_region": "R1"})
    obs.index = obs.index.astype(str)
    adata = ad.AnnData(X=np.zeros((n, 0), dtype=np.float32), obs=obs)
    P = np.tile([0.9, 0.05, 0.05], (n, 1))  # background cells: single membership (not border)
    P[:50] = [0.45, 0.45, 0.10]  # first 50 are the (clustered) two-way border cells
    adata.obsm["neighborhood_probabilities"] = P
    adata.uns["neighborhood_probability_neighborhoods"] = ["N0", "N1", "N2"]

    out = spatial_border_clustering_null(adata, threshold=0.25, n_permutations=100, seed=0)
    assert 0.0 <= out["p_value_clustered"] <= 1.0
    assert out["observed_mean_nn_distance"] < out["null_mean"]  # clustered => closer than chance
    assert out["p_value_clustered"] < 0.05
