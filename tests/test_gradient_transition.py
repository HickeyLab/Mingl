"""Tests for the gradient/transition-clustering validation tooling (R1.5 / R1.9).

Covers the synthetic generator, the steepness metric, ground-truth recovery
(sharp > gradual), and run-to-run determinism.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import mingl.tl as tl
from mingl.tl.grad import mingl_neighborhoods_scverse


UNIT1 = "Inner Follicle"
UNIT2 = "Outer Follicle"

pytestmark = pytest.mark.filterwarnings("ignore")


def _run_pipeline(adata, seed=0):
    a = adata.copy()
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        mingl_neighborhoods_scverse(
            a,
            tu1=UNIT1,
            tu2=UNIT2,
            ks=(20,),
            k=20,
            distance_max="none",
            target_neighborhoods=(UNIT1, UNIT2),
            n_bins=5,
            n_clusters=5,
            random_state=seed,
        )
    return a


def test_simulate_schema_and_ground_truth():
    adata = tl.simulate_transition_tissue(n_cells=1200, sharpness="sharp", seed=0)
    for col in ("x", "y", "unique_region", "Neighborhood", "Community", "Tissue Unit", "Cell Type", UNIT1, UNIT2):
        assert col in adata.obs.columns
    # probabilities are complementary and finite
    p1 = adata.obs[UNIT1].to_numpy(dtype=float)
    p2 = adata.obs[UNIT2].to_numpy(dtype=float)
    assert np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))
    assert np.allclose(p1 + p2, 1.0, atol=1e-6)
    # both units are present
    assert set(adata.obs["Neighborhood"].unique()) == {UNIT1, UNIT2}
    # sharper transition => larger ground-truth sharpness
    sim = adata.uns["sim_params"]
    assert sim["ground_truth_sharpness"] == pytest.approx(1.0 / sim["transition_width_frac"])


def test_sharpness_preset_ordering():
    sharp = tl.simulate_transition_tissue(sharpness="sharp").uns["sim_params"]["ground_truth_sharpness"]
    medium = tl.simulate_transition_tissue(sharpness="medium").uns["sim_params"]["ground_truth_sharpness"]
    gradual = tl.simulate_transition_tissue(sharpness="gradual").uns["sim_params"]["ground_truth_sharpness"]
    assert sharp > medium > gradual


def test_steepness_score_keys_and_ordering():
    adata = tl.simulate_transition_tissue(n_cells=1500, sharpness="sharp", seed=0)
    out = _run_pipeline(adata)
    res = tl.steepness_score(out)
    for key in ("steepness", "slope", "ordered_clusters", "ordered_means", "second_derivative", "n_clusters"):
        assert key in res
    # ordering returns a permutation of the observed cluster labels
    observed = set(out.obs["Probability_Bin_Cluster"].dropna().astype(str))
    assert set(res["ordered_clusters"]) == observed
    assert res["n_clusters"] == len(observed)
    assert np.isfinite(res["steepness"])


def test_pipeline_is_deterministic():
    adata = tl.simulate_transition_tissue(n_cells=1500, sharpness="medium", seed=3)
    a1 = _run_pipeline(adata, seed=0)
    a2 = _run_pipeline(adata, seed=0)
    assert np.allclose(a1.obs["Score"].to_numpy(float), a2.obs["Score"].to_numpy(float))
    assert (
        a1.obs["Probability_Bin_Cluster"].astype(str).to_numpy()
        == a2.obs["Probability_Bin_Cluster"].astype(str).to_numpy()
    ).all()
    assert tl.steepness_score(a1)["steepness"] == tl.steepness_score(a2)["steepness"]


def test_ground_truth_recovery_sharp_gt_gradual():
    df = tl.validate_ground_truth_recovery(
        sharpness_levels=("sharp", "gradual"),
        n_cells=2500,
        n_repeats=3,
        seed=0,
    )
    assert df["error"].isna().all()
    means = df.groupby("sharpness_label")["steepness"].mean()
    assert means["sharp"] > means["gradual"]


def test_sensitivity_runs_without_error():
    adata = tl.simulate_transition_tissue(n_cells=1500, sharpness="sharp", seed=0)
    df = tl.gradient_sensitivity_analysis(
        adata,
        param_grid={"k": [10, 20], "n_clusters": [4, 5]},
        seed=0,
    )
    assert not df.empty
    assert "steepness" in df.columns
    assert df["error"].isna().all()
    assert (df["param"] == "baseline").sum() == 1
