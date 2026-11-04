"""Tests for the 3D-capable tools in :mod:`mingl.tl3d`.

Three kinds of checks:

* **2D parity** -- every tl3d function with ``z_key=None`` reproduces the
  original :mod:`mingl.tl` function exactly.
* **Constant-z equivalence** -- the 3D code path on data with a flat ``z``
  (so 3D distances equal 2D distances) reproduces the 2D result.
* **True 3D behavior** -- constructions where only a 3D neighbor search gives
  the right answer (layered tissues, z-transitions, known 3D gradient fields).
"""

import contextlib
import importlib
import io
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mingl.tl as tl
from mingl import tl3d
from mingl.tl import emission_models as em2d
from mingl.tl import model_comparison as mc2d

# mingl.tl re-exports a *function* named ``gb`` that shadows the submodule.
gb2d = importlib.import_module("mingl.tl.gb")

pytestmark = pytest.mark.filterwarnings("ignore")

CT, NB, REG = "cell_type", "neighborhood", "unique_region"
UNIT1, UNIT2 = "Inner Follicle", "Outer Follicle"


def _quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _bands(seed: int = 0, n_per: int = 400, z: str | None = None) -> ad.AnnData:
    """Three x-bands (distinct cell-type mixes) split across two regions.

    ``z=None``: 2D. ``z="flat"``: a constant z column. ``z="random"``: z ~ U(0, 100).
    """
    rng = np.random.default_rng(seed)
    types = ["A", "B", "C", "D"]
    comps = {
        "N0": [0.70, 0.10, 0.10, 0.10],
        "N1": [0.10, 0.70, 0.10, 0.10],
        "N2": [0.10, 0.10, 0.40, 0.40],
    }
    frames = []
    for i, (nb, p) in enumerate(comps.items()):
        frames.append(
            pd.DataFrame(
                {
                    "x": rng.uniform(i * 100.0, i * 100.0 + 100.0, n_per),
                    "y": rng.uniform(0.0, 100.0, n_per),
                    NB: nb,
                    CT: rng.choice(types, size=n_per, p=p),
                }
            )
        )
    obs = pd.concat(frames, ignore_index=True)
    obs[REG] = np.where(obs["y"] < 50.0, "R1", "R2")
    if z == "flat":
        obs["z"] = 7.0
    elif z == "random":
        obs["z"] = rng.uniform(0.0, 100.0, len(obs))
    obs.index = obs.index.astype(str)
    return ad.AnnData(X=np.zeros((len(obs), 0), dtype=np.float32), obs=obs)


def _layers(seed: int = 0, n_per: int = 300, gap: float = 100.0) -> ad.AnnData:
    """Two z-layers with identical x/y spread; cell type is set by the layer only."""
    rng = np.random.default_rng(seed)
    frames = []
    for i, ct in enumerate(["A", "B"]):
        frames.append(
            pd.DataFrame(
                {
                    "x": rng.uniform(0.0, 100.0, n_per),
                    "y": rng.uniform(0.0, 100.0, n_per),
                    "z": np.full(n_per, i * gap),
                    CT: ct,
                    REG: "R1",
                }
            )
        )
    obs = pd.concat(frames, ignore_index=True)
    obs.index = obs.index.astype(str)
    return ad.AnnData(X=np.zeros((len(obs), 0), dtype=np.float32), obs=obs)


def _count_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c in {"A", "B", "C", "D"}]


# ---------------------------------------------------------------------------
# Coordinate helper
# ---------------------------------------------------------------------------
def test_get_coords_shapes_and_missing_z_raises():
    a = _bands(z="random")
    assert tl3d.get_coords(a.obs).shape == (a.n_obs, 2)
    assert tl3d.get_coords(a.obs, z_key="z").shape == (a.n_obs, 3)
    with pytest.raises(KeyError, match="z"):
        tl3d.get_coords(_bands().obs, z_key="z")


# ---------------------------------------------------------------------------
# KNN / KNN2
# ---------------------------------------------------------------------------
def test_knn2_2d_parity():
    a = _bands()
    w2 = _quiet(tl.KNN2, a, cluster_col=CT)
    w3 = tl3d.KNN2(a, cluster_col=CT)
    assert w2.keys() == w3.keys()
    for k in w2:
        pd.testing.assert_frame_equal(w2[k], w3[k])


def test_knn_legacy_2d_parity():
    a = _bands()
    w2 = tl.KNN(a, cluster_col=CT)
    w3 = tl3d.KNN(a, cluster_col=CT)
    for k in w2:
        pd.testing.assert_frame_equal(w2[k], w3[k])


def test_knn2_flat_z_matches_2d():
    a = _bands(z="flat")
    w2 = _quiet(tl.KNN2, a, cluster_col=CT)
    w3 = tl3d.KNN2(a, cluster_col=CT, z_key="z")
    for k in w2:
        cols = _count_cols(w2[k])
        pd.testing.assert_frame_equal(w2[k][cols], w3[k][cols])
        assert "z" in w3[k].columns


def test_knn2_3d_separates_z_layers_2d_does_not():
    a = _layers()
    w3 = tl3d.KNN2(a, cluster_col=CT, z_key="z", ks=(10,))[10]
    w2 = tl3d.KNN2(a, cluster_col=CT, ks=(10,))[10]
    own_3d = np.where(a.obs[CT] == "A", w3["A"], w3["B"])
    own_2d = np.where(a.obs[CT] == "A", w2["A"], w2["B"])
    assert np.all(own_3d == 10)  # 3D windows are layer-pure
    assert own_2d.mean() < 7  # 2D windows mix the layers (~5 expected)


def test_knn2_missing_z_raises():
    with pytest.raises(KeyError, match="z"):
        tl3d.KNN2(_bands(), cluster_col=CT, z_key="z")


# ---------------------------------------------------------------------------
# Centroids + GMM
# ---------------------------------------------------------------------------
def test_centroids_and_cpu_gmm_2d_parity():
    a2, a3 = _bands(), _bands()
    c2 = _quiet(tl.centroid_Calculation, a2, cluster_col=CT, neighborhood_col=NB)
    c3 = tl3d.centroid_Calculation(a3, cluster_col=CT, neighborhood_col=NB)
    np.testing.assert_array_equal(c2.X, c3.X)
    assert list(c2.var_names) == list(c3.var_names)

    _quiet(tl.cpu_gmm_probability, a2, c2, cluster_col=CT, neighborhood_col=NB)
    tl3d.cpu_gmm_probability(a3, c3, cluster_col=CT, neighborhood_col=NB)
    np.testing.assert_array_equal(
        a2.obsm["neighborhood_probabilities"], a3.obsm["neighborhood_probabilities"]
    )
    assert a2.uns["neighborhood_probability_neighborhoods"] == a3.uns["neighborhood_probability_neighborhoods"]


def test_cpu_gmm_flat_z_matches_2d_and_accepts_custom_coord_names():
    a2 = _bands()
    a3 = _bands(z="flat")
    a3.obs = a3.obs.rename(columns={"x": "X_um", "y": "Y_um", "z": "Z_um"})
    c2 = tl3d.centroid_Calculation(a2, cluster_col=CT, neighborhood_col=NB)
    c3 = tl3d.centroid_Calculation(
        a3, cluster_col=CT, neighborhood_col=NB, x_key="X_um", y_key="Y_um", z_key="Z_um"
    )
    np.testing.assert_array_equal(c2.X, c3.X)
    tl3d.cpu_gmm_probability(a2, c2, cluster_col=CT, neighborhood_col=NB)
    tl3d.cpu_gmm_probability(
        a3, c3, cluster_col=CT, neighborhood_col=NB, x_key="X_um", y_key="Y_um", z_key="Z_um"
    )
    np.testing.assert_array_equal(
        a2.obsm["neighborhood_probabilities"], a3.obsm["neighborhood_probabilities"]
    )


def test_gpu_gmm_requires_cupy():
    pytest.importorskip("cupy")
    a = _bands(z="random")
    c = tl3d.centroid_Calculation(a, cluster_col=CT, neighborhood_col=NB, z_key="z")
    tl3d.gpu_gmm_probability(a, c, cluster_col=CT, neighborhood_col=NB, z_key="z")
    assert a.obsm["neighborhood_probability"].shape == (a.n_obs, 3)


# ---------------------------------------------------------------------------
# Emission models / model comparison
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model", ["diagonal_gaussian", "full_gaussian", "multinomial"])
def test_membership_probabilities_2d_parity(model):
    a2, a3 = _bands(), _bands()
    _quiet(em2d.mingl_membership_probabilities, a2, model=model, cluster_col=CT, neighborhood_col=NB)
    tl3d.mingl_membership_probabilities(a3, model=model, cluster_col=CT, neighborhood_col=NB)
    np.testing.assert_array_equal(a2.obsm["neighborhood_probabilities"], a3.obsm["neighborhood_probabilities"])
    d2, d3 = a2.uns[f"mingl_emission_{model}"], a3.uns[f"mingl_emission_{model}"]
    assert d3["z_key"] is None
    assert d2["bic"] == d3["bic"]


def test_window_matrix_flat_z_matches_2d():
    X2, f2 = tl3d.mingl_window_matrix(_bands(), cluster_col=CT)
    X3, f3 = tl3d.mingl_window_matrix(_bands(z="flat"), cluster_col=CT, z_key="z")
    np.testing.assert_array_equal(X2, X3)
    assert f2 == f3


def test_compare_and_attach_2d_parity():
    models = ("diagonal_gaussian", "multinomial")
    r2 = _quiet(mc2d.compare_emission_models, _bands(), models=models, cluster_col=CT, neighborhood_col=NB)
    r3 = tl3d.compare_emission_models(_bands(), models=models, cluster_col=CT, neighborhood_col=NB)
    pd.testing.assert_frame_equal(r2.drop(columns="runtime_s"), r3.drop(columns="runtime_s"))

    a2, a3 = _bands(), _bands()
    _quiet(mc2d.attach_all_model_probabilities, a2, models=models, cluster_col=CT, neighborhood_col=NB)
    tl3d.attach_all_model_probabilities(a3, models=models, cluster_col=CT, neighborhood_col=NB)
    for m in models:
        np.testing.assert_array_equal(a2.obsm[f"mingl_prob_{m}"], a3.obsm[f"mingl_prob_{m}"])


def test_emission_3d_runs_and_differs_from_2d_on_real_z():
    a3 = tl3d.mingl_membership_probabilities(
        _bands(z="random"), model="diagonal_gaussian", cluster_col=CT, neighborhood_col=NB, z_key="z"
    )
    a2 = tl3d.mingl_membership_probabilities(
        _bands(z="random"), model="diagonal_gaussian", cluster_col=CT, neighborhood_col=NB
    )
    p3 = a3.obsm["neighborhood_probabilities"]
    assert np.allclose(p3.sum(axis=1), 1.0)
    assert not np.array_equal(p3, a2.obsm["neighborhood_probabilities"])


# ---------------------------------------------------------------------------
# Neighborhood-count selection
# ---------------------------------------------------------------------------
def test_run_mingl_over_n_clusters_adds_z_and_matches_2d():
    def _windows(a):
        w = tl3d.KNN2(a, cluster_col=CT, ks=(10,))[10]
        wa = ad.AnnData(obs=w.copy())
        wa.obs_names = wa.obs.index.astype(str)
        return wa

    base = _bands(z="random")
    feats = ["A", "B", "C", "D"]
    w2, w3 = _windows(base), _windows(base)
    w3.obs["z"] = base.obs["z"].values

    s2, pc2 = _quiet(tl.run_mingl_over_n_clusters, w2, feats, n_range=range(1, 4), plot_summary=False)
    s3, pc3 = _quiet(tl3d.run_mingl_over_n_clusters, w3, feats, n_range=range(1, 4), plot_summary=False, z_key="z")
    pd.testing.assert_frame_equal(s2, s3)
    assert list(pc3.columns[:4]) == ["x", "y", "z", REG]
    pd.testing.assert_frame_equal(pc2, pc3.drop(columns="z"))
    with pytest.raises(KeyError):
        tl3d.run_mingl_over_n_clusters(w2, feats, n_range=range(1, 2), plot_summary=False, z_key="z")


def _sweep_windows(z=None):
    base = _bands(z=z)
    w = tl3d.KNN2(base, cluster_col=CT, ks=(10,))[10]
    wa = ad.AnnData(obs=w.copy())
    wa.obs_names = wa.obs.index.astype(str)
    if z is not None:
        wa.obs["z"] = base.obs["z"].values
    return base, wa


def test_sweep_reimplementation_matches_original_exactly():
    """The tl3d sweep loop with clusterer="minibatch" is bit-identical to mingl.tl."""
    from mingl.tl3d import n_neighbors as nn3

    feats = ["A", "B", "C", "D"]
    _, w_ref = _sweep_windows()
    _, w_new = _sweep_windows()
    s_ref, pc_ref = _quiet(tl.run_mingl_over_n_clusters, w_ref, feats, n_range=range(1, 5), plot_summary=False)
    s_new, pc_new = nn3._sweep(
        w_new, feats, range(1, 5), "Neighborhood_{}", True, coord_keys=["x", "y"], region_key=REG,
        results_uns_key="mingl_n_clusters", cluster_row_order="original", order_key=None,
        clusterer="minibatch", random_state=0, n_init=10,
    )
    pd.testing.assert_frame_equal(s_ref, s_new)
    pd.testing.assert_frame_equal(pc_ref, pc_new)
    for n in range(1, 5):
        for col in (f"Neighborhood_{n}", f"log_likelihood_n{n}", f"assigned_prob_n{n}"):
            pd.testing.assert_series_equal(w_ref.obs[col], w_new.obs[col])
    assert w_new.uns["mingl_n_clusters"]["clusterer"] == "minibatch"


def test_score_partition_reproduces_original_per_cell_scores():
    feats = ["A", "B", "C", "D"]
    _, w = _sweep_windows()
    _quiet(tl.run_mingl_over_n_clusters, w, feats, n_range=range(3, 4), plot_summary=False)
    ll, ap = tl3d.score_partition(w.obs[feats], w.obs["Neighborhood_3"].to_numpy(), 3)
    np.testing.assert_array_equal(np.asarray(ll, np.float32), w.obs["log_likelihood_n3"].to_numpy())
    np.testing.assert_array_equal(np.asarray(ap, np.float32), w.obs["assigned_prob_n3"].to_numpy())


def test_kmeans_clusterer_runs_adds_z_and_is_seed_stable():
    from sklearn.metrics import adjusted_rand_score

    feats = ["A", "B", "C", "D"]
    base, w0 = _sweep_windows(z="random")
    _, w1 = _sweep_windows(z="random")
    s0, pc0 = _quiet(tl3d.run_mingl_over_n_clusters, w0, feats, n_range=range(1, 5), plot_summary=False,
                     z_key="z", clusterer="kmeans", random_state=0)
    _quiet(tl3d.run_mingl_over_n_clusters, w1, feats, n_range=range(1, 5), plot_summary=False,
           z_key="z", clusterer="kmeans", random_state=7)
    assert list(pc0.columns[:4]) == ["x", "y", "z", REG]
    assert list(s0.columns) == ["n_clusters", "avg_log_likelihood", "avg_assigned_probability"]
    assert w0.uns["mingl_n_clusters"]["clusterer"] == "kmeans"
    assert w0.uns["mingl_n_clusters"]["n_init"] == 10
    # same partition from a different seed, and it recovers the three bands
    assert adjusted_rand_score(w0.obs["Neighborhood_3"], w1.obs["Neighborhood_3"]) > 0.95
    # bitwise reproducible with the same seed
    X = np.asarray(w0.obs[feats], np.float32)
    np.testing.assert_array_equal(tl3d.fit_neighborhood_labels(X, 3, clusterer="kmeans"),
                                  tl3d.fit_neighborhood_labels(X, 3, clusterer="kmeans"))
    assert adjusted_rand_score(w0.obs["Neighborhood_3"], base.obs[NB].to_numpy()) > 0.7
    with pytest.raises(ValueError):
        tl3d.run_mingl_over_n_clusters(w0, feats, n_range=range(1, 2), plot_summary=False, clusterer="agglomerative")
    with pytest.raises(ValueError):
        tl3d.fit_neighborhood_labels(np.zeros((5, 2), np.float32), 2, clusterer="x")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def test_simulate_2d_parity():
    a2 = tl.simulate_transition_tissue(n_cells=800, sharpness="sharp", seed=4)
    a3 = tl3d.simulate_transition_tissue(n_cells=800, sharpness="sharp", seed=4)
    pd.testing.assert_frame_equal(a2.obs, a3.obs)
    assert a3.uns["sim_params"]["n_dims"] == 2


def test_simulate_3d_z_transition():
    a = tl3d.simulate_transition_tissue(n_cells=2000, sharpness="sharp", depth=80.0, transition_axis="z", seed=1)
    assert "z" in a.obs and a.obs["z"].between(0, 80).all()
    p1 = a.obs[UNIT1].to_numpy(float)
    z = a.obs["z"].to_numpy(float)
    assert np.corrcoef(p1, z)[0, 1] > 0.8  # transition runs along z ...
    assert abs(np.corrcoef(p1, a.obs["x"].to_numpy(float))[0, 1]) < 0.1  # ... not along x
    with pytest.raises(ValueError):
        tl3d.simulate_transition_tissue(transition_axis="z")  # z needs depth


# ---------------------------------------------------------------------------
# Gradient / transition pipeline
# ---------------------------------------------------------------------------
def _pipeline(fn, adata, **kw):
    a = adata.copy()
    _quiet(
        fn, a, tu1=UNIT1, tu2=UNIT2, ks=(20,), k=20, distance_max="none",
        target_neighborhoods=(UNIT1, UNIT2), n_bins=5, n_clusters=5, random_state=0, **kw,
    )
    return a


def test_neighborhoods_pipeline_2d_parity():
    adata = tl.simulate_transition_tissue(n_cells=1500, sharpness="medium", seed=2)
    a2 = _pipeline(tl.grad.mingl_neighborhoods_scverse, adata)
    a3 = _pipeline(tl3d.mingl_neighborhoods_scverse, adata)
    for col in ("Score", "Probability_Level", "neighborhood20", "Probability_Bin_Cluster"):
        pd.testing.assert_series_equal(a2.obs[col], a3.obs[col])
    pd.testing.assert_frame_equal(a2.uns["mingl_fc"], a3.uns["mingl_fc"])


def test_neighborhoods_pipeline_flat_z_matches_2d():
    adata = tl.simulate_transition_tissue(n_cells=1500, sharpness="medium", seed=2)
    flat = adata.copy()
    flat.obs["z"] = 3.0
    a2 = _pipeline(tl.grad.mingl_neighborhoods_scverse, adata)
    a3 = _pipeline(tl3d.mingl_neighborhoods_scverse, flat, Z="z")
    pd.testing.assert_series_equal(a2.obs["Probability_Bin_Cluster"], a3.obs["Probability_Bin_Cluster"])
    assert "z" in a3.uns["mingl_windows"][20].columns


def test_ground_truth_recovery_along_z():
    df = tl3d.validate_ground_truth_recovery(
        sharpness_levels=("sharp", "gradual"), n_cells=2500, n_repeats=3,
        depth=100.0, transition_axis="z", seed=0,
    )
    assert df["error"].isna().all()
    assert (df["z_key"] == "z").all()
    means = df.groupby("sharpness_label")["steepness"].mean()
    assert means["sharp"] > means["gradual"]


def test_z_transition_needs_z_for_spatially_coherent_clusters():
    """A z-transition seen in 2D: transition clusters no longer track the Score."""
    adata = tl3d.simulate_transition_tissue(
        n_cells=2500, sharpness="medium", depth=100.0, transition_axis="z", seed=0
    )

    def _cluster_score_r2(z_key):
        a = _pipeline(tl3d.mingl_neighborhoods_scverse, adata, Z=z_key)
        obs = a.obs.dropna(subset=["Probability_Bin_Cluster"])
        score = obs["Score"].astype(float)
        fitted = score.groupby(obs["Probability_Bin_Cluster"]).transform("mean")
        return 1.0 - ((score - fitted) ** 2).sum() / ((score - score.mean()) ** 2).sum()

    r2_3d, r2_2d = _cluster_score_r2("z"), _cluster_score_r2(None)
    assert r2_3d > 0.6
    assert r2_3d > r2_2d + 0.3


def test_sensitivity_3d_runs():
    adata = tl3d.simulate_transition_tissue(n_cells=1500, sharpness="sharp", depth=100.0, transition_axis="z", seed=0)
    df = tl3d.gradient_sensitivity_analysis(adata, z_key="z", param_grid={"k": [10, 20]}, seed=0)
    assert df["error"].isna().all()
    assert (df["z_key"] == "z").all()


# ---------------------------------------------------------------------------
# Local Score gradients
# ---------------------------------------------------------------------------
def _scored(seed=0, n=1500, depth=None):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame(
        {
            "x": rng.uniform(0, 100, n),
            "y": rng.uniform(0, 100, n),
            REG: "R1",
        }
    )
    if depth is not None:
        obs["z"] = rng.uniform(0, depth, n)
    obs.index = obs.index.astype(str)
    return ad.AnnData(obs=obs)


def test_local_gradients_2d_delegates_to_original():
    a2, a3 = _scored(), _scored()
    for a in (a2, a3):
        a.obs["Score"] = np.sin(a.obs["x"] / 20.0) + 0.01 * a.obs["y"]
    gb2d.gb_local_score_gradients(a2, make_plots=False, use_progress=False)
    tl3d.gb_local_score_gradients(a3, make_plots=False, use_progress=False)
    pd.testing.assert_frame_equal(a2.obs, a3.obs)


def test_batched_solver_matches_original_per_cell_regression_in_2d():
    a = _scored(seed=3)
    a.obs["Score"] = np.sin(a.obs["x"] / 15.0) * np.cos(a.obs["y"] / 25.0)
    gb2d.gb_local_score_gradients(a, make_plots=False, use_progress=False, k_neighbors=15)

    from scipy.spatial import cKDTree

    coords = a.obs[["x", "y"]].to_numpy(float)
    _, nidx = cKDTree(coords).query(coords, k=16)
    grads, valid = tl3d.local_linear_gradients(coords, a.obs["Score"].to_numpy(float), nidx)
    np.testing.assert_array_equal(valid, a.obs["grad_valid"].to_numpy(bool))
    np.testing.assert_allclose(grads[:, 0], a.obs["grad_x"].to_numpy(float), rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(grads[:, 1], a.obs["grad_y"].to_numpy(float), rtol=1e-7, atol=1e-10)


def test_local_gradients_recover_3d_linear_field():
    a = _scored(depth=60.0)
    g = np.array([0.02, -0.05, 0.10])
    a.obs["Score"] = a.obs[["x", "y", "z"]].to_numpy(float) @ g
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = tl3d.gb_local_score_gradients(a, z_key="z", make_plots=False)
    assert out["summary"]["n_dims"] == 3
    assert a.obs["grad_valid"].all()
    got = a.obs[["grad_x", "grad_y", "grad_z"]].to_numpy(float)
    np.testing.assert_allclose(got, np.broadcast_to(g, got.shape), atol=1e-9)
    np.testing.assert_allclose(a.obs["grad_mag"], np.linalg.norm(g), atol=1e-9)


def test_local_gradients_3d_warns_on_make_plots():
    a = _scored(depth=60.0)
    a.obs["Score"] = a.obs["z"] * 0.1
    with pytest.warns(UserWarning, match="not implemented"):
        tl3d.gb_local_score_gradients(a, z_key="z")


def test_local_gradients_flat_z_is_rank_deficient():
    a = _scored()
    a.obs["z"] = 0.0
    a.obs["Score"] = a.obs["x"] * 0.1
    tl3d.gb_local_score_gradients(a, z_key="z", make_plots=False)
    assert not a.obs["grad_valid"].any()  # offsets never span z


# ---------------------------------------------------------------------------
# Border / threshold tools
# ---------------------------------------------------------------------------
def _with_probs(z="random"):
    a = _bands(z=z)
    return tl3d.mingl_membership_probabilities(
        a, model="diagonal_gaussian", cluster_col=CT, neighborhood_col=NB, z_key="z" if z else None
    )


def test_border_tools_require_requested_coords():
    a2 = _with_probs(z=None)
    with pytest.raises(KeyError, match="z"):
        tl3d.border_metrics_at_threshold(a2, threshold=0.25)
    with pytest.raises(KeyError, match="z"):
        tl3d.threshold_sensitivity_analysis(a2)
    with pytest.raises(KeyError, match="z"):
        tl3d.spatial_border_clustering_null(a2)


def test_border_tools_3d_outputs():
    a = _with_probs()
    m = tl3d.border_metrics_at_threshold(a, threshold=0.1, cell_type_col=CT)
    assert set(m["border_centroid"]) == {"x", "y", "z"}
    res = tl3d.threshold_sensitivity_analysis(a, thresholds=(0.1, 0.2), cell_type_col=CT)
    assert {"centroid_z", "spread_z"} <= set(res["summary"].columns)
    null = tl3d.spatial_border_clustering_null(a, threshold=0.1, n_permutations=20)
    assert np.isfinite(null["observed_mean_nn_distance"])


# ---------------------------------------------------------------------------
# crd2 pass-through
# ---------------------------------------------------------------------------
def test_crd2_carries_z(tmp_path):
    a = _with_probs()
    features = ["A", "B", "C", "D"]
    w = tl3d.KNN2(a, cluster_col=CT, z_key="z", ks=(10,))[10]
    w["cellid"] = a.obs_names.to_numpy()
    w["region"] = a.obs[REG].to_numpy()
    w["neigh_name"] = a.obs[NB].to_numpy()
    names = list(a.uns["neighborhood_probability_neighborhoods"])
    probs = pd.DataFrame(a.obsm["neighborhood_probabilities"], columns=names)
    probs["cellid"] = a.obs_names.to_numpy()
    assigned = w[["cellid", "neigh_name", "region"]].set_index("cellid")
    kw = dict(probabilities_df=probs, assigned_df=assigned, neighborhoods=names,
              cell_type_features=features, copy_cells=w)

    p2, d2 = _quiet(tl.crd2, windows2=w, out_probs_path=str(tmp_path / "p2.csv"),
                    out_delta_path=str(tmp_path / "d2.csv"), **kw)
    p3, d3 = _quiet(tl3d.crd2, windows2=w, out_probs_path=str(tmp_path / "p3.csv"),
                    out_delta_path=str(tmp_path / "d3.csv"), **kw)
    z_by_id = a.obs["z"]
    for two, three in ((p2, p3), (d2, d3)):
        assert list(three.columns[-3:]) == ["x", "y", "z"]
        np.testing.assert_array_equal(three["z"].to_numpy(), z_by_id.loc[three["cellid"]].to_numpy())
        pd.testing.assert_frame_equal(two, three.drop(columns="z"))
    assert "z" in pd.read_csv(tmp_path / "p3.csv").columns


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
def _toy_volume():
    lab = np.zeros((4, 6, 8), dtype=np.uint16)
    lab[0:2, 0:2, 0:4] = 5  # 16 voxels, center (z, y, x) = (0.5, 0.5, 1.5)
    lab[1:4, 3:6, 5:8] = 9  # 27 voxels, center (2, 4, 6)
    return lab


def test_label_centroids_physical_units():
    cent = tl3d.label_centroids(_toy_volume(), voxel_size=(2.0, 1.0, 0.5))
    assert list(cent.index) == [5, 9]
    assert list(cent["n_voxels"]) == [16, 27]
    np.testing.assert_allclose(cent.loc[5, ["z", "y", "x"]].to_numpy(float), [1.0, 0.5, 0.75])
    np.testing.assert_allclose(cent.loc[9, ["z", "y", "x"]].to_numpy(float), [4.0, 4.0, 3.0])


def test_cells_from_label_volume_joins_intensities():
    inten = pd.DataFrame({"CD3": [1.0, 2.0], "CK19": [3.0, 4.0]}, index=[9, 42])
    a = tl3d.cells_from_label_volume(_toy_volume(), inten, voxel_size=(2.0, 1.0, 1.0), region="M1")
    assert a.n_obs == 1 and list(a.obs["label"]) == [9]
    assert list(a.obs_names) == ["0"]
    np.testing.assert_array_equal(a.X, [[1.0, 3.0]])
    assert a.obs[REG].iloc[0] == "M1"
    info = a.uns["tl3d_ingest"]
    assert info["n_labels_without_intensities"] == 1 and info["n_intensity_rows_without_label"] == 1


def test_load_kuett2022_model_from_synthetic_folder(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    model = tmp_path / "ToyModel"
    model.mkdir()
    tifffile.imwrite(model / "measured_mask_final_segmentation_toy.tif", _toy_volume().astype(np.uint32))
    pd.DataFrame({"label": [5, 9], "area": [16, 27]}).to_csv(model / "toy_labels_area.csv", index=False)
    pd.DataFrame({"CD3": [0.5, 2.0], "panCK": [9.0, 1.0]}).to_csv(model / "toy_mean_intensities.csv", index=False)

    a = tl3d.load_kuett2022_model(model)
    assert list(a.obs["label"]) == [5, 9]
    assert list(a.var_names) == ["CD3", "panCK"]
    np.testing.assert_array_equal(a.X, [[0.5, 9.0], [2.0, 1.0]])
    np.testing.assert_allclose(a.obs.loc["1", ["x", "y", "z"]].to_numpy(float), [6.0, 4.0, 4.0])  # z scaled by 2 µm
    assert (a.obs[REG] == "ToyModel").all()

    pd.DataFrame({"label": [5, 9], "area": [16, 26]}).to_csv(model / "toy_labels_area.csv", index=False)
    with pytest.raises(ValueError, match="do not match"):
        tl3d.load_kuett2022_model(model)
