import anndata as ad
import numpy as np
import pandas as pd

import mingl.tl.gmm as gmm


def _make_cells_adata() -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "cell_type": ["T", "B", "T"],
            "neighborhood": ["N1", "N2", "N1"],
            "unique_region": ["R1", "R1", "R1"],
        },
        index=["cell_1", "cell_2", "cell_3"],
    )
    var = pd.DataFrame(index=["placeholder"])
    return ad.AnnData(X=np.zeros((len(obs), 1), dtype=np.float64), obs=obs, var=var)


def _make_centroids_adata() -> ad.AnnData:
    obs = pd.DataFrame({"neighborhood": ["N1", "N2"]}, index=["N1", "N2"])
    var = pd.DataFrame(index=["T_mean", "B_mean", "T_std", "B_std"])
    x = np.array(
        [
            [1.0, 0.0, 0.1, 0.1],
            [0.0, 1.0, 0.1, 0.1],
        ],
        dtype=np.float64,
    )
    return ad.AnnData(X=x, obs=obs, var=var)


def _mock_knn2_result(cells: ad.AnnData) -> dict[int, pd.DataFrame]:
    return {
        10: pd.DataFrame(
            {
                "T": [1.0, 0.0, 1.0],
                "B": [0.0, 1.0, 0.0],
            },
            index=cells.obs_names,
        )
    }


def _old_scalar_probability(cell_row: np.ndarray, centroid_means: np.ndarray, centroid_stds: np.ndarray) -> np.ndarray:
    probabilities = []
    for centroid_index in range(len(centroid_means)):
        total_probability = 1.0
        for feature_index in range(len(cell_row)):
            mean = centroid_means[centroid_index, feature_index]
            std = centroid_stds[centroid_index, feature_index]
            value = cell_row[feature_index]

            if std == 0:
                feature_probability = 1.0 if value == mean else 0.0
            else:
                feature_probability = (
                    1.0
                    / (std * np.sqrt(2 * np.pi))
                    * np.exp(-0.5 * ((value - mean) / std) ** 2)
                )

            total_probability *= feature_probability

        probabilities.append(total_probability)

    probabilities = np.array(probabilities, dtype=np.float64)
    total = probabilities.sum()
    if total == 0:
        return np.zeros_like(probabilities)

    return probabilities / total


def test_cpu_gmm_probability_assigns_expected_neighborhood_probabilities(monkeypatch):
    cells = _make_cells_adata()
    centroids = _make_centroids_adata()

    monkeypatch.setattr(gmm, "KNN2", lambda *args, **kwargs: _mock_knn2_result(cells))

    result = gmm.cpu_gmm_probability(
        CELLS_ADATA=cells,
        CENTROIDS_ADATA=centroids,
        ks=(10,),
        k=10,
    )

    probabilities = result.obsm["neighborhood_probabilities"]

    assert probabilities.shape == (3, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(3))
    assert result.uns["neighborhood_probability_neighborhoods"] == ["N1", "N2"]
    assert probabilities[0, 0] > probabilities[0, 1]
    assert probabilities[1, 1] > probabilities[1, 0]
    assert probabilities[2, 0] > probabilities[2, 1]


def test_cpu_gmm_probability_parallel_workers_match_single_worker(monkeypatch):
    cells = _make_cells_adata()
    centroids = _make_centroids_adata()

    monkeypatch.setattr(gmm, "KNN2", lambda *args, **kwargs: _mock_knn2_result(cells))

    single_worker = gmm.cpu_gmm_probability(
        CELLS_ADATA=cells.copy(),
        CENTROIDS_ADATA=centroids,
        ks=(10,),
        k=10,
        num_processes=1,
    )
    parallel_workers = gmm.cpu_gmm_probability(
        CELLS_ADATA=cells.copy(),
        CENTROIDS_ADATA=centroids,
        ks=(10,),
        k=10,
        num_processes=2,
    )

    np.testing.assert_allclose(
        single_worker.obsm["neighborhood_probabilities"],
        parallel_workers.obsm["neighborhood_probabilities"],
    )


def test_batched_probability_kernel_matches_old_scalar_math():
    window_batch = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.5, 1.5, 0.0],
            [2.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    centroid_means = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    centroid_stds = np.array(
        [
            [0.1, 0.0, 0.3],
            [0.5, 0.2, 0.0],
        ],
        dtype=np.float64,
    )

    expected = np.vstack(
        [
            _old_scalar_probability(cell_row, centroid_means, centroid_stds)
            for cell_row in window_batch
        ]
    )
    observed = gmm._compute_probability_batch(window_batch, centroid_means, centroid_stds)

    np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=1e-12)
