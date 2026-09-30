"""End-to-end smoke tests: both drivers run on the synthetic tissue and write output.

These exercise the real code paths (window computation, posterior scoring, the
pair analysis, every statistic and both figures) on a deliberately small tissue,
so a refactor that breaks the pipeline fails here rather than on the lab server.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from border_figures import figure1_emission_models as fig1
from border_figures import figure2_threshold_effects as fig2
from border_figures.loading import make_null, synthetic_tissue
from border_figures.config import get_level, resolve_dataset

SMALL = ["--synthetic-regions", "4", "--synthetic-cells", "900"]


def test_synthetic_tissue_has_a_consistent_hierarchy():
    adata = synthetic_tissue(n_regions=3, n_cells_per_region=500, seed=0)
    obs = adata.obs
    assert {"x", "y", "unique_region", "Cell Type", "Neighborhood", "Community",
            "Tissue Unit"} <= set(obs.columns)
    # Every neighborhood belongs to exactly one community, and each community to
    # exactly one tissue unit -- otherwise the hierarchy levels are meaningless.
    assert (obs.groupby("Neighborhood", observed=True)["Community"].nunique() == 1).all()
    assert (obs.groupby("Community", observed=True)["Tissue Unit"].nunique() == 1).all()
    assert obs["unique_region"].nunique() == 3


def test_make_null_preserves_geography_and_abundance_but_shuffles_identity():
    adata = synthetic_tissue(n_regions=2, n_cells_per_region=800, seed=0)
    spec = resolve_dataset("synthetic")
    level = get_level("synthetic", "neighborhood")
    null = make_null(adata, mode="celltype", spec=spec, level=level, seed=0)

    # Coordinates, regions and unit labels are untouched...
    for col in ("x", "y", "unique_region", "Neighborhood"):
        pd.testing.assert_series_equal(adata.obs[col], null.obs[col], check_names=False)
    # ...cell-type abundances are preserved...
    pd.testing.assert_series_equal(
        adata.obs["Cell Type"].value_counts().sort_index(),
        null.obs["Cell Type"].value_counts().sort_index(),
        check_names=False,
    )
    # ...but the spatial arrangement of cell types is not.
    assert (adata.obs["Cell Type"].to_numpy() != null.obs["Cell Type"].to_numpy()).mean() > 0.5


def test_make_null_rejects_unknown_mode():
    adata = synthetic_tissue(n_regions=1, n_cells_per_region=200, seed=0)
    with pytest.raises(ValueError, match="Unknown null mode"):
        make_null(adata, mode="nonsense", spec=resolve_dataset("synthetic"),
                  level=get_level("synthetic", "neighborhood"), seed=0)


@pytest.mark.slow
def test_figure1_driver_writes_figure_and_tables(tmp_path):
    exit_code = fig1.main(
        [*SMALL, "--synthetic", "--focus-level", "neighborhood", "--no-cache",
         "--n-boot", "20", "--min-border-cells", "5",
         "--out-dir", str(tmp_path)]
    )
    assert exit_code == 0
    for name in (
        "figure1_emission_models.png",
        "focus_border_summary.csv",
        "border_composition_by_model.csv",
        "assigned_probability_per_neighborhood.csv",
        "statistics.csv",
    ):
        assert (tmp_path / name).exists(), name

    summary = pd.read_csv(tmp_path / "focus_border_summary.csv")
    assert len(summary) == 5  # one row per emission model
    assert summary["n_border"].ge(0).all()
    # Probabilities of the assigned neighborhood must be valid probabilities.
    per_unit = pd.read_csv(tmp_path / "assigned_probability_per_neighborhood.csv")
    assert per_unit["median_p_assigned"].between(0, 1).all()


@pytest.mark.slow
def test_figure2_driver_writes_figure_and_monotone_counts(tmp_path):
    exit_code = fig2.main(
        [*SMALL, "--synthetic", "--no-cache", "--n-boot", "20",
         "--min-border-cells", "5", "--thresholds", "0.05", "0.15", "0.25", "0.4",
         "--out-dir", str(tmp_path)]
    )
    assert exit_code == 0
    assert (tmp_path / "figure2_threshold_effects.png").exists()

    conditions = pd.read_csv(tmp_path / "conditions.csv")
    # null + three hierarchy levels, each with its own focus border.
    assert set(conditions["level"]) == {"neighborhood", "community", "tissue_unit"}
    assert conditions["is_null"].sum() == 1
    # The manuscript's k per level (Methods 4.2), not one window re-used everywhere.
    by_level = conditions.drop_duplicates("level").set_index("level")["k"].to_dict()
    assert by_level == {"neighborhood": 10, "community": 30, "tissue_unit": 60}
    assert (conditions["cn1"] != conditions["cn2"]).all()

    summary = pd.read_csv(tmp_path / "threshold_summary.csv")
    for _, block in summary.groupby("condition"):
        counts = block.sort_values("threshold")["n_border"].to_numpy()
        assert np.all(np.diff(counts) <= 0), "border count must not rise with threshold"

    stats_frame = pd.read_csv(tmp_path / "statistics.csv")
    assert "real vs null" not in stats_frame.columns  # schema sanity
    assert stats_frame["p_value"].dropna().between(0, 1).all()


@pytest.mark.slow
def test_figure2_null_borders_are_further_from_the_interface(tmp_path):
    """The point of the null: its border cells are not at the anatomical interface."""
    assert fig2.main(
        [*SMALL, "--synthetic", "--no-cache", "--n-boot", "20", "--min-border-cells", "5",
         "--thresholds", "0.1", "0.25", "--out-dir", str(tmp_path)]
    ) == 0

    summary = pd.read_csv(tmp_path / "threshold_summary.csv")
    at_ref = summary[summary["threshold"] == 0.25].set_index("condition")
    null_distance = at_ref.loc["synthetic_null", "median_dist_to_interface"]
    real = at_ref.drop(index="synthetic_null")["median_dist_to_interface"].dropna()
    assert len(real) >= 2
    assert (real < null_distance).all(), (real.to_dict(), null_distance)
