"""Tests for the statistics helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from border_figures import stats as st


# ---------------------------------------------------------------------------
# Multiple-testing correction
# ---------------------------------------------------------------------------
def test_holm_matches_hand_computed_values():
    # m=4: sorted p = .01, .02, .03, .04 -> 4*.01, 3*.02, 2*.03, 1*.04, monotone.
    adj = st.holm([0.01, 0.02, 0.03, 0.04])
    np.testing.assert_allclose(adj, [0.04, 0.06, 0.06, 0.06], atol=1e-12)


def test_holm_is_monotone_and_bounded():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50)
    adj = st.holm(p)
    assert np.all(adj <= 1.0)
    assert np.all(adj >= p - 1e-12)  # adjustment never makes a p-value smaller
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-12)  # monotone in the sorted order


def test_holm_preserves_position_and_propagates_nan():
    adj = st.holm([0.5, np.nan, 0.01])
    assert np.isnan(adj[1])
    assert adj[2] < adj[0]


def test_p_stars_thresholds():
    assert st.p_stars(1e-4) == "***"
    assert st.p_stars(5e-3) == "**"
    assert st.p_stars(0.03) == "*"
    assert st.p_stars(0.2) == "ns"
    assert st.p_stars(np.nan) == ""


# ---------------------------------------------------------------------------
# Paired tests
# ---------------------------------------------------------------------------
def test_wilcoxon_effect_size_sign_follows_the_difference():
    values = np.arange(10, dtype=float) + 5.0
    reference = np.arange(10, dtype=float)
    res = st.wilcoxon_vs_reference(values, reference, comparison="up")
    assert res.p_value < 0.05
    assert res.effect_size == pytest.approx(1.0)  # values always above reference
    assert res.unit == "region"

    flipped = st.wilcoxon_vs_reference(reference, values, comparison="down")
    assert flipped.effect_size == pytest.approx(-1.0)


def test_wilcoxon_reports_degenerate_input_instead_of_raising():
    res = st.wilcoxon_vs_reference([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], comparison="same")
    assert np.isnan(res.p_value)
    assert "degenerate" in res.note


def test_wilcoxon_requires_equal_length_pairs():
    with pytest.raises(ValueError, match="equal length"):
        st.wilcoxon_vs_reference([1.0, 2.0], [1.0], comparison="bad")


def test_paired_condition_tests_returns_omnibus_plus_adjusted_posthocs():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 20)
    matrix = pd.DataFrame(
        {"ref": base, "same": base + rng.normal(0, 0.01, 20), "shifted": base + 3.0}
    )
    results = st.paired_condition_tests(matrix, reference="ref", label="demo")

    omnibus, posthocs = results[0], results[1:]
    assert omnibus.test == "friedman"
    assert omnibus.effect_name == "kendall_w"
    assert len(posthocs) == 2
    assert all(r.p_adjusted is not None for r in posthocs)
    shifted = next(r for r in posthocs if "shifted" in r.comparison)
    assert shifted.p_adjusted < 0.05
    assert shifted.effect_size == pytest.approx(1.0)


def test_paired_condition_tests_rejects_unknown_reference():
    matrix = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
    with pytest.raises(KeyError):
        st.paired_condition_tests(matrix, reference="missing", label="demo")


def test_friedman_needs_three_conditions():
    matrix = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0) + 1})
    res = st.friedman_across_conditions(matrix, comparison="two only")
    assert np.isnan(res.p_value)
    assert "needs >=3 conditions" in res.note


# ---------------------------------------------------------------------------
# Unpaired / effect sizes
# ---------------------------------------------------------------------------
def test_cliffs_delta_is_one_for_fully_separated_groups():
    assert st.cliffs_delta([10, 11, 12], [1, 2, 3]) == pytest.approx(1.0)
    assert st.cliffs_delta([1, 2, 3], [10, 11, 12]) == pytest.approx(-1.0)
    assert st.cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_cell_level_tests_are_flagged_descriptive():
    rng = np.random.default_rng(2)
    a, b = rng.normal(0, 1, 500), rng.normal(1, 1, 500)
    for res in (
        st.mannwhitney(a, b, comparison="mw"),
        st.ks_vs_reference(a, b, comparison="ks"),
    ):
        assert res.unit == "cell"
        assert "descriptive only" in res.note
        assert res.p_value < 0.05


def test_chi2_homogeneity_detects_a_shifted_composition():
    table = pd.DataFrame({"T": [900, 100], "B": [100, 900]}, index=["m1", "m2"])
    res = st.chi2_homogeneity(table, comparison="chi2")
    assert res.p_value < 1e-10
    assert 0.0 <= res.effect_size <= 1.0


# ---------------------------------------------------------------------------
# Composition distance + bootstrap
# ---------------------------------------------------------------------------
def test_jensen_shannon_bounds_and_symmetry():
    p = [0.5, 0.5, 0.0]
    q = [0.0, 0.0, 1.0]
    assert st.jensen_shannon_divergence(p, p) == pytest.approx(0.0, abs=1e-12)
    assert st.jensen_shannon_divergence(p, q) == pytest.approx(1.0, abs=1e-9)
    assert st.jensen_shannon_divergence(p, q) == pytest.approx(
        st.jensen_shannon_divergence(q, p)
    )


def test_jensen_shannon_renormalizes_unnormalized_input():
    assert st.jensen_shannon_divergence([2, 2], [1, 1]) == pytest.approx(0.0, abs=1e-12)


def test_jensen_shannon_is_nan_for_an_empty_composition():
    assert np.isnan(st.jensen_shannon_divergence([0, 0], [1, 1]))


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(3)
    values = rng.normal(5.0, 1.0, 40)
    point, lo, hi = st.bootstrap_ci(
        lambda idx: float(np.mean([values[int(i)] for i in idx])),
        list(range(40)), n_boot=400, seed=0,
    )
    assert lo <= point <= hi
    assert hi - lo < 2.0  # a mean of 40 draws is not that uncertain


def test_bootstrap_ci_returns_nan_ci_for_too_few_units():
    point, lo, hi = st.bootstrap_ci(lambda idx: float(len(idx)), [1, 2], n_boot=50)
    assert point == 2
    assert np.isnan(lo) and np.isnan(hi)


def test_results_to_frame_has_the_expected_schema():
    frame = st.results_to_frame(
        [st.TestResult("c", "t", "region", 5, 1.0, 0.01, 0.02, "e", 0.5)]
    )
    assert list(frame.columns)[:6] == ["comparison", "test", "unit", "n", "statistic", "p_value"]
    assert frame.loc[0, "stars"] == "*"


def test_results_to_frame_handles_no_results():
    assert st.results_to_frame([]).empty
