"""Tests for the pair-specific border definitions and the interface coordinate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from border_figures import pair_borders as pb


@pytest.fixture
def toy_probs() -> pd.DataFrame:
    """Six cells x three units, hand-built so every group is checkable by eye."""
    return pd.DataFrame(
        [
            [0.60, 0.30, 0.10],  # 0: A only (0.30 > 0.25 -> also B!) -> A+B border
            [0.80, 0.15, 0.05],  # 1: A only
            [0.10, 0.85, 0.05],  # 2: B only
            [0.40, 0.35, 0.25],  # 3: A+B border (C exactly at 0.25, not > )
            [0.30, 0.30, 0.40],  # 4: A+B+C (all above 0.25)
            [0.05, 0.05, 0.90],  # 5: C only
        ],
        columns=["A", "B", "C"],
        index=[f"c{i}" for i in range(6)],
    )


def test_positive_matrix_inclusive_vs_strict(toy_probs):
    strict = pb.positive_matrix(toy_probs, 0.25, inclusive=False)
    inclusive = pb.positive_matrix(toy_probs, 0.25, inclusive=True)
    # Cell 3 has C == 0.25 exactly: excluded by ">", included by ">=".
    assert not strict[3, 2]
    assert inclusive[3, 2]
    assert strict.sum() + 1 == inclusive.sum()


def test_pair_masks_groups_are_disjoint_and_correct(toy_probs):
    m = pb.pair_masks(toy_probs, "A", "B", threshold=0.25)
    assert list(np.where(m["border"])[0]) == [0, 3, 4]
    assert list(np.where(m["cn1_only"])[0]) == [1]
    assert list(np.where(m["cn2_only"])[0]) == [2]
    # The three groups never overlap.
    assert not (m["border"] & m["cn1_only"]).any()
    assert not (m["border"] & m["cn2_only"]).any()
    assert not (m["cn1_only"] & m["cn2_only"]).any()


def test_require_exactly_two_drops_triple_positive_cells(toy_probs):
    m = pb.pair_masks(toy_probs, "A", "B", threshold=0.25, require_exactly_two=True)
    # Cell 4 is positive for A, B and C, so it is no longer an A|B-only border.
    assert list(np.where(m["border"])[0]) == [0, 3]


def test_pair_masks_rejects_unknown_unit(toy_probs):
    with pytest.raises(KeyError, match="not among the scored units"):
        pb.pair_masks(toy_probs, "A", "Nope", threshold=0.25)


def test_select_top_pair_returns_most_co_positive_pair(toy_probs):
    # A&B co-occur in cells 0, 3, 4; A&C and B&C only in cell 4.
    assert pb.select_top_pair(toy_probs, threshold=0.25) == ("A", "B")


def test_border_count_is_monotone_non_increasing_in_threshold(toy_probs):
    counts = [
        int(pb.pair_masks(toy_probs, "A", "B", threshold=t)["border"].sum())
        for t in (0.0, 0.1, 0.25, 0.35, 0.45)
    ]
    assert all(b <= a for a, b in zip(counts, counts[1:])), counts


# ---------------------------------------------------------------------------
# Interface coordinate
# ---------------------------------------------------------------------------
def _two_band_obs(n_per_band: int = 40, spacing: float = 1.0) -> pd.DataFrame:
    """Two adjacent unit bands along y, one region, on a regular lattice."""
    rows = []
    for i in range(n_per_band):
        rows.append({"x": (i % 8) * spacing, "y": (i // 8) * spacing, "unit": "A"})
    offset = (n_per_band // 8) * spacing
    for i in range(n_per_band):
        rows.append({"x": (i % 8) * spacing, "y": offset + (i // 8) * spacing, "unit": "B"})
    df = pd.DataFrame(rows)
    df["region"] = "R1"
    df.index = df.index.astype(str)
    return df


def test_interface_coordinate_sign_and_zero_crossing():
    obs = _two_band_obs()
    loc = pb.interface_coordinate(
        obs, unit_col="unit", cn1="A", cn2="B", region_key="region", x_key="x", y_key="y"
    )
    a_side = loc.loc[obs["unit"] == "A", "interface_coord"]
    b_side = loc.loc[obs["unit"] == "B", "interface_coord"]
    # Never the wrong sign: CN1 cells are <= 0, CN2 cells >= 0. Cells sitting on
    # the boundary row are equidistant from both units and land exactly at 0.
    assert (a_side <= 0).all()
    assert (b_side >= 0).all()

    y_a = obs.loc[obs["unit"] == "A", "y"]
    deepest_a = obs.index[obs["y"] == y_a.min()]
    boundary_a = obs.index[obs["y"] == y_a.max()]
    # Deep cells are strictly on their own side; boundary cells sit at the interface.
    assert (loc.loc[deepest_a, "interface_coord"] < 0).all()
    np.testing.assert_allclose(loc.loc[boundary_a, "interface_coord"].to_numpy(), 0.0, atol=1e-9)
    assert (
        loc.loc[boundary_a, "dist_to_interface"].mean()
        < loc.loc[deepest_a, "dist_to_interface"].mean()
    )


def test_interface_coordinate_is_nan_when_a_unit_is_absent_from_a_region():
    obs = _two_band_obs()
    lonely = obs.copy()
    lonely["region"] = "R2"
    lonely["unit"] = "A"  # region R2 has no B cells at all
    combined = pd.concat([obs, lonely], ignore_index=True)
    combined.index = combined.index.astype(str)

    loc = pb.interface_coordinate(
        combined, unit_col="unit", cn1="A", cn2="B", region_key="region", x_key="x", y_key="y"
    )
    in_r2 = (combined["region"] == "R2").to_numpy()
    assert loc.loc[in_r2, "interface_coord"].isna().all()
    assert loc.loc[~in_r2, "interface_coord"].notna().all()


def test_interface_coordinate_is_scale_free_when_normalized():
    """Doubling every coordinate must not change the normalized coordinate."""
    a = pb.interface_coordinate(
        _two_band_obs(spacing=1.0), unit_col="unit", cn1="A", cn2="B",
        region_key="region", x_key="x", y_key="y",
    )
    b = pb.interface_coordinate(
        _two_band_obs(spacing=3.0), unit_col="unit", cn1="A", cn2="B",
        region_key="region", x_key="x", y_key="y",
    )
    np.testing.assert_allclose(
        a["interface_coord"].to_numpy(), b["interface_coord"].to_numpy(), rtol=1e-9
    )


# ---------------------------------------------------------------------------
# Composition + enrichment (Methods 4.5)
# ---------------------------------------------------------------------------
def test_pair_enrichment_matches_the_manuscript_formula():
    # border: 8 T, 2 B ; CN1-only: 4 T, 16 B ; CN2-only: 10 T, 10 B
    cell_types = ["T"] * 8 + ["B"] * 2 + ["T"] * 4 + ["B"] * 16 + ["T"] * 10 + ["B"] * 10
    n = len(cell_types)
    masks = {
        "border": np.array([True] * 10 + [False] * (n - 10)),
        "cn1_only": np.array([False] * 10 + [True] * 20 + [False] * 20),
        "cn2_only": np.array([False] * 30 + [True] * 20),
    }
    comp = pb.pair_composition(cell_types, masks)
    enr = pb.pair_enrichment(comp, min_cells=1).set_index("cell_type")

    # p_T: border 0.8, CN1-only 0.2, CN2-only 0.5
    assert enr.loc["T", "log2_vs_cn1"] == pytest.approx(np.log2(0.8 / 0.2), abs=1e-6)
    assert enr.loc["T", "log2_vs_cn2"] == pytest.approx(np.log2(0.8 / 0.5), abs=1e-6)
    # Enriched only if positive against BOTH single-unit groups.
    assert bool(enr.loc["T", "enriched_both"])
    assert not bool(enr.loc["B", "enriched_both"])
    assert enr.loc["T", "min_enrichment"] == pytest.approx(np.log2(0.8 / 0.5), abs=1e-6)


def test_pair_enrichment_applies_mingls_three_way_count_filter():
    """All three groups must clear min_cells, matching mingl.pl.plot_border_enrichment."""
    # T: border 20, cn1-only 20, cn2-only 20  -> reportable
    # Rare: border 20, cn1-only 20, cn2-only 1 -> dropped (cn2 denominator too thin)
    cell_types = (
        ["T"] * 20 + ["Rare"] * 20      # border
        + ["T"] * 20 + ["Rare"] * 20    # cn1-only
        + ["T"] * 20 + ["Rare"] * 1     # cn2-only
    )
    n = len(cell_types)
    border = np.zeros(n, dtype=bool); border[:40] = True
    cn1 = np.zeros(n, dtype=bool); cn1[40:80] = True
    cn2 = np.zeros(n, dtype=bool); cn2[80:] = True
    masks = {"border": border, "cn1_only": cn1, "cn2_only": cn2}

    enr = pb.pair_enrichment(pb.pair_composition(cell_types, masks), min_cells=5)
    enr = enr.set_index("cell_type")
    assert not np.isnan(enr.loc["T", "log2_vs_cn1"])
    # Rare has plenty of border cells but only one cn2-only cell -> NaN, not a
    # spuriously huge enrichment.
    assert np.isnan(enr.loc["Rare", "log2_vs_cn1"])
    assert np.isnan(enr.loc["Rare", "log2_vs_cn2"])
    assert not bool(enr.loc["Rare", "enriched_both"])


def test_pair_composition_proportions_sum_to_one_per_group():
    cell_types = ["T", "B", "T", "B", "T", "B"]
    masks = {
        "border": np.array([True, True, False, False, False, False]),
        "cn1_only": np.array([False, False, True, True, False, False]),
        "cn2_only": np.array([False, False, False, False, True, True]),
    }
    comp = pb.pair_composition(cell_types, masks)
    totals = comp.groupby("group")["proportion"].sum()
    np.testing.assert_allclose(totals.to_numpy(), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Region-level aggregation
# ---------------------------------------------------------------------------
def test_per_region_stats_exclude_small_and_interface_free_regions():
    rng = np.random.default_rng(0)
    n_per = 60
    obs = pd.DataFrame(
        {
            "x": rng.uniform(0, 10, 2 * n_per),
            "y": rng.uniform(0, 10, 2 * n_per),
            "region": ["R1"] * n_per + ["R2"] * n_per,
            "unit": ["A", "B"] * n_per,
            "Cell Type": rng.choice(["T", "B"], 2 * n_per),
        }
    )
    obs.index = obs.index.astype(str)
    location = pb.interface_coordinate(
        obs, unit_col="unit", cn1="A", cn2="B", region_key="region", x_key="x", y_key="y"
    )
    # R1 gets 30 border cells, R2 only 3.
    border = np.zeros(2 * n_per, dtype=bool)
    border[:30] = True
    border[n_per : n_per + 3] = True
    masks = {"border": border, "cn1_only": ~border, "cn2_only": np.zeros(2 * n_per, dtype=bool)}

    out = pb.per_region_border_stats(
        obs, masks, location, region_key="region", cell_type_col="Cell Type",
        min_border_cells=20,
    )
    usable = out["summary"].set_index("region")["usable"]
    assert bool(usable["R1"]) and not bool(usable["R2"])
    assert list(out["composition"].index) == ["R1"]


def test_mingl_composition_matches_mask_composition():
    """The shipped compute_grouped_proportions must agree with our mask counts.

    analyze_pair reports MINGL's proportions but our counts; that is only sound
    if both identify the same three groups. This pins it.
    """
    import anndata as ad

    rng = np.random.default_rng(0)
    n = 400
    obs = pd.DataFrame(
        {
            "x": rng.uniform(0, 50, n),
            "y": rng.uniform(0, 50, n),
            "unique_region": "R1",
            "Cell Type": rng.choice(["T", "B", "Mac"], n),
        }
    )
    obs.index = obs.index.astype(str)
    adata = ad.AnnData(X=np.zeros((n, 0), dtype=np.float32), obs=obs)

    probs = pd.DataFrame(rng.dirichlet([1.0, 1.0, 1.0], n), index=obs.index,
                         columns=["A", "B", "C"])
    pb.attach_posterior(adata, probs)

    masks = pb.pair_masks(probs, "A", "B", threshold=0.25)
    mask_comp = pb.pair_composition(obs["Cell Type"], masks).set_index(["group", "cell_type"])
    mingl_comp = pb.composition_via_mingl(
        adata, cn1="A", cn2="B", threshold=0.25, cell_type_col="Cell Type"
    ).set_index(["group", "cell_type"])

    shared = mask_comp.index.intersection(mingl_comp.index)
    assert len(shared) > 0
    np.testing.assert_allclose(
        mask_comp.loc[shared, "proportion"].to_numpy(),
        mingl_comp.loc[shared, "proportion"].to_numpy(),
        atol=1e-12,
    )
