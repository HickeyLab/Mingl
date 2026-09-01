"""Statistics used by both border figures.

Design notes
------------
**Cells are not independent replicates.** Every cell in a tissue shares its k-NN
window with its neighbors, and the same cells are re-scored by every emission
model / at every threshold, so a cell-level test across models or thresholds has
a wildly inflated n and is not interpretable as evidence. Every *primary* test
here therefore uses the **tissue region as the experimental unit** and is
**paired**, because the same regions are measured under every condition:

* >= 3 conditions -> Friedman test (+ Kendall's W as the effect size),
* post-hoc / 2 conditions -> Wilcoxon signed-rank against the reference
  condition, Holm-corrected across conditions, with the matched-pairs
  rank-biserial correlation as the effect size.

Cell-level tests (Mann-Whitney, Kolmogorov-Smirnov, chi-square) are still
provided and reported, but always labeled ``unit="cell"`` so they are read as
descriptive, not inferential.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "TestResult",
    "holm",
    "p_stars",
    "wilcoxon_vs_reference",
    "friedman_across_conditions",
    "mannwhitney",
    "ks_vs_reference",
    "chi2_homogeneity",
    "cliffs_delta",
    "jensen_shannon_divergence",
    "bootstrap_ci",
    "paired_condition_tests",
    "results_to_frame",
]

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Containers / helpers
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    """One statistical test, flattened so a list of these becomes a tidy table."""

    comparison: str
    test: str
    unit: str  # "region" (paired, primary) or "cell" (descriptive)
    n: int
    statistic: float
    p_value: float
    p_adjusted: float | None = None
    effect_name: str | None = None
    effect_size: float | None = None
    note: str = ""

    @property
    def stars(self) -> str:
        p = self.p_adjusted if self.p_adjusted is not None else self.p_value
        return p_stars(p)


def results_to_frame(results: Sequence[TestResult]) -> pd.DataFrame:
    """Tidy DataFrame from a list of :class:`TestResult` (adds a ``stars`` column)."""
    if not results:
        return pd.DataFrame(
            columns=[
                "comparison", "test", "unit", "n", "statistic", "p_value",
                "p_adjusted", "effect_name", "effect_size", "note", "stars",
            ]
        )
    rows = []
    for r in results:
        row = asdict(r)
        row["stars"] = r.stars
        rows.append(row)
    return pd.DataFrame(rows)


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (NaNs propagate, order preserved)."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx], kind="stable")]
    m = order.size
    running = 0.0
    for rank, j in enumerate(order):
        adj = (m - rank) * p[j]
        running = max(running, adj)  # enforce monotonicity
        out[j] = min(1.0, running)
    return out


def p_stars(p: float | None) -> str:
    """``***`` / ``**`` / ``*`` / ``ns`` (``''`` when p is missing)."""
    if p is None or not np.isfinite(p):
        return ""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _clean_pair(a: Sequence[float], b: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Paired arrays must have equal length, got {a.shape} vs {b.shape}.")
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


# ---------------------------------------------------------------------------
# Paired, region-level tests (primary)
# ---------------------------------------------------------------------------
def _rank_biserial_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation in [-1, 1] (positive: a > b)."""
    d = a - b
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    total = ranks.sum()
    if total <= 0:
        return 0.0
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / total)


def wilcoxon_vs_reference(
    values: Sequence[float],
    reference: Sequence[float],
    *,
    comparison: str,
    note: str = "",
) -> TestResult:
    """Paired Wilcoxon signed-rank of ``values`` against ``reference`` across regions."""
    a, b = _clean_pair(values, reference)
    n = int(a.size)
    if n < 3 or np.allclose(a, b):
        return TestResult(
            comparison=comparison,
            test="wilcoxon_signed_rank",
            unit="region",
            n=n,
            statistic=float("nan"),
            p_value=float("nan"),
            effect_name="rank_biserial",
            effect_size=_rank_biserial_paired(a, b) if n else float("nan"),
            note=(note + " insufficient/degenerate pairs").strip(),
        )
    res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return TestResult(
        comparison=comparison,
        test="wilcoxon_signed_rank",
        unit="region",
        n=n,
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        effect_name="rank_biserial",
        effect_size=_rank_biserial_paired(a, b),
        note=note,
    )


def friedman_across_conditions(
    matrix: pd.DataFrame,
    *,
    comparison: str,
    note: str = "",
) -> TestResult:
    """Friedman test over a (regions x conditions) matrix; effect size = Kendall's W.

    Rows with any missing value are dropped (the test needs complete blocks).
    """
    m = matrix.dropna(axis=0, how="any")
    n_blocks, n_cond = m.shape
    if n_cond < 3 or n_blocks < 3:
        return TestResult(
            comparison=comparison,
            test="friedman",
            unit="region",
            n=int(n_blocks),
            statistic=float("nan"),
            p_value=float("nan"),
            effect_name="kendall_w",
            effect_size=float("nan"),
            note=(note + f" needs >=3 conditions and >=3 regions (got {n_cond}, {n_blocks})").strip(),
        )
    columns = [m.iloc[:, j].to_numpy(dtype=float) for j in range(n_cond)]
    if all(np.allclose(columns[0], c) for c in columns[1:]):
        return TestResult(
            comparison=comparison, test="friedman", unit="region", n=int(n_blocks),
            statistic=float("nan"), p_value=float("nan"), effect_name="kendall_w",
            effect_size=0.0, note=(note + " all conditions identical").strip(),
        )
    res = stats.friedmanchisquare(*columns)
    kendall_w = float(res.statistic / (n_blocks * (n_cond - 1))) if n_blocks and n_cond > 1 else float("nan")
    return TestResult(
        comparison=comparison,
        test="friedman",
        unit="region",
        n=int(n_blocks),
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        effect_name="kendall_w",
        effect_size=kendall_w,
        note=note,
    )


def paired_condition_tests(
    matrix: pd.DataFrame,
    *,
    reference: str,
    label: str,
) -> list[TestResult]:
    """Omnibus Friedman + Holm-corrected Wilcoxon post-hocs against ``reference``.

    ``matrix`` is (regions x conditions); ``reference`` names one of its columns.
    Returns the omnibus result first, then one post-hoc per remaining condition.
    """
    if reference not in matrix.columns:
        raise KeyError(f"Reference condition {reference!r} not in columns {list(matrix.columns)}.")

    results = [friedman_across_conditions(matrix, comparison=f"{label}: across conditions")]
    others = [c for c in matrix.columns if c != reference]
    posthoc = [
        wilcoxon_vs_reference(
            matrix[c], matrix[reference], comparison=f"{label}: {c} vs {reference}"
        )
        for c in others
    ]
    for res, adj in zip(posthoc, holm([r.p_value for r in posthoc])):
        res.p_adjusted = None if not np.isfinite(adj) else float(adj)
        res.note = (res.note + " Holm-corrected across conditions").strip()
    results.extend(posthoc)
    return results


# ---------------------------------------------------------------------------
# Unpaired / cell-level tests (descriptive)
# ---------------------------------------------------------------------------
def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta in [-1, 1] (positive: ``a`` stochastically larger than ``b``)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    # delta = 2*U/(n1*n2) - 1 with U from the rank-sum identity (O(n log n)).
    u = float(stats.mannwhitneyu(a, b, alternative="two-sided").statistic)
    return float(2.0 * u / (a.size * b.size) - 1.0)


def mannwhitney(
    a: Sequence[float], b: Sequence[float], *, comparison: str, note: str = ""
) -> TestResult:
    """Two-sided Mann-Whitney U with Cliff's delta (cell-level, descriptive)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return TestResult(
            comparison=comparison, test="mann_whitney_u", unit="cell",
            n=int(a.size + b.size), statistic=float("nan"), p_value=float("nan"),
            effect_name="cliffs_delta", effect_size=float("nan"),
            note=(note + " too few observations").strip(),
        )
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    return TestResult(
        comparison=comparison, test="mann_whitney_u", unit="cell",
        n=int(a.size + b.size), statistic=float(res.statistic), p_value=float(res.pvalue),
        effect_name="cliffs_delta", effect_size=cliffs_delta(a, b),
        note=(note + " cells are not independent; descriptive only").strip(),
    )


def ks_vs_reference(
    values: Sequence[float], reference: Sequence[float], *, comparison: str, note: str = ""
) -> TestResult:
    """Two-sample Kolmogorov-Smirnov (cell-level, descriptive).

    The KS statistic doubles as an interpretable effect size: the maximum vertical
    gap between the two cumulative distributions.
    """
    a = np.asarray(values, dtype=float)
    b = np.asarray(reference, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return TestResult(
            comparison=comparison, test="ks_2samp", unit="cell", n=int(a.size + b.size),
            statistic=float("nan"), p_value=float("nan"),
            note=(note + " too few observations").strip(),
        )
    res = stats.ks_2samp(a, b)
    return TestResult(
        comparison=comparison, test="ks_2samp", unit="cell", n=int(a.size + b.size),
        statistic=float(res.statistic), p_value=float(res.pvalue),
        effect_name="ks_D", effect_size=float(res.statistic),
        note=(note + " cells are not independent; descriptive only").strip(),
    )


def chi2_homogeneity(
    table: pd.DataFrame, *, comparison: str, note: str = ""
) -> TestResult:
    """Chi-square test of homogeneity on a counts table (rows = conditions).

    Effect size is Cramer's V. Zero rows/columns are dropped first.
    """
    t = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        return TestResult(
            comparison=comparison, test="chi2_homogeneity", unit="cell",
            n=int(table.to_numpy().sum()), statistic=float("nan"), p_value=float("nan"),
            note=(note + " degenerate table").strip(),
        )
    chi2, p, _dof, _exp = stats.chi2_contingency(t.to_numpy())
    n = float(t.to_numpy().sum())
    v = float(np.sqrt(chi2 / (n * (min(t.shape) - 1)))) if n > 0 and min(t.shape) > 1 else float("nan")
    return TestResult(
        comparison=comparison, test="chi2_homogeneity", unit="cell", n=int(n),
        statistic=float(chi2), p_value=float(p), effect_name="cramers_v", effect_size=v,
        note=(note + " same cells appear under every condition; descriptive only").strip(),
    )


# ---------------------------------------------------------------------------
# Composition distance + bootstrap
# ---------------------------------------------------------------------------
def jensen_shannon_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence in bits (0 = identical, 1 = disjoint support).

    Inputs are renormalized to sum to 1; missing/negative entries are treated as 0.
    """
    p = np.nan_to_num(np.asarray(p, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    q = np.nan_to_num(np.asarray(q, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(x, y):
        nz = x > 0
        return float(np.sum(x[nz] * np.log2(x[nz] / np.clip(y[nz], _EPS, None))))

    return float(np.clip(0.5 * _kl(p, m) + 0.5 * _kl(q, m), 0.0, 1.0))


def bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    items: Sequence,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap over ``items`` (resampled with replacement).

    ``items`` should be the *independent* units -- here, regions. Returns
    ``(point_estimate, lo, hi)``; the CI is NaN when fewer than 3 units are
    available or the statistic is undefined on the full sample.
    """
    items = list(items)
    n = len(items)
    point = float(statistic(np.asarray(items, dtype=object)))
    if n < 3 or not np.isfinite(point):
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.asarray(items, dtype=object)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draws[i] = statistic(arr[rng.integers(0, n, size=n)])
    finite = draws[np.isfinite(draws)]
    if finite.size < max(10, n_boot // 20):
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)
