"""Dataset / hierarchy-level / border-pair presets for the border figures.

Everything that is dataset-specific lives here so the figure scripts stay generic.

Hierarchy levels follow the manuscript Methods 4.2 exactly: each level scores a
cell's window of the *lower* level's labels, with a level-specific ``k``::

    cellular neighborhood : k = 10  window of  cell-type   labels
    community             : k = 100 window of  neighborhood labels
    tissue unit           : k = 300 window of  community    labels

This matters -- the shipped reviewer drivers re-score every level with the
cell-type window at ``k=10``, which is not the hierarchy the paper defines. Here
``feature_col``/``k`` are per level, while ``cell_type_col`` (used for border
*composition* and *enrichment* readouts) always stays at the cell-type level, as
in Figure 2k.

Column names in ``DATASETS`` were verified against the real lab files rather than
assumed; ``loading.load_dataset`` re-validates them on read and fails loudly if a
file's schema has drifted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "LevelSpec",
    "DatasetSpec",
    "DATASETS",
    "DEFAULT_PAIRS",
    "MODELS",
    "MODEL_LABELS",
    "MODEL_COLORS",
    "BASELINE_MODEL",
    "DEFAULT_THRESHOLDS",
    "REFERENCE_THRESHOLD",
    "MANUSCRIPT_REGIONS",
    "manuscript_region",
    "get_level",
    "default_pair",
]


# ---------------------------------------------------------------------------
# Hierarchy levels
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LevelSpec:
    """One hierarchical organizational level of a dataset.

    Attributes
    ----------
    name
        Short key used on the command line and in output paths.
    unit_col
        ``obs`` column holding the discrete organizational-unit label for this
        level (the mixture components MINGL scores against).
    feature_col
        ``obs`` column whose labels are counted in each cell's k-NN window (the
        level *below* this one).
    k
        Number of nearest neighbors in the window (Methods 4.2).
    label
        Human-readable name for figure titles.
    """

    name: str
    unit_col: str
    feature_col: str
    k: int
    label: str


@dataclass(frozen=True)
class DatasetSpec:
    """Column-name preset and available hierarchy levels for one dataset."""

    name: str
    label: str
    cell_type_col: str
    region_key: str
    x_key: str = "x"
    y_key: str = "y"
    levels: dict[str, LevelSpec] = field(default_factory=dict)
    default_level: str = "neighborhood"
    #: Rough cell count of the shipped file; used only to warn about runtime.
    approx_n_cells: int | None = None


DATASETS: dict[str, DatasetSpec] = {
    "intestine": DatasetSpec(
        name="intestine",
        label="Healthy human intestine (CODEX)",
        cell_type_col="Cell Type",
        region_key="unique_region",
        approx_n_cells=2_512_002,
        levels={
            "neighborhood": LevelSpec(
                "neighborhood", "Neighborhood", "Cell Type", 10, "Cellular neighborhood"
            ),
            "community": LevelSpec(
                "community", "Community", "Neighborhood", 100, "Community"
            ),
            "tissue_unit": LevelSpec(
                "tissue_unit", "Tissue Unit", "Community", 300, "Tissue unit"
            ),
        },
    ),
    "melanoma": DatasetSpec(
        name="melanoma",
        label="Human melanoma (CODEX)",
        cell_type_col="Cell_Type",
        region_key="filename",
        approx_n_cells=5_019_159,
        levels={
            "neighborhood": LevelSpec(
                "neighborhood", "Neighborhood", "Cell_Type", 10, "Cellular neighborhood"
            ),
        },
    ),
    # Mouse brain spatial-transcriptomics dataset (Reviewer Response/Data/
    # Spatial Transcriptomics/..._n5_neighborhood_annotations.csv). Verified
    # 2026-08-15: 378,918 cells, 31 regions (min 7,215 cells), 13 cell types,
    # 4 neighborhoods. Only one annotated hierarchy level.
    "spatial": DatasetSpec(
        name="spatial",
        label="Mouse brain (spatial transcriptomics)",
        cell_type_col="cell_type",
        region_key="unique_region",
        approx_n_cells=378_918,
        levels={
            "neighborhood": LevelSpec(
                "neighborhood", "Neighborhood_5", "cell_type", 10, "Cellular neighborhood"
            ),
            # Anatomical regions (striatum, cortical layers, corpus callosum, ...).
            # 8 units, verified 2026-08-20. This is the level that carries named
            # anatomy, so it is the mouse-brain analogue of the intestine's
            # tissue units rather than of its MINGL neighborhoods.
            "tissue": LevelSpec(
                "tissue", "tissue", "cell_type", 10, "Anatomical region"
            ),
        },
        default_level="tissue",
    ),
    # Barrett's esophagus -- the previous "spatial" dataset, kept addressable by
    # name so earlier runs stay reproducible.
    "esophagus": DatasetSpec(
        name="esophagus",
        label="Barrett's esophagus (CODEX)",
        cell_type_col="Cell Type",
        region_key="region",
        approx_n_cells=645_661,
        levels={
            "neighborhood": LevelSpec(
                "neighborhood", "neigh_name", "Cell Type", 10, "Cellular neighborhood"
            ),
            "community": LevelSpec(
                "community", "community", "neigh_name", 100, "Community"
            ),
        },
    ),
    # Local, lab-data-free validation target (see loading.synthetic_tissue).
    "synthetic": DatasetSpec(
        name="synthetic",
        label="Synthetic tissue",
        cell_type_col="Cell Type",
        region_key="unique_region",
        levels={
            "neighborhood": LevelSpec(
                "neighborhood", "Neighborhood", "Cell Type", 10, "Cellular neighborhood"
            ),
            "community": LevelSpec(
                "community", "Community", "Neighborhood", 30, "Community"
            ),
            "tissue_unit": LevelSpec(
                "tissue_unit", "Tissue Unit", "Community", 60, "Tissue unit"
            ),
        },
    ),
}

_DATASET_ALIASES = {
    "barretts": "esophagus",
    "mousebrain": "spatial",
    "mouse_brain": "spatial",
    "brain": "spatial",
    "spatial_transcriptomics": "spatial",
    "gut": "intestine",
}


def resolve_dataset(name: str) -> DatasetSpec:
    """Look up a :class:`DatasetSpec`, accepting the common aliases."""
    key = str(name).strip().lower()
    key = _DATASET_ALIASES.get(key, key)
    if key not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Choose from {sorted(DATASETS)}.")
    return DATASETS[key]


def get_level(dataset: str, level: str | None = None) -> LevelSpec:
    """Return the :class:`LevelSpec` for ``dataset`` (default level if ``None``)."""
    spec = resolve_dataset(dataset)
    key = (level or spec.default_level).strip().lower().replace(" ", "_").replace("-", "_")
    if key not in spec.levels:
        raise KeyError(
            f"Dataset {spec.name!r} has no level {level!r}. Available: {sorted(spec.levels)}."
        )
    return spec.levels[key]


# ---------------------------------------------------------------------------
# Border pairs
# ---------------------------------------------------------------------------
#: One focus border per (dataset, level), taken from the manuscript where it
#: names one. ``None`` means "auto-select the pair sharing the most border cells"
#: (see :func:`border_figures.pair_borders.select_top_pair`). Every default is
#: verified against the data at run time and falls back to auto-selection with a
#: printed warning if the unit names are absent.
DEFAULT_PAIRS: dict[tuple[str, str], tuple[str, str] | None] = {
    # Chosen deliberately, not auto-selected: composition and enrichment are only
    # interpretable against a named interface, and an auto-picked pair makes the
    # panels incomparable across conditions.
    ("intestine", "neighborhood"): ("Inner Follicle", "Outer Follicle"),
    ("intestine", "community"): ("Secretory Epithelial", "Plasma Cell Enriched"),
    ("intestine", "tissue_unit"): ("Mucosa", "Muscularis mucosa"),
    ("melanoma", "neighborhood"): ("Inflamed Tumor", "Productive T cell & Tumor"),
    # Grey matter meeting white matter -- a real anatomical interface. Lives in
    # the `tissue` column, not Neighborhood_5.
    ("spatial", "tissue"): ("cortical layer VI", "corpus callosum"),
    ("spatial", "neighborhood"): None,
    ("esophagus", "neighborhood"): None,
    ("esophagus", "community"): None,
    ("synthetic", "tissue_unit"): ("Mucosa", "Muscularis Mucosa"),
    ("synthetic", "neighborhood"): None,
    ("synthetic", "community"): None,
}


def default_pair(dataset: str, level: str) -> tuple[str, str] | None:
    """Manuscript-derived default border pair for a dataset/level, if any."""
    spec = resolve_dataset(dataset)
    return DEFAULT_PAIRS.get((spec.name, level))


# ---------------------------------------------------------------------------
# Emission models
# ---------------------------------------------------------------------------
#: ``diagonal_gaussian`` *is* the shipped MINGL scorer -- the "our GMM" baseline
#: every other distribution is contrasted against.
BASELINE_MODEL = "diagonal_gaussian"

MODELS: tuple[str, ...] = (
    "diagonal_gaussian",
    "full_gaussian",
    "multinomial",
    "dirichlet_multinomial",
    "logistic_normal",
)

MODEL_LABELS: dict[str, str] = {
    "diagonal_gaussian": "MINGL GMM\n(diagonal)",
    "full_gaussian": "Gaussian\n(full cov.)",
    "multinomial": "Multinomial",
    "dirichlet_multinomial": "Dirichlet-\nmultinomial",
    "logistic_normal": "Logistic-\nnormal",
}

#: Baseline in a neutral dark grey, alternatives in a colorblind-safe sequence.
MODEL_COLORS: dict[str, str] = {
    "diagonal_gaussian": "#4d4d4d",
    "full_gaussian": "#0173b2",
    "multinomial": "#029e73",
    "dirichlet_multinomial": "#d55e00",
    "logistic_normal": "#cc78bc",
}


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
#: The manuscript sweeps 0 -> 0.25 (Fig. 2g/h) and the reviewer response extends
#: to 0.49 (above 0.5 two probabilities can no longer both be positive, so no
#: border cell can exist).
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.33, 0.4, 0.49)

#: Region plotted in the manuscript's own spatial figures, so the catplots show
#: the same tissue the paper discusses in detail. Verified from the tutorial
#: notebooks: fig2_melanoma_neighborhood.ipynb calls spatial_loc_region with
#: "05_06_23_reg003.tsv" ("14_06_23_reg002.tsv" is commented out beside it);
#: fig2_intestine_tissue_unit.ipynb uses "B008_Sigmoid".
MANUSCRIPT_REGIONS: dict[str, str] = {
    # Verified spans: reg002 is 22,251 x 23,475 (aspect 0.95, square) while
    # reg003 is 18,044 x 9,964 (aspect 1.81). The manuscript panels use the
    # square one; reg003 appears uncommented in the notebook but renders wide.
    "melanoma": "14_06_23_reg002.tsv",
    "intestine": "B008_Sigmoid",
}


def manuscript_region(dataset: str) -> str | None:
    """Region the manuscript plots for this dataset, if it names one."""
    return MANUSCRIPT_REGIONS.get(resolve_dataset(dataset).name)


#: The threshold used throughout the manuscript; every "vs reference" statistic
#: is computed against this one.
REFERENCE_THRESHOLD = 0.25
