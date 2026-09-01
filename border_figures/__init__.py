"""Modular figure package for MINGL border analyses.

Two self-contained figures, each with its own driver:

``figure1_emission_models``
    Contrast alternative mixture/emission distributions against the shipped
    MINGL GMM: border location, border cell-type composition, and assigned
    neighborhood probability distributions.
``figure2_threshold_effects``
    How the probability threshold changes border location, border-cell number,
    border composition and border cell-type enrichment, across a null condition,
    three hierarchical scales of the intestine, and a second platform.

Shared pieces: :mod:`config` (dataset/level/pair presets), :mod:`loading`
(reading, subsampling, nulls, cached posteriors), :mod:`pair_borders` (the
manuscript's pair-specific border definitions plus the interface coordinate),
:mod:`stats` (region-paired tests) and :mod:`plotting` (style helpers).

Nothing here mutates the package under ``src/mingl`` or writes anywhere except
this folder's ``outputs/``.
"""

from __future__ import annotations

__all__ = ["config", "loading", "pair_borders", "plotting", "stats"]
