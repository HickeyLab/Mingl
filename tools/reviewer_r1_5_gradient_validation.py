"""Reviewer R1.5 validation: gradient/transition steepness on synthetic data.

Generates the ground-truth-recovery and sensitivity-analysis evidence requested by
Reviewer #1 (R1.5) for the MINGL gradient/transition-clustering workflow:

* synthetic tissues with KNOWN sharp/medium/gradual transitions,
* a check that recovered steepness tracks the ground-truth sharpness,
* a run-to-run determinism check, and
* a one-at-a-time sensitivity sweep over neighbor count (k), number of probability
  bins, number of transition clusters, and spatial sampling density.

Outputs CSV tables and PNG figures to ``tools/reviewer_r1_5_outputs/`` and prints a
summary. Nothing here modifies the shipped pipeline; it only exercises it.

Usage:
    python tools/reviewer_r1_5_gradient_validation.py [--n-cells 3000] [--repeats 5]
"""

from __future__ import annotations

import argparse
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mingl.tl as tl

warnings.filterwarnings("ignore")

LEVELS = ("sharp", "medium", "gradual")
LEVEL_COLORS = {"sharp": "#d62728", "medium": "#ff7f0e", "gradual": "#1f77b4"}


def ground_truth_recovery(out_dir, n_cells, repeats, seed):
    df = tl.validate_ground_truth_recovery(
        sharpness_levels=LEVELS, n_cells=n_cells, n_repeats=repeats, seed=seed
    )
    df.to_csv(os.path.join(out_dir, "ground_truth_recovery.csv"), index=False)

    summ = (
        df.groupby("sharpness_label")
        .agg(
            ground_truth_sharpness=("ground_truth_sharpness", "first"),
            steepness_mean=("steepness", "mean"),
            steepness_std=("steepness", "std"),
        )
        .reset_index()
        .sort_values("ground_truth_sharpness")
    )

    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
    for lvl in LEVELS:
        sub = df[df["sharpness_label"] == lvl]
        ax.scatter(
            sub["ground_truth_sharpness"], sub["steepness"],
            color=LEVEL_COLORS[lvl], label=lvl, s=40, alpha=0.85, edgecolor="k", linewidth=0.4,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Ground-truth sharpness (1 / transition width fraction)")
    ax.set_ylabel("Recovered steepness (mean |2nd derivative|)")
    ax.set_title("Steepness recovers ground-truth transition sharpness")
    ax.legend(title="Transition", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ground_truth_recovery.png"))
    plt.close(fig)
    return df, summ


def determinism_check(n_cells, seed):
    adata = tl.simulate_transition_tissue(n_cells=n_cells, sharpness="medium", seed=seed + 99)
    r1 = tl.run_gradient_transition(adata, seed=0)
    r2 = tl.run_gradient_transition(adata, seed=0)
    return {
        "steepness_run1": r1["steepness"],
        "steepness_run2": r2["steepness"],
        "identical": bool(r1["steepness"] == r2["steepness"]),
    }


def sensitivity(out_dir, n_cells, seed):
    frames = []
    for lvl in LEVELS:
        adata = tl.simulate_transition_tissue(n_cells=n_cells, sharpness=lvl, seed=seed)
        frames.append(tl.gradient_sensitivity_analysis(adata, seed=seed))
    alldf = pd.concat(frames, ignore_index=True)
    alldf.to_csv(os.path.join(out_dir, "sensitivity_analysis.csv"), index=False)

    params = ["k", "n_bins", "n_clusters", "subsample_frac"]
    fig, axes = plt.subplots(1, len(params), figsize=(4 * len(params), 3.6), dpi=200)
    for ax, pname in zip(axes, params):
        sub = alldf[alldf["param"] == pname]
        for lvl in LEVELS:
            s = sub[sub["transition"] == lvl].sort_values("value")
            ax.plot(s["value"], s["steepness"], marker="o", color=LEVEL_COLORS[lvl], label=lvl)
        ax.set_title(pname)
        ax.set_xlabel(pname)
        ax.set_ylabel("steepness")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(title="Transition", frameon=False, fontsize=8)
    fig.suptitle("Steepness sensitivity to analysis parameters (one-at-a-time)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sensitivity_analysis.png"))
    plt.close(fig)

    # Ordering robustness: does sharp > gradual hold at each setting?
    piv = alldf.pivot_table(index=["param", "value"], columns="transition", values="steepness")
    piv = piv[[c for c in LEVELS if c in piv.columns]]
    ordering_ok = (piv["sharp"] > piv["gradual"])
    return alldf, piv, ordering_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cells", type=int, default=3000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "reviewer_r1_5_outputs"),
    )
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Output directory: {args.out_dir}\n")

    print("== 1. Ground-truth recovery ==")
    _, summ = ground_truth_recovery(args.out_dir, args.n_cells, args.repeats, args.seed)
    print(summ.to_string(index=False))

    print("\n== 2. Determinism (same input, two runs) ==")
    det = determinism_check(args.n_cells, args.seed)
    print(det)

    print("\n== 3. Sensitivity analysis ==")
    _, piv, ordering_ok = sensitivity(args.out_dir, args.n_cells, args.seed)
    print(piv.round(3).to_string())
    print(
        f"\nsharp > gradual holds at {int(ordering_ok.sum())}/{len(ordering_ok)} parameter settings"
    )
    violations = piv[~ordering_ok]
    if len(violations):
        print("Ordering violations:")
        print(violations.round(3).to_string())

    print(f"\nArtifacts written to {args.out_dir}")


if __name__ == "__main__":
    main()
