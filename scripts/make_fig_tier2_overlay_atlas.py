"""Overlay Tier-2 (box-model) mechanistic curves on the Tier-1 atlas.

This is the 'closure' figure that ties the three-tier story together:
  - Tier 1: published CR curve families in nondimensional (x,y) space
  - Tier 2: mechanistic curves emergent from tunable land--atmosphere coupling

Visually, the figure demonstrates which regions of the atlas correspond to
strong vs weak atmospheric adjustment (mixing depth, relaxation, ventilation).

Outputs
-------
figures/fig_tier2_overlay_atlas.png
outputs/tier2_overlay_curves.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # <-- added

from src.cr_models import default_model_set
from src.box_model import BoxParams, run_rs_sweep, fit_asymmetry_b
from src.diagnostics import max_chord_deviation
from src.plotting import savefig


def _summarize_curve(label: str, out: dict, params: BoxParams) -> dict:
    b = fit_asymmetry_b(out["E_pa"], out["E_p0"], out["E"])
    dev = max_chord_deviation(out["x"], out["y"])
    return {
        "label": label,
        "b_fit": float(b),
        "chord_dev": float(dev),
        "h_m": params.h,
        "tau_s": params.tau_q,
        "r_a": params.r_a,
        "alpha_PT": params.alpha_PT,
    }


def main():
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)

    # --- Tier 1: atlas background ---
    models = default_model_set()

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.set_xlabel(r"$x = E_{p0}/E_{pa}$")
    ax.set_ylabel(r"$y = E/E_{pa}$")
    ax.set_title("Tier-2 mechanistic curves overlaid on Tier-1 CR atlas")
    ax.grid(True, alpha=0.25)

    # Plot atlas curves (thin gray) WITHOUT legend labels (we'll add a proxy handle later)
    for m in models:
        x = np.linspace(m.x_min, m.x_max, 800)
        y = m.func(x)
        ax.plot(x, np.clip(y, 0.0, 1.25), linewidth=1.0, alpha=0.7, color="0.75")

    ax.axhline(1.0, linewidth=0.8, color="k", alpha=0.35, linestyle="--")
    ax.axvline(1.0, linewidth=0.8, color="k", alpha=0.35, linestyle="--")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(-0.02, 1.28)

    # --- Tier 2: selected coupling regimes ---
    rs_values = np.logspace(-3, 3.7, 260)

    regimes: list[tuple[str, BoxParams]] = [
        (
            "Strong adjustment: shallow h, moderate ventilation",
            BoxParams(h=300.0, tau_q=21600.0, tau_T=21600.0, r_a=30.0),
        ),
        (
            "Intermediate adjustment: baseline",
            BoxParams(h=1000.0, tau_q=21600.0, tau_T=21600.0, r_a=50.0),
        ),
        (
            "Weak adjustment: deep h, faster relaxation, weaker ventilation",
            BoxParams(h=2000.0, tau_q=7200.0, tau_T=7200.0, r_a=70.0),
        ),
    ]

    curve_rows = []
    summary_rows = []
    mech_lines = []  # <-- store thick regime line handles for legend

    for label, params in regimes:
        out = run_rs_sweep(params, rs_values=rs_values, definition="ML_wetconst")
        x = out["x"]
        y = out["y"]

        # thick mechanistic curve (store handle)
        (line,) = ax.plot(x, y, linewidth=3.0, alpha=0.9, label=label)
        mech_lines.append(line)

        # markers (these are the orange circles you wanted to explain in the legend)
        ax.plot(
            x[::24],
            y[::24],
            marker="o",
            linestyle="None",
            markersize=3.0,
            alpha=0.9,
        )

        # write raw curve points
        for xi, yi in zip(x, y):
            curve_rows.append({"regime": label, "x": float(xi), "y": float(yi)})

        summary_rows.append(_summarize_curve(label, out, params))

    # --- Proxy handles for legend (gray atlas + orange circles) ---
    atlas_proxy = Line2D(
        [0],
        [0],
        color="0.75",
        lw=2.5,       # thicker than actual atlas curves so it's visible in legend
        alpha=0.9,
    )

    orange_circles_proxy = Line2D(
        [0],
        [0],
        linestyle="None",
        marker="o",
        markersize=4.0,
        color="C1",   # Matplotlib default orange
        alpha=0.9,
    )

    handles = [atlas_proxy] + mech_lines + [orange_circles_proxy]
    labels = (
        ["Tier-1 atlas (published CR families)"]
        + [ln.get_label() for ln in mech_lines]
        + ["Sampled sweep points (markers)"]
    )

    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        fontsize=7,
        ncols=1,          # stacked vertically
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.4,
        handlelength=2.2,
        handletextpad=0.6,
    )

    savefig(fig, ROOT / "figures" / "fig_tier2_overlay_atlas.png", dpi=300)

    pd.DataFrame(curve_rows).to_csv(ROOT / "outputs" / "tier2_overlay_curves.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(ROOT / "outputs" / "tier2_overlay_summary.csv", index=False)


if __name__ == "__main__":
    main()
