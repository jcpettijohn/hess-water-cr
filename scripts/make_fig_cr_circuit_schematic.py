"""scripts/make_fig_cr_circuit_schematic.py

Create a circuit-style schematic for the CR resistance/transport analogy.

This version is layout-safe (no overlapping text) and uses consistent fonts and
weights for a more professional appearance.

Outputs
-------
- figures/fig_cr_circuit_schematic.png
- outputs/fig_cr_circuit_schematic.png
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from src.plotting import savefig


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )


def _save_to_targets(fig, filename: str, dpi: int = 300) -> None:
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "outputs").mkdir(exist_ok=True)
    savefig(fig, ROOT / "figures" / filename, dpi=dpi)
    savefig(fig, ROOT / "outputs" / filename, dpi=dpi)


def _add_resistor(ax, x: float, y: float, w: float, h: float, *, lw: float = 2.2) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=lw,
        edgecolor="0.15",
        facecolor="white",
        zorder=2,
    )
    ax.add_patch(patch)


def main() -> None:
    _set_style()

    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # --- Core geometry ---
    y_wire = 0.70
    x_right = 0.93

    # Battery plates
    ax.add_patch(
        Rectangle((0.030, 0.52), 0.015, 0.36, fill=False, linewidth=2.2, edgecolor="0.15")
    )
    ax.add_patch(
        Rectangle((0.052, 0.56), 0.015, 0.28, fill=False, linewidth=2.2, edgecolor="0.15")
    )

    # Connect battery to the main wire
    ax.plot([0.067, 0.12], [y_wire, y_wire], linewidth=3.0, color="0.15")

    # Resistor geometry
    # Slightly widen r_a so the '(aerodynamic)' label fits comfortably.
    ra_x, ra_w = 0.24, 0.18
    gap = 0.07
    rs_w = 0.15
    rs_x = ra_x + ra_w + gap
    box_h = 0.18
    box_y = y_wire - box_h / 2

    # Wire up to r_a
    ax.plot([0.12, ra_x], [y_wire, y_wire], linewidth=3.0, color="0.15")

    # r_a (aerodynamic)
    _add_resistor(ax, ra_x, box_y, ra_w, box_h)
    ax.text(
        ra_x + ra_w / 2,
        y_wire,
        "$r_a$\n(aerodynamic)",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="semibold",
    )

    # Wire between r_a and r_s
    ax.plot([ra_x + ra_w, rs_x], [y_wire, y_wire], linewidth=3.0, color="0.15")

    # r_s (surface)
    _add_resistor(ax, rs_x, box_y, rs_w, box_h)
    ax.text(
        rs_x + rs_w / 2,
        y_wire,
        "$r_s$\n(surface)",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="semibold",
    )
    ax.text(
        rs_x + rs_w / 2,
        box_y - 0.08,
        r"(wet $\rightarrow$ dry)",
        ha="center",
        va="center",
        fontsize=12,
        color="0.25",
    )

    # Output wire
    ax.plot([rs_x + rs_w, x_right], [y_wire, y_wire], linewidth=3.0, color="0.15")

    # Flux arrow and label
    x_arrow0 = rs_x + rs_w + 0.10
    x_arrow1 = x_right - 0.01
    ax.add_patch(
        FancyArrowPatch(
            (x_arrow0, y_wire + 0.02),
            (x_arrow1, y_wire + 0.02),
            arrowstyle="->",
            mutation_scale=18,
            linewidth=2.2,
            color="0.15",
        )
    )
    ax.text(
        0.5 * (x_arrow0 + x_arrow1),
        y_wire + 0.09,
        r"$E$ (actual)",
        ha="center",
        va="bottom",
        fontsize=16,
        fontweight="semibold",
    )

    # Driving potential
    # Move this label above the circuit and add a pointer so it's clear it drives
    # the left side of the circuit (battery / initial wire segment).
    drive_x, drive_y = 0.09, 0.95
    ax.text(
        drive_x,
        drive_y,
        "Driving potential\n(humidity/temperature gradient)",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="semibold",
    )
    ax.add_patch(
        FancyArrowPatch(
            (drive_x + 0.10, drive_y - 0.12),
            (0.050, y_wire),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=2.0,
            color="0.15",
        )
    )

    # Definitions / diagnostics (bottom)
    # Increase size and bold the symbols so they are readable at figure scale.
    defs_fs = 14.5
    ax.text(
        0.18,
        0.20,
        r"$\mathbf{E_{pa}}$: set $r_s=0$\n(saturated patch)\n(same air state)",
        ha="left",
        va="center",
        fontsize=defs_fs,
        linespacing=1.15,
    )
    ax.text(
        0.61,
        0.20,
        r"$\mathbf{E_{p0}}$: wet-environment reference\n(counterfactual)\n(fully wet region)",
        ha="left",
        va="center",
        fontsize=defs_fs,
        linespacing=1.15,
    )

    _save_to_targets(fig, "fig_cr_circuit_schematic.png", dpi=300)


if __name__ == "__main__":
    main()
