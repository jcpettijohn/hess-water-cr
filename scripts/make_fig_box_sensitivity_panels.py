"""Create the Tier-2 sensitivity summary figure (synthetic; no external data).

This script explores how the *geometry* of the mechanistic complementary
relationship responds to key land--atmosphere adjustment controls in the
mixed-layer box model.

We vary three control axes:
  - mixed-layer depth h (m)
  - relaxation timescale tau (s), with tau_T=tau_q=tau
  - aerodynamic resistance r_a (s m-1)

For each axis value, we sweep the surface resistance r_s from wet to dry and
compute atlas coordinates (x,y) = (E_p0/E_pa, E/E_pa). We then estimate:
  - an effective linear asymmetry coefficient b (least squares through origin), and
  - a curvature proxy: maximum deviation from the endpoint chord between
    (x_min,0) and (1,1) in atlas space.

Outputs
-------
figures/fig_box_sensitivity_panels.png
outputs/box_sensitivity_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Plot styling tuned for manuscript figures (readable after LaTeX scaling) ---
# Copernicus/HESS figures are often reduced in the PDF; use larger default font
# sizes so tick labels remain legible after reduction.
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 15,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2.4,
        "lines.markersize": 6.0,
    }
)

# Ensure project root (…/cr_paper_final_lab) is on the import path so `import src...` works
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.box_model import BoxParams, run_rs_sweep, fit_asymmetry_b
from src.diagnostics import max_chord_deviation

FIGDIR = ROOT / "figures"
OUTDIR = ROOT / "outputs"
FIGDIR.mkdir(parents=True, exist_ok=True)
OUTDIR.mkdir(parents=True, exist_ok=True)


def rs_grid(n: int = 260) -> np.ndarray:
    """Log-spaced r_s sweep from near-wet to strongly limited."""
    return np.logspace(-3, 3.7, n)  # 1e-3 to ~5000 s m-1


def compute_metrics(params: BoxParams, definition: str) -> dict:
    out = run_rs_sweep(params, rs_values=rs_grid(), definition=definition)
    b = fit_asymmetry_b(out["E_pa"], out["E_p0"], out["E"])
    curv = max_chord_deviation(out["x"], out["y"])
    return {
        "b": float(b),
        "curv": float(curv),
        "x_min": float(np.nanmin(out["x"])),
        "x_max": float(np.nanmax(out["x"])),
        "y_min": float(np.nanmin(out["y"])),
        "y_max": float(np.nanmax(out["y"])),
    }


def _panel_label(ax: plt.Axes, s: str) -> None:
    """Readable (a)–(f) panel tag that survives LaTeX downscaling."""
    ax.text(
        0.50,  # centered
        0.95,  # near top
        s,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontweight="bold",
        fontsize=14,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0),
    )


def main() -> None:
    base = BoxParams()

    # Axes values: chosen to span weak-to-strong adjustment without numerical pathologies
    h_vals = np.array([250, 400, 600, 1000, 1600, 2500], dtype=float)
    tau_vals = np.array([1800, 3600, 7200, 14400, 28800, 43200], dtype=float)  # 0.5 h .. 12 h
    ra_vals = np.array([20, 30, 50, 70, 100, 140], dtype=float)

    defs = ["PT_dry", "ML_wetconst"]

    # Legend label wrapping:
    # - "PT from dry-state air" fits on one line at this figure width
    # - keep "(box wet state)" on a second line for readability
    label_map = {
        "PT_dry": "PT from dry-state air",
        "ML_wetconst": "Wet-environment consistent\n(box wet state)",
    }

    records: list[dict] = []

    for definition in defs:
        for h in h_vals:
            params = BoxParams(**{**base.__dict__, "h": float(h)})
            m = compute_metrics(params, definition)
            records.append({"axis": "h", "value": float(h), "definition": definition, **m})

        for tau in tau_vals:
            params = BoxParams(**{**base.__dict__, "tau_q": float(tau), "tau_T": float(tau)})
            m = compute_metrics(params, definition)
            records.append({"axis": "tau", "value": float(tau), "definition": definition, **m})

        for ra in ra_vals:
            params = BoxParams(**{**base.__dict__, "r_a": float(ra)})
            m = compute_metrics(params, definition)
            records.append({"axis": "r_a", "value": float(ra), "definition": definition, **m})

    df = pd.DataFrame.from_records(records)
    df.to_csv(OUTDIR / "box_sensitivity_summary.csv", index=False)

    # ---- Plot: 3 rows x 2 cols (rows: h, tau, r_a; cols: b, curvature) ----
    # Wider-than-tall canvas so the exported figure fills \linewidth efficiently
    # (reduced side whitespace) while keeping fonts readable.
    fig_w = 12.0
    fig_h = 8.7
    fig, axs = plt.subplots(3, 2, figsize=(fig_w, fig_h), constrained_layout=False)

    # Manual spacing (instead of constrained_layout) gives predictable control
    # when y-labels are multi-line and fonts are large.
    fig.subplots_adjust(
        left=0.08,
        right=0.99,
        bottom=0.09,
        top=0.90,
        wspace=0.34,  # extra room for wrapped right-column y-labels
        hspace=0.42,
    )

    def plot_row(ax_b: plt.Axes, ax_c: plt.Axes, axis: str, xlabel: str, xscale: str = "linear") -> None:
        for definition in defs:
            sub = df[(df["axis"] == axis) & (df["definition"] == definition)].sort_values("value")
            ax_b.plot(sub["value"].values, sub["b"].values, marker="o", label=label_map[definition])
            ax_c.plot(sub["value"].values, sub["curv"].values, marker="o", label=label_map[definition])

        for ax in (ax_b, ax_c):
            ax.set_xlabel(xlabel)
            ax.grid(True, alpha=0.30)
            ax.set_xscale(xscale)

        ax_b.set_ylabel(r"inferred $b$")
        # Wrapped label prevents overlap between right-column panels at manuscript-ready font sizes.
        ax_c.set_ylabel("curvature\n(max chord deviation)")

    # Row 1: h
    plot_row(
        axs[0, 0],
        axs[0, 1],
        "h",
        r"mixed-layer depth $h$ (m)",
        xscale="linear",
    )

    # Row 2: tau (log scale)
    plot_row(
        axs[1, 0],
        axs[1, 1],
        "tau",
        r"relaxation timescale $\tau$ (s)",
        xscale="log",
    )

    # Row 3: r_a
    plot_row(
        axs[2, 0],
        axs[2, 1],
        "r_a",
        r"aerodynamic resistance $r_a$ (s m$^{-1}$)",
        xscale="linear",
    )

    # Panel labels: (a)-(f) in reading order
    panel_tags = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    k = 0
    for r in range(3):
        for c in range(2):
            _panel_label(axs[r, c], panel_tags[k])
            k += 1

    # Legend in panel (b): top-right, but nudged down so it sits below the (b)
    # panel label and above the curves.
    #
    # Requested tweak: move it down by ~0.002 on the curvature y-scale.
    ax_leg = axs[0, 1]
    y0, y1 = ax_leg.get_ylim()
    yr = max(y1 - y0, 1e-12)
    dy_axes = 0.002 / yr
    legend_anchor_y = 0.98 - dy_axes  # ~0.002 downward shift in data units

    ax_leg.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.98, legend_anchor_y),
        borderaxespad=0.0,
        handlelength=2.0,
        labelspacing=0.35,
    )

    fig.suptitle(
        "Tier-2 box model: emergent asymmetry and curvature vs atmospheric adjustment",
        fontsize=16,
    )

    # bbox_inches='tight' trims outer whitespace so LaTeX uses page real estate
    # efficiently when the image is constrained by \linewidth.
    fig.savefig(
        FIGDIR / "fig_box_sensitivity_panels.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
