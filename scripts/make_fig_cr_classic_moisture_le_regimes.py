"""Generate a classical Bouchet--Morton moisture-availability CR diagram.

Output: figures/fig_cr_classic_moisture_le_regimes.png

This script produces a single-axis version of the earlier multi-panel figure by
overlaying all three coupling regimes (strong / baseline / weak) on one plot.

Each coupling regime contributes three curves:
  * Actual latent heat flux:            lambda * E
  * Wet benchmark (PT, scaled):         lambda * E_p0
  * Apparent potential (PM, r_s = 0):   lambda * E_pa

Coupling regime is encoded by color; curve type is encoded by line style.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.box_model import BoxParams, run_rs_sweep, priestley_taylor_Ep0
from src.thermo import latent_heat_vaporization
from src.plotting import savefig


def _scaled_pt_LE(params: BoxParams, T_wet_C: float, E_wet: float) -> float:
    """Return constant wet-benchmark LE using PT scaled to match wet equilibrium."""
    base = priestley_taylor_Ep0(params.RnG, T_wet_C, params.p_kpa, alpha_PT=1.0)
    alpha_eff = E_wet / base
    Ep0 = priestley_taylor_Ep0(params.RnG, T_wet_C, params.p_kpa, alpha_PT=alpha_eff)
    return float(latent_heat_vaporization(T_wet_C) * Ep0)


def _curves(params: BoxParams, r_s: np.ndarray) -> dict[str, np.ndarray]:
    """Compute curves for one coupling regime."""
    out = run_rs_sweep(params, r_s, definition="PT_wetref")

    T = out["T_C"]
    lam = latent_heat_vaporization(T)

    LE = lam * out["E"]
    LE_pa = lam * out["E_pa"]

    # Wet benchmark: Priestley–Taylor scaled to match wet equilibrium of this regime.
    T_wet = float(out["T_C"][0])
    E_wet = float(out["E"][0])
    LE_p0 = _scaled_pt_LE(params, T_wet_C=T_wet, E_wet=E_wet)

    # Moisture availability proxy: r_a / (r_a + r_s)
    avail = params.r_a / (params.r_a + out["r_s"])

    # Sort to plot left(dry) -> right(wet)
    idx = np.argsort(avail)
    return {
        "avail": avail[idx],
        "LE": LE[idx],
        "LE_pa": LE_pa[idx],
        "LE_p0": np.full_like(LE[idx], LE_p0),
    }


def main() -> None:
    # Surface resistance sweep used to span dry -> wet conditions.
    rs = np.concatenate(([0.0], np.logspace(-1, 4, 140)))

    regimes: list[tuple[str, BoxParams]] = [
        ("Strong adjustment", BoxParams(h=500.0, tau_T=8 * 3600.0, tau_q=8 * 3600.0)),
        ("Baseline", BoxParams()),
        ("Weak adjustment", BoxParams(h=2000.0, tau_T=1 * 3600.0, tau_q=1 * 3600.0)),
    ]

    # Single panel.
    fig_w = 6.2
    fig_h = 5.9
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0.16, right=0.985, top=0.955, bottom=0.13)

    # Square plot box (matches style of the original stacked panels).
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(1.0)

    # Regime colors.
    regime_colors: dict[str, str] = {
        "Strong adjustment": "C0",
        "Baseline": "C1",
        "Weak adjustment": "C2",
    }

    # Curve styles (same across regimes).
    style_actual = dict(ls="-", lw=2.6)
    style_bench = dict(ls="--", lw=2.6)
    style_potential = dict(ls="-.", lw=2.6)

    # Plot all regimes.
    for title, params in regimes:
        C = _curves(params, rs)
        color = regime_colors[title]
        ax.plot(C["avail"], C["LE"], color=color, **style_actual)
        ax.plot(C["avail"], C["LE_p0"], color=color, **style_bench)
        ax.plot(C["avail"], C["LE_pa"], color=color, **style_potential)

    # Axes / labels.
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Regional moisture availability (proxy)")
    ax.set_ylabel(r"Latent heat flux $\lambda E$ (W m$^{-2}$)")

    # Dry -> wet arrow annotation.
    arrow_y_frac = 0.93
    ax.annotate(
        "",
        xy=(0.1, arrow_y_frac),
        xytext=(0.9, arrow_y_frac),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1.6),
    )
    ax.text(0.02, arrow_y_frac, "dry", transform=ax.transAxes, va="center")
    ax.text(0.92, arrow_y_frac, "wet", transform=ax.transAxes, va="center")

    # Legends: one for curve type (line style), one for regime (color).
    curve_handles = [
        Line2D([0], [0], color="k", **style_actual),
        Line2D([0], [0], color="k", **style_bench),
        Line2D([0], [0], color="k", **style_potential),
    ]
    curve_labels = [
        r"Actual $\lambda E$",
        r"Wet benchmark $\lambda E_{p0}$ (PT, scaled)",
        r"Apparent potential $\lambda E_{pa}$ (PM, $r_s=0$)",
    ]

    regime_handles = [
        Line2D([0], [0], color=regime_colors[name], lw=3.0, ls="-")
        for name, _ in regimes
    ]
    regime_labels = [name for name, _ in regimes]

    # Move the coupling-regime legend down so it does not overlap the dry<->wet arrow.
    # The request is to position the *top* of this legend around 260–265 W m^-2.
    # We anchor the legend using data coordinates and clamp into that range while
    # also ensuring it sits below the arrow line.
    ymin, ymax = ax.get_ylim()
    y_arrow_data = ymin + arrow_y_frac * (ymax - ymin)
    y_regime_top = y_arrow_data - 6.0  # keep a small gap below the arrow
    y_regime_top = min(265.0, y_regime_top)
    y_regime_top = max(260.0, y_regime_top)
    # Keep anchor inside the visible y-range (defensive).
    y_regime_top = min(y_regime_top, ymax - 1e-6)
    y_regime_top = max(y_regime_top, ymin + 1e-6)

    leg_regime = ax.legend(
        regime_handles,
        regime_labels,
        title="Coupling regime",
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, y_regime_top),
        bbox_transform=ax.transData,
        borderaxespad=0.0,
    )
    ax.add_artist(leg_regime)

    ax.legend(
        curve_handles,
        curve_labels,
        title="Curve",
        frameon=False,
        loc="lower right",
    )

    outpath = ROOT / "figures" / "fig_cr_classic_moisture_le_regimes.png"
    savefig(fig, outpath)
    plt.close(fig)


if __name__ == "__main__":
    main()
