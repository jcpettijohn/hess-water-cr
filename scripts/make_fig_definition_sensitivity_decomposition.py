"""
scripts/make_fig_definition_sensitivity_decomposition.py

Generates:
  figures/fig_definition_sensitivity_decomposition.png
  outputs/fig_definition_sensitivity_decomposition.csv

Purpose:
  Exact decomposition of definition-induced shifts in
    x = E_p0 / E_pa

Relative to a chosen baseline definition (default: PT_wetref), for each sampled state i:
  x0_i = Ep0_0_i / Epa_0_i          (baseline)
  x1_i = Ep0_1_i / Epa_1_i          (alternative)

Total shift:
  Δx = x1 - x0

Exact "two-factor" decomposition:
  Δx_p0  = (Ep0_1 / Epa_0) - (Ep0_0 / Epa_0)      [change Ep0 only, hold Epa at baseline]
  Δx_pa  = (Ep0_0 / Epa_1) - (Ep0_0 / Epa_0)      [change Epa only, hold Ep0 at baseline]
  Δx_int = Δx - (Δx_p0 + Δx_pa)                   [interaction remainder; exact closure]

Notes:
  With your current Tier-1B mechanistic test (PT_dry vs PT_wetref),
  E_pa is defined identically across the two runs, so Δx_pa ≈ 0 and Δx_int ≈ 0,
  making the "Ep0 dominates" claim *quantitative*.

"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.box_model import BoxParams, run_rs_sweep

try:
    from src.plotting import savefig  # your repo helper
except Exception:
    savefig = None


def decompose_dx(Ep0_base, Epa_base, Ep0_alt, Epa_alt):
    Ep0_base = np.asarray(Ep0_base, dtype=float)
    Epa_base = np.asarray(Epa_base, dtype=float)
    Ep0_alt = np.asarray(Ep0_alt, dtype=float)
    Epa_alt = np.asarray(Epa_alt, dtype=float)

    x_base = Ep0_base / Epa_base
    x_alt = Ep0_alt / Epa_alt
    dx_total = x_alt - x_base

    dx_p0 = (Ep0_alt / Epa_base) - (Ep0_base / Epa_base)
    dx_pa = (Ep0_base / Epa_alt) - (Ep0_base / Epa_base)
    dx_int = dx_total - (dx_p0 + dx_pa)

    return x_base, x_alt, dx_total, dx_p0, dx_pa, dx_int


def main():
    params = BoxParams()

    # Match your existing definition-sensitivity sampling:
    # rs includes 0 plus logspace(0,4,40). (This is what your current script uses.)
    rs = np.concatenate([np.array([0.0]), np.logspace(0, 4, 40)])

    baseline_def = "PT_wetref"
    alt_defs = ["PT_dry", "ML_wetconst"]  # ML_wetconst will be skipped if not implemented

    out_base = run_rs_sweep(params, rs, definition=baseline_def)
    Ep0_base = out_base["E_p0"]
    Epa_base = out_base["E_pa"]

    # We'll plot Δx components vs baseline x (sorted), which avoids log(rs) issues and is atlas-relevant.
    fig, axes = plt.subplots(
        nrows=len(alt_defs),
        ncols=1,
        figsize=(7.2, 3.3 * len(alt_defs)),
        sharex=True,
    )
    if len(alt_defs) == 1:
        axes = [axes]

    rows = []
    used_axes = 0

    for ddef in alt_defs:
        try:
            out_alt = run_rs_sweep(params, rs, definition=ddef)
        except Exception as e:
            print(f"[skip] definition={ddef!r} not available or failed: {e}")
            continue

        Ep0_alt = out_alt["E_p0"]
        Epa_alt = out_alt["E_pa"]

        x_base, x_alt, dx_total, dx_p0, dx_pa, dx_int = decompose_dx(
            Ep0_base, Epa_base, Ep0_alt, Epa_alt
        )

        # Archive raw decomposition values (one row per sampled state)
        for i in range(len(rs)):
            rows.append(
                {
                    "definition_baseline": baseline_def,
                    "definition_alt": ddef,
                    "rs": float(rs[i]),
                    "x_base": float(x_base[i]),
                    "x_alt": float(x_alt[i]),
                    "dx_total": float(dx_total[i]),
                    "dx_p0": float(dx_p0[i]),
                    "dx_pa": float(dx_pa[i]),
                    "dx_int": float(dx_int[i]),
                }
            )

        ax = axes[used_axes]
        order = np.argsort(x_base)
        xb = x_base[order]

        ax.axhline(0.0, linewidth=1.0)
        markevery = 3  # show markers often enough to reveal overlaps without clutter

        ax.plot(
            xb, dx_total[order],
            linewidth=2.0,
            marker="o", markersize=3, markevery=markevery,
            label=r"$\Delta x$ (total)"
        )
        ax.plot(
            xb, dx_p0[order],
            linewidth=1.6, linestyle="--",
            marker="x", markersize=3, markevery=markevery,
            label=r"$\Delta x_{p0}$ (vary $E_{p0}$ only)"
        )
        ax.plot(
            xb, dx_pa[order],
            linewidth=1.3, linestyle=":",
            marker="o", markersize=3, markerfacecolor="none", markevery=markevery,
            label=r"$\Delta x_{pa}$ (vary $E_{pa}$ only)"
        )
        ax.plot(
            xb, dx_int[order],
            linewidth=1.3, linestyle="-.",
            marker="+", markersize=4, markevery=markevery,
            label=r"$\Delta x_{\mathrm{int}}$ (remainder)"
        )


        ax.set_ylabel(r"$\Delta x$")
        ax.set_title(f"{ddef} relative to {baseline_def}")
        ax.grid(True, alpha=0.3)

        if used_axes == 0:
            ax.legend(frameon=False, fontsize=9)

        used_axes += 1

    if used_axes == 0:
        raise RuntimeError(
            "No alternative definitions could be plotted. "
            "Edit alt_defs or confirm run_rs_sweep supports those definition strings."
        )

    # Remove unused axes if any definitions were skipped
    for k in range(used_axes, len(axes)):
        fig.delaxes(axes[k])

    # Label x-axis on final used axis
    axes[used_axes - 1].set_xlabel(r"baseline $x = E_{p0}/E_{pa}$")

    fig.suptitle(r"Tier-1B definition sensitivity: exact decomposition of $\Delta x$", y=0.985)
    fig.tight_layout()

    out_csv = ROOT / "outputs" / "fig_definition_sensitivity_decomposition.csv"
    out_png = ROOT / "figures" / "fig_definition_sensitivity_decomposition.png"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[write] {out_csv}")

    if savefig is not None:
        savefig(fig, out_png)
    else:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[write] {out_png}")


if __name__ == "__main__":
    main()
