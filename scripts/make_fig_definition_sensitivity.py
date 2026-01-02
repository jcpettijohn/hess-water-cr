"""
scripts/make_fig_definition_sensitivity.py

Illustrate "definition sensitivity" (Hypothesis H1) using the same box-model-generated air states
but two Ep0 definitions:

- PT_dry: Ep0 computed using the dry-state temperature at each r_s
- PT_wetref: Ep0 computed using the wet-reference (r_s=0) temperature

Output:
  outputs/fig_definition_sensitivity_xy.png
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.box_model import BoxParams, run_rs_sweep, fit_asymmetry_b
from src.plotting import savefig

def main():
    params = BoxParams()
    rs = np.concatenate([np.array([0.0]), np.logspace(0, 4, 40)])

    out_dry = run_rs_sweep(params, rs, definition="PT_dry")
    out_wet = run_rs_sweep(params, rs, definition="PT_wetref")

    b_dry = fit_asymmetry_b(out_dry["E_pa"], out_dry["E_p0"], out_dry["E"])
    b_wet = fit_asymmetry_b(out_wet["E_pa"], out_wet["E_p0"], out_wet["E"])

    fig, ax = plt.subplots()
    ax.set_xlabel(r"$x = E_{p0}/E_{pa}$")
    ax.set_ylabel(r"$y = E/E_{pa}$")
    ax.set_title("Definition sensitivity: same physics, different $E_{p0}$ definition")
    ax.grid(True, alpha=0.3)

    ax.plot(out_dry["x"], out_dry["y"], marker="o", markersize=3, linewidth=1.5,
            label=f"PT_dry (b_fit={b_dry:.2f})")
    ax.plot(out_wet["x"], out_wet["y"], marker="o", markersize=3, linewidth=1.5,
            label=f"PT_wetref (b_fit={b_wet:.2f})")

    ax.axhline(1.0, linewidth=1.0)
    ax.axvline(1.0, linewidth=1.0)

    ax.legend(frameon=False)
    savefig(fig, ROOT/"outputs"/"fig_definition_sensitivity_xy.png")

if __name__ == "__main__":
    main()
