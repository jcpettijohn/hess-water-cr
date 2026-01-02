"""
scripts/make_fig_cr_xy_atlas.py

Generate the Tier-1 "atlas" figure: unified nondimensional y=f(x) curves across CR families.

Output:
  - outputs/fig_cr_xy_atlas.png
  - outputs/fig_cr_xy_atlas_diagnostics.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

# allow "src/" imports when running as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.cr_models import default_model_set
from src.diagnostics import compute_curve_diagnostics
from src.plotting import savefig

def main():
    models = default_model_set()

    fig, ax = plt.subplots()
    ax.set_xlabel(r"$x = E_{p0}/E_{pa}$")
    ax.set_ylabel(r"$y = E/E_{pa}$")
    ax.set_title("Unified nondimensional CR atlas (Tier 1)")
    ax.grid(True, alpha=0.3)

    diag_rows = []

    for model in models:
        x = np.linspace(model.x_min, model.x_max, 800)
        y = model.func(x)

        # Clip only for plot readability
        ax.plot(x, np.clip(y, 0.0, 1.2), label=model.name)

        d = compute_curve_diagnostics(x, np.clip(y, 0.0, 1.0))
        diag_rows.append({
            "model": model.name,
            "x_min": d.x_min,
            "wet_slope": d.wet_slope,
            "int_abs_curvature": d.int_abs_curv,
            "max_chord_deviation": d.chord_dev,
        })

    # Explicit boundary constraints
    ax.axhline(1.0, linewidth=0.6, color="0.75", zorder=0)
    ax.axvline(1.0, linewidth=0.6, color="0.75", zorder=0)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(-0.02, 1.25)

    ax.legend(loc="upper left", fontsize=9, ncols=1, frameon=False)

    out_png = ROOT / "outputs" / "fig_cr_xy_atlas.png"
    savefig(fig, out_png, dpi=300)

    df = pd.DataFrame(diag_rows).sort_values(["x_min", "model"])
    df.to_csv(ROOT / "outputs" / "fig_cr_xy_atlas_diagnostics.csv", index=False)

if __name__ == "__main__":
    main()