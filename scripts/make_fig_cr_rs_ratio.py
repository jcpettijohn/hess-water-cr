"""
scripts/make_fig_cr_rs_ratio.py

Plot implied resistance ratio r_s/r_a as a transform of y = E/Epa for the same model set used in
the atlas figure.

We use the identity (for the PM saturated-patch definition):
  r_s/r_a = (Δ+γ)/γ * (1/y - 1)

For this conceptual figure we set (Δ+γ)/γ = K as a constant (default K=2).
The purpose is to show how different CR curves imply different *effective* resistance behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.cr_models import default_model_set
from src.plotting import savefig

def main(K: float = 2.0):
    models = default_model_set()

    fig, ax = plt.subplots()
    ax.set_xlabel(r"$x = E_{p0}/E_{pa}$")
    ax.set_ylabel(r"Implied $r_s/r_a$ (dimensionless)")
    ax.set_title(r"Resistance-ratio interpretation ($r_s/r_a = K(1/y - 1)$)")
    ax.grid(True, alpha=0.3)

    for model in models:
        x = np.linspace(model.x_min, model.x_max, 800)
        y = model.func(x)

        y_safe = np.clip(y, 1e-3, 1.0)  # avoid divergence at y=0
        rs_ra = K * (1.0 / y_safe - 1.0)

        ax.plot(x, rs_ra, label=model.name)

    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 100.0)
    # One-column legend, anchored at the upper-right of the axes.
    ax.legend(fontsize=7, ncols=1, frameon=False, loc="upper right")

    out_png = ROOT / "outputs" / "fig_cr_rs_ratio.png"
    savefig(fig, out_png, dpi=300)

if __name__ == "__main__":
    main()
