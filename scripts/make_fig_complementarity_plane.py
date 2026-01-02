"""Mechanistic complementarity plane: (E_pa-E_p0) vs (E_p0-E).

This plot is the most direct visualization of the Bouchet/Morton linear
complementarity form and its asymmetric extension.

Outputs
-------
figures/fig_complementarity_plane.png
outputs/complementarity_plane_points.csv
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.box_model import BoxParams, run_rs_sweep, fit_asymmetry_b
from src.plotting import savefig


def main() -> None:
    fig_dir = ROOT / 'figures'
    out_dir = ROOT / 'outputs'
    fig_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)

    rs_values = np.logspace(-3, 4, 260)

    regimes = {
        'Shallow + ventilated (strong adjustment)': BoxParams(r_a=30.0, h=300.0, tau_q=21600.0, tau_T=21600.0),
        'Baseline (moderate adjustment)': BoxParams(r_a=50.0, h=1000.0, tau_q=21600.0, tau_T=21600.0),
        'Deep + weak ventilation (weak adjustment)': BoxParams(r_a=70.0, h=2000.0, tau_q=7200.0, tau_T=7200.0),
    }

    definition = 'ML_wetconst'

    rows = []
    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    for label, params in regimes.items():
        out = run_rs_sweep(params, rs_values=rs_values, definition=definition)
        x = out['E_p0'] - out['E']
        y = out['E_pa'] - out['E_p0']
        b = fit_asymmetry_b(out['E_pa'], out['E_p0'], out['E'])

        ax.plot(x, y, linewidth=2.0, label=f"{label} (b={b:.2f})")

        for i in range(len(x)):
            rows.append({
                'regime': label,
                'rs': rs_values[i],
                'Ep0_minus_E': x[i],
                'Epa_minus_Ep0': y[i],
                'b_fit': b,
            })

    # 1:1 and axes
    xmax = max(r['Ep0_minus_E'] for r in rows)
    ax.plot([0, xmax], [0, xmax], linestyle='--', linewidth=1.0, color='k', alpha=0.35, label='symmetric (b=1)')

    ax.set_xlabel(r"$E_{p0} - E$")
    ax.set_ylabel(r"$E_{pa} - E_{p0}$")
    ax.set_title("Mechanistic complementarity plane (definition-consistent $E_{p0}$)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=9)

    savefig(fig, fig_dir / 'fig_complementarity_plane.png', dpi=220)

    pd.DataFrame(rows).to_csv(out_dir / 'complementarity_plane_points.csv', index=False)


if __name__ == '__main__':
    main()
