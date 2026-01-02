"""
scripts/make_fig_box_sensitivity.py

Run Tier-2 box model sweeps and produce diagnostic figures that show how
an emergent linear asymmetry coefficient b and curvature diagnostics vary with:

- mixed-layer depth h
- relaxation timescale tau (tau_T = tau_q)
- aerodynamic resistance r_a

This is intended as a *preliminary* figure generator for the manuscript outline.

Outputs:
  outputs/fig_box_b_vs_h.png
  outputs/fig_box_b_vs_tau.png
  outputs/fig_box_b_vs_ra.png
  outputs/fig_box_curv_vs_h.png
  outputs/fig_box_curv_vs_tau.png
  outputs/fig_box_curv_vs_ra.png
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.box_model import BoxParams, run_rs_sweep, fit_asymmetry_b
from src.diagnostics import integrated_abs_curvature
from src.plotting import savefig

def sweep_and_diagnostics(params: BoxParams, definition: str) -> tuple[float, float]:
    # sweep r_s from wet to dry (log spacing; include 0 exactly)
    rs = np.concatenate([np.array([0.0]), np.logspace(0, 4, 30)])  # s/m (31 points total)
    out = run_rs_sweep(params, rs_values=rs, definition=definition)

    b = fit_asymmetry_b(out["E_pa"], out["E_p0"], out["E"])

    # curvature in (x,y); sort by x increasing
    x = out["x"]
    y = out["y"]
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if len(x) < 10:
        return b, float("nan")

    order = np.argsort(x)
    x = x[order]
    y = np.clip(y[order], 0.0, 1.0)
    curv = integrated_abs_curvature(x, y)
    return b, curv

def plot_param_sensitivity(param_name: str, xvals: np.ndarray, b_vals: dict, curv_vals: dict, x_label: str):
    # b plot
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel("Fitted asymmetry coefficient b")
    ax.set_title(f"Box-model emergent b vs {x_label}")
    ax.grid(True, alpha=0.3)
    for label, vals in b_vals.items():
        ax.plot(xvals, vals, marker="o", label=label)
    ax.legend(frameon=False)
    savefig(fig, ROOT/"outputs"/f"fig_box_b_vs_{param_name}.png")

    # curvature plot
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\int |d^2y/dx^2|\,dx$")
    ax.set_title(f"Box-model curvature vs {x_label}")
    ax.grid(True, alpha=0.3)
    for label, vals in curv_vals.items():
        ax.plot(xvals, vals, marker="o", label=label)
    ax.legend(frameon=False)
    savefig(fig, ROOT/"outputs"/f"fig_box_curv_vs_{param_name}.png")

def main():
    base = BoxParams()

    # Two Ep0 definition modes to demonstrate definition sensitivity
    defs = ["PT_dry", "PT_wetref"]

    # 1) h sensitivity
    h_vals = np.array([300.0, 800.0, 1500.0, 3000.0])
    b_vals = {d: [] for d in defs}
    curv_vals = {d: [] for d in defs}
    for h in h_vals:
        for d in defs:
            p = BoxParams(**{**base.__dict__, "h": float(h)})
            b, curv = sweep_and_diagnostics(p, d)
            b_vals[d].append(b)
            curv_vals[d].append(curv)
    plot_param_sensitivity("h", h_vals, b_vals, curv_vals, x_label="Mixed-layer depth h (m)")

    # 2) tau sensitivity (tau_T=tau_q)
    tau_hours = np.array([1.0, 4.0, 10.0, 24.0])
    tau_vals = tau_hours * 3600.0
    b_vals = {d: [] for d in defs}
    curv_vals = {d: [] for d in defs}
    for tau in tau_vals:
        for d in defs:
            p = BoxParams(**{**base.__dict__, "tau_q": float(tau), "tau_T": float(tau)})
            b, curv = sweep_and_diagnostics(p, d)
            b_vals[d].append(b)
            curv_vals[d].append(curv)
    plot_param_sensitivity("tau", tau_hours, b_vals, curv_vals, x_label="Relaxation timescale τ (hours)")

    # 3) r_a sensitivity
    ra_vals = np.array([20.0, 50.0, 120.0, 250.0])
    b_vals = {d: [] for d in defs}
    curv_vals = {d: [] for d in defs}
    for ra in ra_vals:
        for d in defs:
            p = BoxParams(**{**base.__dict__, "r_a": float(ra)})
            b, curv = sweep_and_diagnostics(p, d)
            b_vals[d].append(b)
            curv_vals[d].append(curv)
    plot_param_sensitivity("ra", ra_vals, b_vals, curv_vals, x_label="Aerodynamic resistance r_a (s m$^{-1}$)")

if __name__ == "__main__":
    main()
