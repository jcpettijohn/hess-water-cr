"""scripts/make_fig_box_model_state_space.py

Appendix figure: state space of the Tier-2 coupled land--atmosphere mixed-layer box model.

The box model evolves a mixed-layer air state (T, q) that relaxes toward a
background state (T_b, q_b) while being forced by surface sensible and latent
heat fluxes. By sweeping surface resistance r_s from wet (r_s=0) to dry
(large r_s), we obtain a locus of equilibrated (T*, q*) states.

Output: figures/fig_box_model_state_space.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.box_model import BoxParams, run_rs_sweep
from src.plotting import savefig


def _es_kpa(T_C: np.ndarray) -> np.ndarray:
    """Tetens-type saturation vapor pressure (kPa), vectorized."""
    T_C = np.asarray(T_C, dtype=float)
    return 0.6108 * np.exp(17.27 * T_C / (T_C + 237.3))


def _q_from_e_kpa(e_kpa: np.ndarray, p_kpa: float) -> np.ndarray:
    """Specific humidity from vapor pressure (kPa) at pressure p_kpa."""
    # EPSILON = 0.622 (molecular weight ratio); keep inline for self-containment.
    eps = 0.622
    e_kpa = np.asarray(e_kpa, dtype=float)
    return eps * e_kpa / (p_kpa - (1 - eps) * e_kpa)


def main() -> None:
    # Typography: professional sans + STIX math
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "mathtext.fontset": "stix",
        }
    )

    params = BoxParams()

    # Sweep r_s from wet to dry; include r_s=0 explicitly.
    rs_values = np.concatenate([[0.0], np.logspace(-2, 5, 80)])

    out = run_rs_sweep(params, rs_values, definition="PT_dry")

    T_star = out["T_C"]
    q_star = out["q"]

    # Saturation curve for context (thermodynamic feasibility envelope)
    T_min = float(np.min(T_star)) - 5.0
    T_max = float(np.max(T_star)) + 10.0
    T_grid = np.linspace(T_min, T_max, 250)
    q_sat = _q_from_e_kpa(_es_kpa(T_grid), params.p_kpa)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(
        T_grid,
        q_sat,
        linestyle="--",
        linewidth=1.5,
        label=r"saturation: $q=q_{sat}(T)$",
    )

    ax.plot(
        T_star,
        q_star,
        linewidth=2.0,
        label=r"equilibrium locus (sweep in $r_s$)",
    )

    # Mark key reference points
    ax.scatter(
        [params.T_b_C],
        [params.q_b],
        s=45,
        marker="s",
        label=r"background $(T_b,q_b)$",
    )

    ax.scatter(
        [T_star[0]],
        [q_star[0]],
        s=55,
        marker="o",
        label=r"wet limit ($r_s=0$)",
    )

    ax.scatter(
        [T_star[-1]],
        [q_star[-1]],
        s=55,
        marker="^",
        label=r"dry end (largest $r_s$)",
    )

    # Directional arrow (wet -> dry)
    mid = len(T_star) // 2
    i0 = max(0, mid - 10)
    i1 = min(len(T_star) - 1, mid + 10)
    ax.annotate(
        "",
        xy=(T_star[i1], q_star[i1]),
        xytext=(T_star[i0], q_star[i0]),
        arrowprops=dict(arrowstyle="->", lw=1.5),
    )
    ax.text(T_star[mid], q_star[mid], "wet → dry", va="bottom", ha="left")

    ax.set_xlabel(r"Mixed-layer air temperature $T$ (°C)")
    ax.set_ylabel(r"Mixed-layer specific humidity $q$ (kg kg$^{-1}$)")
    ax.set_title(r"Box-model state space: equilibrated mixed-layer states as $r_s$ increases")
    ax.grid(True, alpha=0.3)

    ax.legend(frameon=True, loc="best", fontsize=9)
    fig.tight_layout()

    out_png = ROOT / "figures" / "fig_box_model_state_space.png"
    savefig(fig, out_png, dpi=300)


if __name__ == "__main__":
    main()
