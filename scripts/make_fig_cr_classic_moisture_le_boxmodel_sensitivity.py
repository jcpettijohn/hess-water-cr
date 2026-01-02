#!/usr/bin/env python3
"""Make classic moisture-availability sensitivity figures for the box model.

This script produces:
  - fig_cr_classic_moisture_le_boxmodel_entrainment_sensitivity.png
  - fig_cr_classic_moisture_le_boxmodel_h_sensitivity.png

It is written to be run from anywhere inside the repo (e.g. from scripts/), and
adds the repo root to sys.path so `import src...` works.

Key design goals (based on feedback):
  * Robust equilibrium solving (adaptive fixed-point + Newton fallback)
  * Okabe–Ito colorblind-safe palette
  * Two-line titles (sensitivity note as 2nd line)
  * Compact legends with no boxes, positioned to avoid covering curves
  * Consistent y-axis limits across both figures
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# Non-interactive backend (safe for headless runs)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# -----------------------------------------------------------------------------
# Path setup so `import src...` works even when running from scripts/
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local imports (repo code)
from src.box_model import (  # noqa: E402
    BoxParams,
    BoxState,
    entrainment_velocity_m_per_s,
    penman_monteith_E,
    priestley_taylor_Ep0,
    solve_equilibrium_fixed_point,
)
from src.thermo import CP_AIR, latent_heat_vaporization  # noqa: E402


# -----------------------------------------------------------------------------
# Styling helpers
# -----------------------------------------------------------------------------
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#999999",
}


def _round_up_to(x: float, base: float = 50.0) -> float:
    return float(base * math.ceil(float(x) / base))


def _moisture_proxy(r_s: np.ndarray, r_s_min: float, r_a: float) -> np.ndarray:
    """Regional moisture availability proxy in [0, 1]."""
    return r_a / (r_a + (r_s - r_s_min))


def _add_dry_wet_arrow(ax: plt.Axes, y: float = 0.92) -> None:
    """Add a 'dry <- wet' arrow in axes-fraction coordinates."""
    x_left, x_right = 0.18, 0.82
    ax.annotate(
        "",
        xy=(x_left, y),
        xytext=(x_right, y),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=3.0, color=OKABE_ITO["black"]),
    )
    ax.text(
        x_left - 0.03,
        y,
        "dry",
        ha="right",
        va="center",
        transform=ax.transAxes,
        fontsize=18,
        color=OKABE_ITO["black"],
    )
    ax.text(
        x_right + 0.03,
        y,
        "wet",
        ha="left",
        va="center",
        transform=ax.transAxes,
        fontsize=18,
        color=OKABE_ITO["black"],
    )


# -----------------------------------------------------------------------------
# Robust equilibrium solver
# -----------------------------------------------------------------------------
def _equilibrium_targets(
    params: BoxParams, r_s: float, T_C: float, q: float
) -> Tuple[float, float, float, float, float, float]:
    """Compute fixed-point targets (T_target, q_target) and diagnostics."""
    E = float(
        penman_monteith_E(
            params.RnG,
            T_C,
            q,
            params.p_kpa,
            params.rho_a,
            params.r_a,
            r_s,
        )
    )  # kg m-2 s-1

    Lv = float(latent_heat_vaporization(T_C))  # J/kg
    LE = E * Lv  # W/m2
    H = float(params.RnG - LE)  # W/m2

    w_e = float(
        entrainment_velocity_m_per_s(H, params)
    )  # m/s

    q_target = float(
        params.q_b
        + params.tau_q
        * (
            E / (params.rho_a * params.h)
            - w_e * (q - params.q_ft) / params.h
        )
    )
    T_target = float(params.T_b_C + params.tau_T * H / (params.rho_a * CP_AIR * params.h))
    return T_target, q_target, E, LE, H, w_e


def _fixed_point_residual(params: BoxParams, r_s: float, st: BoxState) -> float:
    T_t, q_t, *_ = _equilibrium_targets(params, r_s, st.T_C, st.q)
    return float(max(abs(T_t - st.T_C), abs(q_t - st.q)))


def _solve_equilibrium_newton(
    params: BoxParams,
    r_s: float,
    state0: BoxState,
    tol: float,
    max_iter: int = 60,
    debug: bool = False,
) -> Tuple[BoxState, bool, float]:
    """2-D damped Newton solver on F(T,q) = [T - T_target, q - q_target]."""
    T = float(state0.T_C)
    q = float(state0.q)

    eps_T = 1e-3  # K
    eps_q = 1e-6  # kg/kg

    best_res = math.inf
    best = BoxState(T_C=T, q=q)

    for _ in range(max_iter):
        T_t, q_t, *_ = _equilibrium_targets(params, r_s, T, q)
        F = np.array([T - T_t, q - q_t], dtype=float)
        res = float(np.max(np.abs(F)))

        if res < best_res:
            best_res = res
            best = BoxState(T_C=T, q=q)

        if res < tol:
            return BoxState(T_C=T, q=q), True, res

        try:
            T_t1, q_t1, *_ = _equilibrium_targets(params, r_s, T + eps_T, q)
            F_T = np.array([(T + eps_T) - T_t1, q - q_t1], dtype=float)

            T_t2, q_t2, *_ = _equilibrium_targets(params, r_s, T, q + eps_q)
            F_Q = np.array([T - T_t2, (q + eps_q) - q_t2], dtype=float)

            J = np.column_stack(((F_T - F) / eps_T, (F_Q - F) / eps_q))
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            dx = -0.25 * F

        alpha = 1.0
        for _ in range(12):
            T_new = float(T + alpha * dx[0])
            q_new = float(q + alpha * dx[1])
            q_new = float(min(max(q_new, 1e-8), 0.05))

            T_tn, q_tn, *_ = _equilibrium_targets(params, r_s, T_new, q_new)
            F_new = np.array([T_new - T_tn, q_new - q_tn], dtype=float)
            res_new = float(np.max(np.abs(F_new)))

            if res_new < res:
                T, q = T_new, q_new
                break
            alpha *= 0.5

    if debug:
        print(
            f"[newton] did not converge: r_s={r_s:.3g}, h={params.h:.1f} m, best_res={best_res:.3e}"
        )
    return best, False, best_res


def solve_equilibrium_robust(
    params: BoxParams,
    r_s: float,
    state0: BoxState,
    tol: float,
    debug: bool = False,
) -> Tuple[BoxState, bool, float, str]:
    """Robust wrapper: adaptive fixed-point attempts, then Newton fallback."""
    ladder = [
        (0.40, 800),
        (0.25, 2000),
        (0.15, 4000),
        (0.10, 8000),
    ]

    best_state = state0
    best_res = math.inf

    st0 = state0
    for relax, max_iter in ladder:
        st = solve_equilibrium_fixed_point(
            params,
            r_s=float(r_s),
            state0=st0,
            max_iter=int(max_iter),
            relax=float(relax),
            tol=float(tol),
        )
        res = _fixed_point_residual(params, float(r_s), st)
        if res < best_res:
            best_res = res
            best_state = st

        if res < tol:
            return st, True, res, f"fixed_point(relax={relax}, max_iter={max_iter})"

        st0 = st

        if debug:
            print(
                f"[retry] r_s={r_s:.4g}, h={params.h:.1f} m, relax={relax}, max_iter={max_iter}, residual={res:.2e}"
            )

    stN, okN, resN = _solve_equilibrium_newton(
        params, float(r_s), best_state, tol=float(tol), debug=debug
    )
    if okN or (resN < best_res):
        return stN, okN, resN, "newton"

    return best_state, False, best_res, "fixed_point(best)"


# -----------------------------------------------------------------------------
# Sweep utilities
# -----------------------------------------------------------------------------
def run_rs_sweep_robust(
    params: BoxParams,
    r_s_values: np.ndarray,
    r_s_min: float,
    tol: float,
    debug: bool = False,
) -> Dict[str, np.ndarray]:
    """Run an r_s sweep (wet->dry) with robust convergence handling."""
    r_s_values = np.asarray(r_s_values, dtype=float)
    if np.any(np.diff(r_s_values) < 0):
        raise ValueError("r_s_values must be monotonically increasing (wet -> dry)")

    st = BoxState(T_C=float(params.T_b_C), q=float(params.q_b))

    T_list: List[float] = []
    q_list: List[float] = []
    LE_list: List[float] = []
    LE_pa_list: List[float] = []
    LE_unstressed_list: List[float] = []
    method_list: List[str] = []

    for r_s in r_s_values:
        st, ok, res, method = solve_equilibrium_robust(
            params, float(r_s), st, tol=float(tol), debug=debug
        )
        if (not ok) and debug:
            print(
                f"[warn] nonconverged: r_s={r_s:.4g}, h={params.h:.1f} m, residual={res:.2e}, using {method}"
            )

        T_C = float(st.T_C)
        q = float(st.q)

        # Actual LE
        _, _, _, LE, _, _ = _equilibrium_targets(params, float(r_s), T_C, q)

        # Apparent potential LE_pa (PM with r_s = 0) at the same (T,q)
        E_pa = float(
            penman_monteith_E(
                params.RnG,
                T_C,
                q,
                params.p_kpa,
                params.rho_a,
                params.r_a,
                0.0,
            )
        )
        LE_pa = E_pa * float(latent_heat_vaporization(T_C))

        # Potential with unstressed canopy (PM with r_s = r_s_min) at the same (T,q)
        E_un = float(
            penman_monteith_E(
                params.RnG,
                T_C,
                q,
                params.p_kpa,
                params.rho_a,
                params.r_a,
                float(r_s_min),
            )
        )
        LE_un = E_un * float(latent_heat_vaporization(T_C))

        T_list.append(T_C)
        q_list.append(q)
        LE_list.append(float(LE))
        LE_pa_list.append(float(LE_pa))
        LE_unstressed_list.append(float(LE_un))
        method_list.append(method)

    return {
        "r_s": r_s_values,
        "T_C": np.asarray(T_list),
        "q": np.asarray(q_list),
        "LE": np.asarray(LE_list),
        "LE_pa": np.asarray(LE_pa_list),
        "LE_unstressed": np.asarray(LE_unstressed_list),
        "method": np.asarray(method_list, dtype=object),
    }


def _wet_pt_scaled_benchmark(
    params_ref: BoxParams,
    r_s_min: float,
    tol: float,
    debug: bool = False,
) -> float:
    """Compute a single constant wet benchmark (PT at wet reference, scaled to match actual wet end)."""
    st0 = BoxState(T_C=float(params_ref.T_b_C), q=float(params_ref.q_b))

    # Wet reference (r_s = 0)
    st_wet, _, _, _ = solve_equilibrium_robust(params_ref, 0.0, st0, tol=tol, debug=debug)
    T_wet = float(st_wet.T_C)

    Ep0 = float(
        priestley_taylor_Ep0(
            params_ref.RnG, T_wet, params_ref.p_kpa, alpha_PT=float(params_ref.alpha_PT)
        )
    )
    LE_pt = Ep0 * float(latent_heat_vaporization(T_wet))

    # Actual wet end (finite canopy at r_s = r_s_min)
    st_rsmin, _, _, _ = solve_equilibrium_robust(
        params_ref, float(r_s_min), st0, tol=tol, debug=debug
    )
    _, _, _, LE_rsmin, _, _ = _equilibrium_targets(
        params_ref, float(r_s_min), float(st_rsmin.T_C), float(st_rsmin.q)
    )

    if LE_pt <= 0:
        return float(LE_pt)
    scale = float(LE_rsmin / LE_pt)
    return float(scale * LE_pt)


# -----------------------------------------------------------------------------
# Compute datasets for the two figures
# -----------------------------------------------------------------------------
def compute_h_sensitivity(
    *,
    params_base: BoxParams,
    h_values: Iterable[float],
    r_s_values: np.ndarray,
    r_s_min: float,
    tol: float,
    debug: bool = False,
) -> Dict[str, object]:
    h_values = list(h_values)

    # Reference sweep (used for actual + unstressed reference curves)
    params_ref = replace(params_base, h=float(params_base.h))
    sweep_ref = run_rs_sweep_robust(
        params_ref, r_s_values, r_s_min=r_s_min, tol=tol, debug=debug
    )

    # Apparent potential (r_s=0) curves for each h
    by_h: Dict[float, np.ndarray] = {}
    for h in h_values:
        params_h = replace(params_base, h=float(h))
        sweep_h = run_rs_sweep_robust(
            params_h, r_s_values, r_s_min=r_s_min, tol=tol, debug=debug
        )
        by_h[float(h)] = sweep_h["LE_pa"]

    return {
        "params_ref": params_ref,
        "r_s": r_s_values,
        "LE_actual_ref": sweep_ref["LE"],
        "LE_unstressed_ref": sweep_ref["LE_unstressed"],
        "LE_pa_by_h": by_h,
    }


def compute_entrainment_sensitivity(
    *,
    params_base: BoxParams,
    r_s_values: np.ndarray,
    r_s_min: float,
    tol: float,
    debug: bool = False,
) -> Dict[str, object]:
    scenarios: List[Tuple[str, BoxParams]] = [
        ("Off", replace(params_base, w_e0=0.0, alpha_ent=0.0)),
        ("Weak", replace(params_base, w_e0=0.5 * float(params_base.w_e0))),
        ("Nominal", params_base),
        ("Strong", replace(params_base, w_e0=2.0 * float(params_base.w_e0))),
        # Drier/warmer background inflow as a simple "advective" perturbation
        (
            "Advective",
            replace(
                params_base,
                T_b_C=float(params_base.T_b_C) + 1.5,
                q_b=float(params_base.q_b) * 0.9,
            ),
        ),
    ]

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for name, p in scenarios:
        sweep = run_rs_sweep_robust(
            p, r_s_values, r_s_min=r_s_min, tol=tol, debug=debug
        )
        out[name] = {
            "LE": sweep["LE"],
            "LE_pa": sweep["LE_pa"],
            "LE_unstressed": sweep["LE_unstressed"],
        }

    return {
        "scenarios": [name for name, _ in scenarios],
        "params_by_scenario": {name: p for name, p in scenarios},
        "curves": out,
        "r_s": r_s_values,
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def _save(fig: plt.Figure, path: Path, dpi: int = 300, force_save: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force_save:
        print(f"[skip] {path} exists (use --force_save to overwrite)")
        return
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {path}")


def plot_entrainment_sensitivity(
    data: Dict[str, object],
    *,
    moisture: np.ndarray,
    LE_p0: float,
    y_max: float,
    outpath: Path,
    force_save: bool,
) -> None:
    scen_names: List[str] = list(data["scenarios"])  # type: ignore[assignment]
    curves: Dict[str, Dict[str, np.ndarray]] = data["curves"]  # type: ignore[assignment]

    scen_colors = {
        "Off": OKABE_ITO["blue"],
        "Weak": OKABE_ITO["sky"],
        "Nominal": OKABE_ITO["green"],
        "Strong": OKABE_ITO["vermillion"],
        "Advective": OKABE_ITO["purple"],
    }

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, y_max)

    ax.set_title(
        "Finite unstressed canopy resistance ($r_s^{\\min} > 0$)\n"
        "Entrainment/advection sensitivity ($h = 150$ m)",
        fontsize=22,
        pad=18,
    )

    _add_dry_wet_arrow(ax, y=0.915)

    # Wet benchmark (constant)
    ax.plot(
        moisture,
        np.full_like(moisture, LE_p0),
        ls="--",
        lw=3.0,
        color=OKABE_ITO["gray"],
        zorder=1,
    )

    for name in scen_names:
        c = scen_colors.get(name, OKABE_ITO["black"])
        ax.plot(moisture, curves[name]["LE"], color=c, lw=3.0, ls="-", zorder=3)
        ax.plot(moisture, curves[name]["LE_pa"], color=c, lw=3.0, ls="-.", zorder=2)

    ax.set_xlabel("Regional moisture availability (proxy)", fontsize=24)
    ax.set_ylabel(r"Latent heat flux $\lambda E$ (W m$^{-2}$)", fontsize=24)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.35)

    # Legends (two, compact, no boxes)
    curve_handles = [
        Line2D([0], [0], color=OKABE_ITO["black"], lw=3.0, ls="-", label=r"Actual $\lambda E$"),
        Line2D([0], [0], color=OKABE_ITO["black"], lw=3.0, ls="-.", label=r"Apparent 'potential' $\lambda E_{pa}$ (PM, $r_s=0$)"),
        Line2D([0], [0], color=OKABE_ITO["gray"], lw=3.0, ls="--", label=r"Wet benchmark $\lambda E_{p0}$ (PT, scaled)"),
    ]
    leg_curve = ax.legend(
        handles=curve_handles,
        title="Curve",
        loc="upper left",
        bbox_to_anchor=(0.08, 0.80),
        frameon=False,
        fontsize=12,
        title_fontsize=12,
        handlelength=3.0,
        borderaxespad=0.0,
    )
    ax.add_artist(leg_curve)

    scen_handles = [Line2D([0], [0], color=scen_colors[n], lw=3.0, ls="-", label=n) for n in scen_names]
    ax.legend(
        handles=scen_handles,
        title="Entrainment scenario",
        loc="upper right",
        bbox_to_anchor=(0.98, 0.80),
        ncol=2,
        frameon=False,
        fontsize=12,
        title_fontsize=12,
        handlelength=2.8,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    fig.tight_layout()
    _save(fig, outpath, force_save=force_save)
    plt.close(fig)


def plot_h_sensitivity(
    data: Dict[str, object],
    *,
    moisture: np.ndarray,
    LE_p0: float,
    y_max: float,
    outpath: Path,
    force_save: bool,
) -> None:
    LE_actual_ref = np.asarray(data["LE_actual_ref"], dtype=float)
    LE_unstressed_ref = np.asarray(data["LE_unstressed_ref"], dtype=float)
    LE_pa_by_h: Dict[float, np.ndarray] = data["LE_pa_by_h"]  # type: ignore[assignment]

    h_sorted = sorted(LE_pa_by_h.keys())
    h_colors = [
        OKABE_ITO["blue"],
        OKABE_ITO["sky"],
        OKABE_ITO["green"],
        OKABE_ITO["orange"],
        OKABE_ITO["vermillion"],
    ]
    color_by_h = {h: h_colors[i % len(h_colors)] for i, h in enumerate(h_sorted)}

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, y_max)

    ax.set_title(
        "Finite unstressed canopy resistance ($r_s^{\\min} > 0$)\n"
        r"$\lambda E_{pa}$ sensitivity to $h$",
        fontsize=22,
        pad=18,
    )
    _add_dry_wet_arrow(ax, y=0.915)

    # Reference curves
    ax.plot(moisture, LE_actual_ref, color=OKABE_ITO["blue"], lw=3.2, ls="-", zorder=3)
    ax.plot(moisture, LE_unstressed_ref, color=OKABE_ITO["purple"], lw=3.0, ls=":", zorder=2)
    ax.plot(moisture, np.full_like(moisture, LE_p0), color=OKABE_ITO["gray"], lw=3.0, ls="--", zorder=1)

    # Apparent potential curves per h
    for h in h_sorted:
        ax.plot(moisture, np.asarray(LE_pa_by_h[h], dtype=float), color=color_by_h[h], lw=3.0, ls="-.", zorder=2)

    ax.set_xlabel("Regional moisture availability (proxy)", fontsize=24)
    ax.set_ylabel(r"Latent heat flux $\lambda E$ (W m$^{-2}$)", fontsize=24)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.35)

    # Legends (two, compact, no boxes)
    curve_handles = [
        Line2D([0], [0], color=OKABE_ITO["blue"], lw=3.2, ls="-", label=r"Actual $\lambda E$ (reference $h$)"),
        Line2D([0], [0], color=OKABE_ITO["purple"], lw=3.0, ls=":", label=r"\"Potential\" with unstressed canopy ($r_s=r_s^{\min}$; ref $h$)"),
        Line2D([0], [0], color=OKABE_ITO["black"], lw=3.0, ls="-.", label=r"Apparent 'potential' $\lambda E_{pa}$ (PM, $r_s=0$)"),
        Line2D([0], [0], color=OKABE_ITO["gray"], lw=3.0, ls="--", label=r"Wet benchmark $\lambda E_{p0}$ (PT, scaled; ref $h$)"),
    ]
    # Nudge the *left* legend column slightly right so it doesn't overlap the
    # dry-end curves. (Do not move the right legend column.)
    leg_curve = ax.legend(
        handles=curve_handles,
        title="Curve",
        loc="upper left",
        bbox_to_anchor=(0.08, 0.80),
        ncol=1,
        frameon=False,
        fontsize=12,
        title_fontsize=12,
        handlelength=3.0,
        borderaxespad=0.0,
    )
    ax.add_artist(leg_curve)

    h_handles = [
        Line2D([0], [0], color=color_by_h[h], lw=3.0, ls="-.", label=f"$h$ = {int(h)} m")
        for h in h_sorted
    ]
    ax.legend(
        handles=h_handles,
        title="Mixed-layer depth",
        loc="upper right",
        bbox_to_anchor=(0.98, 0.80),
        ncol=1,
        frameon=False,
        fontsize=12,
        title_fontsize=12,
        handlelength=2.8,
        borderaxespad=0.0,
    )

    fig.tight_layout()
    _save(fig, outpath, force_save=force_save)
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--force_save", action="store_true", help="Overwrite existing output figures")
    p.add_argument("--debug", action="store_true", help="Verbose solver diagnostics")
    p.add_argument("--tol", type=float, default=1e-6, help="Equilibrium tolerance (default: 1e-6)")
    p.add_argument(
        "--min_h",
        type=float,
        default=50.0,
        help="Minimum h to include in the h-sensitivity sweep (default: 50 m)",
    )

    # Entrainment/advection knobs (BoxParams fields)
    p.add_argument(
        "--w_e0",
        type=float,
        default=0.2,
        help="Reference entrainment velocity parameter w_e0 (BoxParams field; default: 0.2)",
    )
    p.add_argument(
        "--alpha_ent",
        type=float,
        default=2.0,
        help="Entrainment nonlinearity parameter alpha_ent (BoxParams field; default: 2.0)",
    )
    p.add_argument(
        "--dT_ent",
        type=float,
        default=2.0,
        help="Entrainment temperature jump dT_ent [K] (BoxParams field; default: 2.0)",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    debug = bool(args.debug)
    tol = float(args.tol)

    # Model knobs
    r_s_min = 70.0

    # Dense rs grid (helps continuation / avoids branch-jumps)
    r_s_excess = np.geomspace(0.1, 1.0e4, 200)
    r_s_values = np.r_[r_s_min, r_s_min + r_s_excess]

    # Base params
    params0 = BoxParams()
    params0 = replace(
        params0,
        h=150.0,  # reference h for both figures
        w_e0=float(args.w_e0),
        alpha_ent=float(args.alpha_ent),
        dT_ent=float(args.dT_ent),
    )

    # h values for the h-sensitivity figure
    h_values = [50, 100, 150, 200, 250]
    h_values = [h for h in h_values if h >= float(args.min_h)]

    # Compute datasets
    data_h = compute_h_sensitivity(
        params_base=params0,
        h_values=h_values,
        r_s_values=r_s_values,
        r_s_min=r_s_min,
        tol=tol,
        debug=debug,
    )
    data_ent = compute_entrainment_sensitivity(
        params_base=params0,
        r_s_values=r_s_values,
        r_s_min=r_s_min,
        tol=tol,
        debug=debug,
    )

    # X-axis (sorted dry->wet for plotting)
    moisture = _moisture_proxy(r_s_values, r_s_min=r_s_min, r_a=float(params0.r_a))
    order = np.argsort(moisture)
    moisture = moisture[order]

    # Reorder arrays to match moisture sorting
    data_h["LE_actual_ref"] = np.asarray(data_h["LE_actual_ref"])[order]
    data_h["LE_unstressed_ref"] = np.asarray(data_h["LE_unstressed_ref"])[order]
    data_h["LE_pa_by_h"] = {h: np.asarray(arr)[order] for h, arr in data_h["LE_pa_by_h"].items()}  # type: ignore[attr-defined]

    for scen, d in data_ent["curves"].items():  # type: ignore[attr-defined]
        d["LE"] = np.asarray(d["LE"])[order]
        d["LE_pa"] = np.asarray(d["LE_pa"])[order]
        d["LE_unstressed"] = np.asarray(d["LE_unstressed"])[order]

    # Shared wet benchmark (computed from reference params)
    LE_p0 = _wet_pt_scaled_benchmark(params0, r_s_min=r_s_min, tol=tol, debug=debug)

    # Harmonize y-axis limits across both figures
    y_candidates: List[float] = [float(LE_p0)]
    y_candidates.append(float(np.nanmax(data_h["LE_actual_ref"])))
    y_candidates.append(float(np.nanmax(data_h["LE_unstressed_ref"])))
    y_candidates.append(float(np.nanmax(np.vstack(list(data_h["LE_pa_by_h"].values())))))  # type: ignore[arg-type]

    for scen in data_ent["scenarios"]:  # type: ignore[attr-defined]
        y_candidates.append(float(np.nanmax(data_ent["curves"][scen]["LE"])))
        y_candidates.append(float(np.nanmax(data_ent["curves"][scen]["LE_pa"])))

    y_max = _round_up_to(max(y_candidates) * 1.05, base=50.0)

    fig_dir = ROOT / "figures"
    plot_entrainment_sensitivity(
        data_ent,
        moisture=moisture,
        LE_p0=float(LE_p0),
        y_max=float(y_max),
        outpath=fig_dir / "fig_cr_classic_moisture_le_boxmodel_entrainment_sensitivity.png",
        force_save=bool(args.force_save),
    )
    plot_h_sensitivity(
        data_h,
        moisture=moisture,
        LE_p0=float(LE_p0),
        y_max=float(y_max),
        outpath=fig_dir / "fig_cr_classic_moisture_le_boxmodel_h_sensitivity.png",
        force_save=bool(args.force_save),
    )


if __name__ == "__main__":
    main()
