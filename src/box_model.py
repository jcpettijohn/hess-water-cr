"""
box_model.py

Tier-2 mixed-layer "box" model that generates complementarity mechanistically.

Key design goals:
- Minimal, transparent physics (bulk mixed layer + relaxation).
- Deterministic, fast equilibrium solver (fixed-point iteration on the steady-state equations).
- Explicit links to the manuscript's nondimensional (x,y) variables and asymmetry fits.

This is a *numerical laboratory*, not a site-level prediction model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .thermo import (
    CP_AIR,
    latent_heat_vaporization,
    slope_sat_vp_curve_kpa_per_C,
    psychrometric_constant_kpa_per_C,
    vpd_kpa,
)

@dataclass
class BoxParams:
    # Forcing and atmosphere
    RnG: float = 150.0          # W m-2 (net available energy: Rn - G)
    p_kpa: float = 101.3        # kPa
    rho_a: float = 1.2          # kg m-3

    # Aerodynamic resistance
    r_a: float = 50.0           # s m-1

    # Mixed-layer parameters
    h: float = 1000.0           # m
    tau_q: float = 6*3600.0     # s
    tau_T: float = 6*3600.0     # s

    # Background (large-scale) air state
    T_b_C: float = 20.0         # °C
    q_b: float = 0.010          # kg/kg

    # Priestley–Taylor coefficient for Ep0
    alpha_PT: float = 1.26

    # ---------------------------------------------------------------------
    # Optional entrainment / advection drying (Tier-2 extension)
    # ---------------------------------------------------------------------
    # We optionally include an explicit dry-air import term in the mixed-layer
    # moisture budget. It is implemented as a bulk exchange with a drier
    # reservoir of specific humidity q_ft at an effective entrainment (or
    # ventilation) velocity w_e (m s-1):
    #
    #   dq/dt = E/(rho_a h) - (q - q_b)/tau_q - (w_e/h) (q - q_ft)
    #
    # where w_e can be prescribed as a constant (w_e0) and/or scaled with
    # positive sensible heat flux H to mimic stronger entrainment under dry,
    # convective conditions.
    #
    # Defaults (w_e0=0 and alpha_ent=0) recover the original Tier-2 model.
    q_ft: float = 0.010         # kg/kg (humidity of entrained/advected air)
    w_e0: float = 0.0           # m/s  (background entrainment/advection rate)
    alpha_ent: float = 0.0      # (-)  entrainment scaling with sensible heat
    dT_ent: float = 3.0         # K    characteristic inversion temperature jump

@dataclass
class BoxState:
    T_C: float
    q: float

def penman_monteith_E(
    RnG: float,
    T_C: float,
    q: float,
    p_kpa: float,
    rho_a: float,
    r_a: float,
    r_s: float,
) -> float:
    """
    Penman–Monteith-like evaporation rate E (kg m-2 s-1) computed from:
      λE = [Δ RnG + ρ c_p VPD / r_a] / [Δ + γ(1 + r_s/r_a)]

    with:
      - Δ, γ in kPa / °C
      - VPD in kPa
      - RnG in W/m2
      - r_a, r_s in s/m
    """
    lam = latent_heat_vaporization(T_C)  # J/kg
    Delta = slope_sat_vp_curve_kpa_per_C(T_C)  # kPa/°C
    gamma = psychrometric_constant_kpa_per_C(p_kpa, lam)  # kPa/°C
    VPD = vpd_kpa(T_C, q, p_kpa)  # kPa

    # consistent units: numerator has W/m2 * kPa/°C; denominator kPa/°C -> LE in W/m2
    num = Delta * RnG + rho_a * CP_AIR * VPD / max(r_a, 1e-6)
    den = Delta + gamma * (1.0 + max(r_s, 0.0) / max(r_a, 1e-6))
    LE = num / max(den, 1e-12)
    E = LE / lam
    return max(0.0, float(E))

def priestley_taylor_Ep0(
    RnG: float,
    T_C: float,
    p_kpa: float,
    alpha_PT: float = 1.26,
) -> float:
    """
    Priestley–Taylor wet-environment evaporation Ep0 (kg m-2 s-1):
      Ep0 = α * Δ/(Δ+γ) * (RnG/λ)
    """
    lam = latent_heat_vaporization(T_C)
    Delta = slope_sat_vp_curve_kpa_per_C(T_C)
    gamma = psychrometric_constant_kpa_per_C(p_kpa, lam)
    return float(alpha_PT * (Delta / (Delta + gamma)) * (RnG / lam))


def entrainment_velocity_m_per_s(H_Wm2: float, params: BoxParams) -> float:
    """Compute the effective entrainment/advection velocity w_e (m s-1).

    This is a deliberately minimal parameterization for Tier-2 experiments.

    We allow:
      * a background (constant) exchange rate ``w_e0``; and
      * an additional, convectively enhanced term proportional to positive
        sensible heat flux H (only when H>0), scaled by a characteristic
        temperature jump dT_ent:

            w_e = w_e0 + alpha_ent * H / (rho_a * c_p * dT_ent)

    The H/(rho c_p) factor has units of (K m s-1); dividing by dT_ent (K)
    yields a velocity scale (m s-1). The dimensionless ``alpha_ent`` then
    controls how strongly entrainment grows with heating.

    Defaults (w_e0=0, alpha_ent=0) recover the original model.
    """
    w_e = float(params.w_e0)
    if params.alpha_ent != 0.0:
        dT = max(float(params.dT_ent), 1e-6)
        w_e += float(params.alpha_ent) * max(float(H_Wm2), 0.0) / (float(params.rho_a) * CP_AIR * dT)
    return max(0.0, w_e)

def solve_equilibrium_fixed_point(
    params: BoxParams,
    r_s: float,
    state0: Optional[BoxState] = None,
    max_iter: int = 2000,
    relax: float = 0.2,
    tol: float = 1e-8,
) -> BoxState:
    """
    Solve the steady state of the box model using fixed-point iteration.

    At equilibrium the energy closure and mixed-layer budgets imply:
      * H = RnG - λE
      * T = T_b + τ_T * H / (ρ c_p h)

    For moisture, the original Tier-2 model used simple relaxation to a
    background humidity q_b, giving q = q_b + τ_q E/(ρ h). When the optional
    entrainment/advection drying term is enabled (w_e>0), q instead satisfies:

        0 = E/(ρ h) - (q-q_b)/τ_q - (w_e/h)(q-q_ft)

    which we solve in closed form each iteration for a given E and w_e.

    Because E depends on (T,q) through Penman–Monteith, this remains an
    implicit nonlinear fixed point.
    """
    if state0 is None:
        T, q = params.T_b_C, params.q_b
    else:
        T, q = float(state0.T_C), float(state0.q)

    converged = False
    for it in range(max_iter):
        E = penman_monteith_E(params.RnG, T, q, params.p_kpa, params.rho_a, params.r_a, r_s)
        lam = latent_heat_vaporization(T)
        H = params.RnG - lam * E

        # Moisture fixed-point update. With optional entrainment/advection,
        # solve the steady moisture budget in closed form:
        #   0 = E/(rho h) - (q-q_b)/tau_q - (w_e/h)(q-q_ft)
        w_e = entrainment_velocity_m_per_s(H, params)
        a = (1.0 / max(params.tau_q, 1e-12)) + (w_e / max(params.h, 1e-12))
        b = (params.q_b / max(params.tau_q, 1e-12)) + (w_e / max(params.h, 1e-12)) * params.q_ft + E / (params.rho_a * max(params.h, 1e-12))
        q_target = b / max(a, 1e-12)
        T_target = params.T_b_C + params.tau_T * H / (params.rho_a * CP_AIR * params.h)

        # relaxed update for stability
        T_new = (1.0 - relax) * T + relax * T_target
        q_new = (1.0 - relax) * q + relax * q_target

        # enforce basic bounds
        q_new = max(0.0, q_new)

        if abs(T_new - T) < tol and abs(q_new - q) < tol:
            T, q = T_new, q_new
            converged = True
            break

        T, q = T_new, q_new

    if not converged:
        import warnings
        warnings.warn(
            (
                f"solve_equilibrium_fixed_point did not converge in {max_iter} iterations "
                f"(r_s={float(r_s):.3g}, h={params.h}, tau_T={params.tau_T}, tau_q={params.tau_q}, r_a={params.r_a}). "
                "Returning last iterate; try smaller relax (e.g., 0.2 or 0.1) and/or larger max_iter."
            ),
            RuntimeWarning,
        )

    return BoxState(T_C=float(T), q=float(q))

def run_rs_sweep(
    params: BoxParams,
    rs_values: np.ndarray,
    definition: str = "PT_dry",
) -> Dict[str, np.ndarray]:
    """
    Sweep surface resistance r_s across rs_values, generating mechanistic arrays for:
      - E, E_pa, E_p0
      - x = E_p0/E_pa, y = E/E_pa
      - T, q

    definition options:
      - "PT_dry": Ep0 via Priestley–Taylor evaluated at each dry-state temperature T*(r_s)
      - "PT_wetref": Ep0 via Priestley–Taylor evaluated at wet-reference temperature T*(r_s=0)
    """
    rs_values = np.asarray(rs_values, dtype=float)

    wet_state = solve_equilibrium_fixed_point(params, r_s=0.0)
    T_wet = wet_state.T_C
    # definition-consistent wet-environment evaporation for this forcing (counterfactual):
    # the equilibrium areal evaporation if the region were moisture-unlimited (r_s=0).
    E_wet = penman_monteith_E(params.RnG, wet_state.T_C, wet_state.q, params.p_kpa, params.rho_a, params.r_a, 0.0)

    out = {k: np.zeros_like(rs_values, dtype=float) for k in ["r_s","T_C","q","E","E_pa","E_p0","x","y"]}
    out["r_s"] = rs_values.copy()

    # continuation for speed (use previous fixed point as initial guess)
    state = wet_state
    for i, rs in enumerate(rs_values):
        state = solve_equilibrium_fixed_point(params, r_s=float(rs), state0=state)

        T_C, q = state.T_C, state.q
        E = penman_monteith_E(params.RnG, T_C, q, params.p_kpa, params.rho_a, params.r_a, float(rs))
        E_pa = penman_monteith_E(params.RnG, T_C, q, params.p_kpa, params.rho_a, params.r_a, 0.0)

        if definition == "PT_dry":
            Ep0 = priestley_taylor_Ep0(params.RnG, T_C, params.p_kpa, params.alpha_PT)
        elif definition == "PT_wetref":
            Ep0 = priestley_taylor_Ep0(params.RnG, T_wet, params.p_kpa, params.alpha_PT)
        elif definition == "ML_wetconst":
            Ep0 = E_wet
        else:
            raise ValueError(f"Unknown definition='{definition}'")

        out["T_C"][i] = T_C
        out["q"][i] = q
        out["E"][i] = E
        out["E_pa"][i] = E_pa
        out["E_p0"][i] = Ep0
        out["x"][i] = Ep0 / E_pa if E_pa > 0 else np.nan
        out["y"][i] = E / E_pa if E_pa > 0 else np.nan

    return out

def fit_asymmetry_b(E_pa: np.ndarray, E_p0: np.ndarray, E: np.ndarray) -> float:
    """
    Fit effective linear asymmetry coefficient b from:
      E_pa - E_p0 = b (E_p0 - E)
    using least squares through the origin.
    """
    X = (E_p0 - E)
    Y = (E_pa - E_p0)
    m = np.isfinite(X) & np.isfinite(Y)
    X = X[m]
    Y = Y[m]
    denom = float(np.dot(X, X))
    if denom <= 0:
        return float("nan")
    return float(np.dot(X, Y) / denom)
