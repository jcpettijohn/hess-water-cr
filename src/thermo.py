"""
thermo.py

Shared thermodynamic / meteorological helper functions used across the CR numerical laboratory.

All functions are written to be explicit and lightweight, mirroring common reference ET guidance
(e.g., FAO-56 / ASCE) while staying general enough for idealized experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

# -----------------------
# Constants (SI)
# -----------------------
EPSILON = 0.622  # Ratio molecular weight water vapor / dry air
CP_AIR = 1004.0  # J kg-1 K-1
LV0 = 2.501e6    # J kg-1 (latent heat at 0C), used as baseline if temperature dependence not desired

def latent_heat_vaporization(T_C: float) -> float:
    """
    Latent heat of vaporization λ(T), J kg-1, using a common linear approximation:
    λ ≈ 2.501e6 - 2.361e3 * T(°C)

    This is consistent with many hydrometeorology conventions for modest temperature ranges.
    """
    return 2.501e6 - 2.361e3 * T_C

def saturation_vapor_pressure_kpa(T_C: float) -> float:
    """
    Saturation vapor pressure (kPa) using Tetens-style equation (FAO-56 style).
    es = 0.6108 * exp(17.27*T / (T + 237.3))
    """
    return 0.6108 * math.exp(17.27 * T_C / (T_C + 237.3))

def slope_sat_vp_curve_kpa_per_C(T_C: float) -> float:
    """
    Slope of saturation vapor pressure curve Δ (kPa/°C) consistent with the Tetens equation:
    Δ = 4098*es / (T+237.3)^2
    """
    es = saturation_vapor_pressure_kpa(T_C)
    return 4098.0 * es / ((T_C + 237.3) ** 2)

def psychrometric_constant_kpa_per_C(p_kpa: float, lambda_J_kg: float = LV0) -> float:
    """
    Psychrometric constant γ (kPa/°C):
    γ = cp * p / (ε * λ)

    Input pressure in kPa; cp in J/kg/K; λ in J/kg.
    """
    return CP_AIR * (p_kpa * 1000.0) / (EPSILON * lambda_J_kg) / 1000.0  # convert Pa->kPa

def vapor_pressure_from_specific_humidity_kpa(q: float, p_kpa: float) -> float:
    """
    Convert specific humidity q (kg/kg) to vapor pressure e (kPa).
    Using e = q * p / (ε + (1-ε)q) in consistent units.
    """
    denom = EPSILON + (1.0 - EPSILON) * q
    if denom <= 0:
        return 0.0
    return q * p_kpa / denom

def specific_humidity_from_vapor_pressure(q_kpa: float, p_kpa: float) -> float:
    """
    Inverse of vapor_pressure_from_specific_humidity_kpa: given vapor pressure e (kPa),
    return specific humidity q (kg/kg).
    """
    e = q_kpa
    # q = ε e / (p - (1-ε)e)
    denom = p_kpa - (1.0 - EPSILON) * e
    if denom <= 0:
        return 0.0
    return EPSILON * e / denom

def vpd_kpa(T_C: float, q: float, p_kpa: float) -> float:
    """
    Vapor pressure deficit (kPa) given air temperature T_C and specific humidity q.
    """
    es = saturation_vapor_pressure_kpa(T_C)
    ea = vapor_pressure_from_specific_humidity_kpa(q, p_kpa)
    return max(0.0, es - ea)

def clamp(x: float, xmin: float, xmax: float) -> float:
    return max(xmin, min(xmax, x))
