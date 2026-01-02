"""
diagnostics.py

Shared shape metrics for nondimensional complementary curves y(x).

These metrics are *comparative diagnostics* used to quantify how curves differ in wet-limit slope,
dry-end location, and curvature. They are not claimed to be unique invariants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

def wet_limit_slope(x: np.ndarray, y: np.ndarray, n_tail: int = 5) -> float:
    """
    One-sided estimate of dy/dx as x -> 1-.
    Uses a linear fit over the last n_tail points.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < n_tail + 1:
        raise ValueError("Not enough points for slope estimate.")
    xx = x[-n_tail:]
    yy = y[-n_tail:]
    # linear regression slope
    A = np.vstack([xx, np.ones_like(xx)]).T
    m, b = np.linalg.lstsq(A, yy, rcond=None)[0]
    return float(m)

def integrated_abs_curvature(x: np.ndarray, y: np.ndarray) -> float:
    """
    Approximate 222b |d2y/dx2| dx using gradients.

    Notes
    -----
    Many synthetic CR curves are naturally parameterized from wet2192dry, so x can be decreasing. For a proper geometric integral we
    therefore sort by x before integrating.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # sort for monotonic integration
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    # drop duplicate x values (can occur in stiff regimes) to avoid zero spacing in gradients
    x, uniq = np.unique(x, return_index=True)
    y = y[uniq]
    # second derivative via gradient twice (robust on nonuniform grids)
    dy_dx = np.gradient(y, x)
    d2y_dx2 = np.gradient(dy_dx, x)
    return float(np.trapz(np.abs(d2y_dx2), x))

def max_chord_deviation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Maximum vertical deviation from the straight chord connecting endpoints:
    (x_min,0) to (1,1). Assumes endpoints are included in arrays.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x0, y0 = float(x[0]), float(y[0])
    x1, y1 = float(x[-1]), float(y[-1])
    # chord line
    y_chord = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return float(np.max(np.abs(y - y_chord)))

@dataclass
class CurveDiagnostics:
    x_min: float
    wet_slope: float
    int_abs_curv: float
    chord_dev: float

def compute_curve_diagnostics(x: np.ndarray, y: np.ndarray) -> CurveDiagnostics:
    return CurveDiagnostics(
        x_min=float(x[0]),
        wet_slope=wet_limit_slope(x, y),
        int_abs_curv=integrated_abs_curvature(x, y),
        chord_dev=max_chord_deviation(x, y),
    )
