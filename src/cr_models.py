"""
cr_models.py

A small library of complementary relationship (CR) mappings expressed as nondimensional curves
y = f(x), where typically x = Ep0/Epa and y = E/Epa.

These are the "functional axis (F)" elements in the manuscript's experiment matrix.

NOTE:
Some published CR variants use rescaled independent variables (e.g., x_S, x_C) that are not exactly
Ep0/Epa. For the Tier-1 "atlas" figure we include representative forms in a common x in [0,1]
coordinate to visualize curvature/wet-limit/dry-end behavior. The manuscript should label these
curves clearly as rescaled-coordinate comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List
import numpy as np

def symmetric_linear(x: np.ndarray) -> np.ndarray:
    """Bouchet symmetric linear CR in (x,y): y = 2x - 1."""
    return 2.0 * x - 1.0

def asymmetric_linear(x: np.ndarray, b: float) -> np.ndarray:
    """Asymmetric linear CR in (x,y): y = ((1+b)x - 1)/b."""
    b = float(b)
    if b == 0:
        raise ValueError("b must be nonzero.")
    return ((1.0 + b) * x - 1.0) / b

def brutsaert_polynomial(x: np.ndarray, c: float) -> np.ndarray:
    """
    Brutsaert (2015) physically constrained polynomial:
    y = (2-c)x^2 - (1-2c)x^3 - c x^4
    """
    c = float(c)
    return (2.0 - c) * x**2 - (1.0 - 2.0*c) * x**3 - c * x**4

def calibration_free_polynomial(x: np.ndarray) -> np.ndarray:
    """
    Calibration-free cubic (Szilagyi et al. 2017 uses a rescaled coordinate; shape matches c=0 polynomial):
    y = 2x^2 - x^3
    """
    return 2.0 * x**2 - x**3

def sigmoid_family(x: np.ndarray, m: float = 1.0, n: float = 2.0) -> np.ndarray:
    """
    Generic sigmoid-like complementary curve on x in (0,1]:
    y = 1 / (1 + m * (1/x - 1)^n)

    This has y(1)=1 and y->0 as x->0.
    """
    x = np.asarray(x, dtype=float)
    y = np.full_like(x, np.nan, dtype=float)
    eps = 1e-12
    xx = np.clip(x, eps, 1.0)
    y = 1.0 / (1.0 + m * ((1.0/xx) - 1.0)**n)
    return y

@dataclass(frozen=True)
class CRModel:
    name: str
    func: Callable[[np.ndarray], np.ndarray]
    x_min: float
    x_max: float = 1.0

def default_model_set() -> List[CRModel]:
    """
    Return a compact set of model curves for the Tier-1 atlas plot.
    """
    models: List[CRModel] = []
    # Symmetric linear (admissible x >= 0.5 for y>=0)
    models.append(CRModel("Linear symmetric (Bouchet)", symmetric_linear, x_min=0.5))
    # Asymmetric linear examples
    for b in [0.5, 1.0, 2.0, 5.0]:
        xmin = 1.0 / (1.0 + b)
        models.append(CRModel(f"Linear asymmetric (b={b:g})", lambda x, bb=b: asymmetric_linear(x, bb), x_min=xmin))
    # Polynomial family examples
    for c in [0.0, 0.5, 1.0]:
        models.append(CRModel(f"Brutsaert poly (c={c:g})", lambda x, cc=c: brutsaert_polynomial(x, cc), x_min=0.0))
    # Sigmoid examples
    for (m, n) in [(1.0, 2.0), (1.0, 4.0), (0.5, 2.0)]:
        models.append(CRModel(f"Sigmoid (m={m:g}, n={n:g})", lambda x, mm=m, nn=n: sigmoid_family(x, mm, nn), x_min=0.0))
    # Calibration-free cubic (shape)
    models.append(CRModel("Cubic (2x^2-x^3)", calibration_free_polynomial, x_min=0.0))
    return models
