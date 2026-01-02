"""
plotting.py

Plot helpers for the CR numerical laboratory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pathlib

import numpy as np
import matplotlib.pyplot as plt

def savefig(fig: plt.Figure, path: str | pathlib.Path, dpi: int = 300) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")

def make_xy_axes(title: str = "", xlabel: str = "x = Ep0/Epa", ylabel: str = "y = E/Epa") -> plt.Axes:
    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax
