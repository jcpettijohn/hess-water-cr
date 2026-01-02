"""
scripts/make_fig_definition_sensitivity_decomposition.py

Generates:
  figures/fig_definition_sensitivity_decomposition.png
  outputs/fig_definition_sensitivity_decomposition.csv
  outputs/fig_definition_sensitivity_decomposition_summary.csv

Purpose (Reviewer 67):
  Provide a quantitative decomposition of definition-induced shifts in the atlas independent
  variable
      x = E_p0 / E_pa
  into (i) an E_p0-only contribution, (ii) an E_pa-only contribution, and (iii) an exact
  interaction/remainder term.

For a chosen baseline definition (0) and an alternative definition (1), for each sampled state i:
  x0_i = Ep0_0_i / Epa_0_i
  x1_i = Ep0_1_i / Epa_1_i

Total shift:
  Δx = x1 - x0

Exact two-factor decomposition (exact closure):
  Δx_p0  = (Ep0_1 / Epa_0) - (Ep0_0 / Epa_0)      [change Ep0 only, hold Epa at baseline]
  Δx_pa  = (Ep0_0 / Epa_1) - (Ep0_0 / Epa_0)      [change Epa only, hold Ep0 at baseline]
  Δx_int = Δx - (Δx_p0 + Δx_pa)                   [interaction remainder]

Notes:
  - If the baseline/alt pair only changes Ep0 (e.g., PT_dry vs PT_wetref), then Δx_pa≈0 and
    Δx_int≈0, and the script quantifies that directly.
  - If you include an alt definition that changes E_pa (e.g., ET0-like, Penman-like, etc.,
    depending on your run_rs_sweep implementation), you will also get nonzero Δx_pa.

Run from repo root:
  python scripts/make_fig_definition_sensitivity_decomposition.py

Optional:
  python scripts/make_fig_definition_sensitivity_decomposition.py --baseline PT_wetref --alts PT_dry ML_wetconst
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.box_model import BoxParams, run_rs_sweep

try:
    # Your repo helper (preferred, if available)
    from src.plotting import savefig  # type: ignore
except Exception:
    savefig = None


def _as_float(x: Any) -> float:
    """Best-effort float conversion for scalar-like values."""
    try:
        return float(x)
    except Exception:
        return float(np.asarray(x).item())


def _maybe_add_series(
    row: Dict[str, Any],
    out: Dict[str, Any],
    key: str,
    colname: str,
    i: int,
) -> None:
    """
    Add out[key][i] to row[colname] if present. Handles scalar values too.
    Silently skips if key is not present.
    """
    if key not in out:
        return

    v = out[key]
    arr = np.asarray(v)

    if arr.ndim == 0:
        row[colname] = _as_float(arr)
    else:
        if i >= len(arr):
            return
        row[colname] = _as_float(arr[i])


def decompose_dx(
    Ep0_base: np.ndarray,
    Epa_base: np.ndarray,
    Ep0_alt: np.ndarray,
    Epa_alt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Ep0_base = np.asarray(Ep0_base, dtype=float)
    Epa_base = np.asarray(Epa_base, dtype=float)
    Ep0_alt = np.asarray(Ep0_alt, dtype=float)
    Epa_alt = np.asarray(Epa_alt, dtype=float)

    x_base = Ep0_base / Epa_base
    x_alt = Ep0_alt / Epa_alt
    dx_total = x_alt - x_base

    dx_p0 = (Ep0_alt / Epa_base) - (Ep0_base / Epa_base)
    dx_pa = (Ep0_base / Epa_alt) - (Ep0_base / Epa_base)
    dx_int = dx_total - (dx_p0 + dx_pa)

    return x_base, x_alt, dx_total, dx_p0, dx_pa, dx_int


def summarize_components(
    dx_total: np.ndarray,
    dx_p0: np.ndarray,
    dx_pa: np.ndarray,
    dx_int: np.ndarray,
) -> Dict[str, float]:
    """Compact quantitative summary for the paper / caption / response to reviewer."""
    dx_total = np.asarray(dx_total, dtype=float)
    dx_p0 = np.asarray(dx_p0, dtype=float)
    dx_pa = np.asarray(dx_pa, dtype=float)
    dx_int = np.asarray(dx_int, dtype=float)

    abs_total = np.abs(dx_total)
    abs_p0 = np.abs(dx_p0)
    abs_pa = np.abs(dx_pa)
    abs_int = np.abs(dx_int)

    denom = float(np.mean(abs_total)) if float(np.mean(abs_total)) != 0.0 else np.nan

    return {
        "mean_abs_dx": float(np.mean(abs_total)),
        "median_abs_dx": float(np.median(abs_total)),
        "max_abs_dx": float(np.max(abs_total)),
        "mean_abs_dx_p0": float(np.mean(abs_p0)),
        "mean_abs_dx_pa": float(np.mean(abs_pa)),
        "mean_abs_dx_int": float(np.mean(abs_int)),
        "frac_mean_abs_due_to_p0": float(np.mean(abs_p0) / denom) if np.isfinite(denom) else np.nan,
        "frac_mean_abs_due_to_pa": float(np.mean(abs_pa) / denom) if np.isfinite(denom) else np.nan,
        "frac_mean_abs_due_to_int": float(np.mean(abs_int) / denom) if np.isfinite(denom) else np.nan,
        "closure_max_abs_err": float(np.max(np.abs(dx_total - (dx_p0 + dx_pa + dx_int)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tier-1B definition sensitivity decomposition for x=Ep0/Epa."
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="PT_wetref",
        help="Baseline definition string passed to run_rs_sweep (default: PT_wetref).",
    )
    parser.add_argument(
        "--alts",
        nargs="+",
        default=["PT_dry", "ML_wetconst"],
        help="One or more alternative definition strings (default: PT_dry ML_wetconst).",
    )
    parser.add_argument(
        "--rs_n",
        type=int,
        default=40,
        help="Number of log-spaced rs points between 10^0 and 10^4 (default: 40).",
    )
    parser.add_argument(
        "--rs_min",
        type=float,
        default=0.0,
        help="Include an explicit rs_min point (default: 0.0).",
    )
    parser.add_argument(
        "--rs_log_min",
        type=float,
        default=0.0,
        help="log10(rs) lower bound for logspace grid (default: 0 => rs=1).",
    )
    parser.add_argument(
        "--rs_log_max",
        type=float,
        default=4.0,
        help="log10(rs) upper bound for logspace grid (default: 4 => rs=1e4).",
    )

    args = parser.parse_args()

    params = BoxParams()

    # Match your current sampling style: include rs=0 plus a log-spaced sweep.
    rs = np.logspace(args.rs_log_min, args.rs_log_max, int(args.rs_n))
    if args.rs_min is not None:
        rs = np.concatenate([np.array([float(args.rs_min)]), rs])

    baseline_def = args.baseline
    alt_defs = list(args.alts)

    out_base = run_rs_sweep(params, rs, definition=baseline_def)
    if "E_p0" not in out_base or "E_pa" not in out_base:
        raise KeyError(
            f"run_rs_sweep(..., definition={baseline_def!r}) must return keys "
            f"'E_p0' and 'E_pa'. Got keys: {sorted(out_base.keys())}"
        )

    Ep0_base = np.asarray(out_base["E_p0"], dtype=float)
    Epa_base = np.asarray(out_base["E_pa"], dtype=float)
    E_base = np.asarray(out_base.get("E", np.full_like(Ep0_base, np.nan)), dtype=float)

    fig, axes = plt.subplots(
        nrows=len(alt_defs),
        ncols=1,
        figsize=(7.6, 3.4 * len(alt_defs)),
        sharex=True,
    )
    if len(alt_defs) == 1:
        axes = [axes]

    # Optional extra state variables to archive if run_rs_sweep provides them.
    OPTIONAL_KEYS = [
        # common; adjust names if your run_rs_sweep uses different ones
        "T", "q", "T_star", "q_star", "T_w", "q_w",
        "VPD", "RH", "u", "RnG", "R_nG",
    ]

    rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    used_axes = 0

    for ddef in alt_defs:
        try:
            out_alt = run_rs_sweep(params, rs, definition=ddef)
        except Exception as e:
            print(f"[skip] definition={ddef!r} not available or failed: {e}")
            continue

        if "E_p0" not in out_alt or "E_pa" not in out_alt:
            print(
                f"[skip] definition={ddef!r} did not provide required keys "
                f"'E_p0' and 'E_pa'. Got keys: {sorted(out_alt.keys())}"
            )
            continue

        Ep0_alt = np.asarray(out_alt["E_p0"], dtype=float)
        Epa_alt = np.asarray(out_alt["E_pa"], dtype=float)
        E_alt = np.asarray(out_alt.get("E", np.full_like(Ep0_alt, np.nan)), dtype=float)

        x_base, x_alt, dx_total, dx_p0, dx_pa, dx_int = decompose_dx(
            Ep0_base, Epa_base, Ep0_alt, Epa_alt
        )

        s = summarize_components(dx_total, dx_p0, dx_pa, dx_int)
        summary_rows.append(
            {
                "definition_baseline": baseline_def,
                "definition_alt": ddef,
                **s,
            }
        )

        for i in range(len(rs)):
            row: Dict[str, Any] = {
                "definition_baseline": baseline_def,
                "definition_alt": ddef,
                "rs": float(rs[i]),
                "E_base": float(E_base[i]) if np.isfinite(E_base[i]) else np.nan,
                "E_pa_base": float(Epa_base[i]),
                "E_p0_base": float(Ep0_base[i]),
                "E_alt": float(E_alt[i]) if np.isfinite(E_alt[i]) else np.nan,
                "E_pa_alt": float(Epa_alt[i]),
                "E_p0_alt": float(Ep0_alt[i]),
                "x_base": float(x_base[i]),
                "x_alt": float(x_alt[i]),
                "dx_total": float(dx_total[i]),
                "dx_p0": float(dx_p0[i]),
                "dx_pa": float(dx_pa[i]),
                "dx_int": float(dx_int[i]),
            }

            for k in OPTIONAL_KEYS:
                _maybe_add_series(row, out_base, k, f"{k}_base", i)
                _maybe_add_series(row, out_alt, k, f"{k}_alt", i)

            rows.append(row)

        ax = axes[used_axes]
        order = np.argsort(x_base)
        xb = x_base[order]

        ax.axhline(0.0, linewidth=1.0)
        ax.plot(xb, dx_total[order], linewidth=2.0, label=r"$\Delta x$ (total)")
        ax.plot(xb, dx_p0[order], linewidth=1.6, label=r"$\Delta x_{p0}$ (vary $E_{p0}$ only)")
        ax.plot(xb, dx_pa[order], linewidth=1.6, label=r"$\Delta x_{pa}$ (vary $E_{pa}$ only)")
        ax.plot(xb, dx_int[order], linewidth=1.6, label=r"$\Delta x_{\mathrm{int}}$ (remainder)")

        frac_p0 = s["frac_mean_abs_due_to_p0"]
        frac_pa = s["frac_mean_abs_due_to_pa"]
        ax.set_title(
            f"{ddef} relative to {baseline_def}  "
            f"(mean|Δx|={s['mean_abs_dx']:.3g}; "
            f"frac p0={frac_p0:.2g}, frac pa={frac_pa:.2g})"
        )

        ax.set_ylabel(r"$\Delta x$")
        ax.grid(True, alpha=0.3)

        if used_axes == 0:
            ax.legend(frameon=False, fontsize=9)

        used_axes += 1

    if used_axes == 0:
        raise RuntimeError(
            "No alternative definitions could be plotted. "
            "Edit --alts or confirm run_rs_sweep supports those definition strings."
        )

    for k in range(used_axes, len(axes)):
        fig.delaxes(axes[k])

    axes[used_axes - 1].set_xlabel(r"baseline $x = E_{p0}/E_{pa}$")
    fig.suptitle(r"Tier-1B definition sensitivity: exact decomposition of $\Delta x$", y=0.985)
    fig.tight_layout()

    out_csv = ROOT / "outputs" / "fig_definition_sensitivity_decomposition.csv"
    out_sum = ROOT / "outputs" / "fig_definition_sensitivity_decomposition_summary.csv"
    out_png = ROOT / "figures" / "fig_definition_sensitivity_decomposition.png"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    pd.DataFrame(summary_rows).to_csv(out_sum, index=False)

    print(f"[write] {out_csv}")
    print(f"[write] {out_sum}")

    if savefig is not None:
        savefig(fig, out_png)
    else:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")

    print(f"[write] {out_png}")


if __name__ == "__main__":
    main()
