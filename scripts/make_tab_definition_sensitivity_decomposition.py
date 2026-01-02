"""scripts/make_tab_definition_sensitivity_decomposition.py

Purpose
-------
Generate the *numbers* behind the Tier-1B definition-sensitivity decomposition in x = Ep0/Epa,
so the long multi-panel figure can be replaced by a compact table.

For each sampled state i along the standard Tier-1B surface-resistance sweep, relative to a
baseline definition (default: PT_wetref) and an alternative definition D_k:

  x_base = Ep0_base / Epa_base
  x_alt  = Ep0_alt  / Epa_alt
  Δx     = x_alt - x_base

Exact two-factor decomposition (identical to the manuscript):
  Δx_p0  = (Ep0_alt / Epa_base) - (Ep0_base / Epa_base)   # vary Ep0 only
  Δx_pa  = (Ep0_base / Epa_alt) - (Ep0_base / Epa_base)   # vary Epa only
  Δx_int = Δx - (Δx_p0 + Δx_pa)                           # remainder (exact closure)

Outputs
-------
1) outputs/definition_sensitivity_decomposition_points.csv
   One row per sampled state and per alternative definition (useful for debugging/repro).

2) outputs/definition_sensitivity_decomposition_summary.csv
   Compact per-comparison summary metrics suitable for a manuscript table, including:
     - max|Δx| and max|Δx_component|
     - median(|Δx_component/Δx|) over nonzero shifts (|Δx| > eps)
     - closure error max|Δx - (Δx_p0+Δx_pa+Δx_int)|

3) outputs/definition_sensitivity_decomposition_summary.tex
   A LaTeX tabular snippet you can \\input{} directly (optional convenience).

Usage
-----
From repo root:
  python scripts/make_tab_definition_sensitivity_decomposition.py

Optional arguments:
  --eps 1e-10
  --baseline PT_wetref
  --alts PT_dry ML_wetconst
  --no_tex

Notes
-----
- In the current Tier-1B mechanistic comparisons, E_pa is computed identically across contrasted
  definitions, so Δx_pa ≈ 0 and Δx_int ≈ 0 (up to floating-point noise). This script quantifies that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root is importable when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.box_model import BoxParams, run_rs_sweep  # noqa: E402


def decompose_dx(
    Ep0_base: np.ndarray,
    Epa_base: np.ndarray,
    Ep0_alt: np.ndarray,
    Epa_alt: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return arrays for x_base, x_alt and exact Δx decomposition terms."""
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

    # closure check (should be ~ machine precision)
    closure = dx_total - (dx_p0 + dx_pa + dx_int)

    return {
        "x_base": x_base,
        "x_alt": x_alt,
        "dx_total": dx_total,
        "dx_p0": dx_p0,
        "dx_pa": dx_pa,
        "dx_int": dx_int,
        "closure": closure,
    }


def summarize_decomposition(arrs: dict[str, np.ndarray], eps: float) -> dict[str, float]:
    """Compute summary scalars for table reporting."""
    dx = arrs["dx_total"]
    dx_p0 = arrs["dx_p0"]
    dx_pa = arrs["dx_pa"]
    dx_int = arrs["dx_int"]
    closure = arrs["closure"]
    x_base = arrs["x_base"]

    finite = np.isfinite(dx) & np.isfinite(dx_p0) & np.isfinite(dx_pa) & np.isfinite(dx_int)
    nonzero = finite & (np.abs(dx) > eps)

    def _safe_stat(v: np.ndarray, fn, default=np.nan):
        v = v[np.isfinite(v)]
        return float(fn(v)) if v.size else float(default)

    out: dict[str, float] = {}
    out["n_states"] = int(np.sum(finite))
    out["n_nonzero_dx"] = int(np.sum(nonzero))

    out["x_base_min"] = _safe_stat(x_base[finite], np.min)
    out["x_base_max"] = _safe_stat(x_base[finite], np.max)

    out["dx_total_maxabs"] = _safe_stat(np.abs(dx[finite]), np.max)
    out["dx_p0_maxabs"] = _safe_stat(np.abs(dx_p0[finite]), np.max)
    out["dx_pa_maxabs"] = _safe_stat(np.abs(dx_pa[finite]), np.max)
    out["dx_int_maxabs"] = _safe_stat(np.abs(dx_int[finite]), np.max)

    out["closure_maxabs"] = _safe_stat(np.abs(closure[finite]), np.max)

    # Contribution fractions over nonzero shifts
    if np.any(nonzero):
        frac_p0 = np.abs(dx_p0[nonzero] / dx[nonzero])
        frac_pa = np.abs(dx_pa[nonzero] / dx[nonzero])
        frac_int = np.abs(dx_int[nonzero] / dx[nonzero])

        out["median_abs_frac_p0"] = float(np.median(frac_p0))
        out["median_abs_frac_pa"] = float(np.median(frac_pa))
        out["median_abs_frac_int"] = float(np.median(frac_int))

        out["p95_abs_frac_p0"] = float(np.quantile(frac_p0, 0.95))
        out["p95_abs_frac_pa"] = float(np.quantile(frac_pa, 0.95))
        out["p95_abs_frac_int"] = float(np.quantile(frac_int, 0.95))
    else:
        out["median_abs_frac_p0"] = np.nan
        out["median_abs_frac_pa"] = np.nan
        out["median_abs_frac_int"] = np.nan
        out["p95_abs_frac_p0"] = np.nan
        out["p95_abs_frac_pa"] = np.nan
        out["p95_abs_frac_int"] = np.nan

    return out


def df_to_latex_tabular(df: pd.DataFrame) -> str:
    """Return a compact LaTeX tabular (no table wrapper) for \\input{} usage."""
    # Keep the column order stable and human-readable
    cols = [
        "definition_baseline",
        "definition_alt",
        "n_states",
        "n_nonzero_dx",
        "dx_total_maxabs",
        "dx_p0_maxabs",
        "dx_pa_maxabs",
        "dx_int_maxabs",
        "median_abs_frac_p0",
        "closure_maxabs",
    ]
    cols = [c for c in cols if c in df.columns]
    d = df[cols].copy()

    # Friendly header names
    rename = {
        "definition_baseline": "Baseline",
        "definition_alt": "Alternative",
        "n_states": "$N$",
        "n_nonzero_dx": "$N(|\\Delta x|>\\varepsilon)$",
        "dx_total_maxabs": "$\\max|\\Delta x|$",
        "dx_p0_maxabs": "$\\max|\\Delta x_{p0}|$",
        "dx_pa_maxabs": "$\\max|\\Delta x_{pa}|$",
        "dx_int_maxabs": "$\\max|\\Delta x_{int}|$",
        "median_abs_frac_p0": "$\\mathrm{median}(|\\Delta x_{p0}/\\Delta x|)$",
        "closure_maxabs": "max closure err",
    }
    d = d.rename(columns=rename)

    # Format floats (scientific for very small numbers, fixed otherwise)
    def fmt(x):
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        if not np.isfinite(x):
            return "--"
        ax = abs(float(x))
        if ax != 0.0 and (ax < 1e-4 or ax >= 1e3):
            return f"{x:.2e}"
        return f"{x:.6f}"

    for c in d.columns:
        if c in ["Baseline", "Alternative"]:
            continue
        d[c] = d[c].map(fmt)

    # Build a minimal LaTeX tabular manually to avoid requiring optional
    # dependencies (e.g., Jinja2 via pandas Styler in newer pandas versions).
    #
    # Requires \usepackage{booktabs} in your LaTeX preamble (or replace rules with \hline).
    headers = list(d.columns)

    # Left-align text columns, right-align numeric columns
    align = "".join("l" if h in ["Baseline", "Alternative"] else "r" for h in headers)

    def _tex_escape_text(s: str) -> str:
        # Escape only for plain-text columns (not math-mode headers)
        return (
            str(s)
            .replace("\\", r"\textbackslash{}")
            .replace("_", r"\_")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("#", r"\#")
        )

    lines: list[str] = []
    lines.append(r"\begin{tabular}{" + align + r"}")
    lines.append(r"\toprule")
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")
    for _, row in d.iterrows():
        cells: list[str] = []
        for h, v in zip(headers, row.values):
            if h in ["Baseline", "Alternative"]:
                cells.append(_tex_escape_text(v))
            else:
                cells.append(str(v))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=1e-10, help="Exclude |Δx| <= eps from fraction stats.")
    ap.add_argument("--baseline", type=str, default="PT_wetref", help="Baseline definition string.")
    ap.add_argument(
        "--alts",
        nargs="+",
        default=["PT_dry", "ML_wetconst"],
        help="Alternative definition strings.",
    )
    ap.add_argument("--no_tex", action="store_true", help="Do not write the LaTeX snippet output.")
    args = ap.parse_args()

    params = BoxParams()

    # Match existing Tier-1B definition-sensitivity sampling
    rs = np.concatenate([np.array([0.0]), np.logspace(0, 4, 40)])

    out_base = run_rs_sweep(params, rs, definition=args.baseline)
    Ep0_base = out_base["E_p0"]
    Epa_base = out_base["E_pa"]

    # --- Per-point archive (optional but good for auditability)
    point_rows: list[dict[str, float | str]] = []
    summary_rows: list[dict[str, float | str]] = []

    pooled_frac_p0: list[float] = []

    for ddef in args.alts:
        try:
            out_alt = run_rs_sweep(params, rs, definition=ddef)
        except Exception as e:
            print(f"[skip] definition={ddef!r} failed: {e}")
            continue

        arrs = decompose_dx(Ep0_base, Epa_base, out_alt["E_p0"], out_alt["E_pa"])

        # record points
        for i in range(len(rs)):
            point_rows.append(
                {
                    "definition_baseline": args.baseline,
                    "definition_alt": ddef,
                    "rs": float(rs[i]),
                    "x_base": float(arrs["x_base"][i]),
                    "x_alt": float(arrs["x_alt"][i]),
                    "dx_total": float(arrs["dx_total"][i]),
                    "dx_p0": float(arrs["dx_p0"][i]),
                    "dx_pa": float(arrs["dx_pa"][i]),
                    "dx_int": float(arrs["dx_int"][i]),
                    "closure": float(arrs["closure"][i]),
                }
            )

        # summary metrics for table
        summ = summarize_decomposition(arrs, eps=args.eps)
        summ.update(
            {
                "definition_baseline": args.baseline,
                "definition_alt": ddef,
                "eps": float(args.eps),
            }
        )
        summary_rows.append(summ)

        # pooled median(|Δx_p0/Δx|) across all nonzero points (for single-scalar reporting)
        dx = arrs["dx_total"]
        nonzero = np.isfinite(dx) & (np.abs(dx) > args.eps)
        if np.any(nonzero):
            pooled_frac_p0.extend(list(np.abs(arrs["dx_p0"][nonzero] / dx[nonzero])))

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    points_csv = out_dir / "definition_sensitivity_decomposition_points.csv"
    summary_csv = out_dir / "definition_sensitivity_decomposition_summary.csv"
    latex_tex = out_dir / "definition_sensitivity_decomposition_summary.tex"

    if point_rows:
        pd.DataFrame(point_rows).to_csv(points_csv, index=False)
        print(f"[write] {points_csv}")
    else:
        print("[warn] no point rows were generated (all definitions skipped?)")

    if summary_rows:
        df = pd.DataFrame(summary_rows).sort_values(["definition_alt"])
        df.to_csv(summary_csv, index=False)
        print(f"[write] {summary_csv}")
        print("\nSummary (copy/paste-friendly):")
        print(df.to_string(index=False))

        if pooled_frac_p0:
            pooled_median = float(np.median(np.asarray(pooled_frac_p0)))
            print(
                f"\nPooled median(|Δx_p0/Δx|) over all nonzero shifts (eps={args.eps:g}): {pooled_median:.6f}"
            )
        else:
            print("\nPooled median(|Δx_p0/Δx|): n/a (no nonzero shifts after eps filtering)")

        if not args.no_tex:
            latex = df_to_latex_tabular(df)
            latex_tex.write_text(latex)
            print(f"[write] {latex_tex}")
    else:
        print("[warn] no summary rows were generated (all definitions skipped?)")


if __name__ == "__main__":
    main()
