#!/usr/bin/env python3
"""
neon_15_fig15_atlas_overlay_grid_v2.py

Overlay a *Tier-1 nondimensional CR atlas* on top of each NEON site's binned CR signal.

Updates vs. v1 (per your review)
--------------------------------
- Removes the full daily scatter cloud by default (optional --show_scatter).
- Keeps the binned median curve and the IQR envelope (peach) as the primary observation.
- Makes the atlas curves easier to see (configurable gray + alpha).
- Cleans up the legend (custom handles, sensible order).
- Adds per-column x-axis labels + tick labels on the bottom-most *used* panel in each column
  (so both left and right columns get x labels even when the last row has a blank axis).

Inputs
------
Expects per-site daily files from `neon_10_make_daily_cr_GAPFILL_v5.py` (or later):
  <data_root>/<SITE>_<START>_<END>/tables/<SITE>_daily_cr.csv

Key fields used:
  - x_eq or x_pt (moisture availability coordinate)
  - y (= E/Epa), plus good_day boolean

Outputs
-------
Writes figure(s) under --outdir with stem:
  fig15_neon_atlas_overlay_<xdef>_<start>_<end>.(png/pdf)

Run
---
python3 neon_15_fig15_atlas_overlay_grid_v2.py \
  --data_root "/Volumes/CoryMedia1/data/neon" \
  --outdir "figs/neon" \
  --format png \
  --xdef eq
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl

# ---- Matplotlib font + PDF embedding (crisp, LaTeX-friendly) ----
mpl.rcParams["pdf.fonttype"] = 42           # embed TrueType fonts in PDFs (Type 42)
mpl.rcParams["svg.fonttype"] = "none"       # keep text as text in SVGs
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.dpi"] = 300

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# -------------------------
# Helpers: locate + load
# -------------------------
def _find_site_folder(data_root: Path, site: str) -> Optional[Path]:
    """Return the first folder like '<SITE>_YYYY-.._YYYY-..' under data_root."""
    if not data_root.exists():
        return None
    site = site.upper()
    candidates = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.upper().startswith(site + "_")])
    return candidates[0] if candidates else None


def _load_daily(site: str, data_root: Path) -> pd.DataFrame:
    sd = _find_site_folder(data_root, site)
    if sd is None:
        raise FileNotFoundError(f"Could not find a folder starting with {site}_ under {data_root}")
    f = sd / "tables" / f"{site}_daily_cr.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing daily CR file for {site}: {f}")

    df = pd.read_csv(f)
    if "date" not in df.columns:
        raise KeyError(f"{site}: missing 'date' column in {f}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["site"] = site.upper()

    if "good_day" in df.columns:
        df["good_day"] = df["good_day"].astype(bool)
    else:
        # very loose fallback: needs E and Epa at least
        df["good_day"] = (~df[["E_mm", "Epa_mm"]].isna().any(axis=1)).astype(bool)

    return df


def _parse_pair(s: str, *, name: str) -> Tuple[float, float]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"--{name} must be two comma-separated numbers, got: {s!r}")
    a, b = float(parts[0]), float(parts[1])
    if a > b:
        a, b = b, a
    return a, b


# -------------------------
# Filtering + binning
# -------------------------
def _iqr_filter_within_bins(
    x: np.ndarray, y: np.ndarray, nbins: int, iqr_mult: float, binning: str
) -> np.ndarray:
    """Filter y outliers *within x-bins* using IQR.

    Returns a boolean mask (same length as x/y) selecting points to keep.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20:
        return ok

    xs = pd.Series(x[ok])
    try:
        if binning == "quantile":
            cats = pd.qcut(xs, q=nbins, duplicates="drop")
        else:
            cats = pd.cut(xs, bins=nbins)
    except Exception:
        return ok

    # IMPORTANT: observed=False to silence pandas FutureWarning (and keep legacy behavior)
    g = pd.DataFrame({"x": x[ok], "y": y[ok], "bin": cats}).dropna().groupby("bin", observed=False)

    keep = np.zeros(ok.sum(), dtype=bool)
    for _, sub in g:
        yy = sub["y"].to_numpy(dtype=float)
        if yy.size < 8:
            # not enough points: keep them rather than over-filtering
            keep[sub.index.to_numpy()] = True
            continue
        q25, q75 = np.nanpercentile(yy, [25, 75])
        iqr = q75 - q25
        lo = q25 - iqr_mult * iqr
        hi = q75 + iqr_mult * iqr
        keep[sub.index.to_numpy()] = (yy >= lo) & (yy <= hi)

    out = np.zeros_like(ok, dtype=bool)
    out[np.where(ok)[0]] = keep
    return out


def _bin_curve(x: np.ndarray, y: np.ndarray, nbins: int, min_per_bin: int, binning: str) -> pd.DataFrame:
    """Compute binned median curve with IQR envelope."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < max(20, min_per_bin):
        return pd.DataFrame(columns=["x_mid", "n", "y_med", "y_q25", "y_q75"])

    xs = pd.Series(x[ok])
    try:
        if binning == "quantile":
            cats = pd.qcut(xs, q=nbins, duplicates="drop")
        else:
            cats = pd.cut(xs, bins=nbins)
    except Exception:
        return pd.DataFrame(columns=["x_mid", "n", "y_med", "y_q25", "y_q75"])

    g = pd.DataFrame({"x": x[ok], "y": y[ok], "bin": cats}).dropna().groupby("bin", observed=False)

    rows = []
    for _, sub in g:
        if sub.shape[0] < int(min_per_bin):
            continue
        xm = float(np.nanmedian(sub["x"].to_numpy(dtype=float)))
        yy = sub["y"].to_numpy(dtype=float)
        rows.append(
            {
                "x_mid": xm,
                "n": int(sub.shape[0]),
                "y_med": float(np.nanmedian(yy)),
                "y_q25": float(np.nanpercentile(yy, 25)),
                "y_q75": float(np.nanpercentile(yy, 75)),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["x_mid", "n", "y_med", "y_q25", "y_q75"])

    return pd.DataFrame(rows).sort_values("x_mid").reset_index(drop=True)


# -------------------------
# Atlas curves (Tier-1 examples)
# -------------------------
@dataclass(frozen=True)
class AtlasCurve:
    name: str
    x: np.ndarray
    y: np.ndarray
    style: dict


def _atlas_curves(alpha_pt: float = 1.26) -> Sequence[AtlasCurve]:
    """Return a small, representative Tier-1 atlas family (NOT exhaustive).

    These are meant as visual guides to plausible mechanistic CR trajectories,
    not a calibrated site-specific fit.
    """
    x = np.linspace(0.0, 1.0, 501)
    curves: list[AtlasCurve] = []

    # Reference limits / baselines
    curves.append(AtlasCurve("Equilibrium (y=x)", x, x, {"lw": 1.7}))
    curves.append(AtlasCurve("Penman–Monteith (y=1)", x, np.ones_like(x), {"lw": 1.7}))
    curves.append(AtlasCurve(r"Priestley–Taylor ($y=\alpha x$)", x, alpha_pt * x, {"lw": 1.7}))

    # Linear asymmetric family (b values)
    # From complementarity-plane slope: 1 - x = b(x - y) => y = ((b+1)x - 1)/b
    for b in [0.5, 1.0, 2.0, 4.0]:
        y = ((b + 1.0) * x - 1.0) / b
        y = np.where(y >= 0.0, y, np.nan)  # physical domain
        curves.append(AtlasCurve(f"Linear (b={b:g})", x, y, {"lw": 1.0}))

    # Brutsaert-style 4th-order polynomial family example:
    # y = (2-c)x^2 - (1-2c)x^3 - c x^4   (c in [0,1])
    for c in [0.0, 0.5, 1.0]:
        y = (2.0 - c) * x**2 - (1.0 - 2.0 * c) * x**3 - c * x**4
        curves.append(AtlasCurve(f"Brutsaert poly (c={c:g})", x, y, {"lw": 1.0}))

    # Calibration-free cubic (common reference)
    y = 2.0 * x**2 - x**3
    curves.append(AtlasCurve("Cubic (2x^2 - x^3)", x, y, {"lw": 1.0}))

    return curves


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay a Tier-1 atlas on NEON site mean CR curves.")
    ap.add_argument("--data_root", type=str, required=True, help="Root folder containing <SITE>_YYYY-.. folders")
    ap.add_argument("--outdir", type=str, default="figs/neon", help="Output directory")
    ap.add_argument("--format", type=str, default="png", choices=["png", "pdf", "both"], help="Figure output format")
    ap.add_argument("--xdef", type=str, default="eq", choices=["eq", "pt"], help="Which x to use: eq or pt")
    ap.add_argument("--start", type=str, default="2018-09-01", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, default="2024-06-30", help="End date (YYYY-MM-DD)")
    ap.add_argument("--sites", type=str, default="HARV,OSBS,KONZ,SRER,NIWO,WREF,SJER", help="Comma-separated site IDs")
    ap.add_argument("--tag", type=str, default="", help="Optional tag appended to output filename stem")

    ap.add_argument("--nbins", type=int, default=12, help="Number of bins for median curve")
    ap.add_argument("--min_per_bin", type=int, default=40, help="Min points per bin")
    ap.add_argument("--binning", type=str, default="quantile", choices=["quantile", "uniform"], help="Binning type")
    ap.add_argument("--qclip", type=str, default="0.5,99.5", help="Quantile clip percentiles for x,y (e.g., '0.5,99.5')")
    ap.add_argument("--iqr_mult", type=float, default=2.5, help="IQR multiplier for within-bin outlier filtering")

    ap.add_argument("--alpha_pt", type=float, default=1.26, help="Priestley–Taylor alpha for y=alpha*x reference line")
    ap.add_argument("--ylim", type=str, default="0,1.2", help="y limits, e.g. '0,1.2'")
    ap.add_argument("--xlim", type=str, default="0,1", help="x limits, e.g. '0,1' or '0.8,1.0'")

    # Scatter is OFF by default (per your request)
    ap.add_argument("--show_scatter", action="store_true", help="Plot the daily scatter cloud (debug)")
    ap.add_argument("--scatter_alpha", type=float, default=0.10, help="Scatter alpha (if --show_scatter)")
    ap.add_argument("--point_size", type=float, default=7.0, help="Scatter marker size (if --show_scatter)")

    # Atlas readability
    ap.add_argument("--atlas_color", type=str, default="0.45", help="Gray level for atlas family curves (0=black, 1=white)")
    ap.add_argument("--atlas_alpha", type=float, default=0.40, help="Alpha for atlas family curves")
    ap.add_argument("--atlas_lw", type=float, default=1.0, help="Line width for atlas family curves")

    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)

    sites = [s.strip().upper() for s in str(args.sites).split(",") if s.strip()]
    if not sites:
        raise ValueError("No sites provided via --sites")

    ylo, yhi = _parse_pair(args.ylim, name="ylim")
    xlo, xhi = _parse_pair(args.xlim, name="xlim")

    # Dynamic grid: 2 columns, rows as needed (keeps the fig13/fig14 feel)
    ncols = 2
    nrows = int(math.ceil(len(sites) / ncols))
    fig_h = 2.65 * nrows + 1.6  # tuned so 4 rows ~ 12-ish
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11.2, fig_h), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    # Precompute atlas curves
    atlas = _atlas_curves(alpha_pt=float(args.alpha_pt))

    # Consistent colors
    c_eq = "tab:orange"
    c_pt = "tab:green"
    c_pm = "tab:blue"
    c_atlas = str(args.atlas_color)

    # NEON styling
    c_neon_line = "tab:blue"  # keep consistent with your earlier figs
    c_neon_iqr = "#F2C9AC"    # peach

    qlo, qhi = _parse_pair(args.qclip, name="qclip")

    for i, site in enumerate(sites):
        ax = axes[i]

        df = _load_daily(site, data_root)
        df = df[df["good_day"]].copy()
        df = df[(df["date"] >= start) & (df["date"] <= end)].copy()

        # Choose x definition
        if args.xdef == "eq":
            if "x_eq" not in df.columns:
                raise KeyError(f"{site}: missing x_eq in daily file (need neon_10_make_daily_cr_GAPFILL_v5+)")
            x = df["x_eq"].to_numpy(dtype=float)
            xlab = r"Moisture availability, $x = E_{eq}/E_{pa}$"
            title = r"NEON atlas overlay (xdef=eq)"
        else:
            if "x_pt" not in df.columns:
                raise KeyError(f"{site}: missing x_pt in daily file")
            x = df["x_pt"].to_numpy(dtype=float)
            xlab = r"Moisture availability, $x = E_{pt}/E_{pa}$"
            title = r"NEON atlas overlay (xdef=pt)"

        if "y" not in df.columns:
            raise KeyError(f"{site}: missing y (=E/Epa) column in daily file")
        y = df["y"].to_numpy(dtype=float)

        # Hard finite filter + quantile clip (global, before within-bin filtering)
        ok = np.isfinite(x) & np.isfinite(y)
        x0, y0 = x[ok], y[ok]
        if x0.size < 20:
            ax.text(0.5, 0.5, f"{site}: too few points", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue

        # Clip extreme x/y tails (keeps plot stable)
        x_lo, x_hi = np.nanpercentile(x0, [qlo, qhi])
        y_lo, y_hi = np.nanpercentile(y0, [qlo, qhi])
        ok2 = (x0 >= x_lo) & (x0 <= x_hi) & (y0 >= y_lo) & (y0 <= y_hi)
        x1, y1 = x0[ok2], y0[ok2]

        # Within-bin IQR filter on y (removes distracting vertical outliers)
        mask = _iqr_filter_within_bins(
            x1, y1, nbins=int(args.nbins), iqr_mult=float(args.iqr_mult), binning=str(args.binning)
        )
        x2, y2 = x1[mask], y1[mask]
        n = int(np.isfinite(x2).sum())

        # ---- draw atlas curves ----
        # Reference lines are colored; the rest are a visible gray family
        for c in atlas:
            if c.name.startswith("Equilibrium"):
                ax.plot(c.x, c.y, color=c_eq, lw=1.7, zorder=1, label="Equilibrium (y=x)" if i == 0 else None)
            elif c.name.startswith("Penman"):
                ax.plot(c.x, c.y, color=c_pm, lw=1.7, zorder=1, label="Penman–Monteith (y=1)" if i == 0 else None)
            elif "Priestley" in c.name:
                ax.plot(c.x, c.y, color=c_pt, lw=1.7, zorder=1, label=r"Priestley–Taylor ($y=\alpha x$)" if i == 0 else None)
            else:
                ax.plot(
                    c.x,
                    c.y,
                    color=c_atlas,
                    lw=float(args.atlas_lw),
                    alpha=float(args.atlas_alpha),
                    zorder=0,
                )

        # ---- NEON: optional scatter (debug) ----
        if bool(args.show_scatter):
            ax.scatter(
                x2,
                y2,
                s=float(args.point_size),
                alpha=float(args.scatter_alpha),
                color=c_neon_line,
                zorder=2,
                label="NEON daily" if i == 0 else None,
            )

        # ---- NEON: binned curve + IQR envelope ----
        curve = _bin_curve(
            x2,
            y2,
            nbins=int(args.nbins),
            min_per_bin=int(args.min_per_bin),
            binning=str(args.binning),
        )
        if not curve.empty:
            ax.fill_between(
                curve["x_mid"],
                curve["y_q25"],
                curve["y_q75"],
                color=c_neon_iqr,
                alpha=0.35,
                zorder=2,
                label="NEON IQR (binned)" if i == 0 else None,
            )
            ax.plot(
                curve["x_mid"],
                curve["y_med"],
                marker="o",
                ms=5.5,
                lw=2.2,
                color=c_neon_line,
                zorder=3,
                label="NEON binned median" if i == 0 else None,
            )

        ax.set_ylim(ylo, yhi)
        ax.set_xlim(xlo, xhi)
        ax.grid(True, alpha=0.25, zorder=-10)

        # Panel label
        ax.text(
            0.02,
            0.92,
            f"{site}\n$n$={n}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
        )

        # Only show y tick labels on left column to declutter
        if (i % ncols) != 0:
            ax.tick_params(axis="y", labelleft=False)

    # Turn off unused axes (if sites not multiple of ncols)
    for j in range(len(sites), len(axes)):
        axes[j].axis("off")

    # --- Per-column x labels + tick labels on the bottom-most used axis in each column ---
    # This ensures both columns have x tick labels even when the last row has a blank axis.
    bottom_axes = {}
    for col in range(ncols):
        for row in reversed(range(nrows)):
            idx = row * ncols + col
            if idx < len(sites):
                bottom_axes[col] = axes[idx]
                break
    for ax in bottom_axes.values():
        ax.set_xlabel(xlab)
        ax.tick_params(axis="x", labelbottom=True)

    # Global y label (line-wrapped to avoid overlap in tight layouts)
    fig.text(0.02, 0.5, "Nondimensional evaporation,\n$y = E/E_{pa}$", va="center", rotation="vertical")

    # Title
    fig.suptitle(f"{title}, {start.date()} to {end.date()}", y=0.985, fontsize=14)

    # Clean legend (custom handles; avoids jumbled/duplicated entries)
    legend_handles = [
        Line2D([0], [0], color=c_neon_line, lw=2.2, marker="o", ms=5.5, label="NEON binned median"),
        Patch(facecolor=c_neon_iqr, edgecolor="none", alpha=0.35, label="NEON IQR (binned)"),
        Line2D([0], [0], color=c_eq, lw=1.7, label="Equilibrium (y=x)"),
        Line2D([0], [0], color=c_pt, lw=1.7, label=r"Priestley–Taylor ($y=\alpha x$)"),
        Line2D([0], [0], color=c_pm, lw=1.7, label="Penman–Monteith (y=1)"),
        Line2D([0], [0], color=c_atlas, lw=float(args.atlas_lw), alpha=float(args.atlas_alpha), label="Tier-1 atlas (example curves)"),
    ]
    if bool(args.show_scatter):
        legend_handles.insert(2, Line2D([0], [0], color=c_neon_line, lw=0, marker="o", ms=4, alpha=float(args.scatter_alpha), label="NEON daily (subset)"))

    # Layout tuning: reserve bottom space for legend (and x labels)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.16, hspace=0.22, wspace=0.16)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.045),
        fontsize=10,
        columnspacing=1.6,
        handlelength=2.4,
    )

    # Save
    tag = f"_{args.tag.strip()}" if str(args.tag).strip() else ""
    stem = f"fig15_neon_atlas_overlay{tag}_{args.xdef}_{start.date()}_{end.date()}"
    if args.format in ("png", "both"):
        fp = outdir / f"{stem}.png"
        fig.savefig(fp, dpi=300)
        print(f"Wrote: {fp}")
    if args.format in ("pdf", "both"):
        fp = outdir / f"{stem}.pdf"
        fig.savefig(fp)
        print(f"Wrote: {fp}")

    plt.close(fig)


if __name__ == "__main__":
    main()
