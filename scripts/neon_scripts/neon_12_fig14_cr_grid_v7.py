#!/usr/bin/env python3
"""
neon_12_fig14_cr_grid_v6.py

Figure 14: NEON complementarity relationship (CR) small-multiples (multi-site grid).

Key change vs v5
----------------
Your atlas and theory plots use the nondimensional coordinate:

    x = E_eq / E_pa     (0 <= x <= 1)

where E_eq is *equilibrium evaporation* (radiation term only), and E_pa is
apparent potential evaporation (Penman-style / saturated-surface PM).

In the previous script you were often plotting:

    x_pt = E_p0,PT / E_pa = alpha_PT * E_eq / E_pa

With alpha_PT ≈ 1.26, if E_pa is close to E_eq (small aerodynamic term), then
x_pt clusters near ~1.26 and Matplotlib will "zoom" the axis so all tick labels
look identical (e.g., 1.2600).

This v6 script defaults to the atlas-compatible definition (xdef="eq") and a
fixed x-axis range of 0–1 so you can directly compare against the CR atlas.

Inputs
------
Expects daily files produced by your NEON daily pipeline (e.g., neon_10_make_daily_cr_GAPFILL.py):

    <data_root>/<SITE>_YYYY-MM_YYYY-MM/tables/<SITE>_daily_cr.csv

Outputs
-------
Writes to --outdir:

    fig14_neon_cr_grid_<xdef>_<start>_<end>.pdf
    fig14_neon_cr_grid_<xdef>_<start>_<end>.png
    (optional) fig14_neon_cr_grid_<xdef>_<start>_<end>_binned.csv

Usage
-----
python3 neon_12_fig14_cr_grid_v6.py \
  --data_root "/Volumes/CoryMedia1/data/neon" \
  --outdir "figs/neon" \
  --start 2018-09-01 --end 2024-06-30 \
  --format both \
  --write_bins_csv
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl

# ---- Matplotlib font + PDF embedding (crisp, LaTeX-friendly) ----
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.dpi"] = 300

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


CORE_CONUS_7 = ["HARV", "OSBS", "KONZ", "SRER", "NIWO", "WREF", "SJER"]


def _find_site_folder(data_root: Path, site: str) -> Optional[Path]:
    """Return the first folder under data_root that matches 'SITE_*'."""
    site = site.upper()
    candidates = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.upper().startswith(site + "_")])
    return candidates[0] if candidates else None


def _load_daily(site: str, data_root: Path) -> pd.DataFrame:
    """Load <SITE>_daily_cr.csv from the standard folder layout."""
    sd = _find_site_folder(data_root, site)
    if sd is None:
        raise FileNotFoundError(f"Could not find a folder starting with {site}_ under {data_root}")
    f = sd / "tables" / f"{site}_daily_cr.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing daily CR file for {site}: {f}")
    df = pd.read_csv(f)
    if "date" not in df.columns:
        raise KeyError(f"{site}: daily file missing 'date' column: {f}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["site"] = site
    if "good_day" in df.columns:
        df["good_day"] = df["good_day"].astype(bool)
    else:
        df["good_day"] = (~df[["E_mm", "Epa_mm"]].isna().any(axis=1)).astype(bool)
    return df


def _qclip_mask(arr: np.ndarray, qlo: float, qhi: float) -> np.ndarray:
    """Boolean mask keeping values between qlo and qhi percentiles."""
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 10:
        return np.ones_like(arr, dtype=bool)
    lo, hi = np.nanpercentile(a, [qlo, qhi])
    return np.isfinite(arr) & (arr >= lo) & (arr <= hi)


def _iqr_filter_within_bins(x: np.ndarray, y: np.ndarray, nbins: int, iqr_mult: float, binning: str) -> np.ndarray:
    """Return mask after filtering y outliers within x-bins using IQR."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20:
        return ok

    s = pd.Series(x[ok])
    if binning == "quantile":
        cats = pd.qcut(s, q=nbins, duplicates="drop")
    else:
        cats = pd.cut(s, bins=nbins)

    tmp = pd.DataFrame({"x": x[ok], "y": y[ok], "bin": cats}).dropna()

    keep = np.zeros(len(tmp), dtype=bool)
    # observed=True silences the pandas FutureWarning and is correct for categorical bins.
    for _, g in tmp.groupby("bin", observed=True):
        yy = g["y"].to_numpy(dtype=float)
        if yy.size < 8:
            keep[g.index.to_numpy()] = True
            continue
        q1, q3 = np.nanpercentile(yy, [25, 75])
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= 0:
            keep[g.index.to_numpy()] = True
            continue
        lo = q1 - iqr_mult * iqr
        hi = q3 + iqr_mult * iqr
        keep[g.index.to_numpy()] = (yy >= lo) & (yy <= hi)

    # Map back to full-length mask
    out = np.zeros_like(ok)
    out[np.where(ok)[0][tmp.index.to_numpy()]] = keep
    return out


def _bin_curve(x: np.ndarray, y: np.ndarray, nbins: int, min_per_bin: int, binning: str) -> pd.DataFrame:
    """Compute a binned summary curve y(x) with an IQR envelope.

    This is a *display* helper: it does not affect the point-level filtering.
    We bin the daily points in x and compute the median (and IQR) of y in each bin.

    Important robustness note
    -------------------------
    With quantile binning (default), if ``nbins`` is too large relative to the
    available sample size and ``min_per_bin``, it is possible for *every* bin
    to fall below ``min_per_bin`` (e.g., NIWO has fewer good days). Earlier
    versions then crashed when trying to sort an empty table.

    Here we (i) adaptively reduce the number of bins so the requested minimum
    occupancy is achievable, and (ii) always return a well-formed (possibly
    empty) DataFrame with the expected columns.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())

    cols = ["x_mid", "n", "y_med", "y_mean", "y_q25", "y_q75"]

    # Not enough points to bin robustly
    if n < max(20, int(min_per_bin)):
        return pd.DataFrame(columns=cols)

    nbins_req = max(1, int(nbins))
    minreq = max(1, int(min_per_bin))

    # Maximum number of bins that can have >= minreq points *on average*.
    # For quantile binning this is a good proxy; for uniform bins it still
    # prevents pathological 'all bins too small' cases.
    max_bins = max(1, n // minreq)
    nbins_eff = min(nbins_req, max_bins)

    xs = pd.Series(x[ok])

    try:
        if str(binning).lower() == "quantile":
            cats = pd.qcut(xs, q=nbins_eff, duplicates="drop")
        else:
            cats = pd.cut(xs, bins=nbins_eff)
    except Exception:
        # Pathological cases (nearly-constant x, etc.)
        cats = pd.cut(xs, bins=1)

    tmp = pd.DataFrame({"x": x[ok], "y": y[ok], "bin": cats}).dropna()

    rows: List[dict] = []
    for _, g in tmp.groupby("bin", observed=True):
        if len(g) < minreq:
            continue
        yy = g["y"].to_numpy(dtype=float)
        xx = g["x"].to_numpy(dtype=float)
        rows.append(
            {
                "x_mid": float(np.nanmedian(xx)),
                "n": int(len(g)),
                "y_med": float(np.nanmedian(yy)),
                "y_mean": float(np.nanmean(yy)),
                "y_q25": float(np.nanpercentile(yy, 25)),
                "y_q75": float(np.nanpercentile(yy, 75)),
            }
        )

    if not rows:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(rows).sort_values("x_mid").reset_index(drop=True)
    return out
def _prepare_site_points(
    df: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    xdef: str,
    alpha_pt: float,
    min_epa: float,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    qclip: Tuple[float, float],
    iqr_filter: bool,
    iqr_mult: float,
    nbins: int,
    binning: str,
) -> pd.DataFrame:
    """Return filtered points with columns x, y."""
    d = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    d = d[d["good_day"].astype(bool)].copy()

    need = ["E_mm", "Epa_mm", "Ep0_eq_mm", "Ep0_pt_mm"]
    for c in need:
        if c not in d.columns:
            raise KeyError(f"{df['site'].iloc[0]}: missing required column '{c}' in daily file")

    # Base numeric and sign sanity
    for c in need:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["E_mm", "Epa_mm", "Ep0_eq_mm"]).copy()
    d = d[d["Epa_mm"] > float(min_epa)].copy()

    if xdef == "eq":
        # Atlas coordinate: x = E_eq / E_pa  (expected in [0, 1])
        d["x"] = d["Ep0_eq_mm"] / d["Epa_mm"]
    else:
        # PT coordinate: x = E_p0,PT / E_pa = alpha * E_eq / E_pa
        d["x"] = d["Ep0_pt_mm"] / d["Epa_mm"]

    d["y"] = d["E_mm"] / d["Epa_mm"]

    # Physical-ish hard range filter
    d = d[(d["x"] >= x_range[0]) & (d["x"] <= x_range[1]) & (d["y"] >= y_range[0]) & (d["y"] <= y_range[1])].copy()

    # Quantile clip to remove extreme tails (helps with weird instrument days)
    if qclip is not None:
        mx = _qclip_mask(d["x"].to_numpy(dtype=float), float(qclip[0]), float(qclip[1]))
        my = _qclip_mask(d["y"].to_numpy(dtype=float), float(qclip[0]), float(qclip[1]))
        d = d[mx & my].copy()

    # Within-bin IQR filtering of y (keeps main signal, removes vertical outliers)
    if iqr_filter and len(d) >= 40:
        mask = _iqr_filter_within_bins(
            d["x"].to_numpy(dtype=float),
            d["y"].to_numpy(dtype=float),
            nbins=int(nbins),
            iqr_mult=float(iqr_mult),
            binning=str(binning),
        )
        d = d[mask].copy()

    # Helpful for later diagnostics
    if xdef == "pt":
        d["x_eq_equiv"] = d["x"] / float(alpha_pt)

    return d


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Make NEON multi-site CR grid (Figure 14).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_root", required=True, help="Root containing site folders like HARV_2018-01_2024-12/")
    p.add_argument("--outdir", default="figs/neon", help="Output directory for figures")
    p.add_argument("--sites", nargs="+", default=CORE_CONUS_7, help="Sites to include (order matters)")
    p.add_argument("--start", default="2018-09-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default="2024-06-30", help="End date (YYYY-MM-DD)")

    # Coordinate choice
    p.add_argument(
        "--xdef",
        choices=["eq", "pt"],
        default="eq",
        help="x definition: 'eq' uses E_eq/E_pa (atlas coordinate, 0–1). 'pt' uses Ep0,PT/E_pa (~alpha * eq).",
    )
    p.add_argument("--alpha_pt", type=float, default=1.26, help="Priestley–Taylor alpha (for reference line y=alpha*x).")

    # Filtering / QC controls
    p.add_argument("--min_Epa", type=float, default=0.1, help="Minimum Epa (mm/d) to keep a day (avoids divide-by-small).")
    p.add_argument(
        "--x_range",
        nargs=2,
        type=float,
        default=(0.0, 1.0),
        metavar=("XMIN", "XMAX"),
        help="Hard x-range filter applied before robust clipping.",
    )
    p.add_argument(
        "--y_range",
        nargs=2,
        type=float,
        default=(0.0, 1.2),
        metavar=("YMIN", "YMAX"),
        help="Hard y-range filter (0–1.2 keeps occasional closure outliers).",
    )
    p.add_argument(
        "--qclip",
        nargs=2,
        type=float,
        default=(0.5, 99.5),
        metavar=("QLO", "QHI"),
        help="Quantile clip applied independently to x and y (percent). Set to '0 100' to disable.",
    )
    p.add_argument("--iqr_filter", action="store_true", default=True, help="Apply IQR outlier filter within x-bins.")
    p.add_argument("--iqr_mult", type=float, default=3.0, help="IQR multiplier for within-bin y filtering.")
    p.add_argument("--nbins", type=int, default=12, help="Number of x-bins used for filtering and binned curve.")
    p.add_argument(
        "--binning",
        choices=["quantile", "uniform"],
        default="quantile",
        help="Binning strategy for curve and IQR filter.",
    )
    p.add_argument("--min_per_bin", type=int, default=30, help="Minimum points per bin to draw binned curve.")

    # Axes / layout
    p.add_argument("--ncols", type=int, default=2, help="Number of columns in the grid layout.")
    p.add_argument(
        "--format",
        choices=["pdf", "png", "both"],
        default="both",
        help="Write PDF, PNG, or both.",
    )
    p.add_argument("--write_bins_csv", action="store_true", help="Write binned-curve summary CSV.")

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root).expanduser()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    sites = [s.upper() for s in args.sites]
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    # Load + filter per site
    per_site: Dict[str, pd.DataFrame] = {}
    for site in sites:
        df = _load_daily(site, data_root)
        per_site[site] = _prepare_site_points(
            df,
            start=start,
            end=end,
            xdef=str(args.xdef),
            alpha_pt=float(args.alpha_pt),
            min_epa=float(args.min_Epa),
            x_range=(float(args.x_range[0]), float(args.x_range[1])),
            y_range=(float(args.y_range[0]), float(args.y_range[1])),
            qclip=(float(args.qclip[0]), float(args.qclip[1])),
            iqr_filter=bool(args.iqr_filter),
            iqr_mult=float(args.iqr_mult),
            nbins=int(args.nbins),
            binning=str(args.binning),
        )

    # Figure layout
    ncols = max(1, int(args.ncols))
    nrows = int(math.ceil(len(sites) / ncols))

    fig_w = 12.0
    fig_h = 2.7 * nrows + 1.6
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharey=True)
    if isinstance(axes, np.ndarray):
        ax_list = axes.flatten().tolist()
    else:
        ax_list = [axes]

    used_flags = [False] * len(ax_list)

    # Track bottom-most used axis per column so BOTH columns get x tick labels even with an empty last panel.
    bottom_row_for_col = {c: -1 for c in range(ncols)}
    for i, site in enumerate(sites):
        row, col = divmod(i, ncols)
        bottom_row_for_col[col] = max(bottom_row_for_col[col], row)

    # Build binned CSV rows if requested
    bins_rows: List[dict] = []

    for i, site in enumerate(sites):
        ax = ax_list[i]
        used_flags[i] = True
        d = per_site[site]

        x = d["x"].to_numpy(dtype=float)
        y = d["y"].to_numpy(dtype=float)

        # Scatter cloud
        ax.scatter(x, y, s=10, alpha=0.18, rasterized=True, label="Actual (daily)")

        # Binned curve
        curve = _bin_curve(x, y, nbins=int(args.nbins), min_per_bin=int(args.min_per_bin), binning=str(args.binning))
        if not curve.empty:
            ax.plot(curve["x_mid"], curve["y_med"], marker="o", linewidth=2.0, label="Binned median")
            ax.fill_between(curve["x_mid"], curve["y_q25"], curve["y_q75"], alpha=0.12, linewidth=0)

            if bool(args.write_bins_csv):
                for _, r in curve.iterrows():
                    bins_rows.append(
                        {
                            "site": site,
                            "x_mid": float(r["x_mid"]),
                            "n": int(r["n"]),
                            "y_med": float(r["y_med"]),
                            "y_mean": float(r["y_mean"]),
                            "y_q25": float(r["y_q25"]),
                            "y_q75": float(r["y_q75"]),
                            "xdef": str(args.xdef),
                            "start": str(args.start),
                            "end": str(args.end),
                        }
                    )

        # Reference lines (drawn over the whole fixed x-range)
        x0, x1 = float(args.x_range[0]), float(args.x_range[1])
        xs = np.array([x0, x1], dtype=float)

        if args.xdef == "eq":
            # Equilibrium line (E = E_eq): y = x
            ax.plot(xs, xs, linewidth=1.6, label="Equilibrium (y = x)")
            # Priestley–Taylor wet-environment surrogate: E = alpha * E_eq => y = alpha * x
            ax.plot(xs, float(args.alpha_pt) * xs, linewidth=1.6, label=r"Priestley–Taylor (y = $\alpha x$)")
        else:
            # In PT coordinates, PT line is y = x
            ax.plot(xs, xs, linewidth=1.6, label="Priestley–Taylor (y = x)")

        # Penman–Monteith upper limit: E = Epa
        ax.axhline(1.0, linewidth=1.6, label="Penman–Monteith (y = 1)")

        # Cosmetics
        n_used = int(np.isfinite(x).sum())
        ax.text(
            0.02,
            0.95,
            f"{site}\n" + f"n={n_used}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

        ax.grid(True, alpha=0.25)
        ax.set_xlim(float(args.x_range[0]), float(args.x_range[1]))
        ax.set_ylim(float(args.y_range[0]), float(args.y_range[1]))

        # Plain tick formatting (no offsets)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

        # Hide x tick labels except on the bottom-most used axis in each column
        row, col = divmod(i, ncols)
        if row != bottom_row_for_col[col]:
            ax.set_xticklabels([])
        else:
            # Put an xlabel on BOTH bottom panels (left + right columns)
            if args.xdef == "eq":
                ax.set_xlabel(r"Moisture availability, $x = E_{eq}/E_{pa}$")
            else:
                ax.set_xlabel(r"Moisture availability, $x = E_{p0,PT}/E_{pa}$")

    # Turn off unused axes
    for j in range(len(sites), len(ax_list)):
        ax_list[j].set_axis_off()

    # One shared y-label (prevents overlap)
    fig.text(
        0.04,
        0.5,
        "Nondimensional evaporation,\n$y = E/E_{pa}$",
        rotation=90,
        va="center",
        ha="center",
    )

    title_x = "x = E_eq/E_pa" if args.xdef == "eq" else "x = Ep0,PT/E_pa"
    fig.suptitle(
        f"NEON complementarity relationship ({title_x}), {start.date()} to {end.date()}",
        y=0.99,
    )

    # Single legend (outside panels)
    handles, labels = [], []
    for ax in ax_list:
        if not ax.has_data():
            continue
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
        # Stop after first data axis (enough to collect all series)
        if handles:
            break

    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=[0.06, 0.06, 1.0, 0.96])

    tag = f"{start.date()}_{end.date()}"
    outbase = outdir / f"fig14_neon_cr_grid_{args.xdef}_{tag}"

    if args.format in ("png", "both"):
        fig.savefig(outbase.with_suffix(".png"), dpi=300)
        print(f"Wrote: {outbase.with_suffix('.png')}")
    if args.format in ("pdf", "both"):
        fig.savefig(outbase.with_suffix(".pdf"))
        print(f"Wrote: {outbase.with_suffix('.pdf')}")

    if bool(args.write_bins_csv):
        if bins_rows:
            bins = pd.DataFrame(bins_rows)
            bins.to_csv(outbase.with_name(outbase.name + "_binned.csv"), index=False)
            print(f"Wrote: {outbase.with_name(outbase.name + '_binned.csv')}")
        else:
            print("[WARN] No bins written (not enough points after filtering).")

    plt.close(fig)


if __name__ == "__main__":
    main()
