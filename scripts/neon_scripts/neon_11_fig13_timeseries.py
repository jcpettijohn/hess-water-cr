#!/usr/bin/env python3
"""neon_11_fig13_timeseries.py

Create the NEON observational proof-of-concept time-series figure (Figure 13)
from per-site daily CR files produced by `neon_10_make_daily_cr.py`.

This script is intentionally geared toward the *first-look, multi-site*
"hydromet manuscript" style figures that are commonly used when testing a
processing pipeline across several towers:

- small-multiple layout (default: 2 columns) instead of a very tall stack
- shared x-axis and (optionally) shared y-axis for cross-site comparability
- one clean, outside-the-panels legend
- optional highlight of the recommended analysis window (chosen objectively
  from good-day coverage)
- optional coverage heatmap (site × year) saved alongside Figure 13

Inputs
------
Expects the NEON 30-min tables to have been processed to daily files:
  <data_root>/<SITE>_<START>_<END>/tables/<SITE>_daily_cr.csv

Those folders and table names match the downloader/stacker:
  neon_00_fetch_inputs_CONUS_core.py

Outputs
-------
Writes under --outdir (default: figs/neon):
  - fig13_neon_timeseries_<TAG>.png
  - fig13_neon_timeseries_<TAG>.pdf
  - neon_daily_coverage_by_year.csv
  - neon_best_window.txt
  - (optional) fig13_neon_coverage_heatmap.png/pdf

Usage
-----
# Pick the best multi-site full-year window and plot a grid figure
python neon_11_fig13_timeseries.py --data_root data/neon

# Force a specific plotting/analysis window
python neon_11_fig13_timeseries.py --data_root data/neon --start 2019-01-01 --end 2023-12-31

# Plot only the chosen window (no full-record context)
python neon_11_fig13_timeseries.py --data_root data/neon --window_only

Notes on the "best window"
---------------------------
By default, this script selects the *longest contiguous run of full calendar
years* for which each included site exceeds a minimum mean fraction of good
(days) (good_day==True). This is designed to find a manuscript-ready period
that is reasonably complete across all sites.

If you prefer water years (Oct–Sep) or a different completeness rule, add that
as a manuscript decision after the first pass.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Default site ordering used in the CONUS-core 7-site workflow.
CORE_CONUS_7 = ["HARV", "OSBS", "KONZ", "SRER", "NIWO", "WREF", "SJER"]


def _iter_site_dirs(data_root: Path) -> Iterable[Path]:
    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        if re.match(r"^[A-Z0-9]{4}_.+_.+$", p.name):
            yield p


def _site_id_from_folder(folder: Path) -> str:
    return folder.name.split("_")[0].upper()


def _load_daily_file(site_dir: Path) -> pd.DataFrame:
    site = _site_id_from_folder(site_dir)
    path = site_dir / "tables" / f"{site}_daily_cr.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["site"] = site

    # Ensure good_day exists and is boolean
    if "good_day" not in df.columns:
        df["good_day"] = (~df[["E_mm", "Epa_mm"]].isna().any(axis=1)).astype(bool)
    else:
        df["good_day"] = df["good_day"].astype(bool)

    return df


def _coverage_by_year(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for df in dfs:
        site = str(df["site"].iloc[0])
        d = df.copy()
        d["year"] = d["date"].dt.year
        for y, g in d.groupby("year"):
            good = g["good_day"].astype(bool)
            rows.append(
                {
                    "site": site,
                    "year": int(y),
                    "n_days": int(len(g)),
                    "n_good": int(good.sum()),
                    "frac_good": float(good.mean()) if len(g) else np.nan,
                }
            )

    out = pd.DataFrame(rows)
    out = out.sort_values(["site", "year"]).reset_index(drop=True)
    return out


def _pick_best_window(
    cov_year: pd.DataFrame,
    sites: List[str],
    year_cov_thresh: float,
    min_years: int,
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]]:
    """Pick the longest contiguous full-year window meeting per-site coverage."""

    if cov_year.empty:
        return None

    piv = cov_year.pivot(index="year", columns="site", values="frac_good").reindex(columns=sites)

    # Years for which all sites have *some* data
    years_avail = [int(y) for y in piv.index[piv.notna().all(axis=1)].tolist()]
    if not years_avail:
        return None

    years_avail = sorted(set(years_avail))

    # Find contiguous runs
    runs: List[List[int]] = []
    run: List[int] = []
    for y in years_avail:
        if not run or y == run[-1] + 1:
            run.append(y)
        else:
            runs.append(run)
            run = [y]
    if run:
        runs.append(run)

    best: Optional[Tuple[int, int, pd.Series]] = None
    best_score: Optional[Tuple[int, float, int]] = None

    for r in runs:
        if len(r) < min_years:
            continue
        for i in range(len(r)):
            for j in range(i + min_years - 1, len(r)):
                y0, y1 = r[i], r[j]
                sub = piv.loc[y0:y1]
                site_mean = sub.mean(axis=0)
                if (site_mean >= year_cov_thresh).all():
                    length_years = y1 - y0 + 1
                    score = (length_years, float(site_mean.mean()), y1)  # length, coverage, recency
                    if best_score is None or score > best_score:
                        best_score = score
                        best = (y0, y1, site_mean)

    if best is None:
        return None

    y0, y1, site_mean = best
    start = pd.Timestamp(f"{y0}-01-01")
    end = pd.Timestamp(f"{y1}-12-31")
    summary = site_mean.reset_index()
    summary.columns = ["site", "mean_frac_good"]
    return start, end, summary


def _rolling_smooth(series: pd.Series, smooth_days: int) -> pd.Series:
    if smooth_days <= 1:
        return series
    # Centered rolling mean; require at least ~1/3 of the window.
    minp = max(1, smooth_days // 3)
    return series.rolling(window=smooth_days, center=True, min_periods=minp).mean()


def _global_ylim(dfs: List[pd.DataFrame], cols: List[str], start: pd.Timestamp, end: pd.Timestamp) -> Tuple[float, float]:
    """Compute a robust (0, ymax) from pooled values across sites for a window."""
    vals: List[float] = []
    for df in dfs:
        d = df[(df["date"] >= start) & (df["date"] <= end)]
        for c in cols:
            if c in d.columns:
                v = d[c].to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                if v.size:
                    vals.append(np.nanpercentile(v, 99.0))
    if not vals:
        return (0.0, 5.0)
    ymax = float(np.nanmax(vals))
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 5.0
    # Round up to a nice number
    mag = 10 ** math.floor(math.log10(ymax)) if ymax > 0 else 1
    ymax_nice = math.ceil(ymax / (mag / 2)) * (mag / 2)
    return (0.0, float(ymax_nice))


def _plot_coverage_heatmap(
    cov_year: pd.DataFrame,
    sites: List[str],
    out_png: Path,
    out_pdf: Path,
) -> None:
    """Heatmap of good-day fraction by site × year (quick multi-site QA)."""

    if cov_year.empty:
        return

    piv = cov_year.pivot(index="site", columns="year", values="frac_good").reindex(index=sites)
    years = piv.columns.to_list()

    fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * len(years) + 2.0), 0.5 * len(sites) + 2.0))

    im = ax.imshow(piv.to_numpy(), aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)

    ax.set_yticks(range(len(sites)))
    ax.set_yticklabels(sites)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")

    ax.set_title("NEON good-day coverage by year (fraction of days passing QC/coverage)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Site")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Fraction good")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)


def _plot_fig13(
    dfs: List[pd.DataFrame],
    sites: List[str],
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
    smooth_days: int,
    out_png: Path,
    out_pdf: Path,
    ep0_field: str,
    layout: str,
    ncols: int,
    sharey: bool,
    window_only: bool,
    ylim: Optional[Tuple[float, float]],
) -> None:
    """Plot the Figure 13 multi-site time series."""

    # Decide the plot extent
    if window_only:
        plot_start, plot_end = analysis_start, analysis_end
    else:
        plot_start = pd.Timestamp(min(df["date"].min() for df in dfs)).floor("D")
        plot_end = pd.Timestamp(max(df["date"].max() for df in dfs)).floor("D")

    # Prepare per-site frames
    df_by_site: Dict[str, pd.DataFrame] = {}
    for df in dfs:
        site = str(df["site"].iloc[0])
        d = df[(df["date"] >= plot_start) & (df["date"] <= plot_end)].copy()
        d = d.sort_values("date")
        for col in ["E_mm", "Epa_mm", ep0_field]:
            if col in d.columns:
                d[col] = _rolling_smooth(d[col].astype(float), smooth_days)
        df_by_site[site] = d

    # Layout
    if layout == "stack":
        nrows, ncols_eff = len(sites), 1
    else:
        ncols_eff = max(1, int(ncols))
        nrows = int(math.ceil(len(sites) / ncols_eff))

    fig_w = 11.0 if ncols_eff > 1 else 10.0
    fig_h = (2.2 * nrows + 1.6) if layout != "stack" else (1.55 * nrows + 1.8)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols_eff,
        sharex=True,
        sharey=bool(sharey),
        figsize=(fig_w, fig_h),
    )

    # Normalize axes to a flat list
    if isinstance(axes, np.ndarray):
        ax_list = axes.flatten().tolist()
    else:
        ax_list = [axes]

    # Y-limits
    # For manuscript-style multi-site comparison, a shared y-axis is usually
    # more interpretable. When enabled, pick a robust (0, ymax) across the
    # *plotted* extent (full record or window-only).
    if ylim is None and sharey:
        ylim = _global_ylim(dfs, ["E_mm", "Epa_mm", ep0_field], plot_start, plot_end)

    # Plot panels
    for i, site in enumerate(sites):
        ax = ax_list[i]
        d = df_by_site.get(site)

        if d is None or d.empty:
            ax.text(0.5, 0.5, f"{site}: no data", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue

        # Highlight chosen analysis window for manuscript (only useful if we show more than the window)
        if not window_only:
            ax.axvspan(analysis_start, analysis_end, color="0.92", zorder=0)

        # Plot series (order is consistent so colors are consistent across panels)
        ax.plot(d["date"], d["E_mm"], label="E")
        ax.plot(d["date"], d["Epa_mm"], label=r"$E_{pa}$")
        ax.plot(d["date"], d[ep0_field], label=r"$E_{p0}$")

        # Panel label + window coverage annotation
        if "good_day" in d.columns:
            dw = d[(d["date"] >= analysis_start) & (d["date"] <= analysis_end)]
            frac = float(dw["good_day"].mean()) if len(dw) else np.nan
            cov_txt = f"good={frac:.0%}" if np.isfinite(frac) else "good=NA"
        else:
            cov_txt = ""

        ax.text(
            0.01,
            0.98,
            f"{site}  {cov_txt}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

        ax.grid(True, alpha=0.25)
        if ylim is not None:
            ax.set_ylim(*ylim)

        # Only label y-axis on the leftmost column for a grid
        if (layout == "grid") and (i % ncols_eff != 0):
            ax.set_ylabel("")
        else:
            ax.set_ylabel(r"mm d$^{-1}$")

    # Turn off unused axes
    for j in range(len(sites), len(ax_list)):
        ax_list[j].set_axis_off()

    # Shared x-axis formatting
    for ax in ax_list[: len(sites)]:
        ax.set_xlim(plot_start, plot_end)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Legend: one, outside the panels
    handles, labels = ax_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))

    # Title
    win_txt = f"analysis window {analysis_start.date()} to {analysis_end.date()}"
    if window_only:
        title = f"NEON daily evaporation constructs (Figure 13) — {win_txt}"
    else:
        title = f"NEON daily evaporation constructs (Figure 13) — full record, {win_txt} highlighted"
    fig.suptitle(title, y=0.98)

    fig.tight_layout(rect=[0.02, 0.06, 1, 0.95])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Plot NEON Figure 13 time series from daily CR files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data_root", default="data/neon", help="Root containing <SITE>_<START>_<END>/ folders")
    ap.add_argument(
        "--sites",
        nargs="+",
        default=None,
        help="Sites to include (default: the 7 CONUS core sites if present; else all found).",
    )
    ap.add_argument("--outdir", default="figs/neon", help="Output directory for figures and summaries")
    ap.add_argument(
        "--start",
        default=None,
        help="Optional analysis window start date (YYYY-MM-DD). If unset, choose best window.",
    )
    ap.add_argument(
        "--end",
        default=None,
        help="Optional analysis window end date (YYYY-MM-DD). If unset, choose best window.",
    )
    ap.add_argument(
        "--year_cov_thresh",
        type=float,
        default=0.70,
        help="Per-site mean fraction of good days required to accept a full-year window.",
    )
    ap.add_argument(
        "--min_years",
        type=int,
        default=3,
        help="Minimum number of contiguous full years in the recommended window.",
    )
    ap.add_argument(
        "--smooth_days",
        type=int,
        default=7,
        help="Rolling-mean smoothing window (days). Use 1 to disable.",
    )
    ap.add_argument(
        "--ep0",
        choices=["pt", "eq"],
        default="pt",
        help="Which Ep0 series to plot: Priestley–Taylor (pt) or equilibrium (eq).",
    )

    # Figure-style options
    ap.add_argument(
        "--layout",
        choices=["grid", "stack"],
        default="grid",
        help="Figure layout: multi-panel grid (typical manuscript) or tall stack (debug).",
    )
    ap.add_argument(
        "--ncols",
        type=int,
        default=2,
        help="Number of columns for grid layout.",
    )
    # Share-y is a common default for multi-site “first look” figures.
    # Provide both --sharey and --no_sharey switches, with default True.
    sharey_group = ap.add_mutually_exclusive_group()
    sharey_group.add_argument(
        "--sharey",
        dest="sharey",
        action="store_true",
        help="Share y-axis across panels.",
    )
    sharey_group.add_argument(
        "--no_sharey",
        dest="sharey",
        action="store_false",
        help="Do not share y-axis across panels.",
    )
    ap.set_defaults(sharey=True)
    ap.add_argument(
        "--window_only",
        action="store_true",
        default=False,
        help="Plot only the analysis window (instead of full record with the window highlighted).",
    )
    ap.add_argument(
        "--ylim",
        type=float,
        default=None,
        help="If set, use (0, ylim) for all panels.",
    )
    ap.add_argument(
        "--no_coverage_heatmap",
        action="store_true",
        default=False,
        help="Disable the site×year coverage heatmap output.",
    )

    args = ap.parse_args(argv)

    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load per-site daily files
    dfs: List[pd.DataFrame] = []
    found_sites: List[str] = []

    for site_dir in _iter_site_dirs(data_root):
        site = _site_id_from_folder(site_dir)
        try:
            df = _load_daily_file(site_dir)
        except FileNotFoundError:
            continue
        dfs.append(df)
        found_sites.append(site)

    if not dfs:
        raise SystemExit(
            f"No <SITE>_daily_cr.csv files found under {data_root}. Run neon_10_make_daily_cr.py first."
        )

    # Choose site list
    if args.sites:
        sites = [s.upper() for s in args.sites]
    else:
        sites = [s for s in CORE_CONUS_7 if s in set(found_sites)]
        if not sites:
            sites = sorted(set(found_sites))

    dfs = [df for df in dfs if str(df["site"].iloc[0]) in set(sites)]

    # Coverage summary table
    cov_year = _coverage_by_year(dfs)
    cov_year.to_csv(outdir / "neon_daily_coverage_by_year.csv", index=False)

    # Determine analysis window
    if args.start and args.end:
        analysis_start = pd.Timestamp(args.start)
        analysis_end = pd.Timestamp(args.end)
        window_note = f"User-specified analysis window: {analysis_start.date()} to {analysis_end.date()}"
        window_summary = None
    else:
        picked = _pick_best_window(
            cov_year=cov_year,
            sites=sites,
            year_cov_thresh=float(args.year_cov_thresh),
            min_years=int(args.min_years),
        )
        if picked is None:
            # Fallback: use overlapping date extent across sites
            min_date = max(df["date"].min() for df in dfs)
            max_date = min(df["date"].max() for df in dfs)
            analysis_start, analysis_end = pd.Timestamp(min_date).floor("D"), pd.Timestamp(max_date).floor("D")
            window_note = (
                "No full-year window met coverage threshold; using overlapping date extent across sites: "
                f"{analysis_start.date()} to {analysis_end.date()}"
            )
            window_summary = None
        else:
            analysis_start, analysis_end, window_summary = picked
            window_note = (
                f"Recommended window (>= {args.year_cov_thresh:.0%} mean good-day coverage per site): "
                f"{analysis_start.date()} to {analysis_end.date()}"
            )

    (outdir / "neon_best_window.txt").write_text(window_note + "\n")
    if window_summary is not None:
        window_summary.to_csv(outdir / "neon_best_window_site_means.csv", index=False)

    print(window_note)

    # Optional coverage heatmap
    if not bool(args.no_coverage_heatmap):
        _plot_coverage_heatmap(
            cov_year=cov_year,
            sites=sites,
            out_png=outdir / "fig13_neon_coverage_heatmap.png",
            out_pdf=outdir / "fig13_neon_coverage_heatmap.pdf",
        )

    # Main figure
    tag = f"{analysis_start.date()}_{analysis_end.date()}"
    ep0_field = "Ep0_pt_mm" if args.ep0 == "pt" else "Ep0_eq_mm"
    out_png = outdir / f"fig13_neon_timeseries_{tag}.png"
    out_pdf = outdir / f"fig13_neon_timeseries_{tag}.pdf"

    _plot_fig13(
        dfs=dfs,
        sites=sites,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
        smooth_days=int(args.smooth_days),
        out_png=out_png,
        out_pdf=out_pdf,
        ep0_field=ep0_field,
        layout=str(args.layout),
        ncols=int(args.ncols),
        sharey=bool(args.sharey),
        window_only=bool(args.window_only),
        ylim=(0.0, float(args.ylim)) if args.ylim is not None else None,
    )

    print(f"[OK] Wrote {out_png}")
    print(f"[OK] Wrote {out_pdf}")
    print(f"[OK] Wrote {outdir / 'neon_daily_coverage_by_year.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
