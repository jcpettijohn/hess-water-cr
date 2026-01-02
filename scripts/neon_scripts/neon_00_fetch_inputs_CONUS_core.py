#!/usr/bin/env python3
"""neon_00_fetch_inputs_CONUS_core.py

Fetch NEON inputs for a CONUS-only subset of *core terrestrial* tower sites.

Outputs
-------
For each site, outputs are written under:

  <outdir>/<SITE>_<START>_<END>/
    raw/    (NEON downloads + intermediate stacked files)
    tables/ (analysis-ready 30-min tables)

The filenames in `tables/` are matched to the current HARV pipeline used in
`neon_01_make_cr_plots_UPDATED.py`:

  eddy_fluxH2o_dp04_30min.parquet
  rad_DP1_00023_001_30min.parquet
  g_DP1_00040_001_30min.parquet
  rh_DP1_00098_001_30min.parquet
  wind_DP1_00001_001_30min.parquet
  pres_DP1_00004_001_30min.parquet

Optional extras (enabled with --package expanded unless --no_extras is set) are
also downloaded/stacked (e.g., PAR, soil temperature, soil water content, and
precipitation variants), each written as a separate 30-min table when available.

Examples
--------
# Default: 7 CONUS core sites (HARV, OSBS, KONZ, SRER, NIWO, WREF, SJER)
python3 neon_00_fetch_inputs_CONUS_core.py --outdir data/neon --start 2018-01 --end 2024-12

# All 16 CONUS core terrestrial sites
python3 neon_00_fetch_inputs_CONUS_core.py --site_group core_conus_all16 --start 2018-01 --end 2024-12

# Just a couple sites, expanded product bundle
python3 neon_00_fetch_inputs_CONUS_core.py --site HARV OSBS --package expanded --start 2018-01 --end 2024-12

Notes on prompts
----------------
The NEON python package (`neonutilities`) can ask for confirmation before large
downloads when `check_size=True`. This script avoids interactive prompts by
default (check_size=False). If you want prompts, pass --check_size.

The legacy convenience flag --yes is supported and simply forces
check_size=False (non-interactive).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd

# -------------------------------
# Import NEON python utilities
# -------------------------------
try:
    import neonutilities as nu
except Exception as e:  # pragma: no cover
    print(
        "[FATAL] Could not import `neonutilities`. Install it first, e.g.\n"
        "  python -m pip install neonutilities\n\n"
        f"Import error: {e}",
        file=sys.stderr,
    )
    raise SystemExit(2)


# -------------------------------
# Site groups (Core Terrestrial, CONUS only)
# -------------------------------
CORE_CONUS_7 = [
    "HARV",  # Northeast deciduous forest
    "OSBS",  # Southeast pine forest
    "KONZ",  # Tallgrass prairie
    "SRER",  # Sonoran desert / semi-arid shrubland
    "NIWO",  # Subalpine / montane
    "WREF",  # Pacific Northwest forest
    "SJER",  # California oak savanna / Mediterranean
]

CORE_CONUS_ALL16 = [
    "CLBJ",
    "CPER",
    "HARV",
    "KONZ",
    "NIWO",
    "ONAQ",
    "ORNL",
    "OSBS",
    "SCBI",
    "SJER",
    "SRER",
    "TALL",
    "UNDE",
    "WOOD",
    "WREF",
    "YELL",
]


# -------------------------------
# Product definitions
# -------------------------------
DP4_EDDY_BUNDLE = "DP4.00200.001"

# Minimal set required by the current HARV pipeline
DP1_BASIC_PRODUCTS = [
    ("DP1.00023.001", "rad_DP1_00023_001"),  # net radiation
    ("DP1.00040.001", "g_DP1_00040_001"),  # soil heat flux
    ("DP1.00098.001", "rh_DP1_00098_001"),  # air temperature & RH
    ("DP1.00001.001", "wind_DP1_00001_001"),  # wind speed & direction
    ("DP1.00004.001", "pres_DP1_00004_001"),  # barometric pressure
]

# Optional add-ons (download when available; failures are non-fatal)
DP1_EXTRA_PRODUCTS = [
    ("DP1.00024.001", "par_DP1_00024_001"),  # PAR
    ("DP1.00041.001", "soilT_DP1_00041_001"),  # soil temperature
    ("DP1.00094.001", "soilVWC_DP1_00094_001"),  # soil water content
    ("DP1.00044.001", "precipW_DP1_00044_001"),  # weighing gauge precip
    ("DP1.00045.001", "precipT_DP1_00045_001"),  # tipping bucket precip
    ("DP1.00046.001", "throughfall_DP1_00046_001"),  # throughfall precip
]


# -------------------------------
# Helpers
# -------------------------------

def _parse_timeindex(value: str) -> Any:
    """Parse timeindex CLI input.

    neonutilities.load_by_product accepts timeindex='all' or an integer number
    of minutes (e.g., 30).
    """

    v = str(value).strip().lower()
    if v == "all":
        return "all"
    try:
        return int(v)
    except ValueError:
        # fall back to the raw string
        return value


@contextlib.contextmanager
def _pushd(path: Path):
    """Temporarily chdir into path (creating it if needed)."""

    path.mkdir(parents=True, exist_ok=True)
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _find_first_dir(root: Path, name: str) -> Optional[Path]:
    """Return the first directory named `name` under root (including root/name)."""

    candidate = root / name
    if candidate.is_dir():
        return candidate
    for p in root.rglob(name):
        if p.is_dir():
            return p
    return None


def _pick_30min_table(obj: Any) -> Optional[Tuple[str, pd.DataFrame]]:
    """Pick a 30-min DataFrame from a neonutilities load_by_product() result."""

    if isinstance(obj, pd.DataFrame):
        return ("data", obj)
    if not isinstance(obj, dict):
        return None

    items: list[Tuple[str, pd.DataFrame]] = []
    for k, v in obj.items():
        if isinstance(v, pd.DataFrame) and len(v) > 0:
            items.append((str(k), v))

    if not items:
        return None

    # Prefer anything explicitly labeled 30min
    for k, v in items:
        if "30min" in k.lower():
            return (k, v)

    # Else just take the first table
    return items[0]


def _write_table(df: pd.DataFrame, outbase: Path, fmt: str) -> Path:
    """Write df to outbase.{parquet,csv}. Returns the written path."""

    outbase.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        outpath = outbase.with_suffix(".parquet")
        # Use pyarrow if available (typical because neonutilities depends on it)
        df.to_parquet(outpath, index=False)
        return outpath

    if fmt == "csv":
        outpath = outbase.with_suffix(".csv")
        df.to_csv(outpath, index=False)
        return outpath

    raise ValueError(f"Unsupported format: {fmt}")


# -------------------------------
# Download routines
# -------------------------------

def _download_dp4_fluxH2o(
    *,
    site: str,
    start: str,
    end: str,
    rawdir: Path,
    tabdir: Path,
    release: str,
    token: Optional[str],
    include_provisional: bool,
    check_size: bool,
    fmt: str,
    avg_min: int = 30,
) -> None:
    """Download DP4 eddy bundle (raw) and stack fluxH2o to 30-min table."""

    print(f"[INFO] Downloading {DP4_EDDY_BUNDLE} (eddy covariance bundle) for {site}")

    try:
        nu.zips_by_product(
            dpid=DP4_EDDY_BUNDLE,
            site=[site],
            startdate=start,
            enddate=end,
            package="basic",
            release=release,
            check_size=check_size,
            include_provisional=include_provisional,
            savepath=str(rawdir),
            token=token,
            progress=True,
        )
    except Exception as e:
        print(f"[WARN] {site} {DP4_EDDY_BUNDLE} raw download failed: {e}")
        return

    # neonutilities typically writes to rawdir/filesToStack00200
    eddy_dir = _find_first_dir(rawdir, "filesToStack00200")
    if eddy_dir is None:
        print(
            f"[WARN] {site} DP4 download finished, but could not locate filesToStack00200 under {rawdir}. "
            "Skipping stack_eddy()."
        )
        return

    try:
        out = nu.stack_eddy(
            filepath=str(eddy_dir),
            level="dp04",
            var="fluxH2o",
            avg=avg_min,
            metadata=False,
            runLocal=False,
        )
    except Exception as e:
        print(f"[WARN] {site} stack_eddy(fluxH2o) failed: {e}")
        return

    # stack_eddy usually returns a dict keyed by site, but be defensive
    df: Optional[pd.DataFrame] = None
    if isinstance(out, dict):
        df = out.get(site)
        if df is None and len(out) == 1:
            # single-site folder: take the only entry
            df = next(iter(out.values()))
    elif isinstance(out, pd.DataFrame):
        df = out

    if df is None or len(df) == 0:
        print(f"[WARN] {site} stack_eddy returned no usable fluxH2o table.")
        return

    outbase = tabdir / f"eddy_fluxH2o_dp04_{avg_min}min"
    written = _write_table(df, outbase, fmt)
    print(f"[OK] Wrote {written}")


def _download_dp1_product(
    *,
    site: str,
    dpid: str,
    out_prefix: str,
    start: str,
    end: str,
    rawdir: Path,
    tabdir: Path,
    release: str,
    token: Optional[str],
    include_provisional: bool,
    check_size: bool,
    package: str,
    timeindex: Any,
    fmt: str,
) -> None:
    """Download one DP1 product and write a 30-min table to disk."""

    print(f"[INFO] Downloading {dpid} ({out_prefix}) for {site}")

    try:
        # load_by_product does not expose savepath in some versions; to keep
        # downloads site-scoped we temporarily chdir into rawdir.
        with _pushd(rawdir):
            data = nu.load_by_product(
                dpid=dpid,
                site=site,
                startdate=start,
                enddate=end,
                package=package,
                release=release,
                timeindex=timeindex,
                check_size=check_size,
                include_provisional=include_provisional,
                token=token,
                progress=True,
            )

        picked = _pick_30min_table(data)
        if picked is None:
            print(f"[WARN] {site} {dpid} returned no DataFrame tables.")
            return

        table_name, df = picked
        outbase = tabdir / f"{out_prefix}_30min"
        written = _write_table(df, outbase, fmt)
        print(f"[OK] Wrote {written} (from table '{table_name}')")

    except Exception as e:
        print(f"[WARN] {site} {dpid} ({out_prefix}) download/stack failed: {e}")
        return


# -------------------------------
# CLI
# -------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Download NEON inputs for CONUS core terrestrial tower sites.",
    )

    p.add_argument(
        "--site_group",
        choices=["core_conus_7", "core_conus_all16"],
        default="core_conus_7",
        help="Predefined CONUS core-site group to download.",
    )
    p.add_argument(
        "--site",
        dest="sites",
        nargs="+",
        default=None,
        help="Override the site group and download only these site IDs (e.g., HARV OSBS).",
    )
    p.add_argument("--start", default="2018-01", help="Start date (YYYY-MM or YYYY-MM-DD).")
    p.add_argument("--end", default="2024-12", help="End date (YYYY-MM or YYYY-MM-DD).")
    p.add_argument("--outdir", default="data/neon", help="Output root directory.")

    p.add_argument(
        "--package",
        choices=["basic", "expanded"],
        default="expanded",
        help="Which DP1 product bundle to attempt (basic matches HARV pipeline; expanded adds extras).",
    )
    p.add_argument(
        "--timeindex",
        default="30",
        help="Time index for DP1 load_by_product (e.g., 30 or 'all').",
    )

    p.add_argument(
        "--check_size",
        action="store_true",
        default=False,
        help="If set, neonutilities may ask before large downloads.",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help="Non-interactive mode; forces check_size=False (no 'y/n' prompts).",
    )

    p.add_argument(
        "--include_provisional",
        action="store_true",
        default=False,
        help="Include provisional NEON data where available.",
    )
    p.add_argument(
        "--no_extras",
        action="store_true",
        default=False,
        help="Skip the optional DP1 extras even if --package expanded is selected.",
    )
    p.add_argument(
        "--no_dp4",
        action="store_true",
        default=False,
        help="Skip the DP4 eddy bundle (useful to test DP1 downloads first).",
    )

    p.add_argument("--release", default="current", help="NEON release tag (usually 'current').")
    p.add_argument("--token", default=None, help="Optional NEON API token.")
    p.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output table format.",
    )

    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    # Resolve sites
    if args.sites is not None:
        sites = [s.upper() for s in args.sites]
    else:
        sites = CORE_CONUS_7 if args.site_group == "core_conus_7" else CORE_CONUS_ALL16

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Prompt control
    check_size = bool(args.check_size) and (not bool(args.yes))

    timeindex = _parse_timeindex(args.timeindex)

    # Product list
    dp1_products = list(DP1_BASIC_PRODUCTS)
    if args.package == "expanded" and (not args.no_extras):
        dp1_products.extend(DP1_EXTRA_PRODUCTS)

    for site in sites:
        print(f"\n=== {site} ===")

        site_root = out_root / f"{site}_{args.start}_{args.end}"
        rawdir = site_root / "raw"
        tabdir = site_root / "tables"
        rawdir.mkdir(parents=True, exist_ok=True)
        tabdir.mkdir(parents=True, exist_ok=True)

        if not args.no_dp4:
            _download_dp4_fluxH2o(
                site=site,
                start=args.start,
                end=args.end,
                rawdir=rawdir,
                tabdir=tabdir,
                release=args.release,
                token=args.token,
                include_provisional=args.include_provisional,
                check_size=check_size,
                fmt=args.format,
                avg_min=30,
            )

        for dpid, prefix in dp1_products:
            _download_dp1_product(
                site=site,
                dpid=dpid,
                out_prefix=prefix,
                start=args.start,
                end=args.end,
                rawdir=rawdir,
                tabdir=tabdir,
                release=args.release,
                token=args.token,
                include_provisional=args.include_provisional,
                check_size=check_size,
                package="basic" if args.package == "basic" else "expanded",
                timeindex=timeindex,
                fmt=args.format,
            )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
