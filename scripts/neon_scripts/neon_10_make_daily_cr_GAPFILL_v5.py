#!/usr/bin/env python3
"""neon_10_make_daily_cr_GAPFILL.py

Build analysis-ready *daily* evaporation constructs from locally archived NEON
30-min tables (as produced by `neon_00_fetch_inputs_CONUS_core.py`).

What this script does
---------------------
For each NEON site folder under <data_root> (e.g., HARV_2018-01_2024-12),
this script reads the 30-min parquet/CSV tables in `tables/` and produces
a per-site daily file:

  <site_folder>/tables/<SITE>_daily_cr.csv

with daily time series of:
  - E (actual evaporation, mm d-1) from eddy-covariance latent heat flux.
    In this NEON workflow, `fluxH2o` from the dp04 eddy bundle is a latent heat
    flux (W m-2) produced by NEON/eddy4R, so it is treated as LE by default.
  - Epa (apparent potential evaporation; saturated-surface Penman-Monteith; mm d-1)
  - Ep0 (wet-environment surrogate; Priestley-Taylor + equilibrium; mm d-1)

Plus daily coverage metrics and QC flags.

Why coverage can look "too low" for eddy ET
-------------------------------------------
If you require (say) 80% of *all 48 half-hours* to pass eddy QC, many days can
fail simply because nighttime turbulence is weak and QC flags are conservative.
For daily ET, missing/flagged nighttime LE often contributes little to the daily
sum.

This version therefore supports:
  - small-gap interpolation at 30-min resolution (optional),
  - defining coverage using **daylight intervals** (based on incoming shortwave),
  - filling missing nighttime LE with 0 (optional),
  - (optional) scaling the daylight integral by daylight coverage.

These choices are configurable via CLI flags and the output CSV includes
diagnostic columns (E_cov_day, etc) so you can see what is driving omissions.

"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Utilities: column detection
# -----------------------------

def _find_first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Return the first candidate that is an exact column name in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _find_by_regex(df: pd.DataFrame, patterns: Sequence[str]) -> Optional[str]:
    """Return the first column whose name matches any regex (case-insensitive)."""
    cols = list(df.columns)
    for pat in patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        for c in cols:
            if rx.search(str(c)):
                return c
    return None


# -----------------------------------------------------------------------------
# Helpers to avoid accidentally using QC/QM columns as data
# -----------------------------------------------------------------------------
_QC_COL_RE = re.compile(
    r"(finalqf|qffinl|passqm|failqm|naqm|alphaqm|betaqm|"
    r"expuncert|stder|stderr|variance|numpts|nump?ts|uncert|"
    r"range(pass|fail|na)qm|persistence(pass|fail|na)qm|"
    r"null(pass|fail|na)qm|gap(pass|fail|na)qm|spike(pass|fail|na)qm|"
    r"validcal(pass|fail|na)qm|sensorerror(pass|fail|na)qm)",
    flags=re.IGNORECASE,
)


def _is_qc_col(name: str) -> bool:
    """True if a column name looks like QA/QC, uncertainty, or metrics rather than a measurement."""
    return bool(_QC_COL_RE.search(str(name)))


def _find_first_existing_non_qc(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns and not _is_qc_col(c):
            return c
    return None


def _find_by_regex_non_qc(df: pd.DataFrame, patterns: Sequence[str]) -> Optional[str]:
    """Return the first *non-QC* column whose name matches any regex (case-insensitive)."""
    cols = [c for c in df.columns if not _is_qc_col(c)]
    for pat in patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        for c in cols:
            if rx.search(str(c)):
                return c
    return None


def _pick_data_col(df: pd.DataFrame, *, candidates: Sequence[str], regexes: Sequence[str]) -> Optional[str]:
    """Pick a likely measurement column while avoiding QC/QM/uncertainty fields."""
    col = _find_first_existing_non_qc(df, candidates)
    if col is not None:
        return col
    return _find_by_regex_non_qc(df, regexes)


def _pressure_to_kpa(p: pd.Series) -> Tuple[pd.Series, float, str]:
    """Normalize pressure to kPa.

    NEON DP1.00004.001 is documented in kilopascal (kPa), but some workflows
    can deliver pressure in Pa or hPa depending on upstream processing.
    This helper converts common units to kPa using simple magnitude heuristics.

    Returns (p_kPa, factor, note) where p_kPa = p * factor.
    """
    s = pd.to_numeric(p, errors="coerce").astype(float)
    try:
        med = float(np.nanmedian(s))
    except Exception:
        med = np.nan

    factor = 1.0
    note = "kPa"
    if not np.isfinite(med):
        return s * np.nan, np.nan, "missing"

    # Typical station pressure is ~60–105 kPa.
    if med > 2000.0:
        factor = 0.001  # Pa -> kPa
        note = "Pa_to_kPa"
    elif med > 200.0:
        factor = 0.1  # hPa -> kPa
        note = "hPa_to_kPa"
    elif 20.0 <= med <= 200.0:
        factor = 1.0
        note = "kPa"
    elif 0.5 <= med <= 2.0:
        factor = 101.325  # atm -> kPa
        note = "atm_to_kPa"
    elif 0.05 <= med <= 0.2:
        factor = 1000.0  # MPa -> kPa
        note = "MPa_to_kPa"
    elif 0.2 <= med <= 2.0:
        factor = 100.0  # bar -> kPa
        note = "bar_to_kPa"
    else:
        # Too small or otherwise implausible – likely not a pressure measurement.
        factor = 1.0
        note = "unknown_units"

    return s * factor, factor, note


def _pick_pressure_col(pdf: pd.DataFrame) -> Optional[str]:
    """Pick the best station/barometric pressure *Mean* column from DP1.00004.001.

    NEON commonly uses 'staPresMean' (and related stats) but naming varies by
    product version/pipeline. We avoid QC/uncertainty columns and then score
    candidates based on whether the median looks like a plausible station
    pressure after unit normalization.
    """
    # Strongly preferred explicit names
    preferred = [
        "staPresMean",
        "corPresMean",
        "baroPresMean",
        "barometricPressureMean",
        "stationPressureMean",
        "airPresMean",
        "airPressureMean",
        "pressureMean",
        "pressure",
    ]
    col = _find_first_existing_non_qc(pdf, preferred)
    if col is not None:
        return col

    # Candidate pool: non-QC columns mentioning pres/pressure
    cols = [
        c
        for c in pdf.columns
        if (not _is_qc_col(str(c)))
        and re.search(r"(pres|pressure)", str(c), flags=re.IGNORECASE)
    ]
    if not cols:
        return None

    # Prefer mean-like columns when present
    mean_like = [c for c in cols if re.search(r"mean$", str(c), flags=re.IGNORECASE)]
    if mean_like:
        cols = mean_like

    best: Optional[str] = None
    best_score: float = float("inf")

    for c in cols:
        s = pd.to_numeric(pdf[c], errors="coerce").astype(float)
        if s.notna().sum() < 10:
            continue
        s_kpa, _, _ = _pressure_to_kpa(s)
        try:
            med = float(np.nanmedian(s_kpa))
        except Exception:
            med = np.nan
        if not np.isfinite(med):
            continue

        # Score: prefer plausible station pressure range, then closeness to ~95 kPa.
        score = 0.0
        if (med < 20.0) or (med > 200.0):
            score += 1e6 + abs(med - 95.0)
        else:
            score += abs(med - 95.0)

        # Penalize missingness
        miss = float(1.0 - s.notna().mean())
        score += 1000.0 * miss

        # Slightly prefer columns with common stems
        cl = str(c).lower()
        if cl.startswith(("stapres", "baropres", "corpres", "airpres")):
            score -= 10.0

        if score < best_score:
            best_score = score
            best = str(c)

    return best



def _normalize_time(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize time column to 'time' as timezone-naive UTC timestamps.

    NEON tables may contain timezone-aware ISO8601 strings (e.g., with a trailing 'Z'
    or an explicit offset like '-05:00') and different products can mix aware/naive
    datetimes. For consistent joins across DP1 / DP4 tables, we parse with utc=True
    and then drop the timezone (keeping UTC clock time).
    """
    df = df.copy()
    time_candidates = [
        "time",
        "timeBgn",
        "timeBgnUTC",
        "startDateTime",
        "startDateTimeUTC",
        "beginDateTime",
        "datetime",
        "dateTime",
    ]
    tcol = _find_first_existing(df, time_candidates)
    if tcol is None:
        tcol = _find_by_regex(df, [r"^time", r"start.*date.*time", r"date.*time"])
    if tcol is None:
        raise KeyError("Could not find a time column")

    # Parse in UTC and drop tz -> naive timestamps (UTC)
    t = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    df["time"] = t.dt.tz_convert(None)
    df = df.dropna(subset=["time"])
    return df


def _infer_dt_seconds(t: pd.Series) -> int:
    """Infer time step (seconds) from a time vector."""
    tt = pd.to_datetime(t).sort_values()
    if len(tt) < 2:
        return 1800
    dt = tt.diff().dropna()
    if dt.empty:
        return 1800
    sec = int(round(dt.median().total_seconds()))
    # be conservative
    if sec <= 0:
        sec = 1800
    return sec


def _collapse_to_timeseries(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Reduce a NEON table to [time, value] and average duplicates across positions."""
    out = df[["time", value_col]].copy()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    # average duplicates at the same timestamp
    out = out.groupby("time", as_index=False)[value_col].mean()
    out = out.sort_values("time")
    return out


def _make_regular_index(times: pd.Series, dt_seconds: int) -> pd.DatetimeIndex:
    times = pd.to_datetime(times)
    if times.empty:
        return pd.DatetimeIndex([])
    freq = f"{int(dt_seconds)}s"
    start = pd.Timestamp(times.min()).floor(freq)
    end = pd.Timestamp(times.max()).ceil(freq)
    return pd.date_range(start, end, freq=freq)


def _regularize_and_gapfill(
    ts: pd.DataFrame,
    value_col: str,
    dt_seconds: int,
    *,
    time_index: Optional[pd.DatetimeIndex] = None,
    gapfill_hours: float = 0.0,
) -> pd.DataFrame:
    """Reindex to a regular dt grid and (optionally) fill short internal gaps by interpolation."""
    if time_index is None:
        time_index = _make_regular_index(ts["time"], dt_seconds)

    if len(time_index) == 0:
        return pd.DataFrame({"time": pd.to_datetime([]), value_col: pd.Series(dtype=float)})

    s = pd.Series(
        pd.to_numeric(ts[value_col], errors="coerce").values,
        index=pd.to_datetime(ts["time"]),
        dtype=float,
    )
    # ensure uniqueness
    s = s.groupby(level=0).mean()
    s = s.reindex(time_index)

    if gapfill_hours and gapfill_hours > 0:
        limit = int(round((gapfill_hours * 3600.0) / float(dt_seconds)))
        if limit > 0:
            s = s.interpolate(method="time", limit=limit, limit_area="inside")

    return pd.DataFrame({"time": time_index, value_col: s.values})


# -----------------------------
# QC masking helpers (DP1 + Eddy)
# -----------------------------

def _mask_by_final_qf(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Mask a DP1 value column based on a corresponding '*FinalQF' column, if present."""
    df = df.copy()
    # common naming: <var>FinalQF or <var>FinalQFSciRvw etc.
    qf_exact = f"{value_col}FinalQF"
    qf = qf_exact if qf_exact in df.columns else None
    if qf is None:
        # try regex: value col stem + FinalQF
        stem = re.sub(r"(Mean|Avg|Average)$", "", str(value_col), flags=re.IGNORECASE)
        cand = _find_by_regex(df, [rf"^{re.escape(stem)}.*FinalQF$", r"FinalQF$"])
        if cand is not None and cand in df.columns:
            qf = cand

    if qf is None:
        return df

    qfv = pd.to_numeric(df[qf], errors="coerce")
    df.loc[qfv != 0, value_col] = np.nan
    return df


def _mask_eddy_qf(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Mask eddy value_col where the corresponding final quality flag indicates fail.

    NEON stackEddy() outputs qfFinl columns that may be simple ('qfFinl') or
    embedded in dotted names like 'qfqm.fluxH2o.nsae.qfFinl'. The NEON tutorial
    defines qfFinl as: 1=fail, 0=pass.
    """
    df = df.copy()

    # 1) Common/simple names
    qf = _find_first_existing(df, ["qfFinl", "qfFinal", "qfFin", "qf", "QF"])

    # 2) NEON dotted names (e.g., qfqm.fluxH2o.nsae.qfFinl)
    if qf is None:
        qf_candidates = [c for c in df.columns if re.search(r"qfFinl", str(c), flags=re.IGNORECASE)]
        if qf_candidates:
            # Token overlap score between value_col and candidate name
            tokens = [t for t in re.split(r"[^A-Za-z0-9]+", str(value_col)) if t]
            drop = {"data", "flux", "mean", "avg", "average", "value"}
            tokens = [t.lower() for t in tokens if t.lower() not in drop]

            def _score(name: str) -> int:
                nl = name.lower()
                return sum(1 for t in tokens if t and t in nl)

            qf = max(qf_candidates, key=lambda c: (_score(str(c)), -len(str(c))))

    if qf is None:
        return df

    qfv = pd.to_numeric(df[qf], errors="coerce")
    df.loc[qfv != 0, value_col] = np.nan
    return df


# -----------------------------
# Daily aggregation helpers
# -----------------------------

@dataclass
class DailyAgg:
    value: pd.Series
    n: pd.Series
    expected_n: int


def _daily_mean(ts: pd.DataFrame, value_col: str, dt_seconds: int) -> DailyAgg:
    v = pd.to_numeric(ts[value_col], errors="coerce")
    day = pd.to_datetime(ts["time"]).dt.floor("D")
    expected = int(round(86400 / dt_seconds)) if dt_seconds > 0 else 48
    val = v.groupby(day).mean()
    n = v.groupby(day).count()
    return DailyAgg(val, n, expected)


def _daily_integrate_Wm2_to_MJ(ts: pd.DataFrame, value_col: str, dt_seconds: int) -> DailyAgg:
    """Integrate W m-2 over day to MJ m-2 d-1. Returns NaN for days with all-missing values."""
    v = pd.to_numeric(ts[value_col], errors="coerce")
    day = pd.to_datetime(ts["time"]).dt.floor("D")
    expected = int(round(86400 / dt_seconds)) if dt_seconds > 0 else 48
    MJ = (v * float(dt_seconds)).groupby(day).sum(min_count=1) / 1e6
    n = v.groupby(day).count()
    return DailyAgg(MJ, n, expected)


def _daily_cov_over_mask(
    ts: pd.DataFrame, value_col: str, mask: pd.Series
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return (n_valid, n_total, cov) for value_col evaluated only where mask is True.

    IMPORTANT: Use a common DatetimeIndex for both the data series and the mask series
    before grouping. Passing a *Series* of day labels to groupby triggers index-alignment,
    which can silently produce empty groups if indices differ.
    """
    # Put the data on a DatetimeIndex
    t = pd.DatetimeIndex(pd.to_datetime(ts["time"]))
    v = pd.Series(pd.to_numeric(ts[value_col], errors="coerce").astype(float).to_numpy(), index=t)

    # Align mask to the same time base
    m = pd.Series(mask.values, index=pd.to_datetime(mask.index))
    m = m.reindex(t, fill_value=False).astype(bool)

    day = t.floor("D")
    n_valid = v.where(m).groupby(day).count()
    n_total = m.groupby(day).sum().astype(int)

    cov = n_valid / n_total.replace({0: np.nan})
    return n_valid, n_total, cov


def _daily_ET_from_LE_Wm2(
    ts_le: pd.DataFrame,
    le_col: str,
    dt_seconds: int,
    daylight_mask: pd.Series,
    *,
    fill_night_zero: bool = True,
    scale_daylight_missing: bool = True,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute daily ET (mm d-1) from a (possibly gappy) 30-min LE (W m-2) time series.

    Returns:
      E_mm, n_day_valid, n_day_total, cov_day

    Notes:
      - If fill_night_zero=True, missing nighttime LE is set to 0 before integration.
      - If scale_daylight_missing=True, the daylight integral is scaled by (n_total/n_valid).
        (Days with n_valid=0 yield NaN.)

    Implementation detail:
      We index both LE and the daylight mask by the same DatetimeIndex before grouping
      to avoid pandas' index-alignment pitfalls.
    """
    # Put LE on a DatetimeIndex
    t = pd.DatetimeIndex(pd.to_datetime(ts_le["time"]))
    le = pd.Series(pd.to_numeric(ts_le[le_col], errors="coerce").astype(float).to_numpy(), index=t)

    # Align daylight mask to this time base
    dm = pd.Series(daylight_mask.values, index=pd.to_datetime(daylight_mask.index))
    dm = dm.reindex(t, fill_value=False).astype(bool)

    # Optionally: treat missing nighttime LE as 0 (helps when eddy QC mainly fails at night)
    if fill_night_zero:
        le = le.copy()
        le.loc[(~dm) & le.isna()] = 0.0

    # Convert each interval to mm (1 kg m-2 = 1 mm), lambda ≈ 2.45e6 J kg-1
    mm_int = le * float(dt_seconds) / 2.45e6

    day = t.floor("D")

    # Split into day/night integrals so we can optionally scale only daytime
    mm_day = mm_int.where(dm).groupby(day).sum(min_count=1)
    mm_night = mm_int.where(~dm).groupby(day).sum(min_count=1)

    n_day_valid = le.where(dm).groupby(day).count()
    n_day_total = dm.groupby(day).sum().astype(int)

    cov_day = n_day_valid / n_day_total.replace({0: np.nan})

    if scale_daylight_missing:
        scale = n_day_total / n_day_valid.replace({0: np.nan})
        E_mm = (mm_day * scale) + mm_night.fillna(0.0)
    else:
        # full-day sum (NaNs are ignored; days with all-missing become NaN via min_count above)
        E_mm = mm_int.groupby(day).sum(min_count=1)

    return E_mm, n_day_valid, n_day_total, cov_day


# -----------------------------
# Physics helpers
# -----------------------------

def _sat_vp_kpa(Tc: pd.Series) -> pd.Series:
    """Saturation vapour pressure (kPa) for temperature in °C."""
    Tc = pd.to_numeric(Tc, errors="coerce")
    return 0.6108 * np.exp((17.27 * Tc) / (Tc + 237.3))


def _delta_kpa_per_C(Tc: pd.Series) -> pd.Series:
    """Slope of saturation vapour pressure curve Δ (kPa/°C)."""
    es = _sat_vp_kpa(Tc)
    Tc = pd.to_numeric(Tc, errors="coerce")
    return 4098.0 * es / (Tc + 237.3) ** 2


# -----------------------------
# Net radiation: DP1.00023.001
# -----------------------------

def _ensure_net_radiation(rad: pd.DataFrame, site: str, fname: str) -> Tuple[pd.DataFrame, str]:
    """Return (rad_df, rn_col) where rn_col is net radiation in W m-2.

    DP1.00023.001 provides in/out SW and LW components; some tables also provide
    a direct net-radiation column. If not present, compute:

      Rn = (inSW - outSW) + (inLW - outLW)

    """
    rad = rad.copy()

    # direct net column
    rn_col = _find_first_existing(
        rad,
        [
            "netRadMean",
            "netRadiationMean",
            "netRadiation",
            "netRad",
        ],
    ) or _find_by_regex(rad, [r"^net.*rad.*mean$", r"net.*radiation.*mean"])

    if rn_col is not None:
        rad = _mask_by_final_qf(rad, rn_col)
        return rad, rn_col

    # components
    in_sw = _find_first_existing_non_qc(rad, ["inSWMean", "incomingShortwaveRadiationMean"]) or _find_by_regex_non_qc(
        rad, [r"^in.*sw.*mean$", r"incoming.*short.*wave.*mean"]
    )
    out_sw = _find_first_existing_non_qc(rad, ["outSWMean", "outgoingShortwaveRadiationMean"]) or _find_by_regex_non_qc(
        rad, [r"^out.*sw.*mean$", r"outgoing.*short.*wave.*mean"]
    )
    in_lw = _find_first_existing_non_qc(rad, ["inLWMean", "incomingLongwaveRadiationMean"]) or _find_by_regex_non_qc(
        rad, [r"^in.*lw.*mean$", r"incoming.*long.*wave.*mean"]
    )
    out_lw = _find_first_existing_non_qc(rad, ["outLWMean", "outgoingLongwaveRadiationMean"]) or _find_by_regex_non_qc(
        rad, [r"^out.*lw.*mean$", r"outgoing.*long.*wave.*mean"]
    )

    missing = [name for name, col in [("inSW", in_sw), ("outSW", out_sw), ("inLW", in_lw), ("outLW", out_lw)] if col is None]
    if missing:
        raise KeyError(
            f"{site}: Could not find net radiation column in {fname} and could not compute it; "
            f"missing component(s): {missing}"
        )

    # Apply QC to components if available (so derived Rn isn't polluted by obvious fails)
    for col in (in_sw, out_sw, in_lw, out_lw):
        rad = _mask_by_final_qf(rad, col)

    rad["netRadCalc"] = (
        (pd.to_numeric(rad[in_sw], errors="coerce") - pd.to_numeric(rad[out_sw], errors="coerce"))
        + (pd.to_numeric(rad[in_lw], errors="coerce") - pd.to_numeric(rad[out_lw], errors="coerce"))
    )
    return rad, "netRadCalc"


def _find_incoming_sw(rad: pd.DataFrame) -> Optional[str]:
    """Find incoming shortwave mean column in DP1.00023.001 tables."""
    return _find_first_existing_non_qc(rad, ["inSWMean", "incomingShortwaveRadiationMean"]) or _find_by_regex_non_qc(
        rad, [r"^in.*sw.*mean$", r"incoming.*short.*wave.*mean"]
    )


def _build_daylight_mask(
    rad: pd.DataFrame,
    rn_col: str,
    time_index: pd.DatetimeIndex,
    dt_seconds: int,
    *,
    swin_thresh_Wm2: float = 5.0,
    gapfill_hours: float = 0.0,
) -> pd.Series:
    """Return boolean daylight mask indexed by time_index.

    Primary definition: incoming shortwave (inSWMean) > threshold.
    Fallback: Rn > threshold.
    """
    swin_col = _find_incoming_sw(rad)
    if swin_col is not None:
        swin = _collapse_to_timeseries(rad, swin_col)
        swin = _regularize_and_gapfill(
            swin, swin_col, dt_seconds, time_index=time_index, gapfill_hours=gapfill_hours
        )
        s = pd.to_numeric(swin[swin_col], errors="coerce")
        dm = (s > float(swin_thresh_Wm2)).fillna(False)
        dm.index = time_index
        return dm

    # fallback: use Rn itself
    rn = _collapse_to_timeseries(rad, rn_col)
    rn = _regularize_and_gapfill(rn, rn_col, dt_seconds, time_index=time_index, gapfill_hours=gapfill_hours)
    s = pd.to_numeric(rn[rn_col], errors="coerce")
    dm = (s > float(swin_thresh_Wm2)).fillna(False)  # same numeric threshold
    dm.index = time_index
    return dm


# -----------------------------
# Files / IO
# -----------------------------

def _read_any_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in (".csv", ".txt"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def _glob_one(tables_dir: Path, pattern: str) -> Path:
    matches = sorted(tables_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern} in {tables_dir}")
    # prefer parquet
    for p in matches:
        if p.suffix.lower() == ".parquet":
            return p
    return matches[0]


@dataclass
class Inputs:
    eddy_path: Path
    rad_path: Path
    g_path: Path
    rh_path: Path
    wind_path: Path
    pres_path: Path


def _find_inputs(tables_dir: Path) -> Inputs:
    # Eddy file: downloader writes eddy_fluxH2o_dp04_30min.* (see neon_00_fetch_inputs_CONUS_core.py)
    eddy = _glob_one(tables_dir, "eddy_*_dp04_*min.*")
    rad = _glob_one(tables_dir, "rad_DP1_00023_001_30min.*")
    g = _glob_one(tables_dir, "g_DP1_00040_001_30min.*")
    rh = _glob_one(tables_dir, "rh_DP1_00098_001_30min.*")
    wind = _glob_one(tables_dir, "wind_DP1_00001_001_30min.*")
    pres = _glob_one(tables_dir, "pres_DP1_00004_001_30min.*")
    return Inputs(eddy, rad, g, rh, wind, pres)


def _pick_nearest_level(df: pd.DataFrame, prefer_m: float, level_cols: Sequence[str]) -> pd.DataFrame:
    """If multiple heights/depths exist, pick rows closest to prefer_m."""
    df = df.copy()
    col = _find_first_existing(df, level_cols)
    if col is None:
        return df
    lev = pd.to_numeric(df[col], errors="coerce")
    if lev.notna().sum() == 0:
        return df
    # pick one level value (closest to prefer_m) based on unique levels
    levels = sorted({float(x) for x in lev.dropna().unique()})
    best = min(levels, key=lambda x: abs(x - prefer_m))
    return df.loc[np.isclose(lev, best)]


# -----------------------------
# Site processing
# -----------------------------

def _build_daily_site(
    site: str,
    tables_dir: Path,
    *,
    min_cov_full: float = 0.50,
    min_cov_day: float = 0.50,
    alpha_pt: float = 1.26,
    h2o_flux_units: str = "Wm2",  # or "mm" if already mm s-1 (rare)
    gapfill_hours: float = 2.0,
    swin_day_thresh: float = 5.0,
    fill_night_et_zero: bool = True,
    scale_daylight_missing: bool = True,
    assume_G0_if_missing: bool = True,
) -> pd.DataFrame:
    inputs = _find_inputs(tables_dir)

    # --------------------
    # Radiation first (daylight mask + Rn)
    # --------------------
    rad = _normalize_time(_read_any_table(inputs.rad_path))
    rad, rn_col = _ensure_net_radiation(rad, site=site, fname=inputs.rad_path.name)

    # Build a regular rad time base
    rad_ts = _collapse_to_timeseries(rad, rn_col)
    dt_rad = _infer_dt_seconds(rad_ts["time"])
    rad_ts = _regularize_and_gapfill(rad_ts, rn_col, dt_rad, gapfill_hours=gapfill_hours)

    rad_index = pd.DatetimeIndex(pd.to_datetime(rad_ts["time"]))
    daylight_rad = _build_daylight_mask(
        rad,
        rn_col,
        rad_index,
        dt_rad,
        swin_thresh_Wm2=swin_day_thresh,
        gapfill_hours=gapfill_hours,
    )
    # Ensure mask indexed by time
    daylight_rad.index = rad_index

    rn_daily = _daily_integrate_Wm2_to_MJ(rad_ts, rn_col, dt_rad)
    rn_n_day, rn_n_day_total, rn_cov_day = _daily_cov_over_mask(rad_ts, rn_col, daylight_rad)

    # --------------------
    # Eddy covariance (E) — from LE or fluxH2o (treated as LE by default)
    # --------------------
    eddy = _normalize_time(_read_any_table(inputs.eddy_path))

    le_col = _find_first_existing(
        eddy,
        [
            "fluxLatHeatMean",
            "fluxLatHeat",
            "latentHeatFluxMean",
            "latentHeatFlux",
            "LEMean",
            "LE",
        ],
    ) or _find_by_regex(eddy, [r"latent.*heat.*flux", r"\bLE(MEAN)?\b"])

    h2o_col = _find_first_existing(
        eddy,
        [
            "fluxH2oMean",
            "fluxH2o",
            "h2oFluxMean",
            "h2oFlux",
        ],
    ) or _find_by_regex(eddy, [r"flux.*h2o", r"h2o.*flux"])

    if le_col is None and h2o_col is None:
        raise KeyError(f"{site}: Could not find latent heat flux (LE) or fluxH2o column in {inputs.eddy_path.name}")

    if le_col is not None:
        value_col = le_col
        E_source = "LE_Wm2"
    else:
        value_col = h2o_col
        E_source = "fluxH2o"

    # Apply eddy QC
    eddy = _mask_eddy_qf(eddy, value_col)

    # Collapse and regularize
    eddy_ts = _collapse_to_timeseries(eddy, value_col)
    dt_eddy = _infer_dt_seconds(eddy_ts["time"])
    eddy_ts = _regularize_and_gapfill(eddy_ts, value_col, dt_eddy, gapfill_hours=gapfill_hours)

    eddy_index = pd.DatetimeIndex(pd.to_datetime(eddy_ts["time"]))

    # Daylight mask on eddy time base (reindex from rad-based mask)
    daylight_eddy = daylight_rad.reindex(eddy_index, fill_value=False)

    units = str(h2o_flux_units).lower()
    if le_col is not None or units in ("wm2", "w/m2", "w"):
        # Treat as LE in W m-2, convert to daily ET
        E_mm, E_n_day, E_expected_day, E_cov_day = _daily_ET_from_LE_Wm2(
            eddy_ts,
            value_col,
            dt_eddy,
            daylight_eddy,
            fill_night_zero=fill_night_et_zero,
            scale_daylight_missing=scale_daylight_missing,
        )
        E_expected = E_expected_day
        E_n = E_n_day
        E_cov = E_cov_day
        if le_col is None:
            E_source = "fluxH2o_Wm2_as_LE"
    else:
        # Interpret as a mass flux and convert to mm/day by integrating in time.
        v = pd.to_numeric(eddy_ts[value_col], errors="coerce")
        day = pd.to_datetime(eddy_ts["time"]).dt.floor("D")
        expected = int(round(86400 / dt_eddy)) if dt_eddy > 0 else 48

        if units.startswith("mmol"):
            # mmol -> kg: 18e-6 kg per mmol; kg/m2 = mm
            mm_per_s = v * 18e-6
            mm = (mm_per_s * float(dt_eddy)).groupby(day).sum(min_count=1)
            E_source = "fluxH2o_mmol"
        else:
            # assume already mm/s
            mm = (v * float(dt_eddy)).groupby(day).sum(min_count=1)
            E_source = "fluxH2o_mm_per_s"

        E_mm = mm
        E_n = v.groupby(day).count()
        E_expected = pd.Series(expected, index=E_mm.index)
        E_cov = E_n / expected

    # --------------------
    # Soil heat flux (G)
    # --------------------
    gdf = _normalize_time(_read_any_table(inputs.g_path))
    g_col = _find_first_existing_non_qc(
        gdf,
        [
            "SHFMean",           # common DP1.00040.001 naming
            "soilHeatFluxMean",
            "soilHeatFlux",
            "gMean",
            "g",
        ],
    ) or _find_by_regex_non_qc(gdf, [r"^shfmean$", r"soil.*heat.*flux.*mean"])

    if g_col is None:
        g_cols = [c for c in gdf.columns if re.search(r"SHF|heat|flux", str(c), flags=re.IGNORECASE)]
        raise KeyError(
            f"{site}: Could not find soil heat flux column in {inputs.g_path.name}. "
            f"Found possible G/SHF columns: {g_cols[:40]}"
        )

    gdf = _mask_by_final_qf(gdf, g_col)
    # pick representative depth if available (often 0.08 m)
    gdf = _pick_nearest_level(gdf, prefer_m=0.08, level_cols=["zOffset", "verticalPosition", "sensorDepth", "depth"])

    g_ts = _collapse_to_timeseries(gdf, g_col)
    dt_g = _infer_dt_seconds(g_ts["time"])
    g_ts = _regularize_and_gapfill(g_ts, g_col, dt_g, gapfill_hours=gapfill_hours)
    g_daily = _daily_integrate_Wm2_to_MJ(g_ts, g_col, dt_g)

    # daylight coverage diagnostics for G (optional use)
    g_index = pd.DatetimeIndex(pd.to_datetime(g_ts["time"]))
    daylight_g = daylight_rad.reindex(g_index, fill_value=False)
    g_n_day, g_n_day_total, g_cov_day = _daily_cov_over_mask(g_ts, g_col, daylight_g)

    # --------------------
    # RH + Temperature (DP1.00098.001)
    # --------------------
    rhdf = _normalize_time(_read_any_table(inputs.rh_path))

    t_col = _find_first_existing_non_qc(
        rhdf,
        [
            "tempRHMean",
            "airTemperatureMean",
            "airTempMean",
            "tempMean",
            "airTemperature",
            "airTemp",
        ],
    ) or _find_by_regex_non_qc(rhdf, [r"temp.*rh.*mean", r"air.*temp.*mean"])

    rh_col = _find_first_existing_non_qc(
        rhdf,
        [
            "RHMean",
            "relativeHumidityMean",
            "relHumidityMean",
            "relativeHumidity",
        ],
    ) or _find_by_regex_non_qc(rhdf, [r"^rhmean$", r"rel.*humid.*mean"])

    if t_col is None or rh_col is None:
        rh_like = [c for c in rhdf.columns if re.search(r"RH|humid|temp", str(c), flags=re.IGNORECASE)]
        raise KeyError(
            f"{site}: Could not find T and/or RH columns in {inputs.rh_path.name}. "
            f"Found RH/temp-like columns: {rh_like[:60]}"
        )

    rhdf = _mask_by_final_qf(rhdf, t_col)
    rhdf = _mask_by_final_qf(rhdf, rh_col)
    rhdf = _pick_nearest_level(rhdf, prefer_m=2.0, level_cols=["verticalPosition", "zOffset", "sensorHeight", "height"])

    t_ts = _collapse_to_timeseries(rhdf, t_col)
    rh_ts = _collapse_to_timeseries(rhdf, rh_col)
    dt_trh = _infer_dt_seconds(t_ts["time"])
    t_ts = _regularize_and_gapfill(t_ts, t_col, dt_trh, gapfill_hours=gapfill_hours)
    rh_ts = _regularize_and_gapfill(rh_ts, rh_col, dt_trh, gapfill_hours=gapfill_hours)
    T_daily = _daily_mean(t_ts, t_col, dt_trh)
    RH_daily = _daily_mean(rh_ts, rh_col, dt_trh)

    # --------------------
    # Wind speed (DP1.00001.001)
    # --------------------
    wdf = _normalize_time(_read_any_table(inputs.wind_path))
    u_col = _find_first_existing_non_qc(
        wdf,
        [
            "windSpeedMean",
            "horizontalWindSpeedMean",
            "windSpdMean",
            "windSpeed",
        ],
    ) or _find_by_regex_non_qc(wdf, [r"wind.*speed.*mean", r"wind.*speed"])
    if u_col is None:
        w_like = [c for c in wdf.columns if re.search(r"wind|speed", str(c), flags=re.IGNORECASE)]
        raise KeyError(f"{site}: Could not find wind speed column in {inputs.wind_path.name}. Found: {w_like[:50]}")
    wdf = _mask_by_final_qf(wdf, u_col)
    wdf = _pick_nearest_level(wdf, prefer_m=2.0, level_cols=["verticalPosition", "zOffset", "sensorHeight", "height"])
    w_ts = _collapse_to_timeseries(wdf, u_col)
    dt_u = _infer_dt_seconds(w_ts["time"])
    w_ts = _regularize_and_gapfill(w_ts, u_col, dt_u, gapfill_hours=gapfill_hours)
    U_daily = _daily_mean(w_ts, u_col, dt_u)

    # --------------------
    # --------------------
    # Pressure (DP1.00004.001)
    # --------------------
    pdf = _normalize_time(_read_any_table(inputs.pres_path))

    # DP1.00004.001 is documented in kPa and typically uses staPresMean, but naming can vary.
    p_col = _pick_pressure_col(pdf)
    if p_col is None:
        p_like = [c for c in pdf.columns if re.search(r"pres|pressure|baro", str(c), flags=re.IGNORECASE)]
        raise KeyError(
            f"{site}: Could not find a usable station/barometric pressure mean column in {inputs.pres_path.name}. "
            f"Found: {p_like[:50]}"
        )

    pdf = _mask_by_final_qf(pdf, p_col)
    pdf = _pick_nearest_level(pdf, prefer_m=2.0, level_cols=["verticalPosition", "zOffset", "sensorHeight", "height"])

    p_ts = _collapse_to_timeseries(pdf, p_col)
    dt_p = _infer_dt_seconds(p_ts["time"])
    p_ts = _regularize_and_gapfill(p_ts, p_col, dt_p, gapfill_hours=gapfill_hours)

    # Normalize pressure values to kPa (robust to Pa/hPa/MPa in upstream workflows).
    p_ts[p_col], p_fac, p_note = _pressure_to_kpa(p_ts[p_col])

    try:
        p_med = float(np.nanmedian(pd.to_numeric(p_ts[p_col], errors="coerce").astype(float)))
    except Exception:
        p_med = np.nan
    if np.isfinite(p_med) and (p_med < 20.0 or p_med > 200.0):
        print(
            f"[WARN] {site}: pressure column '{p_col}' median={p_med:.3g} kPa after unit normalization "
            f"({p_note}). Epa/x calculations may be unreliable."
        )

    P_daily = _daily_mean(p_ts, p_col, dt_p)

    # --------------------
    # Assemble daily frame
    # --------------------
    all_days = sorted(
        set(pd.to_datetime(E_mm.index))
        | set(pd.to_datetime(rn_daily.value.index))
        | set(pd.to_datetime(g_daily.value.index))
        | set(pd.to_datetime(T_daily.value.index))
        | set(pd.to_datetime(RH_daily.value.index))
        | set(pd.to_datetime(U_daily.value.index))
        | set(pd.to_datetime(P_daily.value.index))
    )
    daily = pd.DataFrame(index=pd.Index(all_days, name="date"))
    daily["site"] = site

    # store stream values and coverage
    daily["E_mm"] = E_mm
    daily["E_source"] = E_source
    daily["E_n_day"] = E_n
    daily["E_expected_day"] = E_expected
    daily["E_cov_day"] = daily["E_n_day"] / daily["E_expected_day"].replace({0: np.nan})

    # for backwards-compatibility, also keep E_n/E_expected_n/E_cov aliases
    daily["E_n"] = daily["E_n_day"]
    daily["E_expected_n"] = daily["E_expected_day"]
    daily["E_cov"] = daily["E_cov_day"]

    daily["Rn_MJ"] = rn_daily.value
    daily["Rn_n"] = rn_daily.n
    daily["Rn_expected_n"] = rn_daily.expected_n
    daily["Rn_cov_full"] = daily["Rn_n"] / daily["Rn_expected_n"]
    daily["Rn_n_day"] = rn_n_day
    daily["Rn_expected_day"] = rn_n_day_total
    daily["Rn_cov_day"] = rn_cov_day

    daily["G_MJ"] = g_daily.value
    daily["G_n"] = g_daily.n
    daily["G_expected_n"] = g_daily.expected_n
    daily["G_cov_full"] = daily["G_n"] / daily["G_expected_n"]
    daily["G_n_day"] = g_n_day
    daily["G_expected_day"] = g_n_day_total
    daily["G_cov_day"] = g_cov_day

    # Optionally assume G=0 when missing/poor coverage (daily scale often does this)
    daily["G_assumed0"] = False
    if assume_G0_if_missing:
        bad_g = daily["G_MJ"].isna() | (daily["G_cov_day"] < min_cov_day)
        daily.loc[bad_g, "G_MJ"] = 0.0
        daily.loc[bad_g, "G_assumed0"] = True

    daily["RnG_MJ"] = daily["Rn_MJ"] - daily["G_MJ"]

    daily["Ta_C"] = T_daily.value
    daily["Ta_n"] = T_daily.n
    daily["Ta_expected_n"] = T_daily.expected_n
    daily["Ta_cov"] = daily["Ta_n"] / daily["Ta_expected_n"]

    daily["RH_pct"] = RH_daily.value
    daily["RH_n"] = RH_daily.n
    daily["RH_expected_n"] = RH_daily.expected_n
    daily["RH_cov"] = daily["RH_n"] / daily["RH_expected_n"]

    daily["U_ms"] = U_daily.value
    daily["U_n"] = U_daily.n
    daily["U_expected_n"] = U_daily.expected_n
    daily["U_cov"] = daily["U_n"] / daily["U_expected_n"]

    daily["P_kPa"] = P_daily.value
    daily["P_col"] = p_col
    daily["P_units_factor"] = p_fac
    daily["P_units_note"] = p_note
    daily["P_n"] = P_daily.n
    daily["P_expected_n"] = P_daily.expected_n
    daily["P_cov"] = daily["P_n"] / daily["P_expected_n"]

    # --------------------
    # Coverage-based flags
    # --------------------
    # met/energy required for Epa/Ep0: use daylight coverage for energy, full coverage for met
    daily["good_met"] = (
        (daily["Rn_cov_day"] >= min_cov_day)
        & (daily["Ta_cov"] >= min_cov_full)
        & (daily["RH_cov"] >= min_cov_full)
        & (daily["U_cov"] >= min_cov_full)
        & (daily["P_cov"] >= min_cov_full)
    )
    # (Optional) if you don't assume G=0, require G day coverage too
    if not assume_G0_if_missing:
        daily["good_met"] = daily["good_met"] & (daily["G_cov_day"] >= min_cov_day)

    daily["good_E"] = (daily["E_cov_day"] >= min_cov_day)

    # mask values accordingly
    daily.loc[~daily["good_E"], "E_mm"] = np.nan
    daily.loc[~daily["good_met"], ["Rn_MJ", "G_MJ", "RnG_MJ", "Ta_C", "RH_pct", "U_ms", "P_kPa"]] = np.nan

    # --------------------
    # Compute evaporation constructs
    # --------------------
    Ta = daily["Ta_C"]
    RH = daily["RH_pct"].clip(lower=0.0, upper=100.0)
    U = daily["U_ms"]
    P = daily["P_kPa"]
    RnG = daily["RnG_MJ"]

    es = _sat_vp_kpa(Ta)
    delta = _delta_kpa_per_C(Ta)
    ea = es * (RH / 100.0)
    vpd = (es - ea).clip(lower=0.0)
    gamma = 0.000665 * P  # kPa/°C if P in kPa

    daily["es_kPa"] = es
    daily["Delta_kPa_C"] = delta
    daily["gamma_kPa_C"] = gamma
    daily["VPD_kPa"] = vpd

    daily["Epa_mm"] = (
        (0.408 * delta * RnG + gamma * (900.0 / (Ta + 273.0)) * U * vpd) / (delta + gamma)
    )

    daily["Ep0_eq_mm"] = (delta / (delta + gamma)) * 0.408 * RnG
    daily["Ep0_pt_mm"] = alpha_pt * (delta / (delta + gamma)) * 0.408 * RnG

    daily.loc[~daily["good_met"], ["Epa_mm", "Ep0_eq_mm", "Ep0_pt_mm"]] = np.nan

    daily["x_eq"] = daily["Ep0_eq_mm"] / daily["Epa_mm"]
    daily["x_pt"] = daily["Ep0_pt_mm"] / daily["Epa_mm"]
    daily["y"] = daily["E_mm"] / daily["Epa_mm"]


    # --------------------
    # Sanity check: if Epa collapses to equilibrium term for (almost) all days,
    # the aerodynamic term is effectively zero (often caused by accidentally
    # selecting a *FinalQF / *QM column instead of the physical variable).
    with np.errstate(invalid="ignore", divide="ignore"):
        aero_mm = daily["Epa_mm"] - daily["Ep0_eq_mm"]
        frac_zero = float((aero_mm.abs() < 1e-6).mean()) if len(aero_mm) else 0.0
    if frac_zero > 0.95:
        u_med = float(pd.to_numeric(daily.get("U_ms"), errors="coerce").median()) if "U_ms" in daily.columns else np.nan
        vpd_med = float(pd.to_numeric(daily.get("VPD_kPa"), errors="coerce").median()) if "VPD_kPa" in daily.columns else np.nan
        print(
            f"[WARN] {site}: Epa ~ Ep0_eq on {frac_zero:.0%} of days (aerodynamic term ~0). "
            f"This usually means wind speed and/or VPD are zero or were read from a QC flag column. "
            f"Median U={u_med:.3g} m/s, median VPD={vpd_med:.3g} kPa."
        )

    daily["good_day"] = daily["good_met"] & daily["good_E"] & daily[["Epa_mm", "Ep0_pt_mm", "E_mm"]].notna().all(axis=1)

    # finalize
    daily = daily.reset_index()
    daily["date"] = pd.to_datetime(daily["date"])

    return daily


# -----------------------------
# CLI
# -----------------------------

def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create daily NEON evaporation constructs (E, Epa, Ep0) from 30-min tables.",
    )
    p.add_argument("--data_root", required=True, help="Root folder containing site folders like HARV_2018-01_2024-12")
    p.add_argument("--sites", nargs="+", default=None, help="Optional list of site IDs to process (e.g., HARV OSBS)")
    p.add_argument("--min_cov_full", type=float, default=0.50, help="Minimum full-day coverage for met variables (fraction of 30-min intervals)")
    p.add_argument("--min_cov_day", type=float, default=0.50, help="Minimum daylight coverage for energy + eddy ET (fraction of daylight intervals)")
    p.add_argument("--gapfill_hours", type=float, default=2.0, help="Linear gap-fill (time interpolation) for short gaps up to this many hours")
    p.add_argument("--swin_day_thresh", type=float, default=5.0, help="Incoming SW threshold (W m-2) to define daylight intervals")
    p.add_argument("--alpha_pt", type=float, default=1.26, help="Priestley–Taylor alpha used for Ep0_pt")
    p.add_argument(
        "--h2o_flux_units",
        choices=["Wm2", "mmol", "mm"],
        default="Wm2",
        help="Units for fluxH2o (dp04). For NEON dp04, fluxH2o is latent heat flux in W m-2, so default Wm2 is usually correct.",
    )
    p.add_argument("--no_fill_night_et_zero", action="store_true", help="Do NOT fill missing nighttime LE with 0 before daily integration")
    p.add_argument("--no_scale_daylight_missing", action="store_true", help="Do NOT scale daylight integral by daylight coverage")
    p.add_argument("--no_assume_G0", action="store_true", help="Require measured soil heat flux; do NOT assume G=0 when missing")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing <SITE>_daily_cr.csv")
    return p.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    root = Path(args.data_root).expanduser()
    if not root.exists():
        print(f"[FATAL] data_root not found: {root}", file=sys.stderr)
        return 2

    site_dirs = sorted([p for p in root.iterdir() if p.is_dir() and re.match(r"^[A-Z0-9]{4}_", p.name)])
    if args.sites is not None:
        wanted = {s.upper() for s in args.sites}
        site_dirs = [p for p in site_dirs if p.name.split("_")[0].upper() in wanted]

    if not site_dirs:
        print(f"[FATAL] No site folders found under {root}", file=sys.stderr)
        return 2

    ok, bad = 0, 0
    for sd in site_dirs:
        site = sd.name.split("_")[0].upper()
        tables_dir = sd / "tables"
        out_csv = tables_dir / f"{site}_daily_cr.csv"
        print(f"\n=== {site} ===")
        try:
            if out_csv.exists() and not args.overwrite:
                print(f"[SKIP] {out_csv} exists (use --overwrite to rebuild)")
                ok += 1
                continue

            daily = _build_daily_site(
                site=site,
                tables_dir=tables_dir,
                min_cov_full=float(args.min_cov_full),
                min_cov_day=float(args.min_cov_day),
                alpha_pt=float(args.alpha_pt),
                h2o_flux_units=str(args.h2o_flux_units),
                gapfill_hours=float(args.gapfill_hours),
                swin_day_thresh=float(args.swin_day_thresh),
                fill_night_et_zero=not bool(args.no_fill_night_et_zero),
                scale_daylight_missing=not bool(args.no_scale_daylight_missing),
                assume_G0_if_missing=not bool(args.no_assume_G0),
            )
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            daily.to_csv(out_csv, index=False)
            print(f"[OK] Wrote {out_csv} ({len(daily)} days)")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {site}: {e!r}")
            bad += 1

    print(f"\nDone. Sites ok: {ok}, failed: {bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
