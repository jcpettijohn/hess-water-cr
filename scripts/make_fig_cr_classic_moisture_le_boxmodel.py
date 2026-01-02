r"""scripts/make_fig_cr_classic_moisture_le_boxmodel.py

Generate a classical Bouchet--Morton moisture-availability CR diagram in
latent-heat-flux space using the Tier-2 mixed-layer box model.

This script produces a *single* panel (finite unstressed canopy resistance,
$r_s^{\min} > 0$) illustrating wet-end closure and the non-convergence that
occurs if the "potential" diagnostic is evaluated with $r_s=0$ while the
wet reference remains canopy-limited.

Output (default): figures/fig_cr_classic_moisture_le_boxmodel.png

Notes on workflow / debugging
-----------------------------
This script is meant to be run as a fresh Python process, e.g.:

    python3 scripts/make_fig_cr_classic_moisture_le_boxmodel.py

To make parameter exploration less error-prone (and avoid editing model
defaults), this script supports both CLI flags and environment variables for
overriding BoxParams:

CLI (highest priority):    --RnG 150 --q_ft 0.004 --alpha_ent 1.0 ...
Env vars (next priority):  RNG=150 QFT=0.004 ALPHA_ENT=1.0 ...
Defaults (lowest):         params = BoxParams()

Use --debug to print import paths, parameters, output location, and a hash of
the written PNG (handy for detecting viewer caching).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.box_model import BoxParams, penman_monteith_E, priestley_taylor_Ep0, run_rs_sweep
from src.thermo import latent_heat_vaporization
from src.plotting import savefig


def _scaled_pt_latent(params: BoxParams, T_wet_C: float, E_wet: float) -> float:
    """Return a constant wet-benchmark latent heat flux using PT scaled to match wet equilibrium."""
    base = priestley_taylor_Ep0(params.RnG, T_wet_C, params.p_kpa, alpha_PT=1.0)
    alpha_eff = float(E_wet / base)
    Ep0 = priestley_taylor_Ep0(params.RnG, T_wet_C, params.p_kpa, alpha_PT=alpha_eff)
    return float(latent_heat_vaporization(T_wet_C) * Ep0)


def _scenario_curves(params: BoxParams, r_s_min: float) -> dict[str, np.ndarray]:
    """Compute availability and latent-heat curves for a finite wet-limit resistance.

    The moisture availability proxy is defined using *excess* resistance above r_s_min:
        M = r_a / (r_a + (r_s - r_s_min))
    which guarantees M=1 at the wet/unstressed limit and M->0 as r_s grows.
    """
    # Sweep total surface resistance from wet (r_s_min) to very dry.
    r_s_excess = np.r_[0.0, np.logspace(-1, 4, 120)]  # s m-1
    r_s = r_s_min + r_s_excess

    out = run_rs_sweep(params, r_s, definition="PT_wetref")
    T = out["T_C"]
    q = out["q"]

    # Moisture availability proxy (wet end at 1.0 by construction)
    avail = params.r_a / (params.r_a + (out["r_s"] - r_s_min))

    lam = latent_heat_vaporization(T)
    LE_actual = lam * out["E"]
    LE_pa_rs0 = lam * out["E_pa"]  # "potential" with r_s=0 (saturated patch)

    # "Potential" with an unstressed canopy resistance r_s=r_s_min (restores wet-end closure)
    E_pa_unstressed = np.array(
        [
            penman_monteith_E(
                params.RnG,
                float(Ti),
                float(qi),
                params.p_kpa,
                params.rho_a,
                r_a=params.r_a,
                r_s=r_s_min,
            )
            for Ti, qi in zip(T, q)
        ]
    )
    LE_pa_unstressed = lam * E_pa_unstressed

    # Wet-environment benchmark: PT scaled to match the wet-limit flux of this scenario.
    T_wet = float(T[0])
    E_wet = float(out["E"][0])
    LE_p0_const = _scaled_pt_latent(params, T_wet_C=T_wet, E_wet=E_wet)
    LE_p0 = np.full_like(LE_actual, LE_p0_const, dtype=float)

    # Sort by availability so x increases to the right (dry->wet)
    idx = np.argsort(avail)
    return {
        "avail": avail[idx],
        "LE_actual": LE_actual[idx],
        "LE_p0": LE_p0[idx],
        "LE_pa_rs0": LE_pa_rs0[idx],
        "LE_pa_unstressed": LE_pa_unstressed[idx],
    }


def _env_float(name: str) -> float | None:
    """Return env var as float, or None if not set."""
    v = os.getenv(name)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError as e:
        raise ValueError(f"Environment variable {name} must be a float, got {v!r}") from e


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate moisture-availability CR diagram (Tier-2 box model).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # BoxParams overrides
    p.add_argument("--RnG", type=float, default=None, help="Override BoxParams.RnG (W m-2). Env: RNG")
    p.add_argument("--q_ft", type=float, default=None, help="Override BoxParams.q_ft (kg/kg). Env: QFT")
    p.add_argument("--w_e0", type=float, default=None, help="Override BoxParams.w_e0 (m/s). Env: WE0")
    p.add_argument("--alpha_ent", type=float, default=None, help="Override BoxParams.alpha_ent (-). Env: ALPHA_ENT")
    p.add_argument("--dT_ent", type=float, default=None, help="Override BoxParams.dT_ent (K). Env: DT_ENT")

    # Figure/scenario controls
    p.add_argument(
        "--r_s_min",
        type=float,
        default=None,
        help="Unstressed canopy resistance (s m-1). Env: RS_MIN",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path. Env: OUT_PNG. Default: figures/fig_cr_classic_moisture_le_boxmodel.png",
    )
    p.add_argument(
        "--force-save",
        action="store_true",
        help="Bypass src.plotting.savefig and call fig.savefig directly (useful if savefig has overwrite logic).",
    )
    p.add_argument("--debug", action="store_true", help="Print import paths, parameters, and output hash.")
    return p


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_figure(fig: plt.Figure, out_png: Path, dpi: int, force: bool) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if force:
        fig.savefig(out_png, dpi=dpi)
    else:
        savefig(fig, out_png, dpi=dpi)


def main() -> None:
    args = _build_parser().parse_args()

    # --- Global style (font sizes tuned for journal-figure readability) ---
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    # Start from defaults, then apply overrides (CLI > env > defaults)
    params = BoxParams()

    overrides: list[tuple[str, float]] = []
    # env-based
    env_map = {
        "RnG": _env_float("RNG"),
        "q_ft": _env_float("QFT"),
        "w_e0": _env_float("WE0"),
        "alpha_ent": _env_float("ALPHA_ENT"),
        "dT_ent": _env_float("DT_ENT"),
    }
    for attr, val in env_map.items():
        if val is not None:
            overrides.append((attr, float(val)))

    # CLI-based
    cli_map = {
        "RnG": args.RnG,
        "q_ft": args.q_ft,
        "w_e0": args.w_e0,
        "alpha_ent": args.alpha_ent,
        "dT_ent": args.dT_ent,
    }
    for attr, val in cli_map.items():
        if val is not None:
            overrides.append((attr, float(val)))

    for attr, val in overrides:
        setattr(params, attr, float(val))

    # Scenario value (r_s_min) with CLI > env > default
    r_s_min_default = 70.0
    r_s_min = r_s_min_default
    env_rs_min = _env_float("RS_MIN")
    if env_rs_min is not None:
        r_s_min = float(env_rs_min)
    if args.r_s_min is not None:
        r_s_min = float(args.r_s_min)

    # Output path with CLI > env > default
    out_png = ROOT / "figures" / "fig_cr_classic_moisture_le_boxmodel.png"
    env_out = os.getenv("OUT_PNG")
    if env_out:
        out_png = Path(env_out)
    if args.out is not None:
        out_png = Path(args.out)
    if not out_png.is_absolute():
        out_png = ROOT / out_png

    if args.debug:
        import src.box_model as bm
        import src.plotting as plotting
        import src.thermo as thermo

        print("FIG SCRIPT:", Path(__file__).resolve())
        print("ROOT:", ROOT)
        print("PYTHON:", sys.executable)
        print("VERSION:", sys.version.split()[0])
        print("IMPORTED src.box_model:", Path(bm.__file__).resolve())
        print("IMPORTED src.plotting:", Path(plotting.__file__).resolve())
        print("IMPORTED src.thermo:", Path(thermo.__file__).resolve())
        print("PARAMS:", params)
        print("r_s_min:", r_s_min)
        print("OUT:", out_png.resolve())
        print("force_save:", bool(args.force_save))

    C = _scenario_curves(params, r_s_min=r_s_min)

    # Figure layout (single square panel)
    fig, ax = plt.subplots(figsize=(6.6, 6.6), constrained_layout=True)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(1)

    # Provide headroom for the wet->dry arrow and top curves
    ymax = float(
        np.nanmax(
            np.concatenate(
                [
                    np.asarray(C["LE_actual"]),
                    np.asarray(C["LE_p0"]),
                    np.asarray(C["LE_pa_rs0"]),
                    np.asarray(C["LE_pa_unstressed"]),
                ]
            )
        )
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax + max(28.0, 0.08 * ymax))
    ax.grid(True, alpha=0.25)

    ax.set_xlabel("Regional moisture availability (proxy)")
    ax.set_ylabel(r"Latent heat flux $\lambda E$ (W m$^{-2}$)")
    ax.set_title(r"Finite unstressed canopy resistance ($r_s^{\min}>0$)")

    # Wet (right) -> Dry (left) annotation in axes-fraction coordinates
    wetdry_fs = 12
    ax.annotate("dry", xy=(0.02, 0.97), xycoords="axes fraction", ha="left", va="top", fontsize=wetdry_fs)
    ax.annotate("wet", xy=(0.98, 0.97), xycoords="axes fraction", ha="right", va="top", fontsize=wetdry_fs)
    ax.annotate(
        "",
        xy=(0.10, 0.94),
        xytext=(0.90, 0.94),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1.2),
    )

    # Curves
    lw = 2.4
    ax.plot(C["avail"], C["LE_actual"], lw=lw, label=r"Actual $\lambda E$")
    ax.plot(
        C["avail"],
        C["LE_p0"],
        lw=lw,
        ls="--",
        label=r"Wet benchmark $\lambda E_{p0}$\n(PT, scaled)",
    )
    ax.plot(
        C["avail"],
        C["LE_pa_rs0"],
        lw=lw,
        label=r'"Potential" without canopy limit\n(PM, $r_s=0$)',
    )
    ax.plot(
        C["avail"],
        C["LE_pa_unstressed"],
        lw=lw,
        ls=":",
        label=r'"Potential" with unstressed canopy\n($r_s=r_s^{\min}$)',
    )

    # Legend: wrapped labels allow larger font without spilling outside the axes.
    ax.legend(frameon=False, loc="lower right", handlelength=2.2, labelspacing=0.5)

    # Callout: point to the non-convergent "potential" (r_s=0) curve near the wet end.
    callout_fs = 12
    x_text = 0.52
    x_target = 0.90

    y_green_at_text = float(np.interp(x_text, C["avail"], C["LE_pa_rs0"]))
    y_red_at_text = float(np.interp(x_text, C["avail"], C["LE_pa_unstressed"]))
    gap = y_green_at_text - y_red_at_text

    # Put the label in the whitespace between the green and red curves.
    y_text = y_red_at_text + 0.50 * gap
    y_text = max(y_text, y_red_at_text + 10.0)
    y_text = min(y_text, y_green_at_text - 10.0)

    y_target = float(np.interp(x_target, C["avail"], C["LE_pa_rs0"]))
    ax.annotate(
        "non-convergence\n" + r"(if $r_s^{\min}$ omitted)",
        xy=(x_target, y_target),
        xytext=(x_text, y_text),
        arrowprops=dict(arrowstyle="->", lw=1.0),
        fontsize=callout_fs,
        ha="left",
        va="center",
    )

    _save_figure(fig, out_png, dpi=300, force=bool(args.force_save))

    if args.debug:
        try:
            st = out_png.stat()
            print("WROTE:", out_png.resolve())
            print(f"  size: {st.st_size} bytes")
            print(f"  mtime: {st.st_mtime}")
            print(f"  sha1: {_sha1(out_png)}")
        except FileNotFoundError:
            print("ERROR: output file not found after save:", out_png.resolve())

    plt.close(fig)


if __name__ == "__main__":
    main()
