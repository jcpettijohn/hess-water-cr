# scripts/make_fig_box_model_schematic.py
# (drop-in replacement; fixes all text overlap + improves typography)

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _setup_typography() -> None:
    # Clean, consistent typography across backends (no TeX dependency)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "mathtext.fontset": "stix",
            "mathtext.default": "it",
            "axes.unicode_minus": False,
        }
    )


def _rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    lw: float = 2.6,
    ec: str = "0.15",
    fc: str = "white",
    pad: float = 0.012,
    rounding: float = 0.018,
    z: float = 1.0,
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size={rounding}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        transform=ax.transAxes,
        zorder=z,
    )
    ax.add_patch(box)
    return box


def _arrow(
    ax,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    arrowstyle: str = "-|>",
    lw: float = 2.2,
    ms: float = 16,
    color: str = "0.15",
    z: float = 2.0,
) -> FancyArrowPatch:
    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle=arrowstyle,
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        transform=ax.transAxes,
        zorder=z,
        shrinkA=0.0,
        shrinkB=0.0,
    )
    ax.add_patch(arr)
    return arr


def _axes_pix(ax, x: float, y: float) -> tuple[float, float]:
    return tuple(ax.transAxes.transform((x, y)))


def _fit_text(
    ax,
    fig,
    *,
    x: float,
    y: float,
    text: str,
    region: tuple[float, float, float, float],  # (x0,y0,w,h) in axes coords
    ha: str = "center",
    va: str = "center",
    fs0: int = 24,
    fs_min: int = 10,
    weight: str | None = "bold",
    color: str = "0.05",
    pad_frac: float = 0.06,
    z: float = 3.0,
) -> None:
    """
    Place text and automatically shrink fontsize until it fits in `region`.
    This is what makes the layout robust (no overlaps when DPI changes).
    """
    rx, ry, rw, rh = region
    # inner padded region
    ix0 = rx + pad_frac * rw
    ix1 = rx + rw - pad_frac * rw
    iy0 = ry + pad_frac * rh
    iy1 = ry + rh - pad_frac * rh

    # available size in pixels
    w_pix = _axes_pix(ax, ix1, y)[0] - _axes_pix(ax, ix0, y)[0]
    h_pix = _axes_pix(ax, x, iy1)[1] - _axes_pix(ax, x, iy0)[1]

    # ensure renderer exists
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for fs in range(fs0, fs_min - 1, -1):
        t = ax.text(
            x,
            y,
            text,
            ha=ha,
            va=va,
            transform=ax.transAxes,
            fontsize=fs,
            fontweight=weight,
            color=color,
            zorder=z,
        )
        fig.canvas.draw()
        bb = t.get_window_extent(renderer=renderer)
        if bb.width <= w_pix and bb.height <= h_pix:
            return
        t.remove()

    # fallback (should rarely hit)
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        transform=ax.transAxes,
        fontsize=fs_min,
        fontweight=weight,
        color=color,
        zorder=z,
    )


def main() -> None:
    _setup_typography()

    root = Path(__file__).resolve().parents[1]
    fig_dir = root / "figures"
    out_dir = root / "outputs"
    fig_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)

    # A slightly larger canvas prevents “cramped” feeling and improves legibility
    fig = plt.figure(figsize=(10.8, 6.0), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- Layout (axes coordinates) ---
    # Keep the relaxation arrow *outside* the mixed-layer box to guarantee no overlap.
    mixed = (0.055, 0.54, 0.69, 0.40)  # x, y, w, h
    land = (0.14, 0.16, 0.46, 0.20)

    _rounded_box(ax, *mixed, lw=2.8, ec="0.15", fc="white", rounding=0.018, z=1.0)
    _rounded_box(ax, *land, lw=2.8, ec="0.15", fc="white", rounding=0.018, z=1.0)

    # Mixed-layer title + state (fit-to-box)
    mx, my, mw, mh = mixed
    _fit_text(
        ax,
        fig,
        x=mx + mw / 2,
        y=my + mh * 0.84,
        text=r"Mixed layer (depth $h$)",
        region=mixed,
        fs0=30,
        fs_min=18,
        weight="bold",
    )
    _fit_text(
        ax,
        fig,
        x=mx + mw / 2,
        y=my + mh * 0.69,
        text=r"State: $(T,\,q)$",
        region=(mx, my + mh * 0.58, mw, mh * 0.22),
        fs0=20,
        fs_min=14,
        weight="semibold",
        color="0.12",
        pad_frac=0.10,
    )

    # Diagnostics callout (separate box inside mixed-layer)
    diag = (mx + mw * 0.04, my + mh * 0.10, mw * 0.78, mh * 0.45)
    _rounded_box(ax, *diag, lw=1.9, ec="0.60", fc="white", rounding=0.012, z=1.2)

    diag_text = (
        "Diagnostics (computed from the equilibrated air state):\n"
        "• $E$: Penman–Monteith with variable $r_s$\n"
        "• $E_{pa}$: PM with $r_s=0$ (same $T,q$)\n"
        "• $E_{p0}$: Priestley–Taylor using dry-state or wet-reference air"
    )
    # Fit diagnostics text to its callout region (prevents overlaps if you edit wording)
    _fit_text(
        ax,
        fig,
        x=diag[0] + diag[2] * 0.04,
        y=diag[1] + diag[3] * 0.92,
        text=diag_text,
        region=diag,
        ha="left",
        va="top",
        fs0=15,
        fs_min=11,
        weight=None,
        color="0.10",
        pad_frac=0.08,
        z=3.0,
    )

    # Land-surface title + energy balance label (fit-to-box)
    lx, ly, lw_, lh = land
    _fit_text(
        ax,
        fig,
        x=lx + lw_ / 2,
        y=ly + lh * 0.70,
        text="Land surface",
        region=land,
        fs0=26,
        fs_min=16,
        weight="bold",
        color="0.05",
    )
    _fit_text(
        ax,
        fig,
        x=lx + lw_ / 2,
        y=ly + lh * 0.38,
        text="energy balance",
        region=(lx, ly, lw_, lh * 0.55),
        fs0=20,
        fs_min=13,
        weight="semibold",
        color="0.12",
    )

    # Put the energy-balance equation *below* the land box (avoids any overlap)
    ax.text(
        lx + lw_ * 0.06,
        ly - 0.040,
        r"$R_n - G = H + \lambda E$",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=19,
        fontweight="semibold",
        color="0.05",
        zorder=3.0,
    )

    # Flux arrows from land surface to mixed layer
    y0 = ly + lh
    y1 = my
    for xx, lab in [
        (lx + lw_ * 0.35, r"$\lambda E$"),
        (lx + lw_ * 0.55, r"$H$"),
    ]:
        _arrow(ax, xx, y0 + 0.01, xx, y1 - 0.01, lw=2.2, ms=16, color="0.15", z=2.0)
        ax.text(
            xx + 0.018,
            (y0 + y1) / 2,
            lab,
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontsize=19,
            fontweight="semibold",
            color="0.05",
            zorder=3.0,
        )

    # ---------------------------------------------------------------------
    # Right-side processes: (i) large-scale relaxation toward background air
    # and (ii) optional entrainment/advection drying (bulk exchange with q_ft).
    #
    # Layout goals (per manuscript + reviewer feedback):
    #   - Keep all text fully outside the rounded mixed-layer box (no overlap).
    #   - Keep (tau_T, tau_q) clear of the entrainment/advection arrow.
    #   - Depict dry-air import as an *exchange* (arrow both in and out of the box).
    # ---------------------------------------------------------------------

    x_box_right = mx + mw
    xA0, xA1 = x_box_right, 0.965

    # (i) Relaxation toward background (T_b, q_b)
    y_relax = my + mh * 0.66
    _arrow(ax, xA0, y_relax, xA1, y_relax, lw=2.2, ms=16, color="0.15", z=2.0)

    # Place the label above the arrow, *left-aligned* so it cannot intrude into the box.
    relax_region = (
        x_box_right,
        y_relax + 0.012,
        1.0 - x_box_right,
        (my + mh) - (y_relax + 0.012),
    )
    _fit_text(
        ax,
        fig,
        x=relax_region[0] + relax_region[2] * 0.06,
        y=relax_region[1] + relax_region[3] * 0.93,
        text="Relax toward background\n$(T_b,\\,q_b)$",
        region=relax_region,
        ha="left",
        va="top",
        fs0=20,
        fs_min=12,
        weight="semibold",
        color="0.05",
        pad_frac=0.06,
        z=3.0,
    )

    # (ii) Entrainment / advection drying (moisture-budget exchange with imported-air humidity q_ft)
    y_ent = my + mh * 0.33
    _arrow(
        ax,
        x_box_right - 0.020,  # cross the box boundary slightly so the exchange is visually clear
        y_ent,
        xA1,
        y_ent,
        arrowstyle="<|-|>",  # exchange (both into and out of the control volume)
        lw=2.2,
        ms=16,
        color="0.15",
        z=2.0,
    )

    ent_region = (
        x_box_right,
        my + 0.010,
        1.0 - x_box_right,
        (y_ent - (my + 0.010)) - 0.010,
    )
    _fit_text(
        ax,
        fig,
        x=ent_region[0] + ent_region[2] * 0.06,
        y=ent_region[1] + ent_region[3] * 0.92,
        # Use \\, for mathtext spacing (avoid invalid escape sequences in Python strings)
        text="Entrainment / advection drying\n$w_e,\\,q_{\\mathrm{ft}}$",
        region=ent_region,
        ha="left",
        va="top",
        fs0=20,
        fs_min=12,
        weight="semibold",
        color="0.05",
        pad_frac=0.06,
        z=3.0,
    )

    # (iii) The relaxation time scales sit between the two arrows
    times_region = (
        x_box_right,
        y_ent + 0.016,
        1.0 - x_box_right,
        (y_relax - y_ent) - 0.032,
    )
    _fit_text(
        ax,
        fig,
        x=times_region[0] + times_region[2] * 0.06,
        y=times_region[1] + times_region[3] * 0.92,
        text="time scales:\n$\\tau_T,\ \\tau_q$",
        region=times_region,
        ha="left",
        va="top",
        fs0=20,
        fs_min=12,
        weight="semibold",
        color="0.05",
        pad_frac=0.06,
        z=3.0,
    )

    # Save (write to BOTH places so your LaTeX can point at figures/)
    out_fig = fig_dir / "fig_box_model_schematic.png"
    out_out = out_dir / "fig_box_model_schematic.png"
    for p in (out_fig, out_out):
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Wrote: {out_fig}")
    print(f"Wrote: {out_out}")


if __name__ == "__main__":
    main()
