"""extract_tier2_regime_table_values.py

Extract the Tier-2 regime parameters and shared forcing/definition settings used
by the figure-generation scripts, and write a small text/LaTeX snippet you can
copy into Table~\ref{tab:tier2-regimes}.

Designed for the *cr_paper_final_lab* workflow:
  - place this file in the project root or in scripts/
  - run from the project root:  python scripts/extract_tier2_regime_table_values.py

It will try to locate these scripts (in scripts/, root/, or /mnt/data):
  - make_fig_complementarity_plane.py
  - make_fig_cr_classic_moisture_le_regimes.py
  - make_fig_tier2_overlay_atlas.py

Outputs:
  outputs/tier2_regime_values.txt
  outputs/tier2_regime_values.tex
"""

from __future__ import annotations

import ast
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _find_script(root: Path, filename: str) -> Path:
    """Find a script in common locations."""
    candidates = [
        root / "scripts" / filename,
        root / filename,
        Path("/mnt/data") / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {filename}. Looked in: "
        + ", ".join(str(c) for c in candidates)
    )


def _get_main_function(tree: ast.AST) -> Optional[ast.FunctionDef]:
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    return None


def _const_str(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _const_num(node: ast.AST) -> Optional[float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    # Allow unary negative constants
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(
        node.operand, ast.Constant
    ):
        if isinstance(node.operand.value, (int, float)):
            return -float(node.operand.value)
    return None


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _is_attr(node: ast.AST, base: str, attr: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == base
        and node.attr == attr
    )


def _extract_boxparams_keywords(call: ast.Call) -> Dict[str, float]:
    """Extract numeric keyword args from BoxParams(h=..., tau_T=..., etc.)"""
    out: Dict[str, float] = {}
    for kw in call.keywords:
        if kw.arg is None:
            continue
        v = _const_num(kw.value)
        if v is None:
            continue
        out[kw.arg] = v
    return out


def _extract_regimes(script_path: Path) -> List[Dict[str, Any]]:
    """Extract regimes list from inside main() in a figure script."""
    tree = ast.parse(script_path.read_text(encoding="utf-8"))

    def _find_regimes_node(stmts: list[ast.stmt]) -> Optional[ast.AST]:
        """Find the AST node assigned to a variable named `regimes`.

        Supports both plain assignments (regimes = [...]) and annotated assignments
        (regimes: list[...] = [...]).
        """
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if _is_name(tgt, "regimes"):
                        return stmt.value
            elif isinstance(stmt, ast.AnnAssign):
                if _is_name(stmt.target, "regimes") and stmt.value is not None:
                    return stmt.value
        return None

    regimes_node: Optional[ast.AST] = None

    # Prefer a `regimes` definition inside main(), but fall back to a module-level
    # `regimes` if the script is structured differently.
    main_fn = _get_main_function(tree)
    if main_fn is not None:
        regimes_node = _find_regimes_node(main_fn.body)
    if regimes_node is None:
        regimes_node = _find_regimes_node([s for s in tree.body if isinstance(s, ast.stmt)])

    if regimes_node is None or not isinstance(regimes_node, (ast.List, ast.Tuple)):
        return []

    elts = regimes_node.elts  # type: ignore[attr-defined]

    regimes: List[Dict[str, Any]] = []
    for elt in regimes_node.elts:
        # Pattern A: dict with keys like "label" or "name" and "params": BoxParams(...)
        if isinstance(elt, ast.Dict):
            d: Dict[str, Any] = {}
            for k_node, v_node in zip(elt.keys, elt.values):
                k = _const_str(k_node) if k_node is not None else None
                if not k:
                    continue
                if k in {"label", "name"}:
                    s = _const_str(v_node)
                    if s:
                        d["label"] = s
                if k == "params" and isinstance(v_node, ast.Call):
                    # BoxParams(...) call
                    if _is_name(v_node.func, "BoxParams"):
                        d["boxparams_kwargs"] = _extract_boxparams_keywords(v_node)
            if d:
                regimes.append(d)
            continue

        # Pattern B: tuple like ("label", BoxParams(...), ...)
        if isinstance(elt, ast.Tuple) and elt.elts:
            d2: Dict[str, Any] = {}
            if (s := _const_str(elt.elts[0])):
                d2["label"] = s
            for sub in elt.elts:
                if isinstance(sub, ast.Call) and _is_name(sub.func, "BoxParams"):
                    d2["boxparams_kwargs"] = _extract_boxparams_keywords(sub)
            if d2:
                regimes.append(d2)

    return regimes


def _find_rs_sweep(script_path: Path) -> Optional[Dict[str, Any]]:
    """Try to infer the r_s sweep from a figure script.

    Recognizes:
      - rs_values = np.logspace(a,b,N)
      - r_s = np.r_[0.0, np.logspace(a,b,N)]

    Returns dict with keys: include_zero, log10_min, log10_max, N.
    """
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    main_fn = _get_main_function(tree)
    nodes = main_fn.body if main_fn is not None else tree.body  # type: ignore[attr-defined]

    # Heuristic: inspect assignments in order.
    for stmt in nodes:
        if not isinstance(stmt, ast.Assign):
            continue
        # only simple Name targets
        targets = [t for t in stmt.targets if isinstance(t, ast.Name)]
        if not targets:
            continue
        tname = targets[0].id
        if tname not in {"rs_values", "r_s", "rs", "rs_sweep", "rs_values"}:
            continue

        v = stmt.value

        # Case 1: np.logspace(a,b,N)
        if isinstance(v, ast.Call) and _is_attr(v.func, "np", "logspace"):
            a = _const_num(v.args[0]) if len(v.args) > 0 else None
            b = _const_num(v.args[1]) if len(v.args) > 1 else None
            n = _const_num(v.args[2]) if len(v.args) > 2 else None
            if a is not None and b is not None and n is not None:
                return {
                    "include_zero": False,
                    "log10_min": a,
                    "log10_max": b,
                    "N": int(round(n)),
                    "var": tname,
                }

        # Case 2: np.r_[0.0, np.logspace(a,b,N)]  (AST: Subscript(np.r_, Tuple(...)))
        if isinstance(v, ast.Subscript) and _is_attr(v.value, "np", "r_"):
            # slice can be Tuple or Index(Tuple) depending on python version
            sl = v.slice
            if isinstance(sl, ast.Tuple):
                elts = sl.elts
            elif isinstance(sl, ast.Index) and isinstance(sl.value, ast.Tuple):  # pragma: no cover
                elts = sl.value.elts
            else:
                elts = []

            include0 = any((_const_num(e) == 0.0) for e in elts)
            ls = next(
                (e for e in elts if isinstance(e, ast.Call) and _is_attr(e.func, "np", "logspace")),
                None,
            )
            if isinstance(ls, ast.Call):
                a = _const_num(ls.args[0]) if len(ls.args) > 0 else None
                b = _const_num(ls.args[1]) if len(ls.args) > 1 else None
                n = _const_num(ls.args[2]) if len(ls.args) > 2 else None
                if a is not None and b is not None and n is not None:
                    return {
                        "include_zero": include0,
                        "log10_min": a,
                        "log10_max": b,
                        "N": int(round(n)) + (1 if include0 else 0),
                        "var": tname,
                    }

    # Fallback: find first np.logspace anywhere
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_attr(node.func, "np", "logspace"):
            a = _const_num(node.args[0]) if len(node.args) > 0 else None
            b = _const_num(node.args[1]) if len(node.args) > 1 else None
            n = _const_num(node.args[2]) if len(node.args) > 2 else None
            if a is not None and b is not None and n is not None:
                return {
                    "include_zero": False,
                    "log10_min": a,
                    "log10_max": b,
                    "N": int(round(n)),
                    "var": "<unknown>",
                }

    return None


def _pick_regime(regimes: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    """Pick the first regime whose label contains key (case-insensitive)."""
    key_l = key.lower()
    for r in regimes:
        lbl = str(r.get("label", "")).lower()
        if key_l in lbl:
            return r
    return None


def _fmt_num(x: float) -> str:
    # format compactly but stably for LaTeX table
    if abs(x) >= 1000 or (abs(x) > 0 and abs(x) < 1e-2):
        return f"{x:.3g}"
    if float(int(round(x))) == x:
        return str(int(round(x)))
    return f"{x:.4g}"


def main() -> None:
    root = Path(__file__).resolve().parent
    # if we are in scripts/, project root is parent
    if (root / "src").exists() and (root / "scripts").exists():
        project_root = root
    else:
        project_root = root.parent

    sys.path.insert(0, str(project_root))
    # Import BoxParams for shared defaults (RnG, p, Tb, qb, alpha_PT, etc.).
    # In the packaged project it's in src/box_model.py; for standalone testing it may
    # be available as box_model.py at the root.
    BoxParams = None  # type: ignore
    try:  # preferred
        from src.box_model import BoxParams as _BoxParams  # type: ignore

        BoxParams = _BoxParams
    except Exception:
        try:  # fallback
            from box_model import BoxParams as _BoxParams  # type: ignore

            BoxParams = _BoxParams
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Could not import BoxParams (tried src.box_model and box_model). "
                "Run this from the project root (where src/ exists) or ensure box_model.py is importable."
            ) from e

    scripts = {
        "comp_plane": _find_script(project_root, "make_fig_complementarity_plane.py"),
        "cr_regimes": _find_script(project_root, "make_fig_cr_classic_moisture_le_regimes.py"),
        "overlay": _find_script(project_root, "make_fig_tier2_overlay_atlas.py"),
    }

    extracted: Dict[str, Any] = {}
    for k, sp in scripts.items():
        regs = _extract_regimes(sp)
        rs = _find_rs_sweep(sp)
        extracted[k] = {"path": str(sp), "regimes": regs, "rs_sweep": rs}

    # Use overlay regimes as the default regime set for the *overlay table*.
    overlay_regs = extracted["overlay"]["regimes"]
    if not overlay_regs:
        raise RuntimeError(
            "Could not extract overlay regimes. "
            "Make sure make_fig_tier2_overlay_atlas.py defines a variable named 'regimes' inside main()."
        )

    # Instantiate BoxParams for the three regimes
    def to_params(r: Dict[str, Any]) -> Any:
        kw = dict(r.get("boxparams_kwargs", {}))
        return BoxParams(**kw)

    strong = _pick_regime(overlay_regs, "strong")
    base = _pick_regime(overlay_regs, "baseline")
    weak = _pick_regime(overlay_regs, "weak")
    if not (strong and base and weak):
        raise RuntimeError(
            "Could not find Strong/Baseline/Weak regimes in overlay script. "
            "Expected labels containing 'Strong', 'Baseline', and 'Weak'."
        )

    p_strong = to_params(strong)
    p_base = to_params(base)
    p_weak = to_params(weak)

    # Check shared settings consistency (everything except the coupling knobs).
    def strip_coupling(d: Dict[str, Any]) -> Dict[str, Any]:
        dd = dict(d)
        for k2 in ["h", "tau_T", "tau_q", "r_a"]:
            dd.pop(k2, None)
        return dd

    s0 = strip_coupling(asdict(p_strong))
    s1 = strip_coupling(asdict(p_base))
    s2 = strip_coupling(asdict(p_weak))
    shared_ok = (s0 == s1 == s2)

    # r_s sweep: prefer the overlay script's sweep
    rs = extracted["overlay"]["rs_sweep"] or extracted["comp_plane"]["rs_sweep"]
    if rs is None:
        raise RuntimeError("Could not infer r_s sweep from overlay/comp-plane scripts.")
    rs_include0 = bool(rs.get("include_zero", False))
    rs_min = 10 ** float(rs["log10_min"])
    rs_max = 10 ** float(rs["log10_max"])
    n_rs = int(rs["N"])

    # Shared settings from baseline params (assumed same across regimes)
    p0 = p_base
    shared = {
        "RnG": p0.RnG,
        "p": p0.p_kpa,
        "Tb": p0.T_b_C,
        "qb": p0.q_b,
        "alphaPT": p0.alpha_PT,
    }

    out_lines: List[str] = []
    out_lines.append("Tier-2 regime values (extracted)\n")
    out_lines.append(f"Overlay script: {extracted['overlay']['path']}")
    out_lines.append(f"Comp-plane script: {extracted['comp_plane']['path']}")
    out_lines.append(f"CR-regimes script: {extracted['cr_regimes']['path']}\n")

    out_lines.append("Regimes used in overlay (and typically comp-plane):")
    out_lines.append(
        f"  Strong   : h={_fmt_num(p_strong.h)} m, tau_T={_fmt_num(p_strong.tau_T)} s, tau_q={_fmt_num(p_strong.tau_q)} s, r_a={_fmt_num(p_strong.r_a)} s m^-1"
    )
    out_lines.append(
        f"  Baseline : h={_fmt_num(p_base.h)} m, tau_T={_fmt_num(p_base.tau_T)} s, tau_q={_fmt_num(p_base.tau_q)} s, r_a={_fmt_num(p_base.r_a)} s m^-1"
    )
    out_lines.append(
        f"  Weak     : h={_fmt_num(p_weak.h)} m, tau_T={_fmt_num(p_weak.tau_T)} s, tau_q={_fmt_num(p_weak.tau_q)} s, r_a={_fmt_num(p_weak.r_a)} s m^-1\n"
    )

    out_lines.append("Shared forcing/definition settings (from baseline BoxParams):")
    out_lines.append(
        "  "
        + ", ".join(
            [
                f"RnG={_fmt_num(shared['RnG'])} W m^-2",
                f"p={_fmt_num(shared['p'])} kPa",
                f"Tb={_fmt_num(shared['Tb'])} C",
                f"qb={_fmt_num(shared['qb'])} kg kg^-1",
                f"alpha_PT={_fmt_num(shared['alphaPT'])}",
            ]
        )
    )
    out_lines.append(
        f"  r_s sweep: min={rs_min:.3g} s m^-1, max={rs_max:.3g} s m^-1, N={n_rs} (includes r_s=0: {'yes' if rs_include0 else 'no'})\n"
    )
    out_lines.append(
        "Shared-settings consistency across regimes (excluding h, tau_T, tau_q, r_a): "
        + ("OK" if shared_ok else "WARNING: mismatch detected")
    )

    # Also report CR-regimes figure regime list if it differs
    cr_regs = extracted["cr_regimes"]["regimes"]
    if cr_regs:
        out_lines.append("\nRegimes found in CR-regimes figure script (for cross-check):")
        for r in cr_regs:
            lbl = r.get("label", "<no label>")
            kw = r.get("boxparams_kwargs", {})
            if kw:
                tmp = BoxParams(**kw)
                out_lines.append(
                    f"  {lbl}: h={_fmt_num(tmp.h)} m, tau_T={_fmt_num(tmp.tau_T)} s, tau_q={_fmt_num(tmp.tau_q)} s, r_a={_fmt_num(tmp.r_a)}"
                )

    # Write outputs
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    (outputs_dir / "tier2_regime_values.txt").write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    # Small LaTeX helper snippet
    tex_lines: List[str] = []
    tex_lines.append("% Auto-generated by extract_tier2_regime_table_values.py")
    tex_lines.append(f"% Source: {extracted['overlay']['path']}")
    tex_lines.append("% Paste values into Table~\\ref{tab:tier2-regimes} placeholders\n")
    tex_lines.append(f"% <h_strong> = {_fmt_num(p_strong.h)}")
    tex_lines.append(f"% <tauT_strong> = {_fmt_num(p_strong.tau_T)}")
    tex_lines.append(f"% <tauq_strong> = {_fmt_num(p_strong.tau_q)}")
    tex_lines.append(f"% <ra_strong> = {_fmt_num(p_strong.r_a)}")
    tex_lines.append(f"% <h_base> = {_fmt_num(p_base.h)}")
    tex_lines.append(f"% <tauT_base> = {_fmt_num(p_base.tau_T)}")
    tex_lines.append(f"% <tauq_base> = {_fmt_num(p_base.tau_q)}")
    tex_lines.append(f"% <ra_base> = {_fmt_num(p_base.r_a)}")
    tex_lines.append(f"% <h_weak> = {_fmt_num(p_weak.h)}")
    tex_lines.append(f"% <tauT_weak> = {_fmt_num(p_weak.tau_T)}")
    tex_lines.append(f"% <tauq_weak> = {_fmt_num(p_weak.tau_q)}")
    tex_lines.append(f"% <ra_weak> = {_fmt_num(p_weak.r_a)}\n")
    tex_lines.append(f"% <RnG> = {_fmt_num(shared['RnG'])}")
    tex_lines.append(f"% <p> = {_fmt_num(shared['p'])}")
    tex_lines.append(f"% <Tb> = {_fmt_num(shared['Tb'])}")
    tex_lines.append(f"% <qb> = {_fmt_num(shared['qb'])}")
    tex_lines.append(f"% <alphaPT> = {_fmt_num(shared['alphaPT'])}")
    tex_lines.append(f"% <rs_min> = {rs_min:.3g}")
    tex_lines.append(f"% <rs_max> = {rs_max:.3g}")
    tex_lines.append(f"% <Nrs> = {n_rs}")
    tex_lines.append(f"% includes r_s=0 explicitly: {'yes' if rs_include0 else 'no'}")

    (outputs_dir / "tier2_regime_values.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    print("\n".join(out_lines))
    print(f"\nWrote: {outputs_dir/'tier2_regime_values.txt'}")
    print(f"Wrote: {outputs_dir/'tier2_regime_values.tex'}")


if __name__ == "__main__":
    main()
