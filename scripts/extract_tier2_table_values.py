#!/usr/bin/env python3
"""
extract_tier2_table_values.py

Run this from the project root (the folder that contains `src/` and `scripts/`):

    python extract_tier2_table_values.py

It prints a console summary and also writes:
    outputs/tier2_table_values.json
    outputs/tier2_table_values.txt

Purpose:
- Provide exact baseline values for Tier-2 Table placeholders (RnG, Tb/qb, etc.)
  from BoxParams() defaults.
- Extract the parameter sweep ranges (h, tau, r_a, r_s grids) from the Tier-2
  figure scripts used to generate Figures 8–11 and related plots.
"""

from __future__ import annotations

import ast
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception as e:
    print("ERROR: numpy is required to run this extractor.")
    print(f"Import error: {e}")
    sys.exit(1)


# -----------------------------
# Helpers
# -----------------------------
def _project_root() -> Path:
    """Find project root as current working directory or script directory."""
    cwd = Path.cwd()
    if (cwd / "src").exists() and (cwd / "scripts").exists():
        return cwd
    here = Path(__file__).resolve().parent
    if (here / "src").exists() and (here / "scripts").exists():
        return here
    # fall back to cwd even if structure isn't perfect
    return cwd


def _jsonable(x: Any) -> Any:
    """Convert numpy types, Paths, etc. into JSON-serializable objects."""
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    return x


def _safe_eval_expr(expr_node: ast.AST) -> Any:
    """
    Evaluate a simple numeric/numpy expression AST node safely.
    Only allow np and a tiny set of builtins.
    """
    code = compile(ast.Expression(expr_node), filename="<ast>", mode="eval")
    allowed_globals = {"np": np, "float": float, "int": int}
    return eval(code, {"__builtins__": {}}, allowed_globals)


def _find_script(root: Path, filename: str) -> Optional[Path]:
    """Search scripts/ recursively for a given filename."""
    scripts_dir = root / "scripts"
    if not scripts_dir.exists():
        return None
    matches = list(scripts_dir.rglob(filename))
    return matches[0] if matches else None


def _parse_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _get_function_def(tree: ast.Module, func_name: str) -> Optional[ast.FunctionDef]:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node
    return None


def _extract_assign_in_func(
    func: ast.FunctionDef, target_names: List[str]
) -> Dict[str, ast.AST]:
    """
    Find direct assignments like:
        name = <expr>
    inside a function body.
    """
    found: Dict[str, ast.AST] = {}
    for node in func.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and tgt.id in target_names:
                found[tgt.id] = node.value
    return found


def _extract_boxparams_calls(tree: ast.Module) -> List[ast.Call]:
    """Return all Call nodes that look like BoxParams(...) in a module."""
    calls: List[ast.Call] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> Any:
            # BoxParams(...)
            if isinstance(node.func, ast.Name) and node.func.id == "BoxParams":
                calls.append(node)
            self.generic_visit(node)

    Visitor().visit(tree)
    return calls


def _call_kwargs_numeric(call: ast.Call) -> Dict[str, Any]:
    """
    Extract keyword args from BoxParams(...) calls when they are numeric-ish.
    Skip **kwargs expansions.
    """
    out: Dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            # **something
            continue
        try:
            val = _safe_eval_expr(kw.value)
            # keep scalar numbers only
            if isinstance(val, (int, float, np.integer, np.floating)):
                out[kw.arg] = float(val)
        except Exception:
            continue
    return out


def _array_summary(arr: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"n": int(arr.size), "min": None, "max": None, "values": []}
    return {
        "n": int(arr.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "values": [float(v) for v in arr.tolist()],
    }


# -----------------------------
# Main extraction logic
# -----------------------------
def main() -> None:
    root = _project_root()
    sys.path.insert(0, str(root))

    # Import BoxParams
    try:
        from src.box_model import BoxParams  # type: ignore
    except Exception as e:
        print("ERROR: Could not import BoxParams from src.box_model")
        print(f"Root used: {root}")
        print(f"Import error: {e}")
        sys.exit(1)

    # Baseline defaults
    base = BoxParams()

    if is_dataclass(base):
        base_dict: Dict[str, Any] = asdict(base)
    else:
        # fallback: best-effort attribute dump
        base_dict = {}
        if hasattr(base, "__dict__"):
            base_dict = dict(vars(base))
        else:
            for k in dir(base):
                if k.startswith("_"):
                    continue
                v = getattr(base, k)
                if callable(v):
                    continue
                base_dict[k] = v

    # Try to locate key scripts
    paths = {
        "complementarity_plane": _find_script(root, "make_fig_complementarity_plane.py"),
        "box_sensitivity_panels": _find_script(root, "make_fig_box_sensitivity_panels.py"),
        "tier2_overlay_atlas": _find_script(root, "make_fig_tier2_overlay_atlas.py"),
        "classic_moisture_regimes": _find_script(root, "make_fig_cr_classic_moisture_le_regimes.py"),
    }

    extracted: Dict[str, Any] = {
        "project_root": str(root),
        "boxparams_defaults": base_dict,
        "scripts_found": {k: (str(v) if v else None) for k, v in paths.items()},
        "tier2_ranges_from_scripts": {},
        "tier2_regimes_from_scripts": {},
        "notes": [],
    }

    # ---- Extract h/tau/ra sweep grids and rs_grid from make_fig_box_sensitivity_panels.py
    p = paths["box_sensitivity_panels"]
    if p and p.exists():
        tree = _parse_ast(p)
        func_main = _get_function_def(tree, "main")
        func_rsgrid = _get_function_def(tree, "rs_grid")

        # Grids (inside main)
        if func_main:
            assigns = _extract_assign_in_func(func_main, ["h_vals", "tau_vals", "ra_vals"])
            for name, node in assigns.items():
                try:
                    val = _safe_eval_expr(node)
                    extracted["tier2_ranges_from_scripts"][name] = _array_summary(np.asarray(val))
                except Exception as e:
                    extracted["notes"].append(f"Could not eval {name} in {p.name}: {e}")

        # rs_grid() return expression
        if func_rsgrid:
            # find first Return node
            rs_expr = None
            for node in func_rsgrid.body:
                if isinstance(node, ast.Return):
                    rs_expr = node.value
                    break
            if rs_expr is not None:
                try:
                    # Evaluate default rs_grid() by simulating n=260 (the script default)
                    # If the expression references "n", replace by constant 260 via eval locals.
                    # Easiest: evaluate a small lambda with n provided.
                    code = compile(ast.Expression(rs_expr), "<ast>", "eval")
                    rs = eval(code, {"__builtins__": {}}, {"np": np, "n": 260, "float": float, "int": int})
                    extracted["tier2_ranges_from_scripts"]["rs_grid_default_n260"] = _array_summary(np.asarray(rs))
                except Exception as e:
                    extracted["notes"].append(f"Could not eval rs_grid() return in {p.name}: {e}")
    else:
        extracted["notes"].append("make_fig_box_sensitivity_panels.py not found; cannot auto-extract h/tau/r_a grids.")

    # ---- Extract rs_values and regimes from make_fig_complementarity_plane.py
    p = paths["complementarity_plane"]
    if p and p.exists():
        tree = _parse_ast(p)
        func_main = _get_function_def(tree, "main")
        if func_main:
            assigns = _extract_assign_in_func(func_main, ["rs_values", "regimes"])
            if "rs_values" in assigns:
                try:
                    rs = _safe_eval_expr(assigns["rs_values"])
                    extracted["tier2_ranges_from_scripts"]["rs_values_complementarity_plane"] = _array_summary(np.asarray(rs))
                except Exception as e:
                    extracted["notes"].append(f"Could not eval rs_values in {p.name}: {e}")

            # regimes dict: {label: BoxParams(...), ...}
            if "regimes" in assigns and isinstance(assigns["regimes"], ast.Dict):
                reg_node: ast.Dict = assigns["regimes"]  # type: ignore
                regimes_out: Dict[str, Any] = {}
                for k_node, v_node in zip(reg_node.keys, reg_node.values):
                    if not isinstance(k_node, ast.Constant) or not isinstance(k_node.value, str):
                        continue
                    label = k_node.value
                    if isinstance(v_node, ast.Call) and isinstance(v_node.func, ast.Name) and v_node.func.id == "BoxParams":
                        regimes_out[label] = _call_kwargs_numeric(v_node)
                extracted["tier2_regimes_from_scripts"]["complementarity_plane_regimes"] = regimes_out
    else:
        extracted["notes"].append("make_fig_complementarity_plane.py not found; cannot auto-extract complementarity-plane rs grid/regimes.")

    # ---- Extract rs_values and regimes from make_fig_tier2_overlay_atlas.py
    p = paths["tier2_overlay_atlas"]
    if p and p.exists():
        tree = _parse_ast(p)
        func_main = _get_function_def(tree, "main")
        if func_main:
            assigns = _extract_assign_in_func(func_main, ["rs_values", "regimes"])
            if "rs_values" in assigns:
                try:
                    rs = _safe_eval_expr(assigns["rs_values"])
                    extracted["tier2_ranges_from_scripts"]["rs_values_tier2_overlay"] = _array_summary(np.asarray(rs))
                except Exception as e:
                    extracted["notes"].append(f"Could not eval rs_values in {p.name}: {e}")

            # regimes list of tuples: [("label", BoxParams(...)), ...]
            reg_list = assigns.get("regimes", None)
            regimes_out: List[Dict[str, Any]] = []
            if isinstance(reg_list, ast.List):
                for elt in reg_list.elts:
                    if isinstance(elt, ast.Tuple) and len(elt.elts) == 2:
                        title_node, params_node = elt.elts
                        if isinstance(title_node, ast.Constant) and isinstance(title_node.value, str):
                            label = title_node.value
                        else:
                            continue
                        if isinstance(params_node, ast.Call) and isinstance(params_node.func, ast.Name) and params_node.func.id == "BoxParams":
                            regimes_out.append({"label": label, "kwargs": _call_kwargs_numeric(params_node)})
            if regimes_out:
                extracted["tier2_regimes_from_scripts"]["tier2_overlay_regimes"] = regimes_out
    else:
        extracted["notes"].append("make_fig_tier2_overlay_atlas.py not found; cannot auto-extract overlay rs grid/regimes.")

    # ---- Extract rs and regimes from make_fig_cr_classic_moisture_le_regimes.py
    p = paths["classic_moisture_regimes"]
    if p and p.exists():
        tree = _parse_ast(p)
        func_main = _get_function_def(tree, "main")
        if func_main:
            assigns = _extract_assign_in_func(func_main, ["rs", "regimes"])
            if "rs" in assigns:
                try:
                    rs = _safe_eval_expr(assigns["rs"])
                    extracted["tier2_ranges_from_scripts"]["rs_values_classic_moisture"] = _array_summary(np.asarray(rs))
                except Exception as e:
                    extracted["notes"].append(f"Could not eval rs in {p.name}: {e}")

            # regimes list: [("Strong adjustment", BoxParams(...)), ...]
            reg_list = assigns.get("regimes", None)
            regimes_out: List[Dict[str, Any]] = []
            if isinstance(reg_list, ast.List):
                for elt in reg_list.elts:
                    if isinstance(elt, ast.Tuple) and len(elt.elts) == 2:
                        title_node, params_node = elt.elts
                        if isinstance(title_node, ast.Constant) and isinstance(title_node.value, str):
                            label = title_node.value
                        else:
                            continue
                        if isinstance(params_node, ast.Call) and isinstance(params_node.func, ast.Name) and params_node.func.id == "BoxParams":
                            regimes_out.append({"label": label, "kwargs": _call_kwargs_numeric(params_node)})
            if regimes_out:
                extracted["tier2_regimes_from_scripts"]["classic_moisture_regimes"] = regimes_out
    else:
        extracted["notes"].append("make_fig_cr_classic_moisture_le_regimes.py not found; cannot auto-extract classic rs/regimes.")

    # ---- Also scan ALL scripts for any explicit BoxParams keyword overrides (captures e.g. alpha_PT or RnG sweeps if present)
    scripts_dir = root / "scripts"
    overrides: Dict[str, List[float]] = {}
    if scripts_dir.exists():
        for py in scripts_dir.rglob("*.py"):
            try:
                tree = _parse_ast(py)
            except Exception:
                continue
            calls = _extract_boxparams_calls(tree)
            for call in calls:
                kw = _call_kwargs_numeric(call)
                for k, v in kw.items():
                    overrides.setdefault(k, []).append(float(v))
    if overrides:
        extracted["boxparams_explicit_overrides_found_in_scripts"] = {
            k: {"min": float(np.min(v)), "max": float(np.max(v)), "values": sorted(set(v))}
            for k, v in overrides.items()
            if len(v) > 0
        }

    # ---- Suggest a single r_s max for table based on all discovered r_s arrays
    rs_candidates = []
    for k, v in extracted["tier2_ranges_from_scripts"].items():
        if isinstance(v, dict) and k.startswith("rs"):
            mx = v.get("max", None)
            if mx is not None:
                rs_candidates.append(float(mx))
    extracted["suggested_rs_max_for_table"] = float(max(rs_candidates)) if rs_candidates else None

    # Output paths
    outdir = root / "outputs"
    outdir.mkdir(exist_ok=True)
    json_path = outdir / "tier2_table_values.json"
    txt_path = outdir / "tier2_table_values.txt"

    # Write JSON
    json_path.write_text(json.dumps(_jsonable(extracted), indent=2), encoding="utf-8")

    # Write a human-readable TXT
    lines: List[str] = []
    lines.append(f"Project root: {root}")
    lines.append("")
    lines.append("=== BoxParams() defaults (baseline) ===")
    for k in sorted(base_dict.keys()):
        lines.append(f"{k}: {base_dict[k]}")
    lines.append("")
    lines.append("=== Key scripts found ===")
    for k, v in extracted["scripts_found"].items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("=== Tier-2 ranges extracted from scripts ===")
    for k, v in extracted["tier2_ranges_from_scripts"].items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("=== Tier-2 regimes extracted from scripts ===")
    for k, v in extracted["tier2_regimes_from_scripts"].items():
        lines.append(f"{k}: {v}")
    lines.append("")
    if "boxparams_explicit_overrides_found_in_scripts" in extracted:
        lines.append("=== Explicit BoxParams keyword overrides found anywhere in scripts/ ===")
        for k, v in extracted["boxparams_explicit_overrides_found_in_scripts"].items():
            lines.append(f"{k}: {v}")
        lines.append("")
    lines.append(f"Suggested single r_s,max for the table: {extracted['suggested_rs_max_for_table']}")
    lines.append("")
    if extracted["notes"]:
        lines.append("=== Notes / warnings ===")
        for n in extracted["notes"]:
            lines.append(f"- {n}")
        lines.append("")

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    # Console summary
    print("\n".join(lines))
    print(f"\nWrote: {json_path}")
    print(f"Wrote: {txt_path}")


if __name__ == "__main__":
    main()
