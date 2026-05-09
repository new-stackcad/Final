"""
user_code_runner.py
===================
Executes user-supplied Python code strings and extracts:
    - CONFIG  dict
    - BOUNDS  dict
    - OUTPUTS dict  (cabin state index mapping)
    - ode_rhs callable
    - constraints callable (optional)

All execution is local (not a public server), so exec() is appropriate here.

Public API
----------
parse_user_code(ode_code, geom_code=None) -> UserModel
validate_user_model(model)                -> List[str]  (empty = OK)
"""

from __future__ import annotations
import traceback
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class UserModel:
    cfg:         Dict                        # CONFIG from editor
    bounds:      Dict                        # BOUNDS from editor
    outputs:     Dict                        # OUTPUTS from editor
    ode_rhs:     Callable                    # user's ode_rhs function
    constraints: Optional[Callable] = None  # user's constraints function (or None)
    n_states:    int = 0                     # auto-detected
    state_names: List[str] = field(default_factory=list)
    geom_enabled: bool = False


# ---------------------------------------------------------------------------
# Safe namespace for exec
# ---------------------------------------------------------------------------
def _make_namespace() -> Dict:
    """Provide a safe set of allowed names for user code."""
    import math
    return {
        "__builtins__": {
            "abs": abs, "min": min, "max": max, "len": len,
            "range": range, "enumerate": enumerate, "zip": zip,
            "list": list, "dict": dict, "tuple": tuple, "float": float,
            "int": int, "bool": bool, "str": str, "print": print,
            "isinstance": isinstance, "hasattr": hasattr,
        },
        "np":    np,
        "numpy": np,
        "math":  math,
    }


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------
def parse_user_code(
    ode_code:  str,
    geom_code: Optional[str] = None,
) -> UserModel:
    """
    Execute user code strings and return a UserModel.

    Parameters
    ----------
    ode_code  : full content of ODE + Config panel (CONFIG, BOUNDS, OUTPUTS, ode_rhs)
    geom_code : content of Geometry panel (constraints function) or None
    """
    ns = _make_namespace()

    # ---- Execute ODE panel code ----
    try:
        exec(compile(ode_code, "<ode_editor>", "exec"), ns)
    except Exception as e:
        raise ValueError(f"ODE code execution error:\n{traceback.format_exc()}") from e

    # ---- Extract CONFIG ----
    cfg = ns.get("CONFIG")
    if cfg is None:
        raise ValueError("CONFIG dict not found. Define CONFIG = {...} in the ODE panel.")
    if not isinstance(cfg, dict):
        raise ValueError("CONFIG must be a Python dict.")

    # ---- Extract BOUNDS ----
    bounds = ns.get("BOUNDS")
    if bounds is None:
        raise ValueError("BOUNDS dict not found. Define BOUNDS = {...} in the ODE panel.")
    if not isinstance(bounds, dict):
        raise ValueError("BOUNDS must be a Python dict.")

    # ---- Extract OUTPUTS ----
    outputs = ns.get("OUTPUTS")
    if outputs is None:
        raise ValueError(
            "OUTPUTS dict not found. Define OUTPUTS = {...} mapping cabin acceleration "
            "state indices. See template."
        )

    # ---- Extract ode_rhs ----
    ode_rhs = ns.get("ode_rhs")
    if ode_rhs is None:
        raise ValueError("ode_rhs function not found. Define def ode_rhs(t, x, params, inputs): ...")
    if not callable(ode_rhs):
        raise ValueError("ode_rhs must be a callable function.")

    # ---- Auto-detect n_states by probing the function ----
    n_states = _detect_n_states(ode_rhs, cfg)

    # ---- Extract state_names if provided ----
    state_names = ns.get("STATE_NAMES", [f"x{i}" for i in range(n_states)])

    # ---- Execute geometry panel code (optional) ----
    constraints = None
    geom_enabled = False
    if geom_code and geom_code.strip():
        geom_ns = _make_namespace()
        geom_ns.update({k: v for k, v in ns.items() if not k.startswith("__")})
        try:
            exec(compile(geom_code, "<geom_editor>", "exec"), geom_ns)
        except Exception as e:
            raise ValueError(f"Geometry code execution error:\n{traceback.format_exc()}") from e

        constraints = geom_ns.get("constraints")
        if constraints is not None:
            if not callable(constraints):
                raise ValueError("constraints must be a callable function.")
            geom_enabled = True

    return UserModel(
        cfg=dict(cfg),
        bounds=dict(bounds),
        outputs=dict(outputs),
        ode_rhs=ode_rhs,
        constraints=constraints,
        n_states=n_states,
        state_names=list(state_names),
        geom_enabled=geom_enabled,
    )


def _detect_n_states(ode_rhs: Callable, cfg: Dict) -> int:
    """
    Call ode_rhs with a zero vector of increasing size until it returns
    without error. Detects the user's state dimension automatically.
    """
    dummy_inputs = {
        "z1f": 0.0, "ph_f": 0.0,
        "z2":  0.0, "ph2":  0.0,
        "z3":  0.0, "ph3":  0.0,
        "dz1f": 0.0, "dph_f": 0.0,
        "dz2":  0.0, "dph2":  0.0,
        "dz3":  0.0, "dph3":  0.0,
    }
    for n in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]:
        try:
            x0  = np.zeros(n)
            res = ode_rhs(0.0, x0, cfg, dummy_inputs)
            if hasattr(res, "__len__") and len(res) == n:
                return n
        except Exception:
            continue
    raise ValueError(
        "Could not detect state dimension from ode_rhs. "
        "Make sure it returns a list/array of the same length as x."
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_user_model(model: UserModel) -> List[str]:
    """
    Run sanity checks on a parsed UserModel.
    Returns a list of violation strings. Empty list = all OK.
    """
    violations: List[str] = []

    def _check(cond: bool, msg: str):
        if not cond:
            violations.append(msg)

    outputs = model.outputs

    # ---- OUTPUTS checks ----
    for key in ("cabin_z_accel_idx", "cabin_th_accel_idx", "cabin_ph_accel_idx"):
        val = outputs.get(key)
        _check(val is not None, f"OUTPUTS missing '{key}'")
        if val is not None:
            _check(
                isinstance(val, int) and 0 <= val < model.n_states,
                f"OUTPUTS['{key}']={val} out of range for {model.n_states} states"
            )

    _check("hcp" in outputs, "OUTPUTS missing 'hcp' (seat height offset [m])")

    # ---- BOUNDS checks ----
    for k, v in model.bounds.items():
        _check(
            isinstance(v, (list, tuple)) and len(v) == 2,
            f"BOUNDS['{k}'] must be [lo, hi]"
        )
        if isinstance(v, (list, tuple)) and len(v) == 2:
            _check(float(v[0]) < float(v[1]), f"BOUNDS['{k}']: lo must be < hi")
        _check(k in model.cfg, f"BOUNDS key '{k}' not found in CONFIG")

    # ---- n_states ----
    _check(model.n_states >= 2, f"State dimension {model.n_states} too small")

    # ---- geometry ----
    if model.geom_enabled:
        _check(
            "baum_omega" in model.cfg and "baum_zeta" in model.cfg,
            "Geometry constraints enabled but baum_omega / baum_zeta missing from CONFIG"
        )

    return violations
