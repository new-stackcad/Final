"""
solver.py  (generic)
====================
Integrates any user-defined ODE system.

Supports:
  - Any number of states (auto-detected from ode_rhs)
  - Optional holonomic constraints with Baumgarte stabilisation
  - Static equilibrium via dynamic relaxation fallback
  - Road signal inputs (from CSV axle data OR empty dict for non-vehicle problems)

Public API
----------
run_one_case(params, model, road, t_eval, log_cb) -> pd.DataFrame
compute_cabin_rms(df, model)                       -> Dict[str, float]
estimate_single_eval_time(model, road, t_sample)   -> float
"""

from __future__ import annotations
import time
import traceback
import numpy as np
import pandas as pd
from numpy.linalg import solve as lin_solve, lstsq
from scipy.integrate import solve_ivp
from typing import Dict, Optional, Callable, List

from user_code_runner import UserModel


# ---------------------------------------------------------------------------
# Road signals helper (generalised — works with any number of axle CSVs)
# ---------------------------------------------------------------------------
def build_road_signals(cfg: Dict) -> Dict:
    """
    Load axle CSV files if present in cfg.
    Returns a dict of interpolant callables keyed by csv key.
    If no CSV keys present (non-vehicle problem), returns empty dict.
    """
    import pandas as pd

    def _load(path: str):
        df   = pd.read_csv(path, skiprows=2, header=None)
        t    = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
        z    = pd.to_numeric(df.iloc[:, 1], errors="coerce").values
        mask = np.isfinite(t) & np.isfinite(z)
        t, z = t[mask].astype(float), z[mask].astype(float)

        def interp(xq):
            xq   = np.asarray(xq, dtype=float)
            xq_c = np.clip(xq, t[0], t[-1])
            idx  = np.clip(np.searchsorted(t, xq_c) - 1, 0, len(t) - 2)
            x0, x1 = t[idx], t[idx + 1]
            y0, y1 = z[idx], z[idx + 1]
            w = (xq_c - x0) / np.maximum(x1 - x0, 1e-12)
            return float(y0 * (1 - w) + y1 * w)
        return interp

    csv_keys = {
        "axlefront_left_csv":  ("f1L", "FA_LH"),
        "axlefront_right_csv": ("f1R", "FA_RH"),
        "axlerear1_left_csv":  ("f2L", "RA1_LH"),
        "axlerear1_right_csv": ("f2R", "RA1_RH"),
        "axlerear2_left_csv":  ("f3L", "RA2_LH"),
        "axlerear2_right_csv": ("f3R", "RA2_RH"),
    }

    signals = {}
    for cfg_key, (sig_key, _) in csv_keys.items():
        path = cfg.get(cfg_key, "")
        if path:
            try:
                signals[sig_key] = _load(path)
            except Exception as e:
                raise RuntimeError(f"Failed to load {cfg_key}: {e}")
    return signals


def _build_inputs(t: float, signals: Dict, cfg: Dict, dt: float = 0.001) -> Dict:
    """
    Build the inputs dict passed into user's ode_rhs at time t.
    Includes axle displacements, roll angles, and their rates.
    """
    if not signals:
        return {}  # non-vehicle problem — user doesn't need inputs

    WT1 = cfg.get("WT1", 1.0)
    WT2 = cfg.get("WT2", 1.0)
    WT3 = cfg.get("WT3", 1.0)

    def _get(key, default=0.0):
        f = signals.get(key)
        return float(f(t)) if f is not None else default

    zr1L, zr1R = _get("f1L"), _get("f1R")
    zr2L, zr2R = _get("f2L"), _get("f2R")
    zr3L, zr3R = _get("f3L"), _get("f3R")

    inp = {
        "z1f":  0.5 * (zr1L + zr1R),
        "ph_f": (zr1L - zr1R) / WT1,
        "z2":   0.5 * (zr2L + zr2R),
        "ph2":  (zr2L - zr2R) / WT2,
        "z3":   0.5 * (zr3L + zr3R),
        "ph3":  (zr3L - zr3R) / WT3,
        # raw signals too
        "zr1L": zr1L, "zr1R": zr1R,
        "zr2L": zr2L, "zr2R": zr2R,
        "zr3L": zr3L, "zr3R": zr3R,
    }

    # Numerical rates (central difference)
    def _get_r(key, default=0.0):
        f = signals.get(key)
        if f is None:
            return default
        return (float(f(t + dt)) - float(f(t - dt))) / (2 * dt)

    zr1L_d, zr1R_d = _get_r("f1L"), _get_r("f1R")
    zr2L_d, zr2R_d = _get_r("f2L"), _get_r("f2R")
    zr3L_d, zr3R_d = _get_r("f3L"), _get_r("f3R")

    inp.update({
        "dz1f":  0.5 * (zr1L_d + zr1R_d),
        "dph_f": (zr1L_d - zr1R_d) / WT1,
        "dz2":   0.5 * (zr2L_d + zr2R_d),
        "dph2":  (zr2L_d - zr2R_d) / WT2,
        "dz3":   0.5 * (zr3L_d + zr3R_d),
        "dph3":  (zr3L_d - zr3R_d) / WT3,
    })
    return inp


# ---------------------------------------------------------------------------
# Generic ODE RHS dispatcher
# ---------------------------------------------------------------------------
def _make_rhs(model: UserModel, params: Dict, signals: Dict):
    """
    Returns a scipy-compatible rhs(t, x) -> xdot that:
      1. Calls user's ode_rhs(t, x, params, inputs)
      2. Optionally enforces holonomic constraints via Baumgarte

    For unconstrained: returns xdot directly.
    For constrained:   assumes first half of x = positions, second half = velocities
                       (standard first-order form for constrained systems).
    """
    cfg        = {**model.cfg, **params}
    n          = model.n_states
    has_constr = model.geom_enabled and model.constraints is not None
    dt         = cfg.get("DT", 0.001)

    if not has_constr:
        # ---- Unconstrained: call user function directly ----
        def rhs(t, x):
            inp  = _build_inputs(t, signals, cfg, dt)
            xdot = model.ode_rhs(t, np.asarray(x), params, inp)
            return np.asarray(xdot, dtype=float)
        return rhs

    else:
        # ---- Constrained: Baumgarte stabilised DAE ----
        # Convention: x[:n//2] = generalised positions q
        #             x[n//2:] = generalised velocities v
        # User's ode_rhs returns [qdot, M^{-1}(F - C^T lam)] but we override
        # with the constraint-corrected acceleration.
        # User's constraints(t, q, params, inputs) -> g (residual vector)
        n2   = n // 2
        w    = float(cfg.get("baum_omega", 10.0))
        zeta = float(cfg.get("baum_zeta",  1.0))

        def rhs(t, x):
            q   = x[:n2]
            v   = x[n2:]
            inp = _build_inputs(t, signals, cfg, dt)

            # Get unconstrained acceleration from user ODE
            xdot_free = np.asarray(
                model.ode_rhs(t, x, params, inp), dtype=float
            )
            qdot_free = xdot_free[:n2]
            vdot_free = xdot_free[n2:]

            # Get constraint residual and Jacobian (numerical)
            try:
                g0 = np.asarray(
                    model.constraints(t, q, params, inp), dtype=float
                )
            except Exception:
                return xdot_free  # constraint eval failed — fall back

            nc = len(g0)
            if nc == 0:
                return xdot_free

            # Numerical Jacobian G = dg/dq
            eps = 1e-6
            G   = np.zeros((nc, n2))
            for j in range(n2):
                qp      = q.copy(); qp[j] += eps
                qm      = q.copy(); qm[j] -= eps
                gp      = np.asarray(model.constraints(t, qp, params, inp), dtype=float)
                gm      = np.asarray(model.constraints(t, qm, params, inp), dtype=float)
                G[:, j] = (gp - gm) / (2 * eps)

            # Baumgarte correction  gamma = G*vdot + w^2*g + 2*zeta*w*(G*v)
            gdot = (G @ v)
            gamma = w**2 * g0 + 2 * zeta * w * gdot

            # Augmented system [M G^T; G 0][vdot; lam] = [F; -gamma]
            # We approximate M as identity (user's vdot_free already is M^-1 F)
            # Correction: vdot = vdot_free - G^T (G G^T)^{-1} (G vdot_free + gamma)
            GGT = G @ G.T
            try:
                lam      = np.linalg.solve(GGT + 1e-12 * np.eye(nc),
                                           G @ vdot_free + gamma)
                vdot_cor = vdot_free - G.T @ lam
            except np.linalg.LinAlgError:
                vdot_cor = vdot_free

            return np.concatenate([v, vdot_cor])

        return rhs


# ---------------------------------------------------------------------------
# Static equilibrium  (generic)
# ---------------------------------------------------------------------------
def _static_equilibrium(model: UserModel, params: Dict, signals: Dict) -> np.ndarray:
    """
    Find static equilibrium state x0 by dynamic relaxation:
    integrate a heavily damped version of the system for a short time.
    Falls back to zeros if it fails.
    """
    cfg = {**model.cfg, **params}

    # Try with heavily scaled damping if user has damping params
    damped_params = dict(params)
    for k in list(params.keys()):
        if k.startswith("C_") or "damp" in k.lower() or "damping" in k.lower():
            damped_params[k] = params[k] * 20.0

    rhs = _make_rhs(model, damped_params, signals)

    try:
        sol = solve_ivp(
            rhs, (0.0, 2.0), np.zeros(model.n_states),
            method="Radau", rtol=1e-5, atol=1e-7, max_step=0.05,
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass

    return np.zeros(model.n_states)


# ---------------------------------------------------------------------------
# Time estimation
# ---------------------------------------------------------------------------
def estimate_single_eval_time(
    model:    UserModel,
    signals:  Dict,
    t_sample: float = 5.0,
) -> float:
    """
    Run a short integration (t_sample seconds) and extrapolate to T_END.
    Returns estimated seconds per full ODE evaluation.
    """
    cfg    = model.cfg
    T_END  = float(cfg.get("T_END", 466.945))
    dt     = float(cfg.get("DT", 0.001))
    params = {k: float(v[0] + v[1]) / 2 for k, v in model.bounds.items()
              if isinstance(v, (list, tuple)) and len(v) == 2}

    rhs    = _make_rhs(model, params, signals)
    x0     = np.zeros(model.n_states)
    t_eval = np.arange(0.0, min(t_sample, T_END) + dt, dt)

    wall0  = time.time()
    try:
        solve_ivp(rhs, (0.0, t_eval[-1]), x0,
                  method="Radau", t_eval=t_eval,
                  rtol=1e-6, atol=1e-8, max_step=0.02)
    except Exception:
        pass
    wall1  = time.time()

    sample_time = wall1 - wall0
    return sample_time * (T_END / t_sample)


# ---------------------------------------------------------------------------
# Main simulation runner
# ---------------------------------------------------------------------------
def run_one_case(
    params:   Dict,
    model:    UserModel,
    signals:  Dict,
    t_eval:   np.ndarray,
    log_cb:   Optional[Callable] = None,
) -> pd.DataFrame:
    """
    Run one ODE integration for a given parameter set.

    Parameters
    ----------
    params   : optimisation parameters (merged into model.cfg)
    model    : parsed UserModel from user_code_runner
    signals  : road signal interpolants from build_road_signals
    t_eval   : time evaluation array
    log_cb   : optional log streaming callback

    Returns
    -------
    DataFrame with columns: t, x0..xN (states), xd0..xdN (velocities),
                            xdd0..xddN (accelerations),
                            cabin_az, cabin_ax, cabin_ay (seat accelerations)
    """
    def _log(msg):
        if log_cb:
            log_cb(msg)

    cfg    = {**model.cfg, **params}
    n      = model.n_states
    rhs    = _make_rhs(model, params, signals)

    _log("Computing static equilibrium …")
    x0   = _static_equilibrium(model, params, signals)
    _log(f"x0 = {np.round(x0, 5).tolist()}")

    t0, tf = float(t_eval[0]), float(t_eval[-1])
    _log(f"Integrating ({n} states, Radau) T=[{t0:.2f}, {tf:.2f}] s")
    wall = time.time()

    sol = solve_ivp(
        rhs,
        (t0, tf),
        x0,
        method="Radau",
        t_eval=t_eval,
        max_step=0.01,
        rtol=1e-6,
        atol=1e-8,
    )

    elapsed = time.time() - wall
    _log(f"Done — success={sol.success}  nfev={sol.nfev}  wall={elapsed:.1f}s")

    if not sol.success:
        raise RuntimeError(f"ODE failed: {sol.message}")
    if not np.all(np.isfinite(sol.y)):
        raise RuntimeError("ODE returned non-finite values — check your equations.")

    # ---- Build output DataFrame ----
    snames = model.state_names
    rows   = []

    # Pre-compute RHS at each time step for accelerations
    for i, t in enumerate(sol.t):
        x    = sol.y[:, i]
        try:
            xdot = rhs(t, x)
        except Exception:
            xdot = np.zeros(n)

        row = {"t": t}
        for j in range(n):
            nm            = snames[j] if j < len(snames) else f"x{j}"
            row[nm]       = float(x[j])
            row[f"d{nm}"] = float(xdot[j])

        # Cabin seat accelerations — from OUTPUTS mapping
        out     = model.outputs
        hcp     = float(out.get("hcp", 0.1))
        iz_idx  = out.get("cabin_z_accel_idx", None)
        ith_idx = out.get("cabin_th_accel_idx", None)
        iph_idx = out.get("cabin_ph_accel_idx", None)

        # For constrained system the acceleration is in xdot[n//2:]
        if model.geom_enabled and n % 2 == 0:
            acc = xdot[n // 2:]
        else:
            acc = xdot

        row["cabin_az"] = float(acc[iz_idx])          if iz_idx  is not None else 0.0
        row["cabin_ax"] = float(-hcp * acc[ith_idx])  if ith_idx is not None else 0.0
        row["cabin_ay"] = float( hcp * acc[iph_idx])  if iph_idx is not None else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Objective / RMS  (always cabin z/x/y — fixed as discussed)
# ---------------------------------------------------------------------------
def compute_cabin_rms(df: pd.DataFrame, model: UserModel) -> Dict[str, float]:
    """
    Compute RMS of cabin seat accelerations.
    Always: RMS_z (vertical), RMS_x (longitudinal/pitch), RMS_y (lateral/roll).
    These are always the objectives — fixed regardless of ODE.
    """
    t_ignore = float(model.cfg.get("T_IGNORE", 0.5))
    mask     = df["t"] >= t_ignore

    az = df.loc[mask, "cabin_az"].values
    ax = df.loc[mask, "cabin_ax"].values
    ay = df.loc[mask, "cabin_ay"].values

    return {
        "rms_z":     float(np.sqrt(np.mean(az ** 2))),
        "rms_x":     float(np.sqrt(np.mean(ax ** 2))),
        "rms_y":     float(np.sqrt(np.mean(ay ** 2))),
        "rms_total": float(np.sqrt(np.mean(az**2) + np.mean(ax**2) + np.mean(ay**2))),
    }
