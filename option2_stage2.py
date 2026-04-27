"""
option2_stage2.py  -  Stage 2 of Option 2.

Given the optimal Kf*, Cf*, K2*, K3* from Stage 1, derives a physically
consistent asymmetric two-stage damper characteristic using the following
industry approach:

1.  Critical damping at each axle:
      Cc_f  = 2 * sqrt(K_f  * m_s * lf_fraction)
      Cc_r1 = 2 * sqrt(K_2  * m_s * l2_fraction)
      Cc_r2 = 2 * sqrt(K_3  * m_s * l3_fraction)

2.  Target ride damping ratio zeta_ride (default 0.35 for comfort):
      cs_target = zeta_ride * Cc  (compression low-speed slope)
      cp_target = asym_ratio * cs_target  (rebound low-speed slope)

3.  Target F-v curve points are generated from the ideal linear damper
    and then the two-stage breakpoints (alpha_c, alpha_r) and high-speed
    slopes (gamma_c, gamma_r) are fitted using scipy least_squares.

4.  The fitted characteristic is verified against the target and the
    RMS is re-evaluated with the full ODE using the complete parameter set.

Usage
-----
# After running Stage 1:
python src/option2_stage2.py --stage1_result checkpoints/opt2/stage1_result.json

# With ODE verification:
python src/option2_stage2.py --stage1_result checkpoints/opt2/stage1_result.json --verify_ode
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))

from config import BASE_CFG, DT
from physics import (
    build_road_signals, compute_seat_rms,
    run_one_case, static_equilibrium,
    TwoStageAsymmetricDamper,
)


# ─────────────────────────────────────────────────────────────
# Step 1: compute critical damping from optimal stiffness
# ─────────────────────────────────────────────────────────────

def critical_damping(K_f: float, K_2: float, K_3: float,
                     cfg: Dict) -> Tuple[float, float, float]:
    """
    Compute critical damping coefficient for each suspension point.

    For a 1-DOF quarter-car:  Cc = 2 * sqrt(K * m_eff)
    where m_eff is the sprung mass contribution at each axle,
    estimated from the pitch-plane geometry.
    """
    m_s  = cfg["m_s"]
    lf   = cfg["lf"]
    l2   = cfg["L12"]
    l3   = cfg["L12"] + cfg["L23"]
    wb   = lf + l3            # approximate wheelbase front-to-rear3

    # Effective sprung mass fraction at each axle (from static load sharing)
    # Front axle carries (l3/wb) of sprung mass, rear axles share the rest
    frac_f  = l3 / wb
    frac_r1 = (lf / wb) * (l3 / (l2 + l3))   # rear1 fraction of rear load
    frac_r2 = (lf / wb) * (l2 / (l2 + l3))   # rear2 fraction of rear load

    m_f  = m_s * frac_f
    m_r1 = m_s * frac_r1
    m_r2 = m_s * frac_r2

    Cc_f  = 2.0 * np.sqrt(K_f  * m_f)
    Cc_r1 = 2.0 * np.sqrt(K_2  * m_r1)
    Cc_r2 = 2.0 * np.sqrt(K_3  * m_r2)

    print(f"  Critical damping:  Cc_f={Cc_f:.1f}  Cc_r1={Cc_r1:.1f}  Cc_r2={Cc_r2:.1f}  N·s/m")
    return Cc_f, Cc_r1, Cc_r2


# ─────────────────────────────────────────────────────────────
# Step 2: target damping coefficients
# ─────────────────────────────────────────────────────────────

def target_damping(Cc_f: float, Cc_r1: float, Cc_r2: float,
                   zeta_ride: float = 0.35,
                   asym_ratio: float = 2.5) -> Dict:
    """
    Compute target cs (compression) and cp (rebound) from ride damping ratio.

    zeta_ride = 0.30–0.40 is the industry comfort-optimised range.
    asym_ratio = cp/cs, typically 2.0–3.5 for commercial vehicles.

    Returns weighted average for the full-vehicle equivalent damper.
    """
    # Use the front axle as the primary reference (cabin is coupled most strongly)
    cs_f  = zeta_ride * Cc_f
    cs_r1 = zeta_ride * Cc_r1
    cs_r2 = zeta_ride * Cc_r2

    # Weighted average cs (weight by stiffness contribution)
    # Front carries more of cabin response so weight it higher
    cs_eq = (2.0*cs_f + cs_r1 + cs_r2) / 4.0
    cp_eq = asym_ratio * cs_eq

    print(f"  Target damping (zeta={zeta_ride}):  "
          f"cs_f={cs_f:.1f}  cs_r1={cs_r1:.1f}  cs_r2={cs_r2:.1f}  N·s/m")
    print(f"  Equivalent:  cs={cs_eq:.1f}  cp={cp_eq:.1f}  "
          f"(asym_ratio={asym_ratio})")

    return {
        "cs_f": cs_f, "cs_r1": cs_r1, "cs_r2": cs_r2,
        "cs_eq": cs_eq, "cp_eq": cp_eq,
    }


# ─────────────────────────────────────────────────────────────
# Step 3: generate target F-v curve and fit two-stage shape
# ─────────────────────────────────────────────────────────────

def _fv_model(v: np.ndarray, params: np.ndarray,
              cs: float, cp: float) -> np.ndarray:
    """
    Two-stage asymmetric damper model for curve fitting.
    params = [alpha_c, alpha_r, gamma_c, gamma_r]
    """
    alpha_c, alpha_r, gamma_c, gamma_r = params
    F = np.zeros_like(v)
    mask_c_ls = (v < 0)  & (v >= alpha_c)
    mask_c_hs = (v < 0)  & (v < alpha_c)
    mask_r_ls = (v >= 0) & (v <= alpha_r)
    mask_r_hs = (v >= 0) & (v > alpha_r)
    F[mask_c_ls] = cs * v[mask_c_ls]
    F[mask_c_hs] = cs * (alpha_c + gamma_c * (v[mask_c_hs] - alpha_c))
    F[mask_r_ls] = cp * v[mask_r_ls]
    F[mask_r_hs] = cp * (alpha_r + gamma_r * (v[mask_r_hs] - alpha_r))
    return F


def _target_fv(v: np.ndarray, cs: float, cp: float,
               target_alpha_c: float = -0.05,
               target_alpha_r: float =  0.13) -> np.ndarray:
    """
    Target F-v characteristic: linear in low-speed, reduced slope high-speed.
    The high-speed slope reduction factor (0.5–0.7) balances ride vs handling.
    """
    gamma_c_target = 0.5   # high-speed compression: 50% of low-speed slope
    gamma_r_target = 0.6   # high-speed rebound:     60% of low-speed slope
    return _fv_model(v,
                     np.array([target_alpha_c, target_alpha_r,
                                gamma_c_target, gamma_r_target]),
                     cs, cp)


def fit_damper_curve(cs: float, cp: float,
                     v_range: Tuple[float, float] = (-0.5, 0.5),
                     n_points: int = 200) -> Dict:
    """
    Generate target F-v points and fit the two-stage shape parameters.

    Returns dict with cs_minus, asym_ratio, gamma_c, gamma_r,
    alpha_c, alpha_r and fit quality metrics.
    """
    v_fit = np.linspace(v_range[0], v_range[1], n_points)
    F_target = _target_fv(v_fit, cs, cp)

    # Initial guess: nominal breakpoints, gamma=0.5
    x0     = np.array([-0.05, 0.13, 0.50, 0.60])
    bounds = (
        [-0.15,  0.05, 0.10, 0.10],   # lower bounds
        [-0.01,  0.30, 0.90, 0.90],   # upper bounds
    )

    def residuals(params):
        return _fv_model(v_fit, params, cs, cp) - F_target

    result = least_squares(residuals, x0, bounds=bounds,
                            method="trf", ftol=1e-10, xtol=1e-10)

    alpha_c, alpha_r, gamma_c, gamma_r = result.x
    F_fitted = _fv_model(v_fit, result.x, cs, cp)

    # Fit quality
    rmse   = float(np.sqrt(np.mean((F_fitted - F_target)**2)))
    r2     = float(1 - np.sum((F_fitted - F_target)**2)
                   / np.sum((F_target - F_target.mean())**2))

    print(f"\n  Damper curve fit:")
    print(f"    cs_minus   = {cs:.4f}  N·s/m")
    print(f"    asym_ratio = {cp/cs:.4f}")
    print(f"    alpha_c    = {alpha_c:.4f}  m/s")
    print(f"    alpha_r    = {alpha_r:.4f}  m/s")
    print(f"    gamma_c    = {gamma_c:.4f}")
    print(f"    gamma_r    = {gamma_r:.4f}")
    print(f"    Fit RMSE   = {rmse:.2f} N   R²={r2:.5f}")

    return {
        "cs_minus":   float(cs),
        "asym_ratio": float(cp / cs),
        "alpha_c":    float(alpha_c),
        "alpha_r":    float(alpha_r),
        "gamma_c":    float(gamma_c),
        "gamma_r":    float(gamma_r),
        "fit_rmse_N": rmse,
        "fit_r2":     r2,
        "_v_fit":     v_fit.tolist(),
        "_F_target":  F_target.tolist(),
        "_F_fitted":  F_fitted.tolist(),
    }


# ─────────────────────────────────────────────────────────────
# Step 4: plot F-v curve
# ─────────────────────────────────────────────────────────────

def plot_fv_curve(damper_params: Dict, out_path: str) -> None:
    v        = np.array(damper_params["_v_fit"])
    F_target = np.array(damper_params["_F_target"])
    F_fitted = np.array(damper_params["_F_fitted"])

    # Also plot nominal damper for reference
    cs_nom  = BASE_CFG["cs_minus"]
    cp_nom  = BASE_CFG["asym_ratio"] * cs_nom
    damper_nom = TwoStageAsymmetricDamper(
        cs_minus=cs_nom, asym_ratio=BASE_CFG["asym_ratio"],
        gamma_c=BASE_CFG["gamma_c"], gamma_r=BASE_CFG["gamma_r"],
    )
    F_nom = np.array([damper_nom.force(vi) * BASE_CFG["C_f"] for vi in v])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(v, F_target, "k--", lw=1.5, label="Target (derived from K*)")
    ax.plot(v, F_fitted, "r-",  lw=2.0,
            label=f"Fitted  (RMSE={damper_params['fit_rmse_N']:.1f} N, "
                  f"R²={damper_params['fit_r2']:.4f})")
    ax.plot(v, F_nom, "b:",  lw=1.2, label="Nominal damper")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.axvline(damper_params["alpha_c"], color="r", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(damper_params["alpha_r"], color="r", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Relative velocity [m/s]")
    ax.set_ylabel("Damper force [N]")
    ax.set_title("Option 2: fitted asymmetric damper F-v characteristic")
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  F-v plot saved → {out_path}")


# ─────────────────────────────────────────────────────────────
# Step 5: ODE verification
# ─────────────────────────────────────────────────────────────

def verify_ode(stage1_params: Dict, damper_params: Dict,
               cfg_base: Dict, downsample: int = 4) -> float:
    """Full ODE solve with optimal K* + fitted damper."""
    params = {**stage1_params, **damper_params}
    # Remove internal fitting keys
    params = {k: v for k, v in params.items()
              if not k.startswith("_") and
              k not in ("fit_rmse_N", "fit_r2")}

    t_eval = np.arange(0.0, float(cfg_base["sim_duration_s"]) + DT, DT)[::downsample]
    print("\n  ODE verification with optimal K* + fitted damper ...")
    df = run_one_case(params, cfg_base, t_eval, verbose=True)
    if df is None or df.empty:
        print("  ODE verification FAILED")
        return float("nan")
    rms = compute_seat_rms(df, {**cfg_base, **params})
    print(f"  ODE verified RMS = {rms:.5f} m/s²")
    return rms


# ─────────────────────────────────────────────────────────────
# Main Stage 2 function
# ─────────────────────────────────────────────────────────────

def run_stage2(stage1_result_path: str,
               out_dir: str = "outputs/opt2",
               cfg_base: Dict = None,
               zeta_ride: float = 0.35,
               asym_ratio: float = 2.5,
               verify: bool = False) -> Dict:

    cfg_base = cfg_base or dict(BASE_CFG)
    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)

    # Load Stage 1 results
    with open(stage1_result_path) as f:
        stage1 = json.load(f)
    p1 = stage1["params"]
    print(f"\n=== Option 2 Stage 2: deriving damper curve ===")
    print(f"  Using Stage 1 optimal: "
          f"Kf={p1['K_f']:.0f}  Cf={p1['C_f']:.0f}  "
          f"K2={p1['K_2']:.0f}  K3={p1['K_3']:.0f}")

    # Step 1: critical damping
    Cc_f, Cc_r1, Cc_r2 = critical_damping(
        p1["K_f"], p1["K_2"], p1["K_3"], cfg_base)

    # Step 2: target damping coefficients
    tgt = target_damping(Cc_f, Cc_r1, Cc_r2,
                          zeta_ride=zeta_ride, asym_ratio=asym_ratio)

    # Step 3: fit F-v curve
    damper = fit_damper_curve(tgt["cs_eq"], tgt["cp_eq"])

    # Step 4: plot
    plot_fv_curve(damper, str(out_path / "fv_curve.png"))

    # Step 5: assemble full result
    result = {
        "stage1_params": p1,
        "rms_surrogate_stage1": stage1["rms_surrogate"],
        "damper_derivation": {
            "Cc_f": Cc_f, "Cc_r1": Cc_r1, "Cc_r2": Cc_r2,
            "zeta_ride": zeta_ride,
            **tgt,
        },
        "fitted_damper": {
            k: v for k, v in damper.items()
            if not k.startswith("_")
        },
        "complete_params": {
            **p1,
            "cs_minus":   damper["cs_minus"],
            "asym_ratio": damper["asym_ratio"],
            "gamma_c":    damper["gamma_c"],
            "gamma_r":    damper["gamma_r"],
        },
    }

    # Nominal for comparison
    nom_rms = stage1.get("rms_nominal", float("nan"))

    # Step 5: ODE verification
    if verify:
        rms_ode = verify_ode(p1, {
            "cs_minus":   damper["cs_minus"],
            "asym_ratio": damper["asym_ratio"],
            "gamma_c":    damper["gamma_c"],
            "gamma_r":    damper["gamma_r"],
        }, cfg_base)
        result["rms_ode_verified"] = rms_ode
    else:
        rms_ode = float("nan")
        result["rms_ode_verified"] = float("nan")

    # Print final summary
    print("\n" + "="*60)
    print("OPTION 2 — COMPLETE RESULT")
    print("="*60)
    print(f"{'Parameter':<16}  {'Nominal':>12}  {'Optimal':>12}  {'Change':>8}")
    print("-"*60)
    nom_vals = {
        "K_f": BASE_CFG["K_f"], "C_f": BASE_CFG["C_f"],
        "K_2": BASE_CFG["K_2"], "K_3": BASE_CFG["K_3"],
        "cs_minus": BASE_CFG["cs_minus"],
        "asym_ratio": BASE_CFG["asym_ratio"],
        "gamma_c": BASE_CFG["gamma_c"],
        "gamma_r": BASE_CFG["gamma_r"],
    }
    for k, v in result["complete_params"].items():
        nom = nom_vals.get(k, float("nan"))
        chg = (v - nom) / nom * 100 if nom else 0
        print(f"{k:<16}  {nom:>12.4f}  {float(v):>12.4f}  {chg:>+7.1f}%")
    print("-"*60)
    if not np.isnan(rms_ode):
        surr_rms = stage1["rms_surrogate"]
        print(f"{'Surrogate RMS':<16}  {nom_rms:>12.5f}  {surr_rms:>12.5f}")
        print(f"{'ODE RMS':<16}  {nom_rms:>12.5f}  {rms_ode:>12.5f}  "
              f"{(nom_rms - rms_ode)/nom_rms*100:>+7.1f}%")

    # Save
    result_save = {k: v for k, v in result.items() if k != "damper_derivation"}
    with open(str(out_path / "stage2_result.json"), "w") as f:
        json.dump(result_save, f, indent=2)

    # Parameter comparison CSV
    rows = []
    for k, v in result["complete_params"].items():
        nom = nom_vals.get(k, float("nan"))
        rows.append({"parameter": k, "nominal": nom, "optimal": float(v),
                     "change_%": (float(v)-nom)/nom*100 if nom else 0})
    import pandas as pd
    pd.DataFrame(rows).to_csv(str(out_path / "param_comparison.csv"), index=False)

    print(f"\nResults saved → {out_path}/")
    print(f"  stage2_result.json")
    print(f"  param_comparison.csv")
    print(f"  fv_curve.png")

    return result


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Option 2 Stage 2: derive asymmetric damper from optimal K*")
    ap.add_argument("--stage1_result",
                    default="checkpoints/opt2/stage1_result.json")
    ap.add_argument("--out",        default="outputs/opt2")
    ap.add_argument("--zeta_ride",  type=float, default=0.35,
                    help="Ride comfort damping ratio (default 0.35)")
    ap.add_argument("--asym_ratio", type=float, default=2.5,
                    help="cp/cs rebound-to-compression ratio (default 2.5)")
    ap.add_argument("--verify_ode", action="store_true")
    ap.add_argument("--data_dir",   default=None)
    args = ap.parse_args()

    cfg = dict(BASE_CFG)
    if args.data_dir:
        for side in ("front_left", "front_right",
                     "rear1_left", "rear1_right",
                     "rear2_left", "rear2_right"):
            ax = "front" if "front" in side else ("rear1" if "1" in side else "rear2")
            sd = "left" if "left" in side else "right"
            cfg[f"axle{ax}_{sd}_csv"] = f"{args.data_dir}/{side}.csv"

    run_stage2(
        stage1_result_path=args.stage1_result,
        out_dir=args.out,
        cfg_base=cfg,
        zeta_ride=args.zeta_ride,
        asym_ratio=args.asym_ratio,
        verify=args.verify_ode,
    )
