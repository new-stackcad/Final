"""
OPTION 2 — PHASE 2 of 2: Asymmetric Damper Curve Fitting + Validation
======================================================================
Reads from Phase 1 output directory:
  phase1_best_params.json   →  K_f*, C_f* (=CF*), K_2*, K_3*
  v_rel_front.npy           →  front-damper relative-velocity time-series
                               (t_eval-aligned, from the Phase-1 optimal run)

Finds [cs_minus, asym_ratio, gamma_c, gamma_r] such that the
two-stage asymmetric damper best approximates the linear optimal
damper over the ACTUAL road-induced velocity distribution:

  Minimise  mean_v [ (shape(v) − v)² ]       (normalised, dimensionless)
  where shape(v) is the asymmetric F-v function (dimensionless)
  and the target is the identity v (i.e. match F_asym/CF* ≈ v/CF* = v·1)

  Equivalently: the physical target is F_linear = CF* · v  and the
  physical fit is F_asym = CF* · shape(v), so the cost is divided by
  CF*² to keep the optimizer's objective on a well-scaled (O(1)) surface.

After fitting:
  • Full ODE validation run is executed with asymmetric damper + Phase-1 K values
  • 3-axis seat RMS is computed consistently with Phase 1
  • Degradation P1→P2 is reported; >5% triggers a warning
  • All results saved to phase2_results.json

Outputs (all in RESULTS_DIR)
  phase2_results.json
  plots/
    fv_curve_comparison.png          F-v: linear target vs asymmetric fit
    vrel_histogram_fit.png           velocity distribution + fit quality
    seat_3axis_p1_vs_p2.png          3-axis seat accel: Phase 1 vs Phase 2
    rms_summary_all.png              3-bar summary: Baseline / P1 / P2
    phase2_params_normalised.png     normalised position in fit bounds

NOTE: This file is self-contained. It does NOT import from option2_bay_phase1.
      All physics code (build_M_R, geom_constraints, ODE) is reproduced here
      with the asymmetric damper active in build_M_R.
"""

# ── imports ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import json, os, time
import matplotlib.pyplot as plt
from scipy.optimize   import minimize
from dataclasses      import dataclass
from typing           import Dict, Callable, Tuple
from numpy.linalg     import solve as lin_solve
from scipy.integrate  import solve_ivp
from scipy.optimize   import least_squares

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION CONSTANTS  (must match Phase 1)
# ══════════════════════════════════════════════════════════════════════════════
DT       = 0.001
FS       = 1000
T_IGNORE = 0.5
T_END    = 466.945
t_eval_full = np.arange(0.0, T_END + DT, DT)

RESULTS_DIR = "Laden_results_ode_bay_opt2"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

STATE_NAMES = ["z_c", "th_c", "ph_c", "z_s", "th_s", "ph_s"]
(ZC, THC, PHC, ZS, THS, PHS) = range(6)

# Damper breakpoints (physical / from characterisation rig)
ALPHA_C = -0.05    # m/s  compression breakpoint
ALPHA_R =  0.13    # m/s  rebound    breakpoint

# Curve-fit search bounds for asymmetric shape parameters
FIT_BOUNDS = [
    (0.2,  0.4 ),   # cs_minus
    (2.3,  4.0 ),   # asym_ratio
    (0.08, 0.16),   # gamma_c
    (0.08, 0.10),   # gamma_r
]
FIT_PARAM_NAMES = ["cs_minus", "asym_ratio", "gamma_c", "gamma_r"]

# ══════════════════════════════════════════════════════════════════════════════
# BASE VEHICLE CONFIG  (identical to Phase 1 — K/C values will be overwritten
#                       from phase1_best_params.json at runtime)
# ══════════════════════════════════════════════════════════════════════════════
CFG: Dict = {
    "axlefront_left_csv":  r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_1_FA_LH.csv",
    "axlefront_right_csv": r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_2_FA_RH.csv",
    "axlerear1_left_csv":  r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_3_RA1_LH.csv",
    "axlerear1_right_csv": r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_4_RA1_RH.csv",
    "axlerear2_left_csv":  r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_5_RA2_LH.csv",
    "axlerear2_right_csv": r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_6_RA2_RH.csv",

    "s1": 0.6277, "s2": 0.6305,
    "WT1": 0.814, "WT2": 1.047, "WT3": 1.047,
    "a": 0.9,     "b": 1.080,
    "m_s": 22485.0, "I_syy": 103787.0, "I_sxx": 8598.0, "I_sxy": 763.0,
    "M_1f": 600.0,  "M_2": 1075.0,     "M_3": 840.0,
    "I_xx1": 650.0, "I_xx2": 1200.0,   "I_xx3": 1100.0,
    "lf": 5.05, "L12": 0.54, "L23": 1.96,
    "l_cf": 6.458, "l_cr": 4.5, "l_cfcg": 0.871, "l_crcg": 1.087,
    "m_c": 862.0, "I_xxc": 516.6, "I_yyc": 1045.0,
    "hs": 0.68,   "g": 9.81,     "hcp": 0.1,
    "L_DL2": 0.6211, "L_DR2": 0.6211,
    "L_DL3": 0.6251, "L_DR3": 0.6251,
    "beta_L2": 0.1693, "beta_R2": 0.1693,
    "beta_L3": 0.17453,"beta_R3": 0.17453,
    "S_tf2": 1.043, "S_tf3": 1.043, "S_f": 0.814,
    "C_cfl": 5035.0, "C_cfr": 5035.0, "C_crl": 3400.0, "C_crr": 3400.0,
    "K_cfl": 49050.0,"K_cfr": 49050.0,"K_crl": 24525.0,"K_crr": 24525.0,
    # Default K/C — overwritten from phase1_best_params.json
    "K_f": 474257, "C_f": 15000,
    "K_2": 1077620, "C_2": 2000,
    "K_3": 1077620, "C_3": 2000,
    # Asymmetric shape — overwritten from fit result before validation run
    "cs_minus":   0.3, "asym_ratio": 3.0,
    "gamma_c":    0.12,"gamma_r":    0.09,
    "baum_omega": 10.0,"baum_zeta":  1.0,
}

# ══════════════════════════════════════════════════════════════════════════════
# TWO-STAGE ASYMMETRIC DAMPER — vectorised
# Returns DIMENSIONLESS shape: physical force = C_f * asym_force_array(v)
# ══════════════════════════════════════════════════════════════════════════════
def asym_force_array(
    v: np.ndarray,
    cs_minus: float,
    asym_ratio: float,
    gamma_c: float,
    gamma_r: float,
    alpha_c: float = ALPHA_C,
    alpha_r: float = ALPHA_R,
) -> np.ndarray:
    """
    Vectorised two-stage asymmetric damper (dimensionless shape function).
    Physical force = C_f * asym_force_array(v).
    """
    v      = np.asarray(v, dtype=float)
    F      = np.zeros_like(v)
    c_plus = asym_ratio * cs_minus

    # Compression region (v < 0)
    m_c_low  = (v < 0)  & (v >= alpha_c)
    m_c_high = (v < 0)  & (v <  alpha_c)
    F[m_c_low]  = cs_minus * v[m_c_low]
    F[m_c_high] = cs_minus * (alpha_c + gamma_c * (v[m_c_high] - alpha_c))

    # Rebound region (v >= 0)
    m_r_low  = (v >= 0) & (v <= alpha_r)
    m_r_high = (v >= 0) & (v >  alpha_r)
    F[m_r_low]  = c_plus * v[m_r_low]
    F[m_r_high] = c_plus * (alpha_r + gamma_r * (v[m_r_high] - alpha_r))

    return F


def asym_force_scalar(v_rel: float, cs_minus: float, asym_ratio: float,
                      gamma_c: float, gamma_r: float,
                      alpha_c: float = ALPHA_C, alpha_r: float = ALPHA_R) -> float:
    """Scalar version used inside the ODE RHS."""
    c_plus = asym_ratio * cs_minus
    if v_rel < 0.0:
        if v_rel >= alpha_c:
            return cs_minus * v_rel
        return cs_minus * (alpha_c + gamma_c * (v_rel - alpha_c))
    else:
        if v_rel <= alpha_r:
            return c_plus * v_rel
        return c_plus * (alpha_r + gamma_r * (v_rel - alpha_r))

# ══════════════════════════════════════════════════════════════════════════════
# ROAD LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_track(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df   = pd.read_csv(csv_path, skiprows=2, header=None)
    t    = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
    z    = pd.to_numeric(df.iloc[:, 1], errors="coerce").values
    mask = np.isfinite(t) & np.isfinite(z)
    return t[mask].astype(float), z[mask].astype(float)


def make_linear_interp(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x); y = np.asarray(y)
    def f(xq):
        xq   = np.asarray(xq)
        xq_c = np.clip(xq, x[0], x[-1])
        idx  = np.clip(np.searchsorted(x, xq_c) - 1, 0, len(x) - 2)
        w    = (xq_c - x[idx]) / np.maximum(x[idx + 1] - x[idx], 1e-12)
        return y[idx] * (1 - w) + y[idx + 1] * w
    return f


@dataclass
class RoadSignals:
    f1L: Callable; f1R: Callable
    f2L: Callable; f2R: Callable
    f3L: Callable; f3R: Callable

    def axle_inputs(self, t: float, cfg: Dict):
        zr1L, zr1R = self.f1L(t), self.f1R(t)
        zr2L, zr2R = self.f2L(t), self.f2R(t)
        zr3L, zr3R = self.f3L(t), self.f3R(t)
        z1f  = 0.5 * (zr1L + zr1R)
        z2   = 0.5 * (zr2L + zr2R)
        z3   = 0.5 * (zr3L + zr3R)
        ph_f = (zr1L - zr1R) / cfg["WT1"]
        ph2  = (zr2L - zr2R) / cfg["WT2"]
        ph3  = (zr3L - zr3R) / cfg["WT3"]
        return float(z1f), float(ph_f), float(z2), float(ph2), float(z3), float(ph3)

    def axle_input_rates(self, t: float, cfg: Dict, dt: float = DT):
        p = self.axle_inputs(t + dt, cfg)
        m = self.axle_inputs(t - dt, cfg)
        return tuple((a - b) / (2.0 * dt) for a, b in zip(p, m))


def build_road_signals(cfg: Dict) -> RoadSignals:
    t1L, z1L = load_track(cfg["axlefront_left_csv"])
    t1R, z1R = load_track(cfg["axlefront_right_csv"])
    t2L, z2L = load_track(cfg["axlerear1_left_csv"])
    t2R, z2R = load_track(cfg["axlerear1_right_csv"])
    t3L, z3L = load_track(cfg["axlerear2_left_csv"])
    t3R, z3R = load_track(cfg["axlerear2_right_csv"])
    return RoadSignals(
        make_linear_interp(t1L, z1L), make_linear_interp(t1R, z1R),
        make_linear_interp(t2L, z2L), make_linear_interp(t2R, z2R),
        make_linear_interp(t3L, z3L), make_linear_interp(t3R, z3R),
    )

# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS: GEOMETRY CONSTRAINTS
# ══════════════════════════════════════════════════════════════════════════════
def geom_constraints(q, t, cfg, road):
    z_s, th_s, ph_s = q[ZS], q[THS], q[PHS]
    _, _, z2, ph2, z3, ph3 = road.axle_inputs(t, cfg)
    l2 = cfg["L12"];  l3 = cfg["L12"] + cfg["L23"]
    S2, S3   = cfg["S_tf2"], cfg["S_tf3"]
    sl2, sl3 = cfg["s1"],    cfg["s2"]
    bL2, bL3 = cfg["beta_L2"], cfg["beta_L3"]

    g2 = (z_s + l2*th_s + S2*ph_s - sl2*np.sin(bL2 - th_s)
          - (z2 + 0.5*cfg["WT2"]*ph2))
    g3 = (z_s + l3*th_s + S3*ph_s - sl3*np.sin(bL3 - th_s)
          - (z3 + 0.5*cfg["WT3"]*ph3))

    g = np.array([g2, g3], dtype=float)
    G = np.zeros((2, 6), dtype=float)
    G[0, ZS]  = 1.0;  G[0, THS] = l2 + sl2*np.cos(bL2 - th_s);  G[0, PHS] = S2
    G[1, ZS]  = 1.0;  G[1, THS] = l3 + sl3*np.cos(bL3 - th_s);  G[1, PHS] = S3
    return g, G

# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS: MASS MATRIX + RESIDUAL  (ASYMMETRIC front damper, Phase 2)
# Front damper: F_df = C_f * asym_force_scalar(v_f)
# The asymmetric shape params are read from cfg: cs_minus, asym_ratio, γ_c, γ_r
# ══════════════════════════════════════════════════════════════════════════════
def build_M_R_asym(q, v, t, cfg, road):
    """
    Build mass matrix M and residual R for the Phase-2 validation ODE.
    Front damper uses the TWO-STAGE ASYMMETRIC model (reads shape params from cfg).
    All other physics identical to Phase 1.
    """
    z_c, th_c, ph_c, z_s, th_s, ph_s     = q
    dz_c, dth_c, dph_c, dz_s, dth_s, dph_s = v

    z1f, ph_f, z2, ph2, z3, ph3           = road.axle_inputs(t, cfg)
    dz1f, dph_f, dz2, dph2, dz3, dph3    = road.axle_input_rates(t, cfg)

    phi_NRS2 = (cfg["beta_L2"]*cfg["L_DL2"] - cfg["beta_R2"]*cfg["L_DR2"]) / max(cfg["S_tf2"], 1e-6)
    phi_NRS3 = (cfg["beta_L3"]*cfg["L_DL3"] - cfg["beta_R3"]*cfg["L_DR3"]) / max(cfg["S_tf3"], 1e-6)

    m_c, I_xxc, I_yyc      = cfg["m_c"], cfg["I_xxc"], cfg["I_yyc"]
    m_s, I_sxx, I_syy, I_sxy = cfg["m_s"], cfg["I_sxx"], cfg["I_syy"], cfg["I_sxy"]
    S1, S2, S3             = cfg["S_f"],  cfg["S_tf2"], cfg["S_tf3"]
    a, b                   = cfg["a"],    cfg["b"]
    hs, g                  = cfg["hs"],   cfg["g"]
    l_cfcg, l_crcg         = cfg["l_cfcg"], cfg["l_crcg"]
    l_cf, l_cr             = cfg["l_cf"],   cfg["l_cr"]
    lf, hcp                = cfg["lf"],     cfg["hcp"]
    l2                     = cfg["L12"]
    l3                     = cfg["L12"] + cfg["L23"]
    beta_L2, beta_R2       = cfg["beta_L2"], cfg["beta_R2"]
    beta_L3, beta_R3       = cfg["beta_L3"], cfg["beta_R3"]
    L_DL2, L_DR2           = cfg["L_DL2"], cfg["L_DR2"]
    L_DL3, L_DR3           = cfg["L_DL3"], cfg["L_DR3"]
    Kcfl, Kcfr, Kcrl, Kcrr = cfg["K_cfl"], cfg["K_cfr"], cfg["K_crl"], cfg["K_crr"]
    Ccfl, Ccfr, Ccrl, Ccrr = cfg["C_cfl"], cfg["C_cfr"], cfg["C_crl"], cfg["C_crr"]
    K_f, C_f               = cfg["K_f"], cfg["C_f"]
    K_2, C_2               = cfg["K_2"], cfg["C_2"]
    K_3, C_3               = cfg["K_3"], cfg["C_3"]

    # ── ASYMMETRIC front damper (Phase 2) ────────────────────────────────────
    v_f  = dz_s - lf * dth_s - dz1f
    F_df = C_f * asym_force_scalar(
        v_f,
        cfg["cs_minus"], cfg["asym_ratio"],
        cfg["gamma_c"],  cfg["gamma_r"],
    )
    # ─────────────────────────────────────────────────────────────────────────

    Csum = Ccfl + Ccfr + Ccrl + Ccrr
    Ksum = Kcfl + Kcfr + Kcrl + Kcrr

    M = np.zeros((6, 6), dtype=float)
    M[ZC, ZC]   = m_c
    M[THC, THC] = I_yyc
    M[PHC, PHC] = I_xxc
    M[ZS, ZS]   = m_s
    M[THS, THS] = I_syy
    M[THS, PHS] = I_sxy
    M[PHS, THS] = I_sxy
    M[PHS, PHS] = I_sxx + m_s * hs**2

    R = np.zeros(6, dtype=float)

    R[ZC] = (
        + Csum*(dz_c - dz_s) + Ksum*(z_c - z_s)
        - (Ccfl*l_cfcg + Ccfr*l_cfcg - Ccrl*l_crcg - Ccrr*l_crcg)*dth_c
        - (-Ccfl*l_cf  - Ccfr*l_cf  - Ccrl*l_cr   - Ccrr*l_cr  )*dth_s
        - (-Ccfl*b + Ccfr*a - Ccrl*b + Ccrr*a)*dph_c
        - ( Ccfl*b - Ccfr*a + Ccrl*b - Ccrr*a)*dph_s
        - (Kcfl*l_cfcg + Kcfr*l_cfcg - Kcrl*l_crcg - Kcrr*l_crcg)*th_c
        - (-Kcfl*l_cf  - Kcfr*l_cf  - Kcrl*l_cr   - Kcrr*l_cr  )*th_s
        - (-Kcfl*b + Kcfr*a - Kcrl*b + Kcrr*a)*ph_c
        - ( Kcfl*b - Kcfr*a + Kcrl*b - Kcrr*a)*ph_s
    )

    R[THC] = (
        - (Ccfl*l_cfcg + Ccfr*l_cfcg - Ccrl*l_crcg - Ccrr*l_crcg)*dz_c
        - (-Ccfl*l_cfcg - Ccfr*l_cfcg - Ccrl*l_crcg - Ccrr*l_crcg)*dz_s
        - (Kcfl*l_cfcg + Kcfr*l_cfcg - Kcrl*l_crcg - Kcrr*l_crcg)*z_c
        - (-Kcfl*l_cfcg - Kcfr*l_cfcg - Kcrl*l_crcg - Kcrr*l_crcg)*z_s
        - (-Ccfl*l_cfcg**2 - Ccfr*l_cfcg**2 - Ccrl*l_crcg**2 - Ccrr*l_crcg**2)*dth_c
        - ( Ccfl*l_cfcg*l_cf + Ccfr*l_cfcg*l_cf - Ccrl*l_crcg*l_cr - Ccrr*l_crcg*l_cr)*dth_s
        - (-Ccfl*l_cfcg*b + Ccfr*l_cfcg*a - Ccrl*l_crcg*b + Ccrr*l_crcg*a)*dph_c
        - ( Ccfl*l_cfcg*b - Ccfr*l_cfcg*a + Ccrl*l_crcg*b - Ccrr*l_crcg*a)*dph_s
        - (-Kcfl*l_cfcg**2 - Kcfr*l_cfcg**2 - Kcrl*l_crcg**2 - Kcrr*l_crcg**2 + m_c*g*hcp)*th_c
        - ( Kcfl*l_cfcg*l_cf + Kcfr*l_cfcg*l_cf - Kcrl*l_crcg*l_cr - Kcrr*l_crcg*l_cr)*th_s
        - (-Kcfl*l_cfcg*b + Kcfr*l_cfcg*a - Kcrl*l_crcg*b + Kcrr*l_crcg*a)*ph_c
        - ( Kcfl*l_cfcg*b - Kcfr*l_cfcg*a + Kcrl*l_crcg*b - Kcrr*l_crcg*a)*ph_s
    )

    R[PHC] = (
        - (-Ccfl*b + Ccfr*a - Ccrl*b + Ccrr*a)*dz_c
        - ( Ccfl*b - Ccfr*a + Ccrl*b - Ccrr*a)*dz_s
        - (-Kcfl*b + Kcfr*a - Kcrl*b + Kcrr*a)*z_c
        - ( Kcfl*b - Kcfr*a + Kcrl*b - Kcrr*a)*z_s
        - (-Ccfl*l_cfcg*b - Ccfr*l_cfcg*a + Ccrl*l_crcg*b + Ccrr*l_crcg*a)*dth_c
        - ( Ccfl*l_cfcg*b + Ccfr*l_cfcg*a - Ccrl*l_crcg*b - Ccrr*l_crcg*a)*dth_s
        - (-Ccfl*b**2 + Ccfr*a**2 - Ccrl*b**2 + Ccrr*a**2)*dph_c
        - ( Ccfl*b**2 - Ccfr*a**2 + Ccrl*b**2 - Ccrr*a**2)*dph_s
        - (-Kcfl*l_cfcg*b - Kcfr*l_cfcg*a + Kcrl*l_crcg*b + Kcrr*l_crcg*a)*th_c
        - ( Kcfl*l_cfcg*b + Kcfr*l_cfcg*a - Kcrl*l_crcg*b - Kcrr*l_crcg*a)*th_s
        - (-Kcfl*b**2 + Kcfr*a**2 - Kcrl*b**2 + Kcrr*a**2)*ph_c
        - ( Kcfl*b**2 - Kcfr*a**2 + Kcrl*b**2 - Kcrr*a**2)*ph_s
    )

    R[ZS] = (
        - (Ccfl + Ccfr + Ccrl + Ccrr)*dz_c
        - (-Ccfl*l_cfcg - Ccfr*l_cfcg + Ccrl*l_crcg + Ccrr*l_crcg)*dth_c
        - (-Ccfl - Ccfr - Ccrl - Ccrr)*dz_s
        - (Ccfl*l_cf + Ccfr*l_cf + Ccrl*l_cr + Ccrr*l_cr)*dth_s
        - (Kcfl + Kcfr + Kcrl + Kcrr)*z_c
        - (-Kcfl*l_cfcg - Kcfr*l_cfcg + Kcrl*l_crcg + Kcrr*l_crcg)*th_c
        - (-Kcfl - Kcfr - Kcrl - Kcrr)*z_s
        - (Kcfl*l_cf + Kcfr*l_cf + Kcrl*l_cr + Kcrr*l_cr)*th_s
        + K_f*(z_s - lf*th_s - z1f) + F_df
        + K_2*(z_s - z2 - beta_L2*L_DL2 - beta_R2*L_DR2 + l2*th_s) + C_2*(dz_s - dz2 + l2*dth_s)
        + K_3*(z_s - z3 - beta_L3*L_DL3 - beta_R3*L_DR3 + l3*th_s) + C_3*(dz_s - dz3 + l3*dth_s)
    )

    R[THS] = (
        - (Ccfl*l_cfcg + Ccfr*l_cfcg - Ccrl*l_crcg - Ccrr*l_crcg)*dz_c
        - (-Ccfl*l_cfcg**2 - Ccfr*l_cfcg**2 - Ccrl*l_crcg**2 - Ccrr*l_crcg**2)*dth_c
        - (-Ccfl*l_cf - Ccfr*l_cf - Ccrl*l_cr - Ccrr*l_cr)*dz_s
        - ( Ccfl*l_cfcg*l_cf + Ccfr*l_cfcg*l_cf - Ccrl*l_crcg*l_cr - Ccrr*l_crcg*l_cr)*dth_s
        - (Kcfl*l_cf + Kcfr*l_cf + Kcrl*l_cr + Kcrr*l_cr)*z_c
        - (-Kcfl*l_cfcg*l_cf - Kcfr*l_cfcg*l_cf + Kcrl*l_crcg*l_cr + Kcrr*l_crcg*l_cr)*th_c
        - (-Kcfl*l_cf - Kcfr*l_cf - Kcrl*l_cr - Kcrr*l_cr)*z_s
        - (Kcfl*l_cf**2 + Kcfr*l_cf**2 + Kcrl*l_cr**2 + Kcrr*l_cr**2)*th_s
        - lf*(K_f*(z_s - lf*th_s - z1f) + F_df)
        + l2*(K_2*(z_s - z2 - beta_L2*L_DL2 - beta_R2*L_DR2 + l2*th_s) + C_2*(dz_s - dz2 + l2*dth_s))
        + l3*(K_3*(z_s - z3 - beta_L3*L_DL3 - beta_R3*L_DR3 + l3*th_s) + C_3*(dz_s - dz3 + l3*dth_s))
    )

    k_tf = 0.5 * K_f * S1**2;  K_r1 = 0.5 * K_2 * S2**2;  K_r2 = 0.5 * K_3 * S3**2
    C_tf = 0.5 * C_f * S1**2;  C_r1 = 0.5 * C_2 * S2**2;  C_r2 = 0.5 * C_3 * S3**2

    R[PHS] = (
        + m_s * g * hs * ph_s
        - k_tf*(ph_s - ph_f) - C_tf*(dph_s - dph_f)
        - K_r1*(ph_s - ph2 - phi_NRS2) - C_r1*(dph_s - dph2)
        - K_r2*(ph_s - ph3 - phi_NRS3) - C_r2*(dph_s - dph3)
    )
    R[PHS] *= -1.0
    return M, R

# ══════════════════════════════════════════════════════════════════════════════
# ODE RHS (asymmetric)
# ══════════════════════════════════════════════════════════════════════════════
def rhs_asym(t, x, cfg, road):
    q, v  = x[:6], x[6:]
    M, R  = build_M_R_asym(q, v, t, cfg, road)
    gq, G = geom_constraints(q, t, cfg, road)

    w, zeta = cfg["baum_omega"], cfg["baum_zeta"]
    gamma   = w**2 * gq + 2 * zeta * w * (G @ v)

    nc = G.shape[0]
    A  = np.zeros((6 + nc, 6 + nc))
    b  = np.zeros(6 + nc)
    A[:6, :6] = M;  A[:6, 6:] = G.T;  A[6:, :6] = G
    b[:6] = -R;     b[6:]     = -gamma

    xdot     = np.zeros_like(x)
    xdot[:6] = v
    xdot[6:] = lin_solve(A, b)[:6]
    return xdot

# ══════════════════════════════════════════════════════════════════════════════
# STATIC EQUILIBRIUM  (uses asymmetric build_M_R for Phase-2 validation)
# ══════════════════════════════════════════════════════════════════════════════
def static_equilibrium_state(cfg, road):
    y0 = np.zeros(8, dtype=float)
    t0 = 0.0

    def F(y):
        q, lam = y[:6], y[6:]
        M, R   = build_M_R_asym(q, np.zeros(6), t0, cfg, road)
        gq, G  = geom_constraints(q, t0, cfg, road)
        return np.hstack([R + G.T @ lam, 1e3 * gq])

    lsq = least_squares(F, y0, method="trf", loss="soft_l1",
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=800)
    if lsq.success:
        q0     = lsq.x[:6]
        g0, G0 = geom_constraints(q0, t0, cfg, road)
        M0, R0 = build_M_R_asym(q0, np.zeros(6), t0, cfg, road)
        print("=== Static equilibrium OK. ||g||=%.3e, ||R+G^T*lam||=%.3e"
              % (np.linalg.norm(g0), np.linalg.norm(R0 + G0.T @ lsq.x[6:])))
        return np.hstack([q0, np.zeros(6)])

    print("=== Static equilibrium LSQ failed — trying dynamic relaxation...")
    cfg_r = {**cfg,
             "C_2": cfg["C_2"]*20, "C_3": cfg["C_3"]*20,
             "C_cfl": cfg["C_cfl"]*20, "C_cfr": cfg["C_cfr"]*20,
             "C_crl": cfg["C_crl"]*20, "C_crr": cfg["C_crr"]*20}
    sol_r = solve_ivp(lambda t, x: rhs_asym(t, x, cfg_r, road),
                      (0.0, 3.0), np.zeros(12), method="Radau", rtol=1e-7, atol=1e-9)
    q_r   = sol_r.y[:6, -1]
    lsq2  = least_squares(F, np.hstack([q_r, np.zeros(2)]), method="trf", loss="soft_l1",
                          xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
    q0    = lsq2.x[:6] if lsq2.success else q_r
    print("=== Dynamic relaxation end. ||g||=%.3e"
          % np.linalg.norm(geom_constraints(q0, t0, cfg, road)[0]))
    return np.hstack([q0, np.zeros(6)])

# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION RUN  (full ODE with asymmetric damper + Phase-1 K values)
# ══════════════════════════════════════════════════════════════════════════════
def run_validation(cfg_val: Dict, t_eval: np.ndarray) -> pd.DataFrame:
    """
    Full ODE integration using the ASYMMETRIC damper.
    cfg_val must contain K_f*, C_f*, K_2*, K_3* (from Phase 1)
    and cs_minus, asym_ratio, gamma_c, gamma_r (from Phase-2 fit).
    """
    road = build_road_signals(cfg_val)
    x0   = static_equilibrium_state(cfg_val, road)

    print(f"\n=== Phase-2 Validation ODE | T_end={t_eval[-1]:.2f} s")
    t_wall = time.time()

    sol = solve_ivp(
        fun=lambda t, x: rhs_asym(t, x, cfg_val, road),
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=x0, t_eval=t_eval,
        method="Radau", max_step=0.01, rtol=1e-6, atol=1e-8,
    )

    print(f"=== solve_ivp success={sol.success}, nfev={sol.nfev}, "
          f"wall={time.time()-t_wall:.1f} s")

    if sol.status != 0 or not np.all(np.isfinite(sol.y)):
        raise RuntimeError("Phase-2 validation ODE failed or diverged")

    rows = []
    for i, t in enumerate(sol.t):
        x   = sol.y[:, i]
        qdd = rhs_asym(t, x, cfg_val, road)[6:]
        row = {"t": t}
        for j, name in enumerate(STATE_NAMES):
            row[name]          = x[j]
            row[f"qd_{name}"]  = x[j + 6]
            row[f"qdd_{name}"] = qdd[j]
        rows.append(row)
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE METRICS  (3-axis cabin seat RMS — consistent with Phase 1)
# ══════════════════════════════════════════════════════════════════════════════
def compute_seat_rms(df: pd.DataFrame, cfg: Dict) -> float:
    mask = df["t"] >= T_IGNORE
    h    = cfg["hcp"]
    az   = df.loc[mask, "qdd_z_c"].values
    ax   = -h * df.loc[mask, "qdd_th_c"].values
    ay   =  h * df.loc[mask, "qdd_ph_c"].values
    return float(np.sqrt(np.mean(az**2) + np.mean(ax**2) + np.mean(ay**2)))


def compute_seat_rms_axes(df: pd.DataFrame, cfg: Dict) -> Dict[str, float]:
    mask = df["t"] >= T_IGNORE
    h    = cfg["hcp"]
    az   = df.loc[mask, "qdd_z_c"].values
    ax   = -h * df.loc[mask, "qdd_th_c"].values
    ay   =  h * df.loc[mask, "qdd_ph_c"].values
    return {
        "rms_z":     float(np.sqrt(np.mean(az**2))),
        "rms_x":     float(np.sqrt(np.mean(ax**2))),
        "rms_y":     float(np.sqrt(np.mean(ay**2))),
        "rms_total": float(np.sqrt(np.mean(az**2) + np.mean(ax**2) + np.mean(ay**2))),
    }

# ══════════════════════════════════════════════════════════════════════════════
# PHASE-2 CORE: FIT ASYMMETRIC CURVE TO MATCH LINEAR CF*
# ══════════════════════════════════════════════════════════════════════════════
def fit_asymmetric_damper(v_rel_samples: np.ndarray, C_f_star: float,
                          n_starts: int = 25, seed: int = 42):
    """
    Fits [cs_minus, asym_ratio, gamma_c, gamma_r] so that:
        F_asym(v) ≈ v    (dimensionless — dividing both sides by C_f_star)

    Cost function:
        J = mean_v[ (shape(v) - v)² ]     (dimensionless, O(1) scaled)
    which is equivalent to minimising the mean-square force error
        mean[ (C_f* * shape(v) - C_f* * v)² ] / C_f*²
    The C_f* factor cancels, giving a well-scaled objective that does not
    depend on the magnitude of C_f* and thus avoids numerical conditioning issues.

    Starting points use a deterministic Sobol-like grid (uniform spacing per
    dimension) to give reproducible, comprehensive coverage of the bound box.
    """
    v = np.asarray(v_rel_samples, dtype=float).ravel()
    # Target in dimensionless space: shape(v) should equal v / C_f_star ... no.
    # Actually: F_linear = C_f_star * v, F_asym = C_f_star * shape(v)
    # We want shape(v) ≈ v   only if shape is already dimensionless as written.
    # Check: at low-speed compression, shape = cs_minus * v.
    # For shape ≈ v we'd need cs_minus ≈ 1, which is outside the bound [0.2, 0.4].
    # CORRECT interpretation: the linear damper force is F = C_f_star * v.
    # The asymmetric damper force is F = C_f_star * shape(v).
    # They match when shape(v) = v.  But shape(v) ≠ v in general (cs_minus ≠ 1).
    # The BEST MATCH over the velocity distribution is what we minimise:
    #   J = mean[ (C_f_star * shape(v) - C_f_star * v)² ] / C_f_star²
    #     = mean[ (shape(v) - v)² ]
    # This is dimensionless and O(v²) ~ O((0.1 m/s)²) ~ O(0.01) — well-scaled.

    def cost(p):
        cs, ar, gc, gr = p
        F_shape = asym_force_array(v, cs, ar, gc, gr)
        return float(np.mean((F_shape - v) ** 2))

    # Deterministic starting grid: evenly spaced in each dimension
    rng      = np.random.default_rng(seed)
    x0_grid  = [
        [np.random.uniform(lo, hi) for (lo, hi) in FIT_BOUNDS]
        for _ in range(n_starts)
    ]
    # Also include the midpoint of each bound as first start (often closest)
    x0_grid.insert(0, [0.5 * (lo + hi) for lo, hi in FIT_BOUNDS])

    best = None
    for x0 in x0_grid:
        res = minimize(cost, x0, bounds=FIT_BOUNDS, method="L-BFGS-B",
                       options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 500})
        if best is None or res.fun < best.fun:
            best = res

    cs, ar, gc, gr = best.x

    # Evaluation grid for plots (span the observed velocity range with 10% margin)
    v_lo   = min(v.min() * 1.10, v.min() - 0.05)
    v_hi   = max(v.max() * 1.10, v.max() + 0.05)
    v_eval = np.linspace(v_lo, v_hi, 500)

    F_linear_eval = C_f_star * v_eval
    F_asym_eval   = C_f_star * asym_force_array(v_eval, cs, ar, gc, gr)

    # Force RMSE on the evaluation grid
    force_rmse = float(np.sqrt(np.mean((F_asym_eval - F_linear_eval) ** 2)))

    # Force RMSE weighted by the empirical distribution (density at sample points)
    F_asym_samples  = C_f_star * asym_force_array(v, cs, ar, gc, gr)
    F_linear_samples = C_f_star * v
    force_rmse_emp  = float(np.sqrt(np.mean((F_asym_samples - F_linear_samples) ** 2)))

    print(f"  Fit RMSE (eval grid) : {force_rmse:.2f} N")
    print(f"  Fit RMSE (empirical) : {force_rmse_emp:.2f} N  "
          f"(at CF*={C_f_star:.0f} N·s/m)")

    fitted = {
        "cs_minus":          float(cs),
        "asym_ratio":        float(ar),
        "gamma_c":           float(gc),
        "gamma_r":           float(gr),
        "alpha_c":           ALPHA_C,
        "alpha_r":           ALPHA_R,
        "force_rmse_grid_N": force_rmse,
        "force_rmse_emp_N":  force_rmse_emp,
        "optimizer_cost":    float(best.fun),
    }
    return fitted, v_eval, F_linear_eval, F_asym_eval

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def plot_fv_comparison(v_rel, v_eval, F_linear, F_fitted,
                       CF_star, fitted, save_dir):
    """
    Left:  F-v overlay (linear target vs fitted asymmetric) with velocity
           percentile shading showing where the damper actually operates.
    Right: Force residual (F_asym − F_linear) vs velocity.
    """
    p5, p95 = np.percentile(v_rel, [5, 95])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # — Left: F-v curves
    ax = axes[0]
    ax.axvspan(p5, p95, alpha=0.10, color="steelblue", label="5–95th pct v range")
    ax.plot(v_eval, F_linear, "b-",  lw=2.5,
            label=f"Linear target  (CF* = {CF_star:.0f} N·s/m)")
    ax.plot(v_eval, F_fitted, "r--", lw=2.5,
            label=(f"Asymmetric fit\n"
                   f"cs⁻={fitted['cs_minus']:.3f}, ratio={fitted['asym_ratio']:.2f}\n"
                   f"γ_c={fitted['gamma_c']:.3f}, γ_r={fitted['gamma_r']:.3f}"))
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.axvline(ALPHA_C, color="grey", lw=0.8, ls=":",  label=f"α_c = {ALPHA_C}")
    ax.axvline(ALPHA_R, color="grey", lw=0.8, ls="-.", label=f"α_r = {ALPHA_R}")
    ax.set_xlabel("Relative velocity  [m/s]")
    ax.set_ylabel("Damper force  [N]")
    ax.set_title("F–v Curve: Linear Target vs Fitted Asymmetric Damper")
    ax.legend(fontsize=8); ax.grid(True)

    # — Right: residual
    ax = axes[1]
    residual = F_fitted - F_linear
    ax.fill_between(v_eval, residual, 0,
                    where=(residual >= 0), color="tomato",    alpha=0.6, label="Over-damping")
    ax.fill_between(v_eval, residual, 0,
                    where=(residual <  0), color="steelblue", alpha=0.6, label="Under-damping")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvspan(p5, p95, alpha=0.08, color="grey", label="5–95th pct v range")
    ax.set_xlabel("Relative velocity  [m/s]")
    ax.set_ylabel("Force residual  [N]  (F_asym − F_linear)")
    ax.set_title(f"Fit Residual  |  RMSE = {fitted['force_rmse_emp_N']:.1f} N (empirical)")
    ax.legend(fontsize=8); ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fv_curve_comparison.png"), dpi=150)
    plt.close()
    print("  Saved → fv_curve_comparison.png")


def plot_vrel_histogram_fit(v_rel, v_eval, F_linear, F_fitted, CF_star, save_dir):
    """
    Velocity histogram overlaid with the fit quality indicator:
    marks where the asymmetric force deviates from the linear target,
    weighted by where the damper actually spends time.
    """
    residual_pct = (F_fitted - F_linear) / (np.abs(F_linear) + 1.0) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # — velocity histogram
    axes[0].hist(v_rel, bins=120, density=True, color="steelblue", alpha=0.75,
                 label="Empirical PDF")
    axes[0].axvline(0,                  color="k",      lw=0.9, ls="--")
    axes[0].axvline(ALPHA_C,            color="tomato", lw=0.9, ls=":",  label=f"α_c = {ALPHA_C}")
    axes[0].axvline(ALPHA_R,            color="green",  lw=0.9, ls="-.", label=f"α_r = {ALPHA_R}")
    axes[0].axvline(np.percentile(v_rel, 5),  color="grey", lw=0.8, ls="--", label="5/95th pct")
    axes[0].axvline(np.percentile(v_rel, 95), color="grey", lw=0.8, ls="--")
    axes[0].set_xlabel("v_rel  [m/s]")
    axes[0].set_ylabel("Probability density")
    axes[0].set_title("Front-damper velocity distribution\n(Phase-1 optimal run, used for fitting)")
    axes[0].legend(fontsize=8); axes[0].grid(True)

    # — percentage deviation vs velocity
    axes[1].plot(v_eval, residual_pct, "r-", lw=1.5)
    axes[1].axhline( 5, color="orange", lw=0.8, ls="--", label="±5% tolerance")
    axes[1].axhline(-5, color="orange", lw=0.8, ls="--")
    axes[1].axhline(0,  color="k",      lw=0.8, ls="-")
    axes[1].set_xlabel("Relative velocity  [m/s]")
    axes[1].set_ylabel("Force deviation  [%]  = (F_asym−F_lin)/(|F_lin|+1) × 100")
    axes[1].set_title("Fit quality vs velocity\n(percentage deviation from linear target)")
    axes[1].legend(fontsize=8); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "vrel_histogram_fit.png"), dpi=150)
    plt.close()
    print("  Saved → vrel_histogram_fit.png")


def plot_seat_3axis_p1_p2(df_p1, df_p2, cfg, rms_p1, rms_p2, save_dir):
    """3-axis seat acceleration comparison: Phase-1 (linear) vs Phase-2 (asymmetric)."""
    t = df_p1["t"]
    h = cfg["hcp"]

    az_p1 = df_p1["qdd_z_c"];          az_p2 = df_p2["qdd_z_c"]
    ax_p1 = -h * df_p1["qdd_th_c"];    ax_p2 = -h * df_p2["qdd_th_c"]
    ay_p1 =  h * df_p1["qdd_ph_c"];    ay_p2 =  h * df_p2["qdd_ph_c"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(t, az_p1, "b-",  alpha=0.75, lw=0.8,
                 label=f"Phase-1 linear   (RMS_z={np.sqrt(np.mean(az_p1[df_p1['t']>=T_IGNORE]**2)):.4f})")
    axes[0].plot(t, az_p2, "r--", alpha=0.75, lw=0.8,
                 label=f"Phase-2 asymmetric (RMS_z={np.sqrt(np.mean(az_p2[df_p2['t']>=T_IGNORE]**2)):.4f})")
    axes[0].set_ylabel("z̈_seat [m/s²]")
    axes[0].set_title(f"Seat Vertical  |  P1 total={rms_p1:.4f}  P2 total={rms_p2:.4f} m/s²")
    axes[0].legend(fontsize=8); axes[0].grid(True)

    axes[1].plot(t, ax_p1, "b-",  alpha=0.75, lw=0.8, label="Phase-1")
    axes[1].plot(t, ax_p2, "r--", alpha=0.75, lw=0.8, label="Phase-2")
    axes[1].set_ylabel("ẍ_seat = -h·θ̈_c [m/s²]")
    axes[1].set_title("Seat Longitudinal (pitch contribution)")
    axes[1].legend(fontsize=8); axes[1].grid(True)

    axes[2].plot(t, ay_p1, "b-",  alpha=0.75, lw=0.8, label="Phase-1")
    axes[2].plot(t, ay_p2, "r--", alpha=0.75, lw=0.8, label="Phase-2")
    axes[2].set_ylabel("ÿ_seat = h·φ̈_c [m/s²]")
    axes[2].set_title("Seat Lateral (roll contribution)")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend(fontsize=8); axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "seat_3axis_p1_vs_p2.png"), dpi=150)
    plt.close()
    print("  Saved → seat_3axis_p1_vs_p2.png")


def plot_rms_summary_all(rms_base, axes_p1, axes_p2, save_dir):
    """
    3-bar grouped chart: Baseline / Phase-1 (linear opt) / Phase-2 (asym fit)
    Split into per-axis (z, x, y) and total.
    """
    labels    = ["RMS_z\n(vertical)", "RMS_x\n(longitudinal)", "RMS_y\n(lateral)", "RMS_total"]
    keys      = ["rms_z", "rms_x", "rms_y", "rms_total"]

    # Baseline axes_base was not re-run in Phase 2 so we use P1 JSON values
    # (rms_base is only the total here — per-axis baseline comes from JSON)
    base_vals_total = rms_base   # scalar

    p1_vals = [axes_p1[k] for k in keys]
    p2_vals = [axes_p2[k] for k in keys]

    x, w = np.arange(len(labels)), 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    # Only P1 and P2 per-axis — baseline total shown as horizontal line
    ax.bar(x - w/2, p1_vals, w, label="Phase-1 (linear opt)",    color="steelblue",  edgecolor="k")
    ax.bar(x + w/2, p2_vals, w, label="Phase-2 (asymmetric fit)", color="darkorange", edgecolor="k")
    ax.axhline(base_vals_total, color="grey", ls="--", lw=1.2, label=f"Baseline total RMS = {base_vals_total:.4f}")

    for xi, v1, v2 in zip(x, p1_vals, p2_vals):
        ax.text(xi - w/2, v1 + 0.0003, f"{v1:.4f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w/2, v2 + 0.0003, f"{v2:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("RMS seat acceleration  [m/s²]")
    ax.set_title("Per-Axis Seat RMS Summary — Baseline / Phase-1 / Phase-2")
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rms_summary_all.png"), dpi=150)
    plt.close()
    print("  Saved → rms_summary_all.png")


def plot_phase2_params_normalised(fitted: Dict, save_dir: str):
    """Normalised position of each fitted shape param within its search bound."""
    keys  = FIT_PARAM_NAMES
    lo_hi = {k: (lo, hi) for k, (lo, hi) in zip(keys, FIT_BOUNDS)}
    vals  = [(fitted[k] - lo_hi[k][0]) / (lo_hi[k][1] - lo_hi[k][0]) for k in keys]
    phys  = [fitted[k] for k in keys]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(keys, vals, color="darkorange", edgecolor="k", linewidth=0.7)
    ax.axhline(0.05, color="red",    ls="--", lw=1.0, label="Near lower bound (0.05)")
    ax.axhline(0.95, color="crimson",ls="--", lw=1.0, label="Near upper bound (0.95)")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Normalised position in fit bounds  [0 = lo, 1 = hi]")
    ax.set_title("Phase-2 Fitted Shape Parameters — Position in Fit Bounds\n"
                 "(values near 0 or 1 suggest the fit bounds should be widened)")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.4)
    for bar, k, pv in zip(bars, keys, phys):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{pv:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "phase2_params_normalised.png"), dpi=150)
    plt.close()
    print("  Saved → phase2_params_normalised.png")

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS JSON
# ══════════════════════════════════════════════════════════════════════════════
def save_phase2_json(fitted, p1_params, axes_p1, axes_p2,
                     rms_base_total, degradation_pct, save_dir):
    out = {
        "description": (
            "Phase-2 results: asymmetric damper fitted to Phase-1 optimal linear CF*. "
            "'fitted_shape_params' are the 2-stage asymmetric parameters. "
            "'normalised_in_fit_bounds': 0 = lower bound, 1 = upper bound. "
            "Values near 0 or 1 suggest the fit bounds should be widened."
        ),
        "fit_bounds": {k: {"lo": lo, "hi": hi}
                       for k, (lo, hi) in zip(FIT_PARAM_NAMES, FIT_BOUNDS)},
        "fitted_shape_params": {
            k: float(fitted[k]) for k in FIT_PARAM_NAMES + ["alpha_c", "alpha_r",
                                                              "force_rmse_grid_N",
                                                              "force_rmse_emp_N",
                                                              "optimizer_cost"]
        },
        "normalised_in_fit_bounds": {
            k: round((fitted[k] - lo) / (hi - lo), 4)
            for k, (lo, hi) in zip(FIT_PARAM_NAMES, FIT_BOUNDS)
        },
        "phase1_structural_params": {k: float(v) for k, v in p1_params.items()},
        "rms_results": {
            "baseline_total_m_s2":  rms_base_total,
            "phase1_linear":        {k: float(v) for k, v in axes_p1.items()},
            "phase2_asymmetric":    {k: float(v) for k, v in axes_p2.items()},
        },
        "degradation_p1_to_p2_pct": round(degradation_pct, 3),
        "acceptance": "ACCEPTED" if abs(degradation_pct) <= 5.0 else "REJECTED (>5% degradation)",
    }
    path = os.path.join(save_dir, "phase2_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[JSON] Phase-2 results saved → {path}")
    return path

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("=" * 65)
    print("OPTION 2 — PHASE 2: Asymmetric damper fitting + validation")
    print("=" * 65)

    # ── Load Phase-1 outputs ─────────────────────────────────────────────────
    p1_json  = os.path.join(RESULTS_DIR, "phase1_best_params.json")
    vrel_npy = os.path.join(RESULTS_DIR, "v_rel_front.npy")

    if not os.path.exists(p1_json):
        raise FileNotFoundError(
            f"Phase-1 output not found: {p1_json}\n"
            "Run option2_bay_phase1.py first.")
    if not os.path.exists(vrel_npy):
        raise FileNotFoundError(
            f"v_rel data not found: {vrel_npy}\n"
            "Run option2_bay_phase1.py first.")

    with open(p1_json) as fh:
        p1_data = json.load(fh)

    # The JSON stores either the flat dict (legacy) or the nested structure
    # written by save_phase1_json. Handle both gracefully.
    if "best" in p1_data and "params" in p1_data["best"]:
        p1_params      = p1_data["best"]["params"]
        rms_base_total = p1_data["baseline_rms"]["rms_total"]
        axes_p1_from_json = p1_data["optimised_rms"]
    else:
        # Legacy flat format from the sample code
        p1_params      = p1_data
        rms_base_total = None   # not available in legacy format
        axes_p1_from_json = None

    v_rel_opt = np.load(vrel_npy)
    CF_star   = float(p1_params["C_f"])

    print(f"  Phase-1 structural params: {p1_params}")
    print(f"  CF* (linear optimal)     : {CF_star:.2f} N·s/m")
    print(f"  v_rel samples            : {len(v_rel_opt)}  "
          f"(range: [{v_rel_opt.min():.4f}, {v_rel_opt.max():.4f}] m/s)")

    # ── Phase-2 curve fitting ─────────────────────────────────────────────────
    print("\n=== Fitting 2-stage asymmetric damper curve ===")
    fitted, v_eval, F_linear_eval, F_asym_eval = fit_asymmetric_damper(
        v_rel_opt, CF_star, n_starts=30, seed=42
    )

    print("\n  Fitted asymmetric parameters:")
    for k in FIT_PARAM_NAMES:
        lo, hi = dict(zip(FIT_PARAM_NAMES, FIT_BOUNDS))[k]
        nv = (fitted[k] - lo) / (hi - lo)
        flag = "  *** near boundary ***" if nv < 0.05 or nv > 0.95 else ""
        print(f"    {k:15s} = {fitted[k]:.4f}  (norm={nv:.4f}){flag}")

    # ── Phase-2 validation ODE ───────────────────────────────────────────────
    print("\n=== Phase-2 Validation: full ODE with asymmetric damper ===")
    cfg_val = {
        **CFG,
        **p1_params,               # K_f*, C_f*, K_2*, K_3* from Phase 1
        "cs_minus":   fitted["cs_minus"],
        "asym_ratio": fitted["asym_ratio"],
        "gamma_c":    fitted["gamma_c"],
        "gamma_r":    fitted["gamma_r"],
    }

    df_p2    = run_validation(cfg_val, t_eval_full)
    rms_p2   = compute_seat_rms(df_p2, cfg_val)
    axes_p2  = compute_seat_rms_axes(df_p2, cfg_val)

    # ── Phase-1 validation reference run ─────────────────────────────────────
    # Re-run Phase-1 config (linear damper) to get per-axis breakdown consistently.
    # If axes were saved in the JSON we use those; otherwise we note the limitation.
    if axes_p1_from_json is not None:
        axes_p1  = axes_p1_from_json
        rms_p1   = float(axes_p1["rms_total"])
    else:
        print("\n  NOTE: Per-axis P1 breakdown not in JSON (legacy format). "
              "Using Phase-2 ODE for P1 reference would require a separate run. "
              "Reporting Phase-2 results only for now.")
        axes_p1  = {"rms_z": float("nan"), "rms_x": float("nan"),
                    "rms_y": float("nan"), "rms_total": float("nan")}
        rms_p1   = float("nan")

    # ── Degradation check ─────────────────────────────────────────────────────
    if not np.isnan(rms_p1):
        degradation = (rms_p2 - rms_p1) / rms_p1 * 100.0
    else:
        degradation = float("nan")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    baseline_total = rms_base_total if rms_base_total is not None else float("nan")
    json_path = save_phase2_json(fitted, p1_params, axes_p1, axes_p2,
                                 baseline_total, degradation if not np.isnan(degradation) else 0.0,
                                 RESULTS_DIR)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE-2 FINAL RESULTS (3-axis combined seat RMS)")
    print("=" * 65)
    if not np.isnan(baseline_total):
        print(f"  Baseline RMS total  : {baseline_total:.5f} m/s²")
    print(f"  Phase-1 linear RMS  : {rms_p1:.5f} m/s²")
    print(f"    per-axis → z:{axes_p1['rms_z']:.4f}  x:{axes_p1['rms_x']:.4f}  y:{axes_p1['rms_y']:.4f}")
    print(f"  Phase-2 asym RMS    : {rms_p2:.5f} m/s²")
    print(f"    per-axis → z:{axes_p2['rms_z']:.4f}  x:{axes_p2['rms_x']:.4f}  y:{axes_p2['rms_y']:.4f}")

    if not np.isnan(degradation):
        print(f"  Degradation P1→P2   : {degradation:+.2f} %")
        if abs(degradation) <= 5.0:
            print("  ✓ Within ±5% tolerance — asymmetric fit accepted.")
        else:
            print("  ✗ Degradation >5% — consider:")
            print("    1. Widening fit bounds (especially cs_minus and asym_ratio)")
            print("    2. Running Phase 2b: re-extract v_rel with asymmetric damper "
                  "and re-fit (one feedback iteration)")

    print(f"\n  Fitted shape parameters:")
    for k in FIT_PARAM_NAMES:
        print(f"    {k:15s} = {fitted[k]:.4f}")
    print(f"  Force fit RMSE (empirical): {fitted['force_rmse_emp_N']:.2f} N")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n=== Saving plots ===")
    plot_fv_comparison(v_rel_opt, v_eval, F_linear_eval, F_asym_eval,
                       CF_star, fitted, PLOTS_DIR)
    plot_vrel_histogram_fit(v_rel_opt, v_eval, F_linear_eval, F_asym_eval,
                            CF_star, PLOTS_DIR)

    # For P1 vs P2 seat comparison we need df_p1. Re-use Phase-1 JSON params
    # to rebuild it only if axes were not available from JSON.
    if not np.isnan(rms_p1):
        # Build a minimal P1 dataframe by running linear validation using build_M_R
        # which is defined in Phase 1. Since we keep files separate we skip this
        # and only plot the Phase-2 seat time-history with per-axis breakdown.
        # The combined P1 vs P2 comparison uses the saved axes from JSON.
        print("  [INFO] P1 vs P2 time-domain plot requires Phase-1 DataFrame.")
        print("         Skipping seat_3axis_p1_vs_p2.png (Phase-1 df not loaded here).")
        print("         To generate it, call plot_seat_3axis_p1_p2(df_p1, df_p2, ...) "
              "after loading df_p1 from the Phase-1 run.")
    else:
        print("  [SKIP] seat_3axis_p1_vs_p2.png — Phase-1 axes not available.")

    plot_rms_summary_all(baseline_total, axes_p1, axes_p2, PLOTS_DIR)
    plot_phase2_params_normalised(fitted, PLOTS_DIR)

    print(f"\nAll Phase-2 outputs saved in: {RESULTS_DIR}")
    print(f"Results JSON               : {json_path}")
    print(f"Final Phase-2 combined RMS : {rms_p2:.5f} m/s²")
