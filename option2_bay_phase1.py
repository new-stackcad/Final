"""
OPTION 2 — BAYESIAN OPTIMISATION  (Phase 1 of 2)
=================================================
Optimises ONLY the 4 linear parameters:  KF, CF_linear, K2, K3
CF here is a plain scalar: F_damper = C_f * v_rel   (no asymmetric shape)

The front-damper relative velocity  v_f  is recorded at every time-step
and saved to  <RESULTS_DIR>/v_rel_front.npy  so Phase 2 can fit the
asymmetric curve to it.

After optimisation a full baseline + optimised run is executed and all
the same plots produced as before, plus a v_rel histogram.
"""

# ── std-lib / third-party ──────────────────────────────────────────────────
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import os
import time
from dataclasses import dataclass
from typing import Dict, Callable, Tuple
from numpy.linalg import solve as lin_solve
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL SIMULATION SETTINGS
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

# ══════════════════════════════════════════════════════════════════════════════
# BASE VEHICLE CONFIG
# (cs_minus / asym_ratio / gamma_c / gamma_r still kept for reference /
#  baseline plots but they are NOT part of the Phase-1 optimisation)
# ══════════════════════════════════════════════════════════════════════════════
CFG: Dict = {
    "axlefront_left_csv":  r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_1_FA_LH.csv",
    "axlefront_right_csv": r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_2_FA_RH.csv",
    "axlerear1_left_csv":  r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_3_RA1_LH.csv",
    "axlerear1_right_csv": r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_4_RA1_RH.csv",
    "axlerear2_left_csv":  r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_5_RA2_LH.csv",
    "axlerear2_right_csv": r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_6_RA2_RH.csv",

    "s1": 0.6277, "s2": 0.6305,
    "WT1": 0.814,  "WT2": 1.047,  "WT3": 1.047,
    "a": 0.9,      "b": 1.080,

    "m_s": 22485.0, "I_syy": 103787.0, "I_sxx": 8598.0, "I_sxy": 763.0,
    "M_1f": 600.0,  "M_2": 1075.0,     "M_3": 840.0,
    "I_xx1": 650.0, "I_xx2": 1200.0,   "I_xx3": 1100.0,

    "lf": 5.05, "L12": 0.54, "L23": 1.96,
    "l_cf": 6.458, "l_cr": 4.5, "l_cfcg": 0.871, "l_crcg": 1.087,

    "m_c": 862.0, "I_xxc": 516.6, "I_yyc": 1045.0,
    "hs": 0.68,   "g": 9.81,      "hcp": 0.1,

    "L_DL2": 0.6211, "L_DR2": 0.6211,
    "L_DL3": 0.6251, "L_DR3": 0.6251,
    "beta_L2": 0.1693, "beta_R2": 0.1693,
    "beta_L3": 0.17453, "beta_R3": 0.17453,
    "S_tf2": 1.043, "S_tf3": 1.043,
    "S_f":   0.814,

    "C_cfl": 5035.0, "C_cfr": 5035.0, "C_crl": 3400.0, "C_crr": 3400.0,
    "K_cfl": 49050.0,"K_cfr": 49050.0,"K_crl": 24525.0,"K_crr": 24525.0,

    # ── Parameters being optimised (Phase 1) ──
    "K_f": 474257, "C_f": 15000,
    "K_2": 1077620, "C_2": 2000,
    "K_3": 1077620, "C_3": 2000,

    # ── Asymmetric shape params (reference only – NOT used in Phase 1) ──
    "cs_minus":   0.3,
    "asym_ratio": 3.0,
    "gamma_c":    0.12,
    "gamma_r":    0.09,

    "baum_omega": 10.0,
    "baum_zeta":  1.0,
}

# ══════════════════════════════════════════════════════════════════════════════
# ROAD LOADING HELPERS  (unchanged from your original)
# ══════════════════════════════════════════════════════════════════════════════
def load_track(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, skiprows=2, header=None)
    t  = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
    z  = pd.to_numeric(df.iloc[:, 1], errors="coerce").values
    mask = np.isfinite(t) & np.isfinite(z)
    return t[mask].astype(float), z[mask].astype(float)


def make_linear_interp(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x); y = np.asarray(y)
    def f(xq):
        xq   = np.asarray(xq)
        xq_c = np.clip(xq, x[0], x[-1])
        idx  = np.clip(np.searchsorted(x, xq_c) - 1, 0, len(x) - 2)
        w    = (xq_c - x[idx]) / np.maximum(x[idx+1] - x[idx], 1e-12)
        return y[idx] * (1 - w) + y[idx+1] * w
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
# GEOMETRY + MASS MATRIX / RESIDUAL
# ══════════════════════════════════════════════════════════════════════════════
def geom_constraints(q, t, cfg, road):
    z_s, th_s, ph_s = q[ZS], q[THS], q[PHS]
    _, _, z2, ph2, z3, ph3 = road.axle_inputs(t, cfg)

    l2 = cfg["L12"]
    l3 = cfg["L12"] + cfg["L23"]
    S2, S3 = cfg["S_tf2"], cfg["S_tf3"]
    sl2, sl3 = cfg["s1"], cfg["s2"]
    bL2, bL3 = cfg["beta_L2"], cfg["beta_L3"]

    g2 = (z_s + l2*th_s + S2*ph_s
          - sl2*np.sin(bL2 - th_s)
          - (z2 + 0.5*cfg["WT2"]*ph2))
    g3 = (z_s + l3*th_s + S3*ph_s
          - sl3*np.sin(bL3 - th_s)
          - (z3 + 0.5*cfg["WT3"]*ph3))

    g = np.array([g2, g3], dtype=float)
    G = np.zeros((2, 6), dtype=float)
    G[0, ZS]  = 1.0
    G[0, THS] = l2 + sl2*np.cos(bL2 - th_s)
    G[0, PHS] = S2
    G[1, ZS]  = 1.0
    G[1, THS] = l3 + sl3*np.cos(bL3 - th_s)
    G[1, PHS] = S3
    return g, G


def build_M_R(q, v, t, cfg, road, record_vrel: list = None):
    """
    PHASE-1 VERSION
    ───────────────
    Front damper force = C_f * v_f   (pure linear — no asymmetric shape)

    If  record_vrel  is a list, v_f is appended to it so Phase 2 can use
    the empirical velocity distribution for curve fitting.
    """
    z_c, th_c, ph_c, z_s, th_s, ph_s = q
    dz_c, dth_c, dph_c, dz_s, dth_s, dph_s = v

    z1f, ph_f, z2, ph2, z3, ph3 = road.axle_inputs(t, cfg)
    dz1f, dph_f, dz2, dph2, dz3, dph3 = road.axle_input_rates(t, cfg)

    phi_NRS2 = (cfg["beta_L2"]*cfg["L_DL2"] - cfg["beta_R2"]*cfg["L_DR2"]) / max(cfg["S_tf2"], 1e-6)
    phi_NRS3 = (cfg["beta_L3"]*cfg["L_DL3"] - cfg["beta_R3"]*cfg["L_DR3"]) / max(cfg["S_tf3"], 1e-6)

    m_c, I_xxc, I_yyc = cfg["m_c"], cfg["I_xxc"], cfg["I_yyc"]
    m_s, I_sxx, I_syy, I_sxy = cfg["m_s"], cfg["I_sxx"], cfg["I_syy"], cfg["I_sxy"]
    S1, S2, S3 = cfg["S_f"], cfg["S_tf2"], cfg["S_tf3"]
    a, b     = cfg["a"], cfg["b"]
    hs, g    = cfg["hs"], cfg["g"]
    l_cfcg, l_crcg = cfg["l_cfcg"], cfg["l_crcg"]
    l_cf, l_cr     = cfg["l_cf"],   cfg["l_cr"]
    lf = cfg["lf"]
    hcp = cfg["hcp"]
    l2  = cfg["L12"]
    l3  = cfg["L12"] + cfg["L23"]
    beta_L2, beta_R2 = cfg["beta_L2"], cfg["beta_R2"]
    beta_L3, beta_R3 = cfg["beta_L3"], cfg["beta_R3"]
    L_DL2, L_DR2, L_DL3, L_DR3 = cfg["L_DL2"], cfg["L_DR2"], cfg["L_DL3"], cfg["L_DR3"]
    Kcfl, Kcfr, Kcrl, Kcrr = cfg["K_cfl"], cfg["K_cfr"], cfg["K_crl"], cfg["K_crr"]
    Ccfl, Ccfr, Ccrl, Ccrr = cfg["C_cfl"], cfg["C_cfr"], cfg["C_crl"], cfg["C_crr"]
    K_f, C_f = cfg["K_f"], cfg["C_f"]
    K_2, C_2 = cfg["K_2"], cfg["C_2"]
    K_3, C_3 = cfg["K_3"], cfg["C_3"]

    # ── PHASE 1: LINEAR front damper ─────────────────────────────────────────
    v_f  = dz_s - lf * dth_s - dz1f          # relative velocity at front mount
    F_df = C_f * v_f                          # <-- plain linear, NO asymmetric

    if record_vrel is not None:               # collect for Phase 2 fitting
        record_vrel.append(float(v_f))
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
        - (-Ccfl*l_cf  - Ccfr*l_cf  - Ccrl*l_cr   - Ccrr*l_cr)*dth_s
        - (-Ccfl*b + Ccfr*a - Ccrl*b + Ccrr*a)*dph_c
        - ( Ccfl*b - Ccfr*a + Ccrl*b - Ccrr*a)*dph_s
        - (Kcfl*l_cfcg + Kcfr*l_cfcg - Kcrl*l_crcg - Kcrr*l_crcg)*th_c
        - (-Kcfl*l_cf  - Kcfr*l_cf  - Kcrl*l_cr   - Kcrr*l_cr)*th_s
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
        - (Ccfl*l_cfcg*l_cf + Ccfr*l_cfcg*l_cf - Ccrl*l_crcg*l_cr - Ccrr*l_crcg*l_cr)*dth_s
        - (Kcfl*l_cf + Kcfr*l_cf + Kcrl*l_cr + Kcrr*l_cr)*z_c
        - (-Kcfl*l_cfcg*l_cf - Kcfr*l_cfcg*l_cf + Kcrl*l_crcg*l_cr + Kcrr*l_crcg*l_cr)*th_c
        - (-Kcfl*l_cf - Kcfr*l_cf - Kcrl*l_cr - Kcrr*l_cr)*z_s
        - (Kcfl*l_cf**2 + Kcfr*l_cf**2 + Kcrl*l_cr**2 + Kcrr*l_cr**2)*th_s
        - lf*(K_f*(z_s - lf*th_s - z1f) + F_df)
        + l2*(K_2*(z_s - z2 - beta_L2*L_DL2 - beta_R2*L_DR2 + l2*th_s) + C_2*(dz_s - dz2 + l2*dth_s))
        + l3*(K_3*(z_s - z3 - beta_L3*L_DL3 - beta_R3*L_DR3 + l3*th_s) + C_3*(dz_s - dz3 + l3*dth_s))
    )

    k_tf = 0.5 * K_f * S1**2
    K_r1 = 0.5 * K_2 * S2**2
    K_r2 = 0.5 * K_3 * S3**2
    C_tf = 0.5 * C_f * S1**2
    C_r1 = 0.5 * C_2 * S2**2
    C_r2 = 0.5 * C_3 * S3**2

    R[PHS] = (
        + m_s * g * hs * ph_s
        - k_tf*(ph_s - ph_f)  - C_tf*(dph_s - dph_f)
        - K_r1*(ph_s - ph2 - phi_NRS2) - C_r1*(dph_s - dph2)
        - K_r2*(ph_s - ph3 - phi_NRS3) - C_r2*(dph_s - dph3)
    )
    R[PHS] *= -1.0

    return M, R


# ══════════════════════════════════════════════════════════════════════════════
# ODE RHS
# ══════════════════════════════════════════════════════════════════════════════
def rhs_first_order(t, x, cfg, road, record_vrel=None):
    q, v = x[:6], x[6:]
    M, R = build_M_R(q, v, t, cfg, road, record_vrel=record_vrel)
    gq, G = geom_constraints(q, t, cfg, road)

    w, zeta = cfg["baum_omega"], cfg["baum_zeta"]
    gamma = w**2 * gq + 2*zeta*w*(G @ v)

    nc = G.shape[0]
    A  = np.zeros((6 + nc, 6 + nc))
    b  = np.zeros(6 + nc)
    A[:6, :6] = M;  A[:6, 6:] = G.T;  A[6:, :6] = G
    b[:6] = -R;     b[6:]     = -gamma

    sol  = lin_solve(A, b)
    xdot = np.zeros_like(x)
    xdot[:6] = v
    xdot[6:] = sol[:6]
    return xdot


# ══════════════════════════════════════════════════════════════════════════════
# STATIC EQUILIBRIUM  (unchanged from your original)
# ══════════════════════════════════════════════════════════════════════════════
def static_equilibrium_state(cfg, road):
    y0 = np.zeros(8, dtype=float)
    t0 = 0.0

    def F(y):
        q, lam = y[:6], y[6:]
        v = np.zeros(6)
        M, R   = build_M_R(q, v, t0, cfg, road)
        gq, G  = geom_constraints(q, t0, cfg, road)
        return np.hstack([R + G.T @ lam, 1e3 * gq])

    lsq = least_squares(F, y0, method="trf", loss="soft_l1",
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=800)

    if lsq.success:
        q0 = lsq.x[:6]
        g0, G0 = geom_constraints(q0, t0, cfg, road)
        M0, R0 = build_M_R(q0, np.zeros(6), t0, cfg, road)
        print("=== Static equilibrium OK. ||g||=%.3e, ||R+G^T*lam||=%.3e"
              % (np.linalg.norm(g0), np.linalg.norm(R0 + G0.T @ lsq.x[6:])))
        return np.hstack([q0, np.zeros(6)])

    print("=== Static equilibrium LSQ failed — trying dynamic relaxation...")
    cfg_relax = {**cfg,
                 "C_2": cfg["C_2"]*20, "C_3": cfg["C_3"]*20,
                 "C_cfl": cfg["C_cfl"]*20, "C_cfr": cfg["C_cfr"]*20,
                 "C_crl": cfg["C_crl"]*20, "C_crr": cfg["C_crr"]*20}
    sol_r = solve_ivp(lambda t, x: rhs_first_order(t, x, cfg_relax, road),
                      (0.0, 3.0), np.zeros(12), method="Radau", rtol=1e-7, atol=1e-9)
    q_r   = sol_r.y[:6, -1]
    lsq2  = least_squares(F, np.hstack([q_r, np.zeros(2)]), method="trf", loss="soft_l1",
                          xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
    q0    = lsq2.x[:6] if lsq2.success else q_r
    print("=== Dynamic relaxation end. ||g||=%.3e"
          % np.linalg.norm(geom_constraints(q0, t0, cfg, road)[0]))
    return np.hstack([q0, np.zeros(6)])


# ══════════════════════════════════════════════════════════════════════════════
# CORE SIMULATION  — returns (DataFrame, v_rel array)
# ══════════════════════════════════════════════════════════════════════════════
def run_one_case(params: Dict, cfg_base: Dict, t_eval: np.ndarray,
                 collect_vrel: bool = False):
    """
    Returns
    -------
    df      : pd.DataFrame  with time, states, velocities, accelerations
    v_rel   : np.ndarray    front-damper relative-velocity history (empty if
                            collect_vrel=False)
    """
    cfg  = {**cfg_base, **params}
    road = build_road_signals(cfg)
    x0   = static_equilibrium_state(cfg, road)

    vrel_buf = [] if collect_vrel else None

    print(f"\n=== Integrating | T_end={t_eval[-1]:.2f} s | dt={t_eval[1]-t_eval[0]:.4f} s")
    t_wall = time.time()

    sol = solve_ivp(
        fun=lambda t, x: rhs_first_order(t, x, cfg, road, record_vrel=vrel_buf),
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=x0, t_eval=t_eval,
        method="Radau", max_step=0.01, rtol=1e-6, atol=1e-8,
    )

    print(f"=== solve_ivp success={sol.success}, nfev={sol.nfev}, "
          f"wall={time.time()-t_wall:.1f} s")

    if sol.status != 0 or not np.all(np.isfinite(sol.y)):
        raise RuntimeError("ODE integration failed or diverged")

    rows = []
    for i, t in enumerate(sol.t):
        x = sol.y[:, i]
        qdd = rhs_first_order(t, x, cfg, road)[6:]
        row = {"t": t}
        for j, name in enumerate(STATE_NAMES):
            row[name]            = x[j]
            row[f"qd_{name}"]    = x[j+6]
            row[f"qdd_{name}"]   = qdd[j]
        rows.append(row)

    df    = pd.DataFrame(rows)
    v_rel = np.array(vrel_buf) if collect_vrel else np.array([])
    return df, v_rel


def compute_seat_rms(df: pd.DataFrame, cfg: Dict) -> float:
    mask   = df["t"] >= T_IGNORE
    a_seat = df.loc[mask, "qdd_z_s"] + cfg["hcp"] * df.loc[mask, "qdd_ph_s"]
    return float(np.sqrt(np.mean(a_seat**2)))


# ══════════════════════════════════════════════════════════════════════════════
# BAYESIAN OPTIMISATION OBJECTIVE  (Phase-1 — 4 variables only)
# ══════════════════════════════════════════════════════════════════════════════
def objective(K_f, C_f, K_2, K_3):
    """
    Only KF, CF (linear scalar), K2, K3 are optimised.
    Asymmetric damper parameters are NOT part of this search.
    """
    params = {"K_f": K_f, "C_f": C_f, "K_2": K_2, "K_3": K_3}
    df, _  = run_one_case(params, CFG, t_eval_full, collect_vrel=False)
    rms    = compute_seat_rms(df, CFG)
    print(f"  → RMS = {rms:.5f}")
    return -rms          # BayesOpt maximises


bounds = {
    "K_f": (0.8789 * CFG["K_f"], 1.1289 * CFG["K_f"]),
    "C_f": (0.80   * CFG["C_f"], 1.20   * CFG["C_f"]),
    "K_2": (0.8920 * CFG["K_2"], 1.1142 * CFG["K_2"]),
    "K_3": (0.8920 * CFG["K_3"], 1.1142 * CFG["K_3"]),
}


# ══════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS  (same as original + new v_rel histogram)
# ══════════════════════════════════════════════════════════════════════════════
def plot_road_inputs(road, cfg, t_eval, save_dir):
    z1, z2, z3 = [], [], []
    for t in t_eval:
        z1f, _, z2i, _, z3i, _ = road.axle_inputs(t, cfg)
        z1.append(z1f); z2.append(z2i); z3.append(z3i)
    plt.figure()
    plt.plot(t_eval, z1, label="Front axle")
    plt.plot(t_eval, z2, label="Rear axle 1")
    plt.plot(t_eval, z3, label="Rear axle 2")
    plt.xlabel("Time [s]"); plt.ylabel("Road displacement [m]")
    plt.title("Road / Axle Vertical Inputs"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(save_dir, "road_inputs.png")); plt.close()


def plot_seat_comparison(df_base, df_opt, cfg, base_rms, opt_rms, save_dir):
    t = df_base["t"]
    a_base = df_base["qdd_z_s"] + cfg["hcp"] * df_base["qdd_ph_s"]
    a_opt  = df_opt["qdd_z_s"]  + cfg["hcp"] * df_opt["qdd_ph_s"]
    plt.figure(figsize=(8, 4))
    plt.plot(t, a_base, label=f"Baseline  (RMS={base_rms:.3f})", alpha=0.8)
    plt.plot(t, a_opt,  label=f"Optimised (RMS={opt_rms:.3f})",  alpha=0.8)
    plt.xlabel("Time [s]"); plt.ylabel("Seat accel [m/s²]")
    plt.title("Seat / Cabin Acceleration — Comparison"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "seat_comparison.png")); plt.close()


def plot_convergence(optimizer, save_dir):
    best = np.minimum.accumulate([-r["target"] for r in optimizer.res])
    plt.figure()
    plt.plot(best)
    plt.xlabel("Iteration"); plt.ylabel("Best seat RMS [m/s²]")
    plt.title("Optimisation Convergence (Phase 1)"); plt.grid(True)
    plt.savefig(os.path.join(save_dir, "convergence_phase1.png")); plt.close()


def plot_vrel_distribution(v_rel: np.ndarray, CF_star: float, save_dir: str):
    """
    Histogram of front-damper relative velocities from the optimal run,
    overlaid with the linear damper force curve F = CF* · v.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # — left: velocity PDF
    axes[0].hist(v_rel, bins=120, density=True, color="steelblue", alpha=0.7)
    axes[0].set_xlabel("v_rel  [m/s]")
    axes[0].set_ylabel("Probability density")
    axes[0].set_title("Front-damper relative-velocity distribution\n(optimal linear run)")
    axes[0].axvline(0, color="k", lw=0.8, ls="--")
    axes[0].grid(True)

    # — right: F-v curve of the optimal linear damper
    v_axis = np.linspace(v_rel.min(), v_rel.max(), 400)
    F_lin  = CF_star * v_axis
    axes[1].plot(v_axis, F_lin, "r-", lw=2, label=f"F = {CF_star:.0f}·v")
    axes[1].set_xlabel("v_rel  [m/s]")
    axes[1].set_ylabel("Damper force  [N]")
    axes[1].set_title("Optimal linear damper F-v curve")
    axes[1].axhline(0, color="k", lw=0.8, ls="--")
    axes[1].axvline(0, color="k", lw=0.8, ls="--")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "vrel_distribution_phase1.png"))
    plt.close()
    print(f"  v_rel stats: mean={v_rel.mean():.4f}, std={v_rel.std():.4f}, "
          f"min={v_rel.min():.4f}, max={v_rel.max():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Phase-1 optimisation ─────────────────────────────────────────────────
    optimizer = BayesianOptimization(f=objective, pbounds=bounds, random_state=123)

    print("=" * 60)
    print("OPTION 2 — PHASE 1: optimising KF, CF_linear, K2, K3")
    print("=" * 60)
    optimizer.maximize(init_points=8, n_iter=30)

    best = optimizer.max["params"]
    CF_star = best["C_f"]

    print("\n=== Phase-1 best parameters ===")
    for k, v in best.items():
        print(f"  {k}: {v:.4f}")

    # ── Save best params so Phase 2 can load them without re-running ─────────
    import json
    with open(os.path.join(RESULTS_DIR, "phase1_best_params.json"), "w") as fh:
        json.dump(best, fh, indent=2)
    print(f"\nPhase-1 results saved → {RESULTS_DIR}/phase1_best_params.json")

    # ── Baseline simulation (original CFG, linear damper) ────────────────────
    print("\n=== Running baseline simulation ===")
    base_params = {"K_f": CFG["K_f"], "C_f": CFG["C_f"],
                   "K_2": CFG["K_2"], "K_3": CFG["K_3"]}
    df_base, _ = run_one_case(base_params, CFG, t_eval_full)
    rms_base   = compute_seat_rms(df_base, CFG)

    # ── Optimised simulation + collect v_rel for Phase 2 ────────────────────
    print("\n=== Running optimised simulation (collecting v_rel) ===")
    df_opt, v_rel_opt = run_one_case(best, CFG, t_eval_full, collect_vrel=True)
    rms_opt = compute_seat_rms(df_opt, CFG)

    # Save v_rel for Phase 2 curve fitting
    np.save(os.path.join(RESULTS_DIR, "v_rel_front.npy"), v_rel_opt)
    print(f"  v_rel saved ({len(v_rel_opt)} samples) → {RESULTS_DIR}/v_rel_front.npy")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n===== PHASE-1 FINAL RESULTS =====")
    print(f"  Baseline RMS  : {rms_base:.5f} m/s²")
    print(f"  Optimised RMS : {rms_opt:.5f} m/s²")
    print(f"  Improvement   : {(rms_base - rms_opt)/rms_base*100:.2f} %")
    print(f"\n  ► CF* (linear optimal) = {CF_star:.2f} N·s/m")
    print(f"  ► This value is the target for Phase-2 asymmetric fitting.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    road = build_road_signals(CFG)
    plot_road_inputs(road, CFG, t_eval_full, PLOTS_DIR)
    plot_seat_comparison(df_base, df_opt, CFG, rms_base, rms_opt, PLOTS_DIR)
    plot_convergence(optimizer, PLOTS_DIR)
    plot_vrel_distribution(v_rel_opt, CF_star, PLOTS_DIR)

    print(f"\nAll plots saved in: {PLOTS_DIR}")
    print("Run  option2_phase2_fit.py  next to fit the asymmetric damper.")
