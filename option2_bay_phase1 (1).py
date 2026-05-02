"""
OPTION 2 — PHASE 1 of 2: Bayesian Optimisation (Linear Damper)
===============================================================
Optimises only the 4 structural parameters:  K_f, C_f (linear), K_2, K_3
Front damper is treated as a pure linear element:  F = C_f * v_rel
(no asymmetric shape in this phase — that is fitted in Phase 2)

Objective: minimise 3-axis combined seat-point RMS acceleration
  RMS = sqrt( mean(z̈_c²) + mean(ẍ_seat²) + mean(ÿ_seat²) )
  where  ẍ_seat = -hcp * θ̈_c   (pitch)
         ÿ_seat =  hcp * φ̈_c   (roll)

After optimisation:
  • v_rel is reconstructed cleanly from sol.y at t_eval (not during integration)
  • Best params + normalised range position saved → phase1_best_params.json
  • v_rel_front.npy saved for Phase 2 curve fitting

Outputs (in RESULTS_DIR)
  phase1_best_params.json
  v_rel_front.npy
  plots/
    road_inputs.png
    seat_comparison_phase1.png
    convergence_phase1.png
    vrel_distribution_phase1.png
    phase1_params_normalised.png
    per_axis_rms_phase1.png
"""

# ── imports ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import json, os, time
import matplotlib.pyplot as plt
from bayes_opt         import BayesianOptimization
from dataclasses       import dataclass
from typing            import Dict, Callable, Tuple
from numpy.linalg      import solve as lin_solve
from scipy.integrate   import solve_ivp
from scipy.optimize    import least_squares

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION CONSTANTS
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

    "K_f": 474257, "C_f": 15000,
    "K_2": 1077620, "C_2": 2000,
    "K_3": 1077620, "C_3": 2000,

    # Asymmetric shape params kept for reference / baseline plots only.
    # They are NOT used in Phase-1 integration (damper is purely linear here).
    "cs_minus":   0.3,
    "asym_ratio": 3.0,
    "gamma_c":    0.12,
    "gamma_r":    0.09,

    "baum_omega": 10.0,
    "baum_zeta":  1.0,
}

# ══════════════════════════════════════════════════════════════════════════════
# OPTIMISATION BOUNDS
# ══════════════════════════════════════════════════════════════════════════════
# NOTE: C_f bounds are ±20% (conservative for linear damper search).
# This is intentionally tighter than Option-1 because the asymmetric
# shape tuning in Phase 2 provides additional freedom.
BOUNDS = {
    "K_f": (0.8789 * CFG["K_f"], 1.1289 * CFG["K_f"]),
    "C_f": (0.80   * CFG["C_f"], 1.20   * CFG["C_f"]),
    "K_2": (0.8920 * CFG["K_2"], 1.1142 * CFG["K_2"]),
    "K_3": (0.8920 * CFG["K_3"], 1.1142 * CFG["K_3"]),
}

# ══════════════════════════════════════════════════════════════════════════════
# ROAD LOADING
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

    l2 = cfg["L12"]
    l3 = cfg["L12"] + cfg["L23"]
    S2, S3   = cfg["S_tf2"], cfg["S_tf3"]
    sl2, sl3 = cfg["s1"],    cfg["s2"]
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
    G[0, THS] = l2 + sl2 * np.cos(bL2 - th_s)
    G[0, PHS] = S2
    G[1, ZS]  = 1.0
    G[1, THS] = l3 + sl3 * np.cos(bL3 - th_s)
    G[1, PHS] = S3
    return g, G

# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS: MASS MATRIX + RESIDUAL  (LINEAR front damper, Phase 1)
# Front damper: F_df = C_f * v_f   — pure linear, no asymmetric shape.
# ══════════════════════════════════════════════════════════════════════════════
def build_M_R(q, v, t, cfg, road):
    z_c, th_c, ph_c, z_s, th_s, ph_s     = q
    dz_c, dth_c, dph_c, dz_s, dth_s, dph_s = v

    z1f, ph_f, z2, ph2, z3, ph3           = road.axle_inputs(t, cfg)
    dz1f, dph_f, dz2, dph2, dz3, dph3    = road.axle_input_rates(t, cfg)

    phi_NRS2 = (cfg["beta_L2"]*cfg["L_DL2"] - cfg["beta_R2"]*cfg["L_DR2"]) / max(cfg["S_tf2"], 1e-6)
    phi_NRS3 = (cfg["beta_L3"]*cfg["L_DL3"] - cfg["beta_R3"]*cfg["L_DR3"]) / max(cfg["S_tf3"], 1e-6)

    m_c, I_xxc, I_yyc   = cfg["m_c"], cfg["I_xxc"], cfg["I_yyc"]
    m_s, I_sxx, I_syy, I_sxy = cfg["m_s"], cfg["I_sxx"], cfg["I_syy"], cfg["I_sxy"]
    S1, S2, S3           = cfg["S_f"],  cfg["S_tf2"], cfg["S_tf3"]
    a, b                 = cfg["a"],    cfg["b"]
    hs, g                = cfg["hs"],   cfg["g"]
    l_cfcg, l_crcg       = cfg["l_cfcg"], cfg["l_crcg"]
    l_cf, l_cr           = cfg["l_cf"],   cfg["l_cr"]
    lf                   = cfg["lf"]
    hcp                  = cfg["hcp"]
    l2                   = cfg["L12"]
    l3                   = cfg["L12"] + cfg["L23"]
    beta_L2, beta_R2     = cfg["beta_L2"], cfg["beta_R2"]
    beta_L3, beta_R3     = cfg["beta_L3"], cfg["beta_R3"]
    L_DL2, L_DR2         = cfg["L_DL2"], cfg["L_DR2"]
    L_DL3, L_DR3         = cfg["L_DL3"], cfg["L_DR3"]
    Kcfl, Kcfr, Kcrl, Kcrr = cfg["K_cfl"], cfg["K_cfr"], cfg["K_crl"], cfg["K_crr"]
    Ccfl, Ccfr, Ccrl, Ccrr = cfg["C_cfl"], cfg["C_cfr"], cfg["C_crl"], cfg["C_crr"]
    K_f, C_f = cfg["K_f"], cfg["C_f"]
    K_2, C_2 = cfg["K_2"], cfg["C_2"]
    K_3, C_3 = cfg["K_3"], cfg["C_3"]

    # ── LINEAR front damper (Phase 1) ────────────────────────────────────────
    v_f  = dz_s - lf * dth_s - dz1f      # relative velocity at front mount
    F_df = C_f * v_f                      # plain linear: no asymmetric shape
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

    k_tf = 0.5 * K_f * S1**2
    K_r1 = 0.5 * K_2 * S2**2
    K_r2 = 0.5 * K_3 * S3**2
    C_tf = 0.5 * C_f * S1**2
    C_r1 = 0.5 * C_2 * S2**2
    C_r2 = 0.5 * C_3 * S3**2

    R[PHS] = (
        + m_s * g * hs * ph_s
        - k_tf*(ph_s - ph_f) - C_tf*(dph_s - dph_f)
        - K_r1*(ph_s - ph2 - phi_NRS2) - C_r1*(dph_s - dph2)
        - K_r2*(ph_s - ph3 - phi_NRS3) - C_r2*(dph_s - dph3)
    )
    R[PHS] *= -1.0

    return M, R

# ══════════════════════════════════════════════════════════════════════════════
# ODE RHS
# ══════════════════════════════════════════════════════════════════════════════
def rhs_first_order(t, x, cfg, road):
    q, v = x[:6], x[6:]
    M, R  = build_M_R(q, v, t, cfg, road)
    gq, G = geom_constraints(q, t, cfg, road)

    w, zeta = cfg["baum_omega"], cfg["baum_zeta"]
    gamma   = w**2 * gq + 2 * zeta * w * (G @ v)

    nc = G.shape[0]
    A  = np.zeros((6 + nc, 6 + nc))
    b  = np.zeros(6 + nc)
    A[:6, :6] = M;  A[:6, 6:] = G.T;  A[6:, :6] = G
    b[:6] = -R;     b[6:]     = -gamma

    xdot      = np.zeros_like(x)
    xdot[:6]  = v
    xdot[6:]  = lin_solve(A, b)[:6]
    return xdot

# ══════════════════════════════════════════════════════════════════════════════
# STATIC EQUILIBRIUM
# ══════════════════════════════════════════════════════════════════════════════
def static_equilibrium_state(cfg, road):
    y0 = np.zeros(8, dtype=float)
    t0 = 0.0

    def F(y):
        q, lam = y[:6], y[6:]
        M, R   = build_M_R(q, np.zeros(6), t0, cfg, road)
        gq, G  = geom_constraints(q, t0, cfg, road)
        return np.hstack([R + G.T @ lam, 1e3 * gq])

    lsq = least_squares(F, y0, method="trf", loss="soft_l1",
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=800)

    if lsq.success:
        q0     = lsq.x[:6]
        g0, G0 = geom_constraints(q0, t0, cfg, road)
        M0, R0 = build_M_R(q0, np.zeros(6), t0, cfg, road)
        print("=== Static equilibrium OK. ||g||=%.3e, ||R+G^T*lam||=%.3e"
              % (np.linalg.norm(g0), np.linalg.norm(R0 + G0.T @ lsq.x[6:])))
        return np.hstack([q0, np.zeros(6)])

    print("=== Static equilibrium LSQ failed — trying dynamic relaxation...")
    cfg_r = {**cfg,
             "C_2": cfg["C_2"]*20, "C_3": cfg["C_3"]*20,
             "C_cfl": cfg["C_cfl"]*20, "C_cfr": cfg["C_cfr"]*20,
             "C_crl": cfg["C_crl"]*20, "C_crr": cfg["C_crr"]*20}
    sol_r = solve_ivp(lambda t, x: rhs_first_order(t, x, cfg_r, road),
                      (0.0, 3.0), np.zeros(12), method="Radau", rtol=1e-7, atol=1e-9)
    q_r   = sol_r.y[:6, -1]
    lsq2  = least_squares(F, np.hstack([q_r, np.zeros(2)]), method="trf", loss="soft_l1",
                          xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
    q0    = lsq2.x[:6] if lsq2.success else q_r
    print("=== Dynamic relaxation end. ||g||=%.3e"
          % np.linalg.norm(geom_constraints(q0, t0, cfg, road)[0]))
    return np.hstack([q0, np.zeros(6)])

# ══════════════════════════════════════════════════════════════════════════════
# CORE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
def run_one_case(params: Dict, cfg_base: Dict, t_eval: np.ndarray) -> pd.DataFrame:
    """
    Run one ODE integration and return a DataFrame of states + accelerations.
    v_rel is reconstructed POST-INTEGRATION from sol.y at t_eval points,
    so it is cleanly aligned with time and avoids internal-step contamination
    from the adaptive Radau solver.
    """
    cfg  = {**cfg_base, **params}
    road = build_road_signals(cfg)
    x0   = static_equilibrium_state(cfg, road)

    print(f"\n=== Integrating | T_end={t_eval[-1]:.2f} s | dt={t_eval[1]-t_eval[0]:.4f} s")
    t_wall = time.time()

    sol = solve_ivp(
        fun=lambda t, x: rhs_first_order(t, x, cfg, road),
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
        x   = sol.y[:, i]
        qdd = rhs_first_order(t, x, cfg, road)[6:]
        row = {"t": t}
        for j, name in enumerate(STATE_NAMES):
            row[name]          = x[j]
            row[f"qd_{name}"]  = x[j + 6]
            row[f"qdd_{name}"] = qdd[j]
        rows.append(row)

    return pd.DataFrame(rows)


def extract_vrel(df: pd.DataFrame, cfg: Dict, road: RoadSignals) -> np.ndarray:
    """
    Reconstruct front-damper relative velocity directly from the solution
    DataFrame at t_eval points.  This is more reliable than collecting
    v_f inside the ODE because the adaptive solver takes many internal
    steps between t_eval points; collecting inside gives a non-uniform,
    over-dense sample biased toward regions where the solver struggled.

    v_f(t) = dz_s(t) - lf * dth_s(t) - dz1f(t)
    """
    lf   = cfg["lf"]
    v_f  = (df["qd_z_s"].values
            - lf * df["qd_th_s"].values
            - np.array([road.f1L(t) * 0.5 + road.f1R(t) * 0.5
                        for t in df["t"].values]))
    # Road velocity term:  dz1f ≈ central diff already available via axle_input_rates,
    # but here we approximate it as the time-derivative of the interpolated road signal.
    # More accurate: use the rate function directly.
    dz1f = np.array([
        road.axle_input_rates(t, cfg)[0]   # index 0 = dz1f
        for t in df["t"].values
    ])
    v_f = df["qd_z_s"].values - lf * df["qd_th_s"].values - dz1f
    return v_f

# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE METRICS
# ══════════════════════════════════════════════════════════════════════════════
def compute_seat_rms(df: pd.DataFrame, cfg: Dict) -> float:
    """
    3-axis combined seat-point RMS acceleration (cabin body, not sprung mass).

      z̈_seat = qdd_z_c              (vertical bounce)
      ẍ_seat = -hcp * qdd_th_c      (longitudinal, from pitch)
      ÿ_seat =  hcp * qdd_ph_c      (lateral, from roll)

    RMS_total = sqrt( mean(z̈²) + mean(ẍ²) + mean(ÿ²) )
    """
    mask = df["t"] >= T_IGNORE
    h    = cfg["hcp"]
    az   = df.loc[mask, "qdd_z_c"].values
    ax   = -h * df.loc[mask, "qdd_th_c"].values
    ay   =  h * df.loc[mask, "qdd_ph_c"].values
    return float(np.sqrt(np.mean(az**2) + np.mean(ax**2) + np.mean(ay**2)))


def compute_seat_rms_axes(df: pd.DataFrame, cfg: Dict) -> Dict[str, float]:
    """Per-axis breakdown for diagnostics."""
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
# BAYESIAN OPTIMISATION OBJECTIVE
# ══════════════════════════════════════════════════════════════════════════════
def objective(K_f, C_f, K_2, K_3):
    params = {"K_f": K_f, "C_f": C_f, "K_2": K_2, "K_3": K_3}
    df     = run_one_case(params, CFG, t_eval_full)
    rms    = compute_seat_rms(df, CFG)
    print(f"  → RMS = {rms:.5f}")
    return -rms   # BayesOpt maximises

# ══════════════════════════════════════════════════════════════════════════════
# BEST-PARAM UTILITIES  (normalised selection, JSON save)
# ══════════════════════════════════════════════════════════════════════════════
def select_best_params(optimizer) -> Dict:
    """
    Returns best params augmented with normalised position in search range.
    Values near 0 or 1 indicate the optimum may lie outside current bounds.
    """
    best_res  = optimizer.max
    raw       = best_res["params"].copy()
    normalised = {
        k: round((v - BOUNDS[k][0]) / (BOUNDS[k][1] - BOUNDS[k][0]), 4)
        for k, v in raw.items()
    }
    return {
        "params":              raw,
        "normalised_in_range": normalised,
        "rms_combined":        float(-best_res["target"]),
    }


def save_phase1_json(best_info: Dict, rms_base: float, axes_base: Dict,
                     axes_opt: Dict) -> str:
    out = {
        "description": (
            "Phase-1 (linear damper) Bayesian optimisation results. "
            "'params' are physical values ready to pass to the ODE. "
            "'normalised_in_range': 0 = lower bound, 1 = upper bound. "
            "Values near 0 or 1 suggest the true optimum may lie outside "
            "the current search bounds — consider widening and re-running."
        ),
        "bounds": {k: {"lo": float(v[0]), "hi": float(v[1])} for k, v in BOUNDS.items()},
        "best": {
            "params":              {k: float(v) for k, v in best_info["params"].items()},
            "normalised_in_range": best_info["normalised_in_range"],
            "rms_combined_m_s2":   best_info["rms_combined"],
        },
        "baseline_rms": {
            "rms_z":     axes_base["rms_z"],
            "rms_x":     axes_base["rms_x"],
            "rms_y":     axes_base["rms_y"],
            "rms_total": axes_base["rms_total"],
        },
        "optimised_rms": {
            "rms_z":     axes_opt["rms_z"],
            "rms_x":     axes_opt["rms_x"],
            "rms_y":     axes_opt["rms_y"],
            "rms_total": axes_opt["rms_total"],
        },
        "improvement_pct": round(
            (axes_base["rms_total"] - axes_opt["rms_total"]) / axes_base["rms_total"] * 100, 3
        ),
        "CF_star_linear": float(best_info["params"]["C_f"]),
        "note_CF_star": (
            "CF_star is the optimal LINEAR damping coefficient. "
            "Phase-2 fits the 2-stage asymmetric curve to match this value "
            "over the actual velocity distribution seen during this run."
        ),
    }
    path = os.path.join(RESULTS_DIR, "phase1_best_params.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[JSON] Phase-1 params saved → {path}")
    return path

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
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
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "road_inputs.png"), dpi=150)
    plt.close()


def plot_seat_comparison_phase1(df_base, df_opt, cfg, rms_base, rms_opt, save_dir):
    """3-axis seat acceleration comparison — baseline vs Phase-1 linear optimal."""
    t = df_base["t"]
    h = cfg["hcp"]

    az_b = df_base["qdd_z_c"];          az_o = df_opt["qdd_z_c"]
    ax_b = -h * df_base["qdd_th_c"];    ax_o = -h * df_opt["qdd_th_c"]
    ay_b =  h * df_base["qdd_ph_c"];    ay_o =  h * df_opt["qdd_ph_c"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(t, az_b, alpha=0.75, lw=0.8, label=f"Baseline  (RMS_z={np.sqrt(np.mean(az_b[df_base['t']>=T_IGNORE]**2)):.4f})")
    axes[0].plot(t, az_o, alpha=0.75, lw=0.8, label=f"Phase-1 opt (RMS_z={np.sqrt(np.mean(az_o[df_opt['t']>=T_IGNORE]**2)):.4f})")
    axes[0].set_ylabel("z̈_seat [m/s²]")
    axes[0].set_title(f"Seat Vertical  |  Combined: Baseline={rms_base:.4f}, Opt={rms_opt:.4f} m/s²")
    axes[0].legend(fontsize=8); axes[0].grid(True)

    axes[1].plot(t, ax_b, alpha=0.75, lw=0.8, label="Baseline")
    axes[1].plot(t, ax_o, alpha=0.75, lw=0.8, label="Phase-1 opt")
    axes[1].set_ylabel("ẍ_seat = -h·θ̈_c [m/s²]")
    axes[1].set_title("Seat Longitudinal (pitch contribution)")
    axes[1].legend(fontsize=8); axes[1].grid(True)

    axes[2].plot(t, ay_b, alpha=0.75, lw=0.8, label="Baseline")
    axes[2].plot(t, ay_o, alpha=0.75, lw=0.8, label="Phase-1 opt")
    axes[2].set_ylabel("ÿ_seat = h·φ̈_c [m/s²]")
    axes[2].set_title("Seat Lateral (roll contribution)")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend(fontsize=8); axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "seat_comparison_phase1.png"), dpi=150)
    plt.close()
    print("  Saved → seat_comparison_phase1.png")


def plot_convergence(optimizer, save_dir):
    best = np.minimum.accumulate([-r["target"] for r in optimizer.res])
    plt.figure()
    plt.plot(best, marker="o", ms=3)
    plt.xlabel("Iteration"); plt.ylabel("Best combined seat RMS [m/s²]")
    plt.title("Bayesian Optimisation Convergence — Phase 1 (linear damper)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "convergence_phase1.png"), dpi=150)
    plt.close()
    print("  Saved → convergence_phase1.png")


def plot_vrel_distribution(v_rel: np.ndarray, CF_star: float, save_dir: str):
    """
    Velocity distribution of the optimal linear run — critical input for Phase 2.
    Left:  probability density of v_rel (shows where the damper actually operates)
    Right: F-v curve of the optimal linear damper with velocity percentile markers
    """
    p5,  p95  = np.percentile(v_rel, [5, 95])
    p25, p75  = np.percentile(v_rel, [25, 75])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # — velocity PDF
    axes[0].hist(v_rel, bins=120, density=True, color="steelblue", alpha=0.7, label="Empirical PDF")
    axes[0].axvline(0,   color="k",      lw=0.9, ls="--", label="v=0")
    axes[0].axvline(p5,  color="tomato", lw=0.9, ls=":",  label=f"5th pct = {p5:.3f} m/s")
    axes[0].axvline(p95, color="tomato", lw=0.9, ls=":",  label=f"95th pct = {p95:.3f} m/s")
    axes[0].set_xlabel("v_rel  [m/s]")
    axes[0].set_ylabel("Probability density")
    axes[0].set_title("Front-damper v_rel distribution\n(Phase-1 optimal run)")
    axes[0].legend(fontsize=8); axes[0].grid(True)

    # — F-v of optimal linear damper with coverage markers
    v_axis = np.linspace(v_rel.min() * 1.05, v_rel.max() * 1.05, 400)
    F_lin  = CF_star * v_axis
    axes[1].plot(v_axis, F_lin, "r-", lw=2.0, label=f"F = {CF_star:.0f}·v (linear opt)")
    axes[1].axvspan(p5, p95, alpha=0.12, color="steelblue", label="5–95th pct velocity range")
    axes[1].axvspan(p25, p75, alpha=0.20, color="steelblue", label="25–75th pct velocity range")
    axes[1].axhline(0, color="k", lw=0.8, ls="--")
    axes[1].axvline(0, color="k", lw=0.8, ls="--")
    axes[1].set_xlabel("v_rel  [m/s]")
    axes[1].set_ylabel("Damper force  [N]")
    axes[1].set_title(f"Optimal linear damper  (CF* = {CF_star:.0f} N·s/m)\nPhase-2 will fit asymmetric shape to this target")
    axes[1].legend(fontsize=8); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "vrel_distribution_phase1.png"), dpi=150)
    plt.close()
    print(f"  v_rel stats: mean={v_rel.mean():.4f}, std={v_rel.std():.4f}, "
          f"min={v_rel.min():.4f}, max={v_rel.max():.4f}")
    print("  Saved → vrel_distribution_phase1.png")


def plot_normalised_params(best_info: Dict, save_dir: str):
    """
    Bar chart: normalised position in search range [0=lower bound, 1=upper bound].
    Red/orange dashed lines flag boundary proximity — key diagnostic for bound adequacy.
    """
    keys = list(best_info["normalised_in_range"].keys())
    vals = [best_info["normalised_in_range"][k] for k in keys]
    phys = [best_info["params"][k]              for k in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(keys, vals, color="steelblue", edgecolor="k", linewidth=0.7)
    ax.axhline(0.05, color="red",    ls="--", lw=1.0, label="Near lower bound (0.05)")
    ax.axhline(0.95, color="orange", ls="--", lw=1.0, label="Near upper bound (0.95)")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Normalised position in search range  [0 = lo, 1 = hi]")
    ax.set_title("Phase-1 Optimal Parameters — Normalised Position in Search Bounds\n"
                 "(values near 0 or 1 suggest bounds should be widened)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.4)
    for bar, k, pv in zip(bars, keys, phys):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{pv:.4g}", ha="center", va="bottom", fontsize=8, rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "phase1_params_normalised.png"), dpi=150)
    plt.close()
    print("  Saved → phase1_params_normalised.png")


def plot_per_axis_rms(axes_base: Dict, axes_opt: Dict, save_dir: str):
    """Grouped bar chart: per-axis RMS for baseline vs Phase-1 optimised."""
    labels    = ["RMS_z\n(vertical)", "RMS_x\n(longitudinal)", "RMS_y\n(lateral)", "RMS_total"]
    keys      = ["rms_z", "rms_x", "rms_y", "rms_total"]
    base_vals = [axes_base[k] for k in keys]
    opt_vals  = [axes_opt[k]  for k in keys]

    x, w = np.arange(len(labels)), 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, base_vals, w, label="Baseline",  color="steelblue",  edgecolor="k")
    ax.bar(x + w/2, opt_vals,  w, label="Phase-1 opt", color="darkorange", edgecolor="k")
    for xi, bv, ov in zip(x, base_vals, opt_vals):
        ax.text(xi - w/2, bv + 0.0003, f"{bv:.4f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w/2, ov + 0.0003, f"{ov:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("RMS seat acceleration  [m/s²]")
    ax.set_title("Per-Axis Seat RMS — Baseline vs Phase-1 (Linear Optimised)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "per_axis_rms_phase1.png"), dpi=150)
    plt.close()
    print("  Saved → per_axis_rms_phase1.png")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("=" * 65)
    print("OPTION 2 — PHASE 1: Optimising K_f, C_f (linear), K_2, K_3")
    print("=" * 65)

    optimizer = BayesianOptimization(f=objective, pbounds=BOUNDS, random_state=123)
    optimizer.maximize(init_points=8, n_iter=30)

    # ── Best param extraction (normalised) ───────────────────────────────────
    best_info = select_best_params(optimizer)
    best      = best_info["params"]
    CF_star   = best["C_f"]

    print("\n=== Phase-1 best parameters (with normalised position in bounds) ===")
    for k, v in best.items():
        nv   = best_info["normalised_in_range"][k]
        flag = "  *** near boundary — consider widening bounds ***" if nv < 0.05 or nv > 0.95 else ""
        print(f"  {k:8s}: {v:.4f}  (norm={nv:.4f}){flag}")

    # ── Baseline run ─────────────────────────────────────────────────────────
    print("\n=== Baseline simulation ===")
    base_params = {"K_f": CFG["K_f"], "C_f": CFG["C_f"],
                   "K_2": CFG["K_2"], "K_3": CFG["K_3"]}
    df_base   = run_one_case(base_params, CFG, t_eval_full)
    rms_base  = compute_seat_rms(df_base, CFG)
    axes_base = compute_seat_rms_axes(df_base, CFG)

    # ── Optimal run + v_rel extraction ───────────────────────────────────────
    print("\n=== Optimal simulation (Phase-1) ===")
    df_opt   = run_one_case(best, CFG, t_eval_full)
    rms_opt  = compute_seat_rms(df_opt, CFG)
    axes_opt = compute_seat_rms_axes(df_opt, CFG)

    # Reconstruct v_rel post-integration: clean, t_eval-aligned, no solver bias
    road_opt = build_road_signals({**CFG, **best})
    v_rel_opt = extract_vrel(df_opt, {**CFG, **best}, road_opt)

    np.save(os.path.join(RESULTS_DIR, "v_rel_front.npy"), v_rel_opt)
    print(f"  v_rel saved ({len(v_rel_opt)} samples) → {RESULTS_DIR}/v_rel_front.npy")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = save_phase1_json(best_info, rms_base, axes_base, axes_opt)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE-1 FINAL RESULTS (3-axis combined seat RMS)")
    print("=" * 65)
    print(f"  Baseline  RMS total : {rms_base:.5f} m/s²")
    print(f"    per-axis → z:{axes_base['rms_z']:.4f}  x:{axes_base['rms_x']:.4f}  y:{axes_base['rms_y']:.4f}")
    print(f"  Phase-1   RMS total : {rms_opt:.5f} m/s²")
    print(f"    per-axis → z:{axes_opt['rms_z']:.4f}  x:{axes_opt['rms_x']:.4f}  y:{axes_opt['rms_y']:.4f}")
    print(f"  Improvement         : {(rms_base - rms_opt)/rms_base*100:.2f} %")
    print(f"\n  ► CF* (linear optimal) = {CF_star:.2f} N·s/m")
    print(f"  ► Pass this to Phase-2 to fit the asymmetric damper shape.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n=== Saving plots ===")
    road_base = build_road_signals(CFG)
    plot_road_inputs(road_base, CFG, t_eval_full, PLOTS_DIR)
    plot_seat_comparison_phase1(df_base, df_opt, CFG, rms_base, rms_opt, PLOTS_DIR)
    plot_convergence(optimizer, PLOTS_DIR)
    plot_vrel_distribution(v_rel_opt, CF_star, PLOTS_DIR)
    plot_normalised_params(best_info, PLOTS_DIR)
    plot_per_axis_rms(axes_base, axes_opt, PLOTS_DIR)

    print(f"\nAll Phase-1 outputs saved in: {RESULTS_DIR}")
    print("Run  option2_bay_phase2.py  next to fit the asymmetric damper.")
