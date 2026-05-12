"""
Seat Acceleration Simulation Runner
====================================
Run up to 4 parameter sets through the vehicle dynamics ODE and generate
a combined 4-panel time-history plot (z, x, y axes + combined RMS) for
each simulation, plus a side-by-side comparison chart.

Usage
-----
1. Edit the PARAM_SETS dict at the top of this file.
2. Run:  python simulation_runner.py
3. Plots land in:  ./sim_runner_results/plots/

Parameter sets included
-----------------------
  SIM_1  – "Axle Displacement Input"  (your named result)
  SIM_2  – "Best Total"               (Pareto best-total)
  SIM_3  – "Best Longitudinal"        (Pareto best longitudinal)
  SIM_4  – "[Reserved – add params]"  (placeholder, skipped if None)
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os
import time
import warnings
from typing import Dict, Optional, Tuple, Callable

# ---------------------------------------------------------------------------
# Numerical / scientific
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from dataclasses import dataclass
from numpy.linalg import solve as lin_solve
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# Matplotlib
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")

# ===========================================================================
# ── OUTPUT DIRECTORIES ──────────────────────────────────────────────────────
# ===========================================================================
RESULTS_DIR = "sim_runner_results"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ===========================================================================
# ── PARAMETER SETS  (edit here) ─────────────────────────────────────────────
# ===========================================================================
#  Set a value to None to skip that simulation slot.
#  Each dict must contain all 8 keys listed in PARAM_KEYS.

PARAM_SETS: Dict[str, Optional[Dict]] = {

    # ── SIM 1 : Axle Displacement Input ─────────────────────────────────────
    "Axle Displacement Input": {
        "K_f":        481103.5568709076,
        "C_f":        20644.515225616524,
        "K_2":        1138522.1932773169,
        "K_3":        1020694.4414728463,
        "cs_minus":   0.33048735315109,
        "asym_ratio": 3.8752837855131004,
        "gamma_c":    0.10428720969215391,
        "gamma_r":    0.08495047405752815,
    },

    # ── SIM 2 : Best Total ───────────────────────────────────────────────────
    "Best Total": {
        "K_f":        481103.5568709076,
        "C_f":        20644.515225616524,
        "K_2":        1138522.1932773169,
        "K_3":        1020694.4414728463,
        "cs_minus":   0.33048735315109,
        "asym_ratio": 3.8752837855131004,
        "gamma_c":    0.10428720969215391,
        "gamma_r":    0.08495047405752815,
        # ↑ Replace with your actual Best Total params from pareto_front.csv
    },

    # ── SIM 3 : Best Longitudinal ────────────────────────────────────────────
    "Best Longitudinal": {
        "K_f":        481103.5568709076,
        "C_f":        20644.515225616524,
        "K_2":        1138522.1932773169,
        "K_3":        1020694.4414728463,
        "cs_minus":   0.33048735315109,
        "asym_ratio": 3.8752837855131004,
        "gamma_c":    0.10428720969215391,
        "gamma_r":    0.08495047405752815,
        # ↑ Replace with your actual Best Longitudinal params from pareto_front.csv
    },

    # ── SIM 4 : Reserved – paste your fourth param set here ─────────────────
    "Custom (Reserved)": None,   # ← Replace None with a params dict to activate
}

# ===========================================================================
# ── SIMULATION SETTINGS  (match your original script) ───────────────────────
# ===========================================================================
DT       = 0.001
FS       = 1000
T_IGNORE = 0.5
T_END    = 466.945

t_eval_full = np.arange(0.0, T_END + DT, DT)

STATE_NAMES = ["z_c", "th_c", "ph_c", "z_s", "th_s", "ph_s"]
(ZC, THC, PHC, ZS, THS, PHS) = range(6)

PARAM_KEYS = ["K_f", "C_f", "K_2", "K_3", "cs_minus", "asym_ratio", "gamma_c", "gamma_r"]

# ===========================================================================
# ── VEHICLE CONFIGURATION  (unchanged from original) ────────────────────────
# ===========================================================================
CFG: Dict = {
    # ── Axle CSV paths – update to match your machine ──────────────────────
    "axlefront_left_csv":  r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Final_ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_1_FA_LH.csv",
    "axlefront_right_csv": r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Final_ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_2_FA_RH.csv",
    "axlerear1_left_csv":  r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Final_ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_3_RA1_LH.csv",
    "axlerear1_right_csv": r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Final_ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_4_RA1_RH.csv",
    "axlerear2_left_csv":  r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Final_ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_5_RA2_LH.csv",
    "axlerear2_right_csv": r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Final_ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_6_RA2_RH.csv",

    "s1": 0.6277, "s2": 0.6305,
    "WT1": 0.814,  "WT2": 1.047,  "WT3": 1.047,

    "m_c": 862.0,   "I_xxc": 516.6,  "I_yyc": 1045.0,
    "M_1f": 600.0,  "M_2": 1075.0,   "M_3": 840.0,
    "I_xx1": 650.0, "I_xx2": 1200.0, "I_xx3": 1100.0,

    "S_tf2": 1.043, "S_tf3": 1.043,
    "S_f":   0.814,

    "C_cfl": 5035.0, "C_cfr": 5035.0, "C_crl": 3400.0, "C_crr": 3400.0,
    "K_cfl": 49050.0,"K_cfr": 49050.0,"K_crl": 24525.0,"K_crr": 24525.0,
    "C_2": 2000,     "C_3": 2000,

    "L_DL2": 0.6211, "L_DR2": 0.6211,
    "L_DL3": 0.6251, "L_DR3": 0.6251,
    "beta_L2": 0.1693,  "beta_R2": 0.1693,
    "beta_L3": 0.17453, "beta_R3": 0.17453,

    "a": 0.9,  "b": 1.080,
    "l_cfcg": 0.871, "l_crcg": 1.087,
    "hcp": 0.1,

    "lf": 5.05, "L12": 0.54, "L23": 1.96,
    "l_cf": 6.458, "l_cr": 4.5,

    "m_s": 22485.0, "I_syy": 103787.0, "I_sxx": 8598.0, "I_sxy": 763.0,
    "hs": 0.68,

    # Baseline damper / stiffness values
    "K_f": 474257,   "C_f": 15000,
    "K_2": 1077620,  "K_3": 1077620,

    "g": 9.81,

    "cs_minus": 0.3, "asym_ratio": 3.0,
    "gamma_c": 0.12, "gamma_r": 0.09,

    # Baumgarte stabilisation
    "baum_omega": 10.0, "baum_zeta": 1.0,
}

# ===========================================================================
# ── PHYSICS  (unchanged from original) ──────────────────────────────────────
# ===========================================================================

@dataclass
class TwoStageAsymmetricDamper:
    cs_minus:   float
    asym_ratio: float
    gamma_c:    float
    gamma_r:    float
    alpha_c:    float = -0.05
    alpha_r:    float =  0.13

    def force(self, v_rel: float) -> float:
        c_plus = self.asym_ratio * self.cs_minus
        if v_rel < 0.0:
            if v_rel >= self.alpha_c:
                return self.cs_minus * v_rel
            else:
                return self.cs_minus * (self.alpha_c + self.gamma_c * (v_rel - self.alpha_c))
        else:
            if v_rel <= self.alpha_r:
                return c_plus * v_rel
            else:
                return c_plus * (self.alpha_r + self.gamma_r * (v_rel - self.alpha_r))


def load_track(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df   = pd.read_csv(csv_path, skiprows=2, header=None)
    t    = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
    z    = pd.to_numeric(df.iloc[:, 1], errors="coerce").values
    mask = np.isfinite(t) & np.isfinite(z)
    return t[mask].astype(float), z[mask].astype(float)


def make_linear_interp(x: np.ndarray, y: np.ndarray) -> Callable:
    x = np.asarray(x); y = np.asarray(y)
    def f(xq):
        xq   = np.asarray(xq)
        xq_c = np.clip(xq, x[0], x[-1])
        idx  = np.clip(np.searchsorted(x, xq_c) - 1, 0, len(x) - 2)
        x0, x1 = x[idx], x[idx + 1]
        y0, y1 = y[idx], y[idx + 1]
        w = (xq_c - x0) / np.maximum(x1 - x0, 1e-12)
        return y0 * (1 - w) + y1 * w
    return f


@dataclass
class RoadSignals:
    f1L: Callable; f1R: Callable
    f2L: Callable; f2R: Callable
    f3L: Callable; f3R: Callable

    def axle_inputs(self, t, cfg):
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

    def axle_input_rates(self, t, cfg, dt=DT):
        p = self.axle_inputs(t + dt, cfg)
        m = self.axle_inputs(t - dt, cfg)
        return tuple((a - b) / (2.0 * dt) for a, b in zip(p, m))


def build_road_signals(cfg) -> RoadSignals:
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


def geom_constraints(q, t, cfg, road):
    z_s, th_s, ph_s = q[ZS], q[THS], q[PHS]
    _, _, z2, ph2, z3, ph3 = road.axle_inputs(t, cfg)
    l2 = cfg["L12"];  l3 = cfg["L12"] + cfg["L23"]
    S2, S3   = cfg["S_tf2"], cfg["S_tf3"]
    sl2, sl3 = cfg["s1"],    cfg["s2"]
    bL2, bL3 = cfg["beta_L2"], cfg["beta_L3"]

    g2 = z_s + l2*th_s + S2*ph_s - sl2*np.sin(bL2 - th_s) - (z2 + 0.5*cfg["WT2"]*ph2)
    g3 = z_s + l3*th_s + S3*ph_s - sl3*np.sin(bL3 - th_s) - (z3 + 0.5*cfg["WT3"]*ph3)

    gq = np.array([g2, g3], dtype=float)
    G  = np.zeros((2, 6), dtype=float)
    G[0, ZS]  = 1.0; G[0, THS] = l2 + sl2*np.cos(bL2 - th_s); G[0, PHS] = S2
    G[1, ZS]  = 1.0; G[1, THS] = l3 + sl3*np.cos(bL3 - th_s); G[1, PHS] = S3
    return gq, G


def build_M_R(q, v, t, cfg, road):
    z_c, th_c, ph_c, z_s, th_s, ph_s = q
    dz_c, dth_c, dph_c, dz_s, dth_s, dph_s = v

    z1f, ph_f, z2, ph2, z3, ph3 = road.axle_inputs(t, cfg)
    dz1f, dph_f, dz2, dph2, dz3, dph3 = road.axle_input_rates(t, cfg)

    phi_NRS2 = (cfg["beta_L2"]*cfg["L_DL2"] - cfg["beta_R2"]*cfg["L_DR2"]) / max(cfg["S_tf2"], 1e-6)
    phi_NRS3 = (cfg["beta_L3"]*cfg["L_DL3"] - cfg["beta_R3"]*cfg["L_DR3"]) / max(cfg["S_tf3"], 1e-6)

    m_c, I_xxc, I_yyc  = cfg["m_c"], cfg["I_xxc"], cfg["I_yyc"]
    m_s, I_sxx, I_syy, I_sxy = cfg["m_s"], cfg["I_sxx"], cfg["I_syy"], cfg["I_sxy"]
    S1, S2, S3 = cfg["S_f"], cfg["S_tf2"], cfg["S_tf3"]
    a, b       = cfg["a"],   cfg["b"]
    hs, g      = cfg["hs"],  cfg["g"]
    l_cfcg, l_crcg, l_cf, l_cr = cfg["l_cfcg"], cfg["l_crcg"], cfg["l_cf"], cfg["l_cr"]
    lf   = cfg["lf"];  hcp = cfg["hcp"]
    l2   = cfg["L12"]; l3  = cfg["L12"] + cfg["L23"]
    beta_L2, beta_R2 = cfg["beta_L2"], cfg["beta_R2"]
    beta_L3, beta_R3 = cfg["beta_L3"], cfg["beta_R3"]
    L_DL2, L_DR2, L_DL3, L_DR3 = cfg["L_DL2"], cfg["L_DR2"], cfg["L_DL3"], cfg["L_DR3"]
    Kcfl,Kcfr,Kcrl,Kcrr = cfg["K_cfl"], cfg["K_cfr"], cfg["K_crl"], cfg["K_crr"]
    Ccfl,Ccfr,Ccrl,Ccrr = cfg["C_cfl"], cfg["C_cfr"], cfg["C_crl"], cfg["C_crr"]
    K_f, C_f = cfg["K_f"], cfg["C_f"]
    K_2, C_2 = cfg["K_2"], cfg["C_2"]
    K_3, C_3 = cfg["K_3"], cfg["C_3"]

    M = np.zeros((6, 6))
    M[ZC, ZC]   = m_c;   M[THC, THC] = I_yyc; M[PHC, PHC] = I_xxc
    M[ZS, ZS]   = m_s;   M[THS, THS] = I_syy
    M[THS, PHS] = I_sxy; M[PHS, THS] = I_sxy
    M[PHS, PHS] = I_sxx + m_s * hs**2

    damp = TwoStageAsymmetricDamper(
        cs_minus=cfg["cs_minus"], asym_ratio=cfg["asym_ratio"],
        gamma_c=cfg["gamma_c"],  gamma_r=cfg["gamma_r"],
        alpha_c=-0.05, alpha_r=0.13,
    )
    v_f  = dz_s - lf * dth_s - dz1f
    F_df = C_f * damp.force(v_f)

    Csum = Ccfl + Ccfr + Ccrl + Ccrr
    Ksum = Kcfl + Kcfr + Kcrl + Kcrr

    R = np.zeros(6)

    R[ZC] = (
        Csum*(dz_c-dz_s) + Ksum*(z_c-z_s)
        - (Ccfl*l_cfcg+Ccfr*l_cfcg-Ccrl*l_crcg-Ccrr*l_crcg)*dth_c
        - (-Ccfl*l_cf-Ccfr*l_cf-Ccrl*l_cr-Ccrr*l_cr)*dth_s
        - (-Ccfl*b+Ccfr*a-Ccrl*b+Ccrr*a)*dph_c
        - (Ccfl*b-Ccfr*a+Ccrl*b-Ccrr*a)*dph_s
        - (Kcfl*l_cfcg+Kcfr*l_cfcg-Kcrl*l_crcg-Kcrr*l_crcg)*th_c
        - (-Kcfl*l_cf-Kcfr*l_cf-Kcrl*l_cr-Kcrr*l_cr)*th_s
        - (-Kcfl*b+Kcfr*a-Kcrl*b+Kcrr*a)*ph_c
        - (Kcfl*b-Kcfr*a+Kcrl*b-Kcrr*a)*ph_s
    )
    R[THC] = (
        -(Ccfl*l_cfcg+Ccfr*l_cfcg-Ccrl*l_crcg-Ccrr*l_crcg)*dz_c
        -(-Ccfl*l_cfcg-Ccfr*l_cfcg-Ccrl*l_crcg-Ccrr*l_crcg)*dz_s
        -(Kcfl*l_cfcg+Kcfr*l_cfcg-Kcrl*l_crcg-Kcrr*l_crcg)*z_c
        -(-Kcfl*l_cfcg-Kcfr*l_cfcg-Kcrl*l_crcg-Kcrr*l_crcg)*z_s
        -(-Ccfl*l_cfcg**2-Ccfr*l_cfcg**2-Ccrl*l_crcg**2-Ccrr*l_crcg**2)*dth_c
        -(Ccfl*l_cfcg*l_cf+Ccfr*l_cfcg*l_cf-Ccrl*l_crcg*l_cr-Ccrr*l_crcg*l_cr)*dth_s
        -(-Ccfl*l_cfcg*b+Ccfr*l_cfcg*a-Ccrl*l_crcg*b+Ccrr*l_crcg*a)*dph_c
        -(Ccfl*l_cfcg*b-Ccfr*l_cfcg*a+Ccrl*l_crcg*b-Ccrr*l_crcg*a)*dph_s
        -(-Kcfl*l_cfcg**2-Kcfr*l_cfcg**2-Kcrl*l_crcg**2-Kcrr*l_crcg**2+m_c*g*hcp)*th_c
        -(Kcfl*l_cfcg*l_cf+Kcfr*l_cfcg*l_cf-Kcrl*l_crcg*l_cr-Kcrr*l_crcg*l_cr)*th_s
        -(-Kcfl*l_cfcg*b+Kcfr*l_cfcg*a-Kcrl*l_crcg*b+Kcrr*l_crcg*a)*ph_c
        -(Kcfl*l_cfcg*b-Kcfr*l_cfcg*a+Kcrl*l_crcg*b-Kcrr*l_crcg*a)*ph_s
    )
    R[PHC] = (
        -(-Ccfl*b+Ccfr*a-Ccrl*b+Ccrr*a)*dz_c
        -(Ccfl*b-Ccfr*a+Ccrl*b-Ccrr*a)*dz_s
        -(-Kcfl*b+Kcfr*a-Kcrl*b+Kcrr*a)*z_c
        -(Kcfl*b-Kcfr*a+Kcrl*b-Kcrr*a)*z_s
        -(-Ccfl*l_cfcg*b-Ccfr*l_cfcg*a+Ccrl*l_crcg*b+Ccrr*l_crcg*a)*dth_c
        -(Ccfl*l_cfcg*b+Ccfr*l_cfcg*a-Ccrl*l_crcg*b-Ccrr*l_crcg*a)*dth_s
        -(-Ccfl*b**2+Ccfr*a**2-Ccrl*b**2+Ccrr*a**2)*dph_c
        -(Ccfl*b**2-Ccfr*a**2+Ccrl*b**2-Ccrr*a**2)*dph_s
        -(-Kcfl*l_cfcg*b-Kcfr*l_cfcg*a+Kcrl*l_crcg*b+Kcrr*l_crcg*a)*th_c
        -(Kcfl*l_cfcg*b+Kcfr*l_cfcg*a-Kcrl*l_crcg*b-Kcrr*l_crcg*a)*th_s
        -(-Kcfl*b**2+Kcfr*a**2-Kcrl*b**2+Kcrr*a**2)*ph_c
        -(Kcfl*b**2-Kcfr*a**2+Kcrl*b**2-Kcrr*a**2)*ph_s
    )
    R[ZS] = (
        -(Ccfl+Ccfr+Ccrl+Ccrr)*dz_c
        -(-Ccfl*l_cfcg-Ccfr*l_cfcg+Ccrl*l_crcg+Ccrr*l_crcg)*dth_c
        -(-Ccfl-Ccfr-Ccrl-Ccrr)*dz_s
        -(Ccfl*l_cf+Ccfr*l_cf+Ccrl*l_cr+Ccrr*l_cr)*dth_s
        -(Kcfl+Kcfr+Kcrl+Kcrr)*z_c
        -(-Kcfl*l_cfcg-Kcfr*l_cfcg+Kcrl*l_crcg+Kcrr*l_crcg)*th_c
        -(-Kcfl-Kcfr-Kcrl-Kcrr)*z_s
        -(Kcfl*l_cf+Kcfr*l_cf+Kcrl*l_cr+Kcrr*l_cr)*th_s
        + K_f*(z_s-lf*th_s-z1f) + F_df
        + K_2*(z_s-z2-cfg["beta_L2"]*cfg["L_DL2"]-cfg["beta_R2"]*cfg["L_DR2"]+l2*th_s) + C_2*(dz_s-dz2+l2*dth_s)
        + K_3*(z_s-z3-cfg["beta_L3"]*cfg["L_DL3"]-cfg["beta_R3"]*cfg["L_DR3"]+l3*th_s) + C_3*(dz_s-dz3+l3*dth_s)
    )
    R[THS] = (
        -(Ccfl*l_cfcg+Ccfr*l_cfcg-Ccrl*l_crcg-Ccrr*l_crcg)*dz_c
        -(-Ccfl*l_cfcg**2-Ccfr*l_cfcg**2-Ccrl*l_crcg**2-Ccrr*l_crcg**2)*dth_c
        -(-Ccfl*l_cf-Ccfr*l_cf-Ccrl*l_cr-Ccrr*l_cr)*dz_s
        -(Ccfl*l_cfcg*l_cf+Ccfr*l_cfcg*l_cf-Ccrl*l_crcg*l_cr-Ccrr*l_crcg*l_cr)*dth_s
        -(Kcfl*l_cf+Kcfr*l_cf+Kcrl*l_cr+Kcrr*l_cr)*z_c
        -(-Kcfl*l_cfcg*l_cf-Kcfr*l_cfcg*l_cf+Kcrl*l_crcg*l_cr+Kcrr*l_crcg*l_cr)*th_c
        -(-Kcfl*l_cf-Kcfr*l_cf-Kcrl*l_cr-Kcrr*l_cr)*z_s
        -(Kcfl*l_cf**2+Kcfr*l_cf**2+Kcrl*l_cr**2+Kcrr*l_cr**2)*th_s
        - lf*(K_f*(z_s-lf*th_s-z1f) + F_df)
        + l2*(K_2*(z_s-z2-beta_L2*L_DL2-beta_R2*L_DR2+l2*th_s) + C_2*(dz_s-dz2+l2*dth_s))
        + l3*(K_3*(z_s-z3-beta_L3*L_DL3-beta_R3*L_DR3+l3*th_s) + C_3*(dz_s-dz3+l3*dth_s))
    )
    k_tf = 0.5*K_f*S1**2; K_r1 = 0.5*K_2*S2**2; K_r2 = 0.5*K_3*S3**2
    C_tf = 0.5*C_f*S1**2; C_r1 = 0.5*C_2*S2**2; C_r2 = 0.5*C_3*S3**2
    R[PHS] = -(
        m_s*g*hs*ph_s
        - k_tf*(ph_s-ph_f) - C_tf*(dph_s-dph_f)
        - K_r1*(ph_s-ph2-phi_NRS2) - C_r1*(dph_s-dph2)
        - K_r2*(ph_s-ph3-phi_NRS3) - C_r2*(dph_s-dph3)
    )
    return M, R


def rhs_first_order(t, x, cfg, road):
    q, v = x[:6], x[6:]
    M, R = build_M_R(q, v, t, cfg, road)
    gq, G = geom_constraints(q, t, cfg, road)
    w, zeta = cfg["baum_omega"], cfg["baum_zeta"]
    gamma = w**2 * gq + 2*zeta*w * (G @ v)
    nc = G.shape[0]
    A = np.zeros((6 + nc, 6 + nc))
    b_vec = np.zeros(6 + nc)
    A[:6, :6] = M; A[:6, 6:] = G.T; A[6:, :6] = G
    b_vec[:6] = -R; b_vec[6:] = -gamma
    sol = lin_solve(A, b_vec)
    xdot = np.zeros_like(x)
    xdot[:6] = v; xdot[6:] = sol[:6]
    return xdot


def static_equilibrium_state(cfg, road):
    t0 = 0.0

    def F(y):
        q, lam = y[:6], y[6:]
        M, R   = build_M_R(q, np.zeros(6), t0, cfg, road)
        gq, G  = geom_constraints(q, t0, cfg, road)
        return np.hstack([R + G.T @ lam, 1e3 * gq])

    lsq = least_squares(F, np.zeros(8), method="trf", loss="soft_l1",
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=800)
    if lsq.success:
        q0 = lsq.x[:6]
        return np.hstack([q0, np.zeros(6)])

    cfg_r = dict(cfg)
    for k in ["C_2", "C_3", "C_cfl", "C_cfr", "C_crl", "C_crr"]:
        cfg_r[k] *= 20
    sol = solve_ivp(lambda t, x: rhs_first_order(t, x, cfg_r, road),
                    (0.0, 3.0), np.zeros(12), method="Radau", rtol=1e-7, atol=1e-9)
    q0 = sol.y[:6, -1]
    return np.hstack([q0, np.zeros(6)])


# ===========================================================================
# ── SIMULATION WRAPPER ───────────────────────────────────────────────────────
# ===========================================================================

def run_one_case(params: Dict, t_eval: np.ndarray) -> pd.DataFrame:
    cfg  = {**CFG, **params}
    road = build_road_signals(cfg)
    x0   = static_equilibrium_state(cfg, road)
    sol  = solve_ivp(
        fun=lambda t, x: rhs_first_order(t, x, cfg, road),
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=x0, t_eval=t_eval,
        method="Radau", max_step=0.01, rtol=1e-6, atol=1e-8,
    )
    if sol.status != 0 or not np.all(np.isfinite(sol.y)):
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    rows = []
    for i, t in enumerate(sol.t):
        x = sol.y[:, i]; q = x[:6]; v = x[6:]
        qdd = rhs_first_order(t, x, cfg, road)[6:]
        row = {"t": t}
        for j, name in enumerate(STATE_NAMES):
            row[name] = q[j]; row[f"qd_{name}"] = v[j]; row[f"qdd_{name}"] = qdd[j]
        rows.append(row)
    return pd.DataFrame(rows)


def compute_rms(df: pd.DataFrame) -> Tuple[float, float, float, float]:
    """Returns (rms_z, rms_x, rms_y, rms_total)."""
    mask = df["t"] >= T_IGNORE
    h    = CFG["hcp"]
    az   = df.loc[mask, "qdd_z_c"].values
    ax   = -h * df.loc[mask, "qdd_th_c"].values
    ay   =  h * df.loc[mask, "qdd_ph_c"].values
    rz   = float(np.sqrt(np.mean(az**2)))
    rx   = float(np.sqrt(np.mean(ax**2)))
    ry   = float(np.sqrt(np.mean(ay**2)))
    rt   = float(np.sqrt(np.mean(az**2) + np.mean(ax**2) + np.mean(ay**2)))
    return rz, rx, ry, rt


# ===========================================================================
# ── PLOTTING ─────────────────────────────────────────────────────────────────
# ===========================================================================

# Colour palette: one accent per simulation slot
SIM_COLORS = {
    0: {"z": "#2E86AB", "x": "#E07B39", "y": "#3BB273", "comb": "#9B5DE5"},
    1: {"z": "#1B4F72", "x": "#CA6F1E", "y": "#1D8348", "comb": "#6C3483"},
    2: {"z": "#5DADE2", "x": "#F0A500", "y": "#52BE80", "comb": "#C39BD3"},
    3: {"z": "#85C1E9", "x": "#FAD7A0", "y": "#A9DFBF", "comb": "#E8DAEF"},
}


def _make_combined_signal(df: pd.DataFrame) -> np.ndarray:
    """Instantaneous combined acceleration magnitude at seat."""
    h  = CFG["hcp"]
    az = df["qdd_z_c"].values
    ax = -h * df["qdd_th_c"].values
    ay =  h * df["qdd_ph_c"].values
    return np.sqrt(az**2 + ax**2 + ay**2)


def plot_single_sim(label: str, params: Dict, sim_idx: int) -> Optional[Dict]:
    """
    Run one simulation and produce a 4-panel time-history plot:
      Row 0 – Vertical   z̈ [m/s²]
      Row 1 – Longitudinal  ẍ [m/s²]
      Row 2 – Lateral    ÿ [m/s²]
      Row 3 – Combined magnitude √(az²+ax²+ay²) [m/s²]

    Returns a result dict or None on failure.
    """
    print(f"\n  ▶  Running ODE for: {label}")
    t0 = time.time()
    try:
        df = run_one_case(params, t_eval_full)
    except Exception as exc:
        print(f"  ✗  ODE failed for '{label}': {exc}")
        return None
    elapsed = time.time() - t0
    print(f"  ✓  Done in {elapsed:.1f}s")

    t    = df["t"].values
    h    = CFG["hcp"]
    az   = df["qdd_z_c"].values
    ax_  = -h * df["qdd_th_c"].values
    ay_  =  h * df["qdd_ph_c"].values
    ac   = _make_combined_signal(df)

    rz, rx, ry, rt = compute_rms(df)
    cols = SIM_COLORS[sim_idx % 4]

    # ── Figure ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 10), facecolor="#FAFAFA")
    fig.patch.set_alpha(1.0)

    gs = gridspec.GridSpec(4, 1, hspace=0.42, top=0.91, bottom=0.07,
                           left=0.10, right=0.97)

    signals = [
        (az,  cols["z"],    "z̈  seat  [m/s²]",   f"RMS = {rz:.4f} m/s²"),
        (ax_, cols["x"],    "ẍ  seat  [m/s²]",   f"RMS = {rx:.4f} m/s²"),
        (ay_, cols["y"],    "ÿ  seat  [m/s²]",   f"RMS = {ry:.4f} m/s²"),
        (ac,  cols["comb"], "|ä|  combined  [m/s²]", f"RMS = {rt:.4f} m/s²"),
    ]
    row_labels = ["Vertical (Z)", "Longitudinal (X)", "Lateral (Y)", "Combined"]

    axes = []
    for row, (sig, color, ylabel, rms_text) in enumerate(signals):
        ax = fig.add_subplot(gs[row])
        ax.plot(t, sig, linewidth=0.55, color=color, alpha=0.9)
        ax.set_ylabel(ylabel, fontsize=9, labelpad=6)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="#CCCCCC")
        ax.tick_params(labelsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        # Annotate RMS as a text box on the right
        ax.text(0.993, 0.88, rms_text, transform=ax.transAxes,
                fontsize=8, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color,
                          alpha=0.85, linewidth=0.8))
        # Row label
        ax.text(0.007, 0.88, row_labels[row], transform=ax.transAxes,
                fontsize=8, ha="left", va="top", color=color,
                fontweight="bold")
        if row < 3:
            ax.tick_params(labelbottom=False)
        axes.append(ax)

    axes[-1].set_xlabel("Time  [s]", fontsize=9)

    # Shared x-limits
    for ax in axes:
        ax.set_xlim(t[0], t[-1])

    # Title
    fig.suptitle(
        f"Seat Acceleration Time History  —  {label}\n"
        f"RMS:  z={rz:.4f}   x={rx:.4f}   y={ry:.4f}   combined={rt:.4f}   [m/s²]",
        fontsize=11, fontweight="bold", y=0.975, color="#1A1A2E"
    )

    safe = label.replace(" ", "_").replace("/", "-")
    fname = f"seat_accel_{safe}.png"
    fpath = os.path.join(PLOTS_DIR, fname)
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊  Saved → {fpath}")

    return {"label": label, "rz": rz, "rx": rx, "ry": ry, "rt": rt}


def plot_comparison_bar(results: list):
    """
    Side-by-side grouped bar chart comparing RMS across all simulations.
    Shows z, x, y, and combined for every simulation that ran successfully.
    """
    if not results:
        return

    labels   = [r["label"] for r in results]
    rz_vals  = [r["rz"] for r in results]
    rx_vals  = [r["rx"] for r in results]
    ry_vals  = [r["ry"] for r in results]
    rt_vals  = [r["rt"] for r in results]

    n   = len(results)
    x   = np.arange(n)
    w   = 0.18
    off = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]

    fig, ax = plt.subplots(figsize=(max(9, n * 2.8 + 2), 5.5), facecolor="#FAFAFA")
    bar_kw = dict(edgecolor="white", linewidth=0.7)

    bars = [
        ax.bar(x + off[0], rz_vals, w, label="z  (vertical)",      color="#2E86AB", **bar_kw),
        ax.bar(x + off[1], rx_vals, w, label="x  (longitudinal)",  color="#E07B39", **bar_kw),
        ax.bar(x + off[2], ry_vals, w, label="y  (lateral)",       color="#3BB273", **bar_kw),
        ax.bar(x + off[3], rt_vals, w, label="combined",           color="#9B5DE5", **bar_kw),
    ]

    # Value labels on top of each bar
    for bar_group in bars:
        for bar in bar_group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.002,
                    f"{h:.4f}", ha="center", va="bottom",
                    fontsize=6.5, color="#333333", rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("RMS Acceleration  [m/s²]", fontsize=10)
    ax.set_title("Simulation Comparison — Per-Axis & Combined RMS",
                 fontsize=12, fontweight="bold", pad=12, color="#1A1A2E")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.5, color="#CCCCCC")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fpath = os.path.join(PLOTS_DIR, "comparison_rms.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  📊  Saved comparison chart → {fpath}")


# ===========================================================================
# ── MAIN ─────────────────────────────────────────────────────────────────────
# ===========================================================================

def main():
    banner = "=" * 65
    print(f"\n{banner}")
    print("  Seat Acceleration Simulation Runner")
    print(f"  Output directory : {os.path.abspath(RESULTS_DIR)}")
    print(f"  Plot directory   : {os.path.abspath(PLOTS_DIR)}")
    print(f"  Simulations      : {sum(v is not None for v in PARAM_SETS.values())}")
    print(f"{banner}\n")

    results = []

    for idx, (label, params) in enumerate(PARAM_SETS.items()):
        if params is None:
            print(f"  ─  '{label}'  skipped  (params = None)")
            continue

        print(f"\n{'─'*55}")
        print(f"  Simulation {idx + 1} / {len(PARAM_SETS)} :  {label}")
        print(f"{'─'*55}")
        print("  Parameters:")
        for k, v in params.items():
            print(f"    {k:15s} = {v:.8g}")

        result = plot_single_sim(label, params, sim_idx=idx)
        if result:
            results.append(result)

    # ── Summary table ───────────────────────────────────────────────────────
    print(f"\n{banner}")
    print("  RESULTS SUMMARY")
    print(f"{banner}")
    if results:
        hdr = f"  {'Simulation':<28}  {'RMS_z':>8}  {'RMS_x':>8}  {'RMS_y':>8}  {'RMS_tot':>9}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in results:
            print(f"  {r['label']:<28}  {r['rz']:>8.4f}  {r['rx']:>8.4f}  {r['ry']:>8.4f}  {r['rt']:>9.4f}")

        # ── Comparison bar chart ─────────────────────────────────────────────
        plot_comparison_bar(results)
    else:
        print("  No simulations completed successfully.")

    print(f"\n{banner}")
    print(f"  Done.  All plots saved to: {os.path.abspath(PLOTS_DIR)}")
    print(f"{banner}\n")


if __name__ == "__main__":
    main()
