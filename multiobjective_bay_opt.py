"""
Multi-Objective Bayesian Optimisation – 3-Axis Seat RMS Pareto Front
=====================================================================
Objectives (all minimised):
    f1 = RMS_z  (vertical seat acceleration)
    f2 = RMS_x  (longitudinal, from cabin pitch)
    f3 = RMS_y  (lateral, from cabin roll)

Optimiser: BoTorch qEHVI  (q-Expected Hypervolume Improvement)
    - State-of-the-art for expensive multi-objective black-boxes
    - 8 quasi-random Sobol init points  +  up to 42 BO iterations = 50 total
    - Each iteration proposes ONE candidate (q=1) → one ODE solve per step

Parameters kept from original:
    K_f, C_f, K_2, K_3, cs_minus, asym_ratio, gamma_c, gamma_r

Outputs (all in RESULTS_DIR):
    pareto_front.csv        – non-dominated solutions with physical params + objectives
    best_params.json        – full run log + Pareto summary
    plots/pareto_f1_f2.png  – 2-D Pareto projections (z vs x, z vs y, x vs y)
    plots/pareto_3d.png     – 3-D Pareto scatter
    plots/hypervolume.png   – hypervolume indicator per iteration
    plots/param_parallel.png – parallel coordinates of Pareto set
    (plus all original per-case response plots)

Dependencies:
    pip install botorch torch gpytorch pandas numpy scipy matplotlib
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os
import json
import time
import warnings
from typing import Dict, Callable, Tuple

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
# BoTorch / PyTorch
# ---------------------------------------------------------------------------
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import unnormalize, normalize
from botorch.optim import optimize_acqf
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

# ---------------------------------------------------------------------------
# Matplotlib
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings("ignore")

# ===========================================================================
# Global simulation settings  (unchanged from original)
# ===========================================================================
DT       = 0.001
FS       = 1000
T_IGNORE = 0.5
T_END    = 466.945

t_eval_full = np.arange(0.0, T_END + DT, DT)

RESULTS_DIR = "Res_Laden_MOBO_qEHVI"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

STATE_NAMES = ["z_c", "th_c", "ph_c", "z_s", "th_s", "ph_s"]
(ZC, THC, PHC, ZS, THS, PHS) = range(6)

# ===========================================================================
# CFG  (unchanged)
# ===========================================================================
CFG: Dict = {
    "axlefront_left_csv":  r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Final_ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_1_FA_LH.csv",
    "axlefront_right_csv": r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Final_ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_2_FA_RH.csv",
    "axlerear1_left_csv":  r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Final_ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_3_RA1_LH.csv",
    "axlerear1_right_csv": r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Final_ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_4_RA1_RH.csv",
    "axlerear2_left_csv":  r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Final_ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_5_RA2_LH.csv",
    "axlerear2_right_csv": r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Final_ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_6_RA2_RH.csv",

    "s1": 0.6277, "s2": 0.6305,
    "WT1": 0.814, "WT2": 1.047, "WT3": 1.047,

    "m_c": 862.0,  "I_xxc": 516.6, "I_yyc": 1045.0,
    "M_1f": 600.0, "M_2": 1075.0,  "M_3": 840.0,
    "I_xx1": 650.0,"I_xx2": 1200.0,"I_xx3": 1100.0,

    "S_tf2": 1.043, "S_tf3": 1.043,
    "S_f":   0.814,

    "C_cfl": 5035.0, "C_cfr": 5035.0, "C_crl": 3400.0, "C_crr": 3400.0,
    "K_cfl": 49050.0,"K_cfr": 49050.0,"K_crl": 24525.0,"K_crr": 24525.0,
    "C_2": 2000, "C_3": 2000,

    "L_DL2": 0.6211, "L_DR2": 0.6211,
    "L_DL3": 0.6251, "L_DR3": 0.6251,
    "beta_L2": 0.1693, "beta_R2": 0.1693,
    "beta_L3": 0.17453,"beta_R3": 0.17453,

    "a": 0.9, "b": 1.080,
    "l_cfcg": 0.871, "l_crcg": 1.087,
    "hcp": 0.1,

    "lf": 5.05, "L12": 0.54, "L23": 1.96,
    "l_cf": 6.458, "l_cr": 4.5,

    "m_s": 22485.0, "I_syy": 103787.0, "I_sxx": 8598.0, "I_sxy": 763.0,
    "hs": 0.68,

    # Baseline values (used as starting point + for baseline run)
    "K_f": 474257,  "C_f": 15000,
    "K_2": 1077620, "K_3": 1077620,

    "g": 9.81,

    # Damper shape
    "cs_minus": 0.3, "asym_ratio": 3.0,
    "gamma_c": 0.12, "gamma_r": 0.09,

    # Baumgarte stabilisation
    "baum_omega": 10.0, "baum_zeta": 1.0,
}

# ===========================================================================
# Search bounds  (same as original)
# ===========================================================================
PARAM_KEYS = ["K_f", "C_f", "K_2", "K_3", "cs_minus", "asym_ratio", "gamma_c", "gamma_r"]

BOUNDS_RAW = {
    "K_f":       (0.879 * CFG["K_f"],  1.126 * CFG["K_f"]),
    "C_f":       (0.44  * CFG["C_f"],  1.4   * CFG["C_f"]),
    "K_2":       (0.892 * CFG["K_2"],  1.116 * CFG["K_2"]),
    "K_3":       (0.892 * CFG["K_3"],  1.116 * CFG["K_3"]),
    "cs_minus":  (0.2,   0.4),
    "asym_ratio":(2.3,   4.0),
    "gamma_c":   (0.08,  0.16),
    "gamma_r":   (0.08,  0.10),
}

# Bounds as (8,) tensors for BoTorch
BOUNDS_LO = torch.tensor([BOUNDS_RAW[k][0] for k in PARAM_KEYS], dtype=torch.double)
BOUNDS_HI = torch.tensor([BOUNDS_RAW[k][1] for k in PARAM_KEYS], dtype=torch.double)
BOUNDS_T  = torch.stack([BOUNDS_LO, BOUNDS_HI])  # shape (2, 8)

# Reference point for hypervolume (slightly worse than expected worst):
# We negate objectives inside BoTorch (it maximises), so ref_point is set
# in the *negated* space: -ref_point_rms.  We use a conservatively high RMS.
REF_POINT = torch.tensor([-5.0, -5.0, -5.0], dtype=torch.double)  # negated

N_INIT   = 8    # Sobol quasi-random initialisations
N_ITER   = 42   # BO iterations
N_TOTAL  = N_INIT + N_ITER  # = 50


# ===========================================================================
# ── Physics (unchanged from original) ──────────────────────────────────────
# ===========================================================================

@dataclass
class TwoStageAsymmetricDamper:
    cs_minus: float
    asym_ratio: float
    gamma_c: float
    gamma_r: float
    alpha_c: float = -0.05
    alpha_r: float =  0.13

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


def make_linear_interp(x: np.ndarray, y: np.ndarray):
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
    l2, l3 = cfg["L12"], cfg["L12"] + cfg["L23"]
    S2, S3 = cfg["S_tf2"], cfg["S_tf3"]
    sl2, sl3 = cfg["s1"], cfg["s2"]
    bL2, bL3 = cfg["beta_L2"], cfg["beta_L3"]
    g2 = z_s + l2*th_s + S2*ph_s - sl2*np.sin(bL2 - th_s) - (z2 + 0.5*cfg["WT2"]*ph2)
    g3 = z_s + l3*th_s + S3*ph_s - sl3*np.sin(bL3 - th_s) - (z3 + 0.5*cfg["WT3"]*ph3)
    gq = np.array([g2, g3], dtype=float)
    G  = np.zeros((2, 6), dtype=float)
    G[0, ZS]  = 1.0
    G[0, THS] = l2 + sl2*np.cos(bL2 - th_s)
    G[0, PHS] = S2
    G[1, ZS]  = 1.0
    G[1, THS] = l3 + sl3*np.cos(bL3 - th_s)
    G[1, PHS] = S3
    return gq, G


def build_M_R(q, v, t, cfg, road):
    z_c, th_c, ph_c, z_s, th_s, ph_s = q
    dz_c, dth_c, dph_c, dz_s, dth_s, dph_s = v
    z1f, ph_f, z2, ph2, z3, ph3 = road.axle_inputs(t, cfg)
    dz1f, dph_f, dz2, dph2, dz3, dph3 = road.axle_input_rates(t, cfg)

    phi_NRS2 = (cfg["beta_L2"]*cfg["L_DL2"] - cfg["beta_R2"]*cfg["L_DR2"]) / max(cfg["S_tf2"], 1e-6)
    phi_NRS3 = (cfg["beta_L3"]*cfg["L_DL3"] - cfg["beta_R3"]*cfg["L_DR3"]) / max(cfg["S_tf3"], 1e-6)

    m_c, I_xxc, I_yyc = cfg["m_c"], cfg["I_xxc"], cfg["I_yyc"]
    m_s, I_sxx, I_syy, I_sxy = cfg["m_s"], cfg["I_sxx"], cfg["I_syy"], cfg["I_sxy"]
    S1, S2, S3 = cfg["S_f"], cfg["S_tf2"], cfg["S_tf3"]
    a, b       = cfg["a"], cfg["b"]
    hs, g      = cfg["hs"], cfg["g"]
    l_cfcg, l_crcg, l_cf, l_cr = cfg["l_cfcg"], cfg["l_crcg"], cfg["l_cf"], cfg["l_cr"]
    lf         = cfg["lf"]
    hcp        = cfg["hcp"]
    l2         = cfg["L12"]
    l3         = cfg["L12"] + cfg["L23"]
    beta_L2, beta_R2 = cfg["beta_L2"], cfg["beta_R2"]
    beta_L3, beta_R3 = cfg["beta_L3"], cfg["beta_R3"]
    L_DL2, L_DR2, L_DL3, L_DR3 = cfg["L_DL2"], cfg["L_DR2"], cfg["L_DL3"], cfg["L_DR3"]
    Kcfl,Kcfr,Kcrl,Kcrr = cfg["K_cfl"], cfg["K_cfr"], cfg["K_crl"], cfg["K_crr"]
    Ccfl,Ccfr,Ccrl,Ccrr = cfg["C_cfl"], cfg["C_cfr"], cfg["C_crl"], cfg["C_crr"]
    K_f, C_f = cfg["K_f"], cfg["C_f"]
    K_2, C_2 = cfg["K_2"], cfg["C_2"]
    K_3, C_3 = cfg["K_3"], cfg["C_3"]

    M = np.zeros((6, 6), dtype=float)
    M[ZC, ZC]   = m_c
    M[THC, THC] = I_yyc
    M[PHC, PHC] = I_xxc
    M[ZS, ZS]   = m_s
    M[THS, THS] = I_syy
    M[THS, PHS] = I_sxy
    M[PHS, THS] = I_sxy
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

    R = np.zeros(6, dtype=float)

    R[ZC] = (
        + Csum*(dz_c - dz_s) + Ksum*(z_c - z_s)
        - (Ccfl*l_cfcg + Ccfr*l_cfcg - Ccrl*l_crcg - Ccrr*l_crcg)*dth_c
        - (-Ccfl*l_cf - Ccfr*l_cf - Ccrl*l_cr - Ccrr*l_cr)*dth_s
        - (-Ccfl*b + Ccfr*a - Ccrl*b + Ccrr*a)*dph_c
        - (Ccfl*b - Ccfr*a + Ccrl*b - Ccrr*a)*dph_s
        - (Kcfl*l_cfcg + Kcfr*l_cfcg - Kcrl*l_crcg - Kcrr*l_crcg)*th_c
        - (-Kcfl*l_cf - Kcfr*l_cf - Kcrl*l_cr - Kcrr*l_cr)*th_s
        - (-Kcfl*b + Kcfr*a - Kcrl*b + Kcrr*a)*ph_c
        - (Kcfl*b - Kcfr*a + Kcrl*b - Kcrr*a)*ph_s
    )
    R[THC] = (
        - (Ccfl*l_cfcg + Ccfr*l_cfcg - Ccrl*l_crcg - Ccrr*l_crcg)*dz_c
        - (-Ccfl*l_cfcg - Ccfr*l_cfcg - Ccrl*l_crcg - Ccrr*l_crcg)*dz_s
        - (Kcfl*l_cfcg + Kcfr*l_cfcg - Kcrl*l_crcg - Kcrr*l_crcg)*z_c
        - (-Kcfl*l_cfcg - Kcfr*l_cfcg - Kcrl*l_crcg - Kcrr*l_crcg)*z_s
        - (-Ccfl*l_cfcg**2 - Ccfr*l_cfcg**2 - Ccrl*l_crcg**2 - Ccrr*l_crcg**2)*dth_c
        - (Ccfl*l_cfcg*l_cf + Ccfr*l_cfcg*l_cf - Ccrl*l_crcg*l_cr - Ccrr*l_crcg*l_cr)*dth_s
        - (-Ccfl*l_cfcg*b + Ccfr*l_cfcg*a - Ccrl*l_crcg*b + Ccrr*l_crcg*a)*dph_c
        - (Ccfl*l_cfcg*b - Ccfr*l_cfcg*a + Ccrl*l_crcg*b - Ccrr*l_crcg*a)*dph_s
        - (-Kcfl*l_cfcg**2 - Kcfr*l_cfcg**2 - Kcrl*l_crcg**2 - Kcrr*l_crcg**2 + m_c*g*hcp)*th_c
        - (Kcfl*l_cfcg*l_cf + Kcfr*l_cfcg*l_cf - Kcrl*l_crcg*l_cr - Kcrr*l_crcg*l_cr)*th_s
        - (-Kcfl*l_cfcg*b + Kcfr*l_cfcg*a - Kcrl*l_crcg*b + Kcrr*l_crcg*a)*ph_c
        - (Kcfl*l_cfcg*b - Kcfr*l_cfcg*a + Kcrl*l_crcg*b - Kcrr*l_crcg*a)*ph_s
    )
    R[PHC] = (
        - (-Ccfl*b + Ccfr*a - Ccrl*b + Ccrr*a)*dz_c
        - (Ccfl*b - Ccfr*a + Ccrl*b - Ccrr*a)*dz_s
        - (-Kcfl*b + Kcfr*a - Kcrl*b + Kcrr*a)*z_c
        - (Kcfl*b - Kcfr*a + Kcrl*b - Kcrr*a)*z_s
        - (-Ccfl*l_cfcg*b - Ccfr*l_cfcg*a + Ccrl*l_crcg*b + Ccrr*l_crcg*a)*dth_c
        - (Ccfl*l_cfcg*b + Ccfr*l_cfcg*a - Ccrl*l_crcg*b - Ccrr*l_crcg*a)*dth_s
        - (-Ccfl*b**2 + Ccfr*a**2 - Ccrl*b**2 + Ccrr*a**2)*dph_c
        - (Ccfl*b**2 - Ccfr*a**2 + Ccrl*b**2 - Ccrr*a**2)*dph_s
        - (-Kcfl*l_cfcg*b - Kcfr*l_cfcg*a + Kcrl*l_crcg*b + Kcrr*l_crcg*a)*th_c
        - (Kcfl*l_cfcg*b + Kcfr*l_cfcg*a - Kcrl*l_crcg*b - Kcrr*l_crcg*a)*th_s
        - (-Kcfl*b**2 + Kcfr*a**2 - Kcrl*b**2 + Kcrr*a**2)*ph_c
        - (Kcfl*b**2 - Kcfr*a**2 + Kcrl*b**2 - Kcrr*a**2)*ph_s
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
    k_tf = 0.5*K_f*S1**2;  K_r1 = 0.5*K_2*S2**2;  K_r2 = 0.5*K_3*S3**2
    C_tf = 0.5*C_f*S1**2;  C_r1 = 0.5*C_2*S2**2;  C_r2 = 0.5*C_3*S3**2
    R[PHS] = -(
        + m_s*g*hs*ph_s
        - k_tf*(ph_s - ph_f) - C_tf*(dph_s - dph_f)
        - K_r1*(ph_s - ph2 - phi_NRS2) - C_r1*(dph_s - dph2)
        - K_r2*(ph_s - ph3 - phi_NRS3) - C_r2*(dph_s - dph3)
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
    A[:6, :6] = M;  A[:6, 6:] = G.T;  A[6:, :6] = G
    b_vec[:6] = -R; b_vec[6:] = -gamma
    sol = lin_solve(A, b_vec)
    xdot = np.zeros_like(x)
    xdot[:6] = v
    xdot[6:] = sol[:6]
    return xdot


def static_equilibrium_state(cfg, road):
    q_seed  = np.zeros(6, dtype=float)
    lam_seed = np.zeros(2, dtype=float)
    y0 = np.hstack([q_seed, lam_seed])
    t0 = 0.0

    def F(y):
        q, lam = y[:6], y[6:]
        v = np.zeros(6)
        M, R = build_M_R(q, v, t0, cfg, road)
        gq, G = geom_constraints(q, t0, cfg, road)
        return np.hstack([R + G.T @ lam, 1e3 * gq])

    lsq = least_squares(F, y0, method="trf", loss="soft_l1",
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=800)
    if lsq.success:
        q0 = lsq.x[:6]
        print(f"  [Static EQ OK] ||g||={np.linalg.norm(geom_constraints(q0, t0, cfg, road)[0]):.3e}")
        return np.hstack([q0, np.zeros(6)])

    # Fallback: dynamic relaxation
    print("  [Static EQ fallback: dynamic relaxation]")
    cfg_r = dict(cfg); cfg_r["C_2"] *= 20; cfg_r["C_3"] *= 20
    cfg_r["C_cfl"] *= 20; cfg_r["C_cfr"] *= 20; cfg_r["C_crl"] *= 20; cfg_r["C_crr"] *= 20
    sol = solve_ivp(lambda t, x: rhs_first_order(t, x, cfg_r, road),
                    (0.0, 3.0), np.hstack([q_seed, np.zeros(6)]),
                    method="Radau", rtol=1e-7, atol=1e-9)
    q0 = sol.y[:6, -1]
    return np.hstack([q0, np.zeros(6)])


# ===========================================================================
# ── Simulation & objective evaluation ──────────────────────────────────────
# ===========================================================================

def run_one_case(params: Dict, cfg_base: Dict, t_eval: np.ndarray) -> pd.DataFrame:
    cfg = {**cfg_base, **params}
    road = build_road_signals(cfg)
    x0   = static_equilibrium_state(cfg, road)
    sol  = solve_ivp(
        fun=lambda t, x: rhs_first_order(t, x, cfg, road),
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=x0, t_eval=t_eval,
        method="Radau", max_step=0.01, rtol=1e-6, atol=1e-8,
    )
    if sol.status != 0 or not np.all(np.isfinite(sol.y)):
        raise RuntimeError(f"ODE failed: {sol.message}")

    rows = []
    for i, t in enumerate(sol.t):
        x = sol.y[:, i]
        q, v = x[:6], x[6:]
        qdd  = rhs_first_order(t, x, cfg, road)[6:]
        row  = {"t": t}
        for j, name in enumerate(STATE_NAMES):
            row[name]           = q[j]
            row[f"qd_{name}"]   = v[j]
            row[f"qdd_{name}"]  = qdd[j]
        rows.append(row)
    return pd.DataFrame(rows)


def compute_per_axis_rms(df: pd.DataFrame, cfg: Dict) -> Tuple[float, float, float]:
    """Returns (rms_z, rms_x, rms_y) – three separate objectives."""
    mask = df["t"] >= T_IGNORE
    h    = cfg["hcp"]
    az   = df.loc[mask, "qdd_z_c"].values
    ax   = -h * df.loc[mask, "qdd_th_c"].values
    ay   =  h * df.loc[mask, "qdd_ph_c"].values
    return (
        float(np.sqrt(np.mean(az**2))),
        float(np.sqrt(np.mean(ax**2))),
        float(np.sqrt(np.mean(ay**2))),
    )


def evaluate_objectives(x_norm: torch.Tensor) -> torch.Tensor:
    """
    x_norm : (1, 8) tensor, normalised to [0, 1]^8
    Returns : (1, 3) tensor of NEGATED objectives (BoTorch maximises)
              returns large penalty tensor on ODE failure
    """
    x_phys = unnormalize(x_norm, BOUNDS_T).squeeze(0).numpy()
    params = {k: float(x_phys[i]) for i, k in enumerate(PARAM_KEYS)}

    try:
        t0 = time.time()
        df = run_one_case(params, CFG, t_eval_full)
        rms_z, rms_x, rms_y = compute_per_axis_rms(df, CFG)
        elapsed = time.time() - t0
        print(f"    ODE OK  ({elapsed:.1f}s)  rms_z={rms_z:.4f}  rms_x={rms_x:.4f}  rms_y={rms_y:.4f}")
        # Negate: BoTorch maximises, we minimise RMS
        return torch.tensor([[-rms_z, -rms_x, -rms_y]], dtype=torch.double)
    except Exception as exc:
        print(f"    ODE FAILED: {exc}  → penalty")
        return torch.tensor([[-99.0, -99.0, -99.0]], dtype=torch.double)


# ===========================================================================
# ── MOBO loop (qEHVI) ──────────────────────────────────────────────────────
# ===========================================================================

def run_mobo() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        train_X : (N_TOTAL, 8) normalised parameter tensor
        train_Y : (N_TOTAL, 3) negated-objective tensor
    """
    print("\n" + "="*60)
    print("  Phase 1 / 2 : Sobol quasi-random initialisation")
    print(f"  Points : {N_INIT}")
    print("="*60)

    sobol = SobolEngine(dimension=8, scramble=True, seed=123)
    init_X = sobol.draw(N_INIT).to(torch.double)   # (N_INIT, 8) in [0,1]^8

    train_X = init_X.clone()
    train_Y = torch.empty((0, 3), dtype=torch.double)

    for i in range(N_INIT):
        print(f"\n  [Init {i+1}/{N_INIT}]")
        y = evaluate_objectives(train_X[i].unsqueeze(0))
        train_Y = torch.cat([train_Y, y], dim=0)

    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("  Phase 2 / 2 : Bayesian optimisation (qEHVI)")
    print(f"  Iterations : {N_ITER}")
    print("="*60)

    hv_history = []

    for iteration in range(N_ITER):
        # --- Fit independent GP per objective ---
        model = SingleTaskGP(
            train_X=normalize(train_X, BOUNDS_T),
            train_Y=train_Y,
            input_transform=Normalize(d=8),
            outcome_transform=Standardize(m=3),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # --- Compute hypervolume of current Pareto front ---
        pareto_mask = is_non_dominated(train_Y)
        pareto_Y    = train_Y[pareto_mask]
        try:
            hv = Hypervolume(ref_point=REF_POINT).compute(pareto_Y)
        except Exception:
            hv = 0.0
        hv_history.append(hv)
        n_pareto = pareto_mask.sum().item()
        print(f"\n  [BO iter {iteration+1}/{N_ITER}]  "
              f"Pareto size={n_pareto}  HV={hv:.6f}")

        # --- qEHVI acquisition ---
        acq = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=REF_POINT.tolist(),
            partitioning=None,   # auto internal
            Y_pmean=train_Y,
        )

        candidate, acq_val = optimize_acqf(
            acq_function=acq,
            bounds=torch.stack([
                torch.zeros(8, dtype=torch.double),
                torch.ones(8,  dtype=torch.double),
            ]),
            q=1,
            num_restarts=10,
            raw_samples=128,
        )

        # Evaluate ODE
        new_y = evaluate_objectives(candidate)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_y],      dim=0)

    # Final HV
    pareto_mask = is_non_dominated(train_Y)
    pareto_Y    = train_Y[pareto_mask]
    try:
        hv_final = Hypervolume(ref_point=REF_POINT).compute(pareto_Y)
    except Exception:
        hv_final = hv_history[-1] if hv_history else 0.0
    hv_history.append(hv_final)

    return train_X, train_Y, hv_history


# ===========================================================================
# ── Pareto extraction & saving ─────────────────────────────────────────────
# ===========================================================================

def extract_pareto(train_X: torch.Tensor, train_Y: torch.Tensor) -> pd.DataFrame:
    """
    Returns a DataFrame of non-dominated solutions with physical params +
    objective values + a simple 'label' column.
    """
    pareto_mask = is_non_dominated(train_Y)
    px = unnormalize(train_X[pareto_mask], BOUNDS_T).numpy()
    py = (-train_Y[pareto_mask]).numpy()       # back to positive RMS

    rows = []
    for i in range(px.shape[0]):
        row = {k: px[i, j] for j, k in enumerate(PARAM_KEYS)}
        row["rms_z"] = py[i, 0]
        row["rms_x"] = py[i, 1]
        row["rms_y"] = py[i, 2]
        row["rms_total"] = float(np.sqrt(py[i, 0]**2 + py[i, 1]**2 + py[i, 2]**2))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("rms_total").reset_index(drop=True)

    # Annotate with readable labels
    n = len(df)
    labels = []
    for i in range(n):
        if i == 0:
            labels.append("Best_total")
        elif df["rms_z"].iloc[i] == df["rms_z"].min():
            labels.append("Best_vertical")
        elif df["rms_x"].iloc[i] == df["rms_x"].min():
            labels.append("Best_longitudinal")
        elif df["rms_y"].iloc[i] == df["rms_y"].min():
            labels.append("Best_lateral")
        else:
            labels.append(f"Pareto_{i+1}")
    df.insert(0, "label", labels)
    return df


def save_pareto_csv(df_pareto: pd.DataFrame) -> str:
    path = os.path.join(RESULTS_DIR, "pareto_front.csv")
    df_pareto.to_csv(path, index=False)
    print(f"[CSV] Pareto front saved → {path}")
    return path


def save_run_json(df_pareto: pd.DataFrame,
                  train_X: torch.Tensor,
                  train_Y: torch.Tensor) -> str:
    """Saves full run log + Pareto summary to JSON."""
    all_x = unnormalize(train_X, BOUNDS_T).numpy()
    all_y = (-train_Y).numpy()

    run_log = []
    for i in range(all_x.shape[0]):
        entry = {k: float(all_x[i, j]) for j, k in enumerate(PARAM_KEYS)}
        entry["rms_z"] = float(all_y[i, 0])
        entry["rms_x"] = float(all_y[i, 1])
        entry["rms_y"] = float(all_y[i, 2])
        entry["phase"] = "init" if i < N_INIT else "bo"
        run_log.append(entry)

    out = {
        "description": (
            "Multi-objective BoTorch qEHVI run. "
            "Objectives: RMS_z (vertical), RMS_x (longitudinal), RMS_y (lateral) "
            "seat accelerations [m/s²]. "
            "pareto_front contains non-dominated solutions. "
            "all_evaluations contains the full run log."
        ),
        "bounds": {k: {"lo": float(v[0]), "hi": float(v[1])} for k, v in BOUNDS_RAW.items()},
        "n_init": N_INIT, "n_iter": N_ITER,
        "pareto_front": df_pareto.to_dict(orient="records"),
        "all_evaluations": run_log,
    }

    path = os.path.join(RESULTS_DIR, "run_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[JSON] Full run log saved → {path}")
    return path


# ===========================================================================
# ── Plotting ───────────────────────────────────────────────────────────────
# ===========================================================================

def plot_pareto_2d(df_pareto: pd.DataFrame, all_y_pos: np.ndarray):
    """Three 2-D Pareto projections: z-x, z-y, x-y."""
    pairs = [("rms_z", "rms_x"), ("rms_z", "rms_y"), ("rms_x", "rms_y")]
    labels_ax = {"rms_z": "RMS_z vertical [m/s²]",
                 "rms_x": "RMS_x longitudinal [m/s²]",
                 "rms_y": "RMS_y lateral [m/s²]"}

    for xa, ya in pairs:
        fig, ax = plt.subplots(figsize=(7, 5))
        # All evaluated points (grey background)
        ax.scatter(all_y_pos[:, list(["rms_z","rms_x","rms_y"]).index(xa)],
                   all_y_pos[:, list(["rms_z","rms_x","rms_y"]).index(ya)],
                   c="lightgrey", s=20, zorder=1, label="All evaluated")
        # Pareto front
        sc = ax.scatter(df_pareto[xa], df_pareto[ya],
                        c=df_pareto["rms_total"], cmap="plasma",
                        s=80, zorder=3, edgecolors="k", linewidths=0.5,
                        label="Pareto front")
        plt.colorbar(sc, ax=ax, label="RMS_total [m/s²]")
        # Annotate labels
        for _, row in df_pareto.iterrows():
            ax.annotate(row["label"], (row[xa], row[ya]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel(labels_ax[xa]); ax.set_ylabel(labels_ax[ya])
        ax.set_title(f"Pareto Front: {xa} vs {ya}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
        plt.tight_layout()
        fname = f"pareto_{xa}_vs_{ya}.png"
        plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
        plt.close()
        print(f"[PLOT] {fname}")


def plot_pareto_3d(df_pareto: pd.DataFrame, all_y_pos: np.ndarray):
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(all_y_pos[:, 0], all_y_pos[:, 1], all_y_pos[:, 2],
               c="lightgrey", s=15, alpha=0.5, label="All evaluated")
    sc = ax.scatter(df_pareto["rms_z"], df_pareto["rms_x"], df_pareto["rms_y"],
                    c=df_pareto["rms_total"], cmap="plasma",
                    s=100, edgecolors="k", linewidths=0.5, label="Pareto")
    plt.colorbar(sc, ax=ax, label="RMS_total", pad=0.1, shrink=0.6)
    for _, row in df_pareto.iterrows():
        ax.text(row["rms_z"], row["rms_x"], row["rms_y"],
                row["label"], fontsize=6)
    ax.set_xlabel("RMS_z [m/s²]")
    ax.set_ylabel("RMS_x [m/s²]")
    ax.set_zlabel("RMS_y [m/s²]")
    ax.set_title("3-D Pareto Front")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pareto_3d.png"), dpi=150)
    plt.close()
    print("[PLOT] pareto_3d.png")


def plot_hypervolume(hv_history: list):
    plt.figure(figsize=(7, 4))
    x = list(range(1, len(hv_history) + 1))
    plt.plot(x, hv_history, marker="o", markersize=4, linewidth=1.5, color="steelblue")
    plt.axvline(N_INIT, color="red", linestyle="--", linewidth=1, label=f"End of init ({N_INIT})")
    plt.xlabel("Iteration"); plt.ylabel("Hypervolume indicator")
    plt.title("Hypervolume Improvement per Iteration")
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "hypervolume.png"), dpi=150)
    plt.close()
    print("[PLOT] hypervolume.png")


def plot_parallel_coordinates(df_pareto: pd.DataFrame):
    """Parallel-coordinates plot of Pareto parameter sets (normalised)."""
    norm_df = df_pareto.copy()
    for k in PARAM_KEYS:
        lo, hi = BOUNDS_RAW[k]
        norm_df[k] = (df_pareto[k] - lo) / (hi - lo)

    cols = PARAM_KEYS + ["rms_z", "rms_x", "rms_y"]
    data = norm_df[PARAM_KEYS].values
    n    = len(df_pareto)
    cmap = cm.get_cmap("plasma", n)

    fig, ax = plt.subplots(figsize=(13, 5))
    xs = range(len(PARAM_KEYS))

    for i in range(n):
        ax.plot(xs, data[i], color=cmap(i), linewidth=1.4, alpha=0.8,
                label=df_pareto["label"].iloc[i])

    ax.set_xticks(list(xs))
    ax.set_xticklabels(PARAM_KEYS, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Normalised value in search range [0–1]")
    ax.set_title("Parallel Coordinates – Pareto Parameter Sets")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.axhline(0.05, color="red",    linestyle=":", linewidth=0.8)
    ax.axhline(0.95, color="orange", linestyle=":", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "param_parallel.png"), dpi=150)
    plt.close()
    print("[PLOT] param_parallel.png")


def plot_pareto_rms_bars(df_pareto: pd.DataFrame):
    """Grouped bar chart: per-axis RMS for each Pareto solution."""
    labels = df_pareto["label"].tolist()
    x      = np.arange(len(labels))
    w      = 0.25

    fig, ax = plt.subplots(figsize=(max(9, len(labels)*1.5), 5))
    ax.bar(x - w,   df_pareto["rms_z"], w, label="RMS_z (vertical)",     color="steelblue",  edgecolor="k")
    ax.bar(x,       df_pareto["rms_x"], w, label="RMS_x (longitudinal)", color="darkorange", edgecolor="k")
    ax.bar(x + w,   df_pareto["rms_y"], w, label="RMS_y (lateral)",      color="seagreen",   edgecolor="k")

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("RMS acceleration [m/s²]")
    ax.set_title("Per-Axis RMS for Each Pareto Solution")
    ax.legend(); ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pareto_rms_bars.png"), dpi=150)
    plt.close()
    print("[PLOT] pareto_rms_bars.png")


# ===========================================================================
# ── Main ───────────────────────────────────────────────────────────────────
# ===========================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # 1. Run MOBO
    # ------------------------------------------------------------------
    train_X, train_Y, hv_history = run_mobo()

    # ------------------------------------------------------------------
    # 2. Extract Pareto front
    # ------------------------------------------------------------------
    df_pareto = extract_pareto(train_X, train_Y)

    print("\n" + "="*60)
    print("  PARETO FRONT  (non-dominated solutions)")
    print("="*60)
    print(df_pareto[["label", "rms_z", "rms_x", "rms_y", "rms_total"]].to_string(index=False))

    # ------------------------------------------------------------------
    # 3. Save outputs
    # ------------------------------------------------------------------
    csv_path  = save_pareto_csv(df_pareto)
    json_path = save_run_json(df_pareto, train_X, train_Y)

    # ------------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------------
    all_y_pos = (-train_Y).numpy()   # (N_TOTAL, 3) positive RMS

    plot_pareto_2d(df_pareto, all_y_pos)
    plot_pareto_3d(df_pareto, all_y_pos)
    plot_hypervolume(hv_history)
    plot_parallel_coordinates(df_pareto)
    plot_pareto_rms_bars(df_pareto)

    # ------------------------------------------------------------------
    # 5. Individual ODE runs for best-total and each single-axis best
    # ------------------------------------------------------------------
    # Find the 3 "corner" solutions
    idx_best_total = df_pareto["rms_total"].idxmin()
    idx_best_z     = df_pareto["rms_z"].idxmin()
    idx_best_x     = df_pareto["rms_x"].idxmin()
    idx_best_y     = df_pareto["rms_y"].idxmin()

    special = {
        df_pareto.loc[idx_best_total, "label"]: df_pareto.loc[idx_best_total],
        df_pareto.loc[idx_best_z,     "label"]: df_pareto.loc[idx_best_z],
        df_pareto.loc[idx_best_x,     "label"]: df_pareto.loc[idx_best_x],
        df_pareto.loc[idx_best_y,     "label"]: df_pareto.loc[idx_best_y],
    }

    print("\n  Running ODE for special Pareto solutions (time history plots)...")
    for label, row in special.items():
        params = {k: row[k] for k in PARAM_KEYS}
        try:
            df_run = run_one_case(params, CFG, t_eval_full)
            t   = df_run["t"]
            h   = CFG["hcp"]
            az  = df_run["qdd_z_c"]
            ax_ = -h * df_run["qdd_th_c"]
            ay  =  h * df_run["qdd_ph_c"]

            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            axes[0].plot(t, az,  linewidth=0.7, color="steelblue")
            axes[0].set_ylabel("z̈ seat [m/s²]"); axes[0].grid(True, alpha=0.4)
            axes[1].plot(t, ax_, linewidth=0.7, color="darkorange")
            axes[1].set_ylabel("ẍ seat [m/s²]"); axes[1].grid(True, alpha=0.4)
            axes[2].plot(t, ay,  linewidth=0.7, color="seagreen")
            axes[2].set_ylabel("ÿ seat [m/s²]"); axes[2].set_xlabel("Time [s]")
            axes[2].grid(True, alpha=0.4)
            rz, rx, ry = compute_per_axis_rms(df_run, CFG)
            fig.suptitle(f"{label}  |  RMS z={rz:.4f}  x={rx:.4f}  y={ry:.4f} [m/s²]")
            plt.tight_layout()
            safe_label = label.replace("/", "_").replace(" ", "_")
            plt.savefig(os.path.join(PLOTS_DIR, f"seat_accel_{safe_label}.png"), dpi=150)
            plt.close()
            print(f"  [PLOT] seat_accel_{safe_label}.png")
        except Exception as exc:
            print(f"  [WARN] ODE failed for {label}: {exc}")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    best_row = df_pareto.loc[idx_best_total]
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"  Total evaluations : {N_TOTAL}")
    print(f"  Pareto front size : {len(df_pareto)}")
    print(f"  Best-total solution:")
    for k in PARAM_KEYS:
        lo, hi = BOUNDS_RAW[k]
        norm_v = (best_row[k] - lo) / (hi - lo)
        flag   = "  ← near boundary" if norm_v < 0.05 or norm_v > 0.95 else ""
        print(f"    {k:15s}: {best_row[k]:.6g}  (norm={norm_v:.3f}){flag}")
    print(f"  RMS_z = {best_row['rms_z']:.4f}  "
          f"RMS_x = {best_row['rms_x']:.4f}  "
          f"RMS_y = {best_row['rms_y']:.4f}  "
          f"RMS_total = {best_row['rms_total']:.4f}  [m/s²]")
    print(f"\n  Outputs saved in : {RESULTS_DIR}/")
    print(f"  Plots saved in   : {PLOTS_DIR}/")
    print(f"  Pareto CSV       : {csv_path}")
    print(f"  Run JSON         : {json_path}")
