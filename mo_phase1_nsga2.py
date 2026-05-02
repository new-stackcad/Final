"""
OPTION 2 — PHASE 1 of 2  (MULTI-OBJECTIVE, pymoo NSGA-II)
==========================================================
Optimises 4 structural parameters with a LINEAR front damper (F = C_f * v):
    K_f,  C_f (linear),  K_2,  K_3

Three SEPARATE objectives (all minimised):
    f1 = RMS_z  – vertical cabin-seat acceleration       [m/s²]
    f2 = RMS_x  – longitudinal seat accel (pitch × hcp) [m/s²]
    f3 = RMS_y  – lateral seat accel (roll × hcp)       [m/s²]

Why separate objectives here vs. one combined scalar?
------------------------------------------------------
In the original code a single combined RMS = √(mean(z²)+mean(x²)+mean(y²))
is minimised, which implicitly weights all axes equally and collapses the
trade-off into one number.  In reality:

  • K_f  strongly affects bounce (z) and pitch (x)
  • K_2, K_3 mainly control pitch (x) and have modest roll (y) coupling
  • C_f  affects all three but disproportionately damps z and x
  • Roll (y) is largely governed by anti-roll bar / track width geometry

So there is a genuine Pareto structure:  a stiffer K_f can reduce pitch
but amplifies bounce; a softer K_2 reduces pitch but may worsen roll.
The Pareto front exposes EXACTLY which structural parameter sets represent
non-dominated trade-offs, giving the engineer a principled choice.

After optimisation:
  • pareto_front_phase1.csv    – non-dominated solutions
  • phase1_run_results.json    – full run log
  • v_rel_front_<label>.npy   – v_rel time series for EACH Pareto solution
                                 (all saved so Phase 2 can pick any of them)
  • phase1_selected_params.json – the "Best_total" solution auto-selected
                                  as the default input to Phase 2

Optimiser: pymoo NSGA-II
Budget:    POP_SIZE × N_GEN  (default 10 × 5 = 50 ODE evaluations)

Dependencies:
    pip install pymoo pandas numpy scipy matplotlib
"""

# ── imports ──────────────────────────────────────────────────────────────────
import os, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401

from dataclasses  import dataclass
from typing       import Dict, Callable, Tuple
from numpy.linalg import solve as lin_solve
from scipy.integrate  import solve_ivp
from scipy.optimize   import least_squares

from pymoo.core.problem            import ElementwiseProblem
from pymoo.algorithms.moo.nsga2    import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm   import PM
from pymoo.operators.sampling.rnd  import FloatRandomSampling
from pymoo.optimize                import minimize as pymoo_minimize
from pymoo.indicators.hv           import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# ══════════════════════════════════════════════════════════════════════════════
# BUDGET  — tune these two numbers to control total ODE evaluations
# ══════════════════════════════════════════════════════════════════════════════
POP_SIZE = 10     # individuals per generation
N_GEN    = 5      # generations
# Total evaluations ≈ POP_SIZE × (N_GEN + 1)  [+1 for initial population]
# Default = 60.  Increase N_GEN to 15-20 for richer Pareto front.

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION CONSTANTS  (must match Phase 2)
# ══════════════════════════════════════════════════════════════════════════════
DT       = 0.001
T_IGNORE = 0.5
T_END    = 466.945
t_eval_full = np.arange(0.0, T_END + DT, DT)

RESULTS_DIR = "Laden_results_ode_bay_opt2"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

STATE_NAMES = ["z_c", "th_c", "ph_c", "z_s", "th_s", "ph_s"]
(ZC, THC, PHC, ZS, THS, PHS) = range(6)

# Hypervolume reference point (must be above all expected objective values)
HV_REF = np.array([5.0, 5.0, 5.0])

# ══════════════════════════════════════════════════════════════════════════════
# BASE CFG
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
    "hs": 0.68,   "g": 9.81,      "hcp": 0.1,

    "L_DL2": 0.6211, "L_DR2": 0.6211,
    "L_DL3": 0.6251, "L_DR3": 0.6251,
    "beta_L2": 0.1693, "beta_R2": 0.1693,
    "beta_L3": 0.17453,"beta_R3": 0.17453,
    "S_tf2": 1.043, "S_tf3": 1.043, "S_f": 0.814,

    "C_cfl": 5035.0, "C_cfr": 5035.0, "C_crl": 3400.0, "C_crr": 3400.0,
    "K_cfl": 49050.0,"K_cfr": 49050.0,"K_crl": 24525.0,"K_crr": 24525.0,

    # Baseline / default structural values
    "K_f": 474257,  "C_f": 15000,
    "K_2": 1077620, "C_2": 2000,
    "K_3": 1077620, "C_3": 2000,

    # Asymmetric shape params — kept in CFG for reference only.
    # Phase 1 ODE uses PURE LINEAR front damper, so these are NOT used here.
    "cs_minus": 0.3, "asym_ratio": 3.0, "gamma_c": 0.12, "gamma_r": 0.09,

    "baum_omega": 10.0, "baum_zeta": 1.0,
}

# ══════════════════════════════════════════════════════════════════════════════
# SEARCH BOUNDS  (same as original Phase 1)
# C_f bounds are ±20% — tighter than Option-1 because Phase 2 provides
# additional freedom via asymmetric shape fitting.
# ══════════════════════════════════════════════════════════════════════════════
PARAM_KEYS  = ["K_f", "C_f", "K_2", "K_3"]
BOUNDS_RAW  = {
    "K_f": (0.8789 * CFG["K_f"], 1.1289 * CFG["K_f"]),
    "C_f": (0.80   * CFG["C_f"], 1.20   * CFG["C_f"]),
    "K_2": (0.8920 * CFG["K_2"], 1.1142 * CFG["K_2"]),
    "K_3": (0.8920 * CFG["K_3"], 1.1142 * CFG["K_3"]),
}
XL = np.array([BOUNDS_RAW[k][0] for k in PARAM_KEYS])
XU = np.array([BOUNDS_RAW[k][1] for k in PARAM_KEYS])

# Global evaluation log: used for plots and JSON output
_eval_log: list = []   # each entry: {K_f, C_f, K_2, K_3, rms_z, rms_x, rms_y, gen}

# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS  (identical to original Phase 1 — linear front damper)
# ══════════════════════════════════════════════════════════════════════════════

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
        z1f  = 0.5*(zr1L+zr1R); z2 = 0.5*(zr2L+zr2R); z3 = 0.5*(zr3L+zr3R)
        ph_f = (zr1L-zr1R)/cfg["WT1"]
        ph2  = (zr2L-zr2R)/cfg["WT2"]
        ph3  = (zr3L-zr3R)/cfg["WT3"]
        return float(z1f), float(ph_f), float(z2), float(ph2), float(z3), float(ph3)

    def axle_input_rates(self, t: float, cfg: Dict, dt: float = DT):
        p = self.axle_inputs(t + dt, cfg)
        m = self.axle_inputs(t - dt, cfg)
        return tuple((a-b)/(2.0*dt) for a, b in zip(p, m))


def build_road_signals(cfg: Dict) -> RoadSignals:
    t1L,z1L = load_track(cfg["axlefront_left_csv"])
    t1R,z1R = load_track(cfg["axlefront_right_csv"])
    t2L,z2L = load_track(cfg["axlerear1_left_csv"])
    t2R,z2R = load_track(cfg["axlerear1_right_csv"])
    t3L,z3L = load_track(cfg["axlerear2_left_csv"])
    t3R,z3R = load_track(cfg["axlerear2_right_csv"])
    return RoadSignals(
        make_linear_interp(t1L,z1L), make_linear_interp(t1R,z1R),
        make_linear_interp(t2L,z2L), make_linear_interp(t2R,z2R),
        make_linear_interp(t3L,z3L), make_linear_interp(t3R,z3R),
    )


def geom_constraints(q, t, cfg, road):
    z_s, th_s, ph_s = q[ZS], q[THS], q[PHS]
    _, _, z2, ph2, z3, ph3 = road.axle_inputs(t, cfg)
    l2 = cfg["L12"]; l3 = cfg["L12"]+cfg["L23"]
    S2,S3   = cfg["S_tf2"],cfg["S_tf3"]
    sl2,sl3 = cfg["s1"],   cfg["s2"]
    bL2,bL3 = cfg["beta_L2"],cfg["beta_L3"]

    g2 = z_s+l2*th_s+S2*ph_s - sl2*np.sin(bL2-th_s) - (z2+0.5*cfg["WT2"]*ph2)
    g3 = z_s+l3*th_s+S3*ph_s - sl3*np.sin(bL3-th_s) - (z3+0.5*cfg["WT3"]*ph3)
    g  = np.array([g2,g3],dtype=float)
    G  = np.zeros((2,6),dtype=float)
    G[0,ZS]=1.; G[0,THS]=l2+sl2*np.cos(bL2-th_s); G[0,PHS]=S2
    G[1,ZS]=1.; G[1,THS]=l3+sl3*np.cos(bL3-th_s); G[1,PHS]=S3
    return g, G


def build_M_R(q, v, t, cfg, road):
    """
    Phase-1 physics: PURELY LINEAR front damper.
    F_df = C_f * v_f  (no asymmetric shape parameters used here).
    """
    z_c,th_c,ph_c,z_s,th_s,ph_s         = q
    dz_c,dth_c,dph_c,dz_s,dth_s,dph_s   = v
    z1f,ph_f,z2,ph2,z3,ph3              = road.axle_inputs(t,cfg)
    dz1f,dph_f,dz2,dph2,dz3,dph3        = road.axle_input_rates(t,cfg)

    phi_NRS2 = (cfg["beta_L2"]*cfg["L_DL2"]-cfg["beta_R2"]*cfg["L_DR2"])/max(cfg["S_tf2"],1e-6)
    phi_NRS3 = (cfg["beta_L3"]*cfg["L_DL3"]-cfg["beta_R3"]*cfg["L_DR3"])/max(cfg["S_tf3"],1e-6)

    m_c,I_xxc,I_yyc     = cfg["m_c"],cfg["I_xxc"],cfg["I_yyc"]
    m_s,I_sxx,I_syy,I_sxy = cfg["m_s"],cfg["I_sxx"],cfg["I_syy"],cfg["I_sxy"]
    S1,S2,S3             = cfg["S_f"],cfg["S_tf2"],cfg["S_tf3"]
    a,b                  = cfg["a"],cfg["b"]
    hs,g                 = cfg["hs"],cfg["g"]
    l_cfcg,l_crcg        = cfg["l_cfcg"],cfg["l_crcg"]
    l_cf,l_cr            = cfg["l_cf"],cfg["l_cr"]
    lf,hcp               = cfg["lf"],cfg["hcp"]
    l2                   = cfg["L12"]; l3 = cfg["L12"]+cfg["L23"]
    bL2,bR2 = cfg["beta_L2"],cfg["beta_R2"]
    bL3,bR3 = cfg["beta_L3"],cfg["beta_R3"]
    L_DL2,L_DR2,L_DL3,L_DR3 = cfg["L_DL2"],cfg["L_DR2"],cfg["L_DL3"],cfg["L_DR3"]
    Kcfl,Kcfr,Kcrl,Kcrr = cfg["K_cfl"],cfg["K_cfr"],cfg["K_crl"],cfg["K_crr"]
    Ccfl,Ccfr,Ccrl,Ccrr = cfg["C_cfl"],cfg["C_cfr"],cfg["C_crl"],cfg["C_crr"]
    K_f,C_f = cfg["K_f"],cfg["C_f"]
    K_2,C_2 = cfg["K_2"],cfg["C_2"]
    K_3,C_3 = cfg["K_3"],cfg["C_3"]

    # ── LINEAR front damper (Phase 1) ────────────────────────────────────────
    v_f  = dz_s - lf*dth_s - dz1f
    F_df = C_f * v_f          # pure linear — asymmetric shape NOT applied here
    # ─────────────────────────────────────────────────────────────────────────

    Csum = Ccfl+Ccfr+Ccrl+Ccrr; Ksum = Kcfl+Kcfr+Kcrl+Kcrr

    M = np.zeros((6,6),dtype=float)
    M[ZC,ZC]=m_c; M[THC,THC]=I_yyc; M[PHC,PHC]=I_xxc
    M[ZS,ZS]=m_s; M[THS,THS]=I_syy
    M[THS,PHS]=I_sxy; M[PHS,THS]=I_sxy
    M[PHS,PHS]=I_sxx+m_s*hs**2

    R = np.zeros(6,dtype=float)
    R[ZC] = (Csum*(dz_c-dz_s)+Ksum*(z_c-z_s)
        -(Ccfl*l_cfcg+Ccfr*l_cfcg-Ccrl*l_crcg-Ccrr*l_crcg)*dth_c
        -(-Ccfl*l_cf-Ccfr*l_cf-Ccrl*l_cr-Ccrr*l_cr)*dth_s
        -(-Ccfl*b+Ccfr*a-Ccrl*b+Ccrr*a)*dph_c
        -(Ccfl*b-Ccfr*a+Ccrl*b-Ccrr*a)*dph_s
        -(Kcfl*l_cfcg+Kcfr*l_cfcg-Kcrl*l_crcg-Kcrr*l_crcg)*th_c
        -(-Kcfl*l_cf-Kcfr*l_cf-Kcrl*l_cr-Kcrr*l_cr)*th_s
        -(-Kcfl*b+Kcfr*a-Kcrl*b+Kcrr*a)*ph_c
        -(Kcfl*b-Kcfr*a+Kcrl*b-Kcrr*a)*ph_s)
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
        -(Kcfl*l_cfcg*b-Kcfr*l_cfcg*a+Kcrl*l_crcg*b-Kcrr*l_crcg*a)*ph_s)
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
        -(Kcfl*b**2-Kcfr*a**2+Kcrl*b**2-Kcrr*a**2)*ph_s)
    R[ZS] = (
        -(Ccfl+Ccfr+Ccrl+Ccrr)*dz_c
        -(-Ccfl*l_cfcg-Ccfr*l_cfcg+Ccrl*l_crcg+Ccrr*l_crcg)*dth_c
        -(-Ccfl-Ccfr-Ccrl-Ccrr)*dz_s
        -(Ccfl*l_cf+Ccfr*l_cf+Ccrl*l_cr+Ccrr*l_cr)*dth_s
        -(Kcfl+Kcfr+Kcrl+Kcrr)*z_c
        -(-Kcfl*l_cfcg-Kcfr*l_cfcg+Kcrl*l_crcg+Kcrr*l_crcg)*th_c
        -(-Kcfl-Kcfr-Kcrl-Kcrr)*z_s
        -(Kcfl*l_cf+Kcfr*l_cf+Kcrl*l_cr+Kcrr*l_cr)*th_s
        +K_f*(z_s-lf*th_s-z1f)+F_df
        +K_2*(z_s-z2-bL2*L_DL2-bR2*L_DR2+l2*th_s)+C_2*(dz_s-dz2+l2*dth_s)
        +K_3*(z_s-z3-bL3*L_DL3-bR3*L_DR3+l3*th_s)+C_3*(dz_s-dz3+l3*dth_s))
    R[THS] = (
        -(Ccfl*l_cfcg+Ccfr*l_cfcg-Ccrl*l_crcg-Ccrr*l_crcg)*dz_c
        -(-Ccfl*l_cfcg**2-Ccfr*l_cfcg**2-Ccrl*l_crcg**2-Ccrr*l_crcg**2)*dth_c
        -(-Ccfl*l_cf-Ccfr*l_cf-Ccrl*l_cr-Ccrr*l_cr)*dz_s
        -(Ccfl*l_cfcg*l_cf+Ccfr*l_cfcg*l_cf-Ccrl*l_crcg*l_cr-Ccrr*l_crcg*l_cr)*dth_s
        -(Kcfl*l_cf+Kcfr*l_cf+Kcrl*l_cr+Kcrr*l_cr)*z_c
        -(-Kcfl*l_cfcg*l_cf-Kcfr*l_cfcg*l_cf+Kcrl*l_crcg*l_cr+Kcrr*l_crcg*l_cr)*th_c
        -(-Kcfl*l_cf-Kcfr*l_cf-Kcrl*l_cr-Kcrr*l_cr)*z_s
        -(Kcfl*l_cf**2+Kcfr*l_cf**2+Kcrl*l_cr**2+Kcrr*l_cr**2)*th_s
        -lf*(K_f*(z_s-lf*th_s-z1f)+F_df)
        +l2*(K_2*(z_s-z2-bL2*L_DL2-bR2*L_DR2+l2*th_s)+C_2*(dz_s-dz2+l2*dth_s))
        +l3*(K_3*(z_s-z3-bL3*L_DL3-bR3*L_DR3+l3*th_s)+C_3*(dz_s-dz3+l3*dth_s)))

    k_tf=0.5*K_f*S1**2; K_r1=0.5*K_2*S2**2; K_r2=0.5*K_3*S3**2
    C_tf=0.5*C_f*S1**2; C_r1=0.5*C_2*S2**2; C_r2=0.5*C_3*S3**2
    R[PHS] = -(
        m_s*g*hs*ph_s
        -k_tf*(ph_s-ph_f)-C_tf*(dph_s-dph_f)
        -K_r1*(ph_s-ph2-phi_NRS2)-C_r1*(dph_s-dph2)
        -K_r2*(ph_s-ph3-phi_NRS3)-C_r2*(dph_s-dph3))
    return M, R


def rhs_first_order(t, x, cfg, road):
    q,v   = x[:6],x[6:]
    M,R   = build_M_R(q,v,t,cfg,road)
    gq,G  = geom_constraints(q,t,cfg,road)
    w,z   = cfg["baum_omega"],cfg["baum_zeta"]
    gamma = w**2*gq + 2*z*w*(G@v)
    nc    = G.shape[0]
    A     = np.zeros((6+nc,6+nc)); b = np.zeros(6+nc)
    A[:6,:6]=M; A[:6,6:]=G.T; A[6:,:6]=G
    b[:6]=-R; b[6:]=-gamma
    xdot=np.zeros_like(x); xdot[:6]=v; xdot[6:]=lin_solve(A,b)[:6]
    return xdot


def static_equilibrium_state(cfg, road):
    t0 = 0.0
    def F(y):
        q,lam = y[:6],y[6:]
        M,R   = build_M_R(q,np.zeros(6),t0,cfg,road)
        gq,G  = geom_constraints(q,t0,cfg,road)
        return np.hstack([R+G.T@lam, 1e3*gq])
    lsq = least_squares(F,np.zeros(8),method="trf",loss="soft_l1",
                        xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=800)
    if lsq.success:
        q0 = lsq.x[:6]
        print(f"    [EQ OK] ||g||={np.linalg.norm(geom_constraints(q0,t0,cfg,road)[0]):.2e}")
        return np.hstack([q0,np.zeros(6)])
    cfg_r = {**cfg, **{k: cfg[k]*20 for k in ["C_2","C_3","C_cfl","C_cfr","C_crl","C_crr"]}}
    sol = solve_ivp(lambda t,x: rhs_first_order(t,x,cfg_r,road),
                    (0.0,3.0),np.zeros(12),method="Radau",rtol=1e-7,atol=1e-9)
    q0 = sol.y[:6,-1]
    print(f"    [EQ fallback] ||g||={np.linalg.norm(geom_constraints(q0,t0,cfg,road)[0]):.2e}")
    return np.hstack([q0,np.zeros(6)])

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def run_one_case(params: Dict, t_eval: np.ndarray):
    """Run ODE, return DataFrame. Also extracts v_rel (t_eval-aligned)."""
    cfg  = {**CFG, **params}
    road = build_road_signals(cfg)
    x0   = static_equilibrium_state(cfg, road)
    sol  = solve_ivp(
        fun=lambda t,x: rhs_first_order(t,x,cfg,road),
        t_span=(float(t_eval[0]),float(t_eval[-1])),
        y0=x0, t_eval=t_eval,
        method="Radau", max_step=0.01, rtol=1e-6, atol=1e-8,
    )
    if sol.status != 0 or not np.all(np.isfinite(sol.y)):
        raise RuntimeError(f"ODE failed: {sol.message}")

    rows = []
    for i,t in enumerate(sol.t):
        x = sol.y[:,i]; qdd = rhs_first_order(t,x,cfg,road)[6:]
        row = {"t": t}
        for j,name in enumerate(STATE_NAMES):
            row[name]=x[j]; row[f"qd_{name}"]=x[j+6]; row[f"qdd_{name}"]=qdd[j]
        rows.append(row)
    df = pd.DataFrame(rows)

    # Reconstruct v_rel post-integration (t_eval-aligned, no adaptive-step bias)
    dz1f_arr = np.array([road.axle_input_rates(t, cfg)[0] for t in df["t"].values])
    v_rel = df["qd_z_s"].values - cfg["lf"] * df["qd_th_s"].values - dz1f_arr

    return df, v_rel


def compute_per_axis_rms(df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Three separate RMS objectives on the CABIN BODY (not sprung mass).
    f1 = RMS_z  : vertical bounce of cabin
    f2 = RMS_x  : longitudinal at seat height  = -hcp * θ̈_c
    f3 = RMS_y  : lateral at seat height       =  hcp * φ̈_c
    """
    mask = df["t"] >= T_IGNORE
    h    = CFG["hcp"]
    az   = df.loc[mask,"qdd_z_c"].values
    ax   = -h * df.loc[mask,"qdd_th_c"].values
    ay   =  h * df.loc[mask,"qdd_ph_c"].values
    return (float(np.sqrt(np.mean(az**2))),
            float(np.sqrt(np.mean(ax**2))),
            float(np.sqrt(np.mean(ay**2))))

# ══════════════════════════════════════════════════════════════════════════════
# pymoo PROBLEM
# ══════════════════════════════════════════════════════════════════════════════

class Phase1Problem(ElementwiseProblem):
    """
    4 variables  → 3 objectives (rms_z, rms_x, rms_y)
    No constraints.
    """
    def __init__(self):
        super().__init__(n_var=4, n_obj=3, xl=XL, xu=XU)
        self._gen_counter = 0

    def _evaluate(self, x, out, *args, **kwargs):
        params = {k: float(x[i]) for i,k in enumerate(PARAM_KEYS)}
        t0 = time.time()
        try:
            df, v_rel = run_one_case(params, t_eval_full)
            rms_z, rms_x, rms_y = compute_per_axis_rms(df)
            print(f"      ODE OK ({time.time()-t0:.1f}s) | "
                  f"z={rms_z:.4f}  x={rms_x:.4f}  y={rms_y:.4f}")
            # Store v_rel on the object so the callback can pick it up
            self._last_vrel  = v_rel
            self._last_params = params
        except Exception as exc:
            print(f"      ODE FAILED: {exc} → penalty")
            rms_z = rms_x = rms_y = 99.0
            self._last_vrel   = None
            self._last_params = params

        out["F"] = [rms_z, rms_x, rms_y]

        entry = {**params, "rms_z": rms_z, "rms_x": rms_x, "rms_y": rms_y,
                 "gen": self._gen_counter}
        _eval_log.append(entry)


class HVCallback:
    def __init__(self):
        self.hv_history = []
        self.ind = HV(ref_point=HV_REF)

    def __call__(self, algorithm):
        algorithm.problem._gen_counter = algorithm.n_gen
        F = algorithm.pop.get("F")
        if F is not None and len(F) > 0:
            nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
            try:    hv = self.ind(F[nds])
            except: hv = 0.0
        else:
            hv = 0.0
        self.hv_history.append(hv)
        print(f"\n  ── Generation {algorithm.n_gen}/{N_GEN}  "
              f"Pareto size={len(nds) if F is not None else 0}  HV={hv:.6f} ──")

# ══════════════════════════════════════════════════════════════════════════════
# PARETO EXTRACTION + LABELLING
# ══════════════════════════════════════════════════════════════════════════════

def extract_pareto(res) -> pd.DataFrame:
    X, F = res.X, res.F
    rows = []
    for i in range(len(X)):
        row = {k: float(X[i,j]) for j,k in enumerate(PARAM_KEYS)}
        row["rms_z"]    = float(F[i,0])
        row["rms_x"]    = float(F[i,1])
        row["rms_y"]    = float(F[i,2])
        row["rms_total"]= float(np.sqrt(F[i,0]**2+F[i,1]**2+F[i,2]**2))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("rms_total").reset_index(drop=True)

    labels = []
    for i in range(len(df)):
        if i == 0:
            labels.append("Best_total")
        elif df["rms_z"].iloc[i] == df["rms_z"].min():
            labels.append("Best_vertical")
        elif df["rms_x"].iloc[i] == df["rms_x"].min():
            labels.append("Best_pitch")
        elif df["rms_y"].iloc[i] == df["rms_y"].min():
            labels.append("Best_roll")
        else:
            labels.append(f"Pareto_{i+1}")
    df.insert(0,"label",labels)
    return df

# ══════════════════════════════════════════════════════════════════════════════
# v_rel COLLECTION — re-run each Pareto solution to get its v_rel
# ══════════════════════════════════════════════════════════════════════════════

def collect_vrel_for_pareto(df_pareto: pd.DataFrame) -> dict:
    """
    Re-run the ODE for each Pareto solution and save the v_rel time-series.
    Returns a dict:  label → v_rel array
    Saves .npy files so Phase 2 can consume any of them.
    """
    vrel_map = {}
    for _, row in df_pareto.iterrows():
        label  = row["label"]
        params = {k: row[k] for k in PARAM_KEYS}
        print(f"\n  [v_rel collection] {label}")
        try:
            df, v_rel = run_one_case(params, t_eval_full)
            vrel_map[label] = v_rel
            npy_path = os.path.join(RESULTS_DIR, f"v_rel_front_{label}.npy")
            np.save(npy_path, v_rel)
            print(f"    Saved → {npy_path}  ({len(v_rel)} samples, "
                  f"range [{v_rel.min():.4f}, {v_rel.max():.4f}] m/s)")
        except Exception as exc:
            print(f"    FAILED: {exc}")
            vrel_map[label] = None
    return vrel_map

# ══════════════════════════════════════════════════════════════════════════════
# SAVE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save_pareto_csv(df_pareto: pd.DataFrame) -> str:
    path = os.path.join(RESULTS_DIR,"pareto_front_phase1.csv")
    df_pareto.to_csv(path,index=False)
    print(f"[CSV] Pareto front → {path}")
    return path


def save_run_json(df_pareto: pd.DataFrame) -> str:
    """
    Saves full run log + Pareto summary.
    The 'phase2_input' section clearly tells Phase 2 which files to load
    and which Pareto solution to use as default.
    """
    valid_logs = [e for e in _eval_log if e.get("rms_z", 99) < 90]
    best_row   = df_pareto.iloc[0]   # Best_total

    # Normalised param position for best solution
    norm = {}
    for k in PARAM_KEYS:
        lo,hi = BOUNDS_RAW[k]
        norm[k] = round((float(best_row[k])-lo)/(hi-lo), 4)

    out = {
        "description": (
            "Phase-1 NSGA-II multi-objective results (linear front damper). "
            "Three objectives: rms_z (vertical), rms_x (pitch), rms_y (roll) "
            "cabin seat accelerations [m/s²]. "
            "pareto_front = non-dominated structural parameter sets. "
            "phase2_input = recommended files/params to pass to Phase-2."
        ),
        "config": {"pop_size": POP_SIZE, "n_gen": N_GEN,
                   "total_evals_approx": POP_SIZE*(N_GEN+1)},
        "bounds": {k: {"lo":float(v[0]),"hi":float(v[1])} for k,v in BOUNDS_RAW.items()},
        "pareto_front": df_pareto.to_dict(orient="records"),
        "best_total": {
            "label":   best_row["label"],
            "params":  {k: float(best_row[k]) for k in PARAM_KEYS},
            "normalised_in_range": norm,
            "rms_z":   float(best_row["rms_z"]),
            "rms_x":   float(best_row["rms_x"]),
            "rms_y":   float(best_row["rms_y"]),
            "rms_total": float(best_row["rms_total"]),
        },
        # ── This section is consumed directly by Phase 2 ──────────────────
        "phase2_input": {
            "default_solution":     best_row["label"],
            "params_file":          "phase1_run_results.json",
            "vrel_file_template":   "v_rel_front_<label>.npy",
            "vrel_file_default":    f"v_rel_front_{best_row['label']}.npy",
            "CF_star":              float(best_row["C_f"]),
            "K_f_star":             float(best_row["K_f"]),
            "K_2_star":             float(best_row["K_2"]),
            "K_3_star":             float(best_row["K_3"]),
            "all_pareto_solutions": [
                {"label": row["label"],
                 "vrel_file": f"v_rel_front_{row['label']}.npy",
                 "CF_star": float(row["C_f"]),
                 "params": {k: float(row[k]) for k in PARAM_KEYS}}
                for _, row in df_pareto.iterrows()
            ],
        },
        # ──────────────────────────────────────────────────────────────────
        "all_evaluations": valid_logs,
    }
    path = os.path.join(RESULTS_DIR,"phase1_run_results.json")
    with open(path,"w") as fh:
        json.dump(out,fh,indent=2)
    print(f"[JSON] Full run log → {path}")
    return path


def save_selected_params_json(df_pareto: pd.DataFrame) -> str:
    """
    Saves phase1_selected_params.json — a simple flat file
    that Phase 2 can load without parsing the full run log.
    Default = Best_total; engineer can swap to any Pareto label manually.
    """
    best = df_pareto.iloc[0]
    out  = {k: float(best[k]) for k in PARAM_KEYS}
    out["C_2"] = CFG["C_2"]
    out["C_3"] = CFG["C_3"]
    # This mirrors the format expected by the original phase2 JSON loader:
    # p1_data["best"]["params"]  →  flatten to top-level here for simplicity
    wrapped = {
        "best": {
            "params": out,
            "label":  best["label"],
            "rms_z":  float(best["rms_z"]),
            "rms_x":  float(best["rms_x"]),
            "rms_y":  float(best["rms_y"]),
            "rms_total": float(best["rms_total"]),
        },
        "baseline_rms": {
            "rms_z": float("nan"),   # baseline not re-run here to save budget
            "rms_x": float("nan"),
            "rms_y": float("nan"),
            "rms_total": float("nan"),
        },
        "optimised_rms": {
            "rms_z":     float(best["rms_z"]),
            "rms_x":     float(best["rms_x"]),
            "rms_y":     float(best["rms_y"]),
            "rms_total": float(best["rms_total"]),
        },
        "all_pareto_labels": df_pareto["label"].tolist(),
    }
    path = os.path.join(RESULTS_DIR,"phase1_best_params.json")
    with open(path,"w") as fh:
        json.dump(wrapped,fh,indent=2)
    print(f"[JSON] Phase-2 input params → {path}")
    return path

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def _all_F_valid():
    return np.array([[e["rms_z"],e["rms_x"],e["rms_y"]]
                     for e in _eval_log if e.get("rms_z",99)<90])


def plot_pareto_2d(df_pareto):
    all_f = _all_F_valid()
    obj   = {"rms_z":0,"rms_x":1,"rms_y":2}
    lax   = {"rms_z":"RMS_z vertical [m/s²]",
             "rms_x":"RMS_x longitudinal [m/s²]",
             "rms_y":"RMS_y lateral [m/s²]"}
    for xa,ya in [("rms_z","rms_x"),("rms_z","rms_y"),("rms_x","rms_y")]:
        fig,ax = plt.subplots(figsize=(7,5))
        if len(all_f):
            ax.scatter(all_f[:,obj[xa]],all_f[:,obj[ya]],
                       c="lightgrey",s=18,zorder=1,label="All evaluated")
        sc = ax.scatter(df_pareto[xa],df_pareto[ya],
                        c=df_pareto["rms_total"],cmap="plasma",
                        s=90,zorder=3,edgecolors="k",linewidths=0.5,
                        label="Pareto front")
        plt.colorbar(sc,ax=ax,label="RMS_total [m/s²]")
        for _,row in df_pareto.iterrows():
            ax.annotate(row["label"],(row[xa],row[ya]),
                        fontsize=7,xytext=(4,4),textcoords="offset points")
        ax.set_xlabel(lax[xa]); ax.set_ylabel(lax[ya])
        ax.set_title(f"Phase-1 Pareto: {xa} vs {ya}")
        ax.legend(fontsize=8); ax.grid(True,alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR,f"p1_pareto_{xa}_vs_{ya}.png"),dpi=150)
        plt.close()


def plot_pareto_3d(df_pareto):
    all_f = _all_F_valid()
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111,projection="3d")
    if len(all_f):
        ax.scatter(all_f[:,0],all_f[:,1],all_f[:,2],c="lightgrey",s=12,alpha=0.4)
    sc = ax.scatter(df_pareto["rms_z"],df_pareto["rms_x"],df_pareto["rms_y"],
                    c=df_pareto["rms_total"],cmap="plasma",
                    s=100,edgecolors="k",linewidths=0.5)
    plt.colorbar(sc,ax=ax,label="RMS_total",pad=0.1,shrink=0.6)
    for _,row in df_pareto.iterrows():
        ax.text(row["rms_z"],row["rms_x"],row["rms_y"],row["label"],fontsize=6)
    ax.set_xlabel("RMS_z"); ax.set_ylabel("RMS_x"); ax.set_zlabel("RMS_y")
    ax.set_title("Phase-1 3-D Pareto Front  (linear damper)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,"p1_pareto_3d.png"),dpi=150)
    plt.close()


def plot_hypervolume(hv_history):
    plt.figure(figsize=(7,4))
    plt.plot(range(1,len(hv_history)+1),hv_history,marker="o",ms=4,color="steelblue")
    plt.xlabel("Generation"); plt.ylabel("Hypervolume")
    plt.title("Phase-1 Hypervolume per Generation")
    plt.grid(True,alpha=0.4); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,"p1_hypervolume.png"),dpi=150)
    plt.close()


def plot_rms_bars(df_pareto):
    labels = df_pareto["label"].tolist()
    x=np.arange(len(labels)); w=0.25
    fig,ax=plt.subplots(figsize=(max(9,len(labels)*1.5),5))
    ax.bar(x-w,  df_pareto["rms_z"],w,label="RMS_z (vertical)",    color="steelblue", edgecolor="k")
    ax.bar(x,    df_pareto["rms_x"],w,label="RMS_x (longitudinal)",color="darkorange",edgecolor="k")
    ax.bar(x+w,  df_pareto["rms_y"],w,label="RMS_y (lateral)",     color="seagreen",  edgecolor="k")
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=20,ha="right")
    ax.set_ylabel("RMS seat acceleration [m/s²]")
    ax.set_title("Phase-1 Per-Axis RMS — Pareto Solutions")
    ax.legend(); ax.grid(True,axis="y",alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,"p1_rms_bars.png"),dpi=150)
    plt.close()


def plot_param_parallel(df_pareto):
    """
    Parallel coordinates (normalised) of Pareto structural params.
    Tells engineer where in the search space each Pareto solution sits.
    """
    norm_df = df_pareto.copy()
    for k in PARAM_KEYS:
        lo,hi = BOUNDS_RAW[k]
        norm_df[k] = (df_pareto[k]-lo)/(hi-lo)
    n    = len(df_pareto)
    cmap = cm.get_cmap("plasma",n)
    xs   = list(range(len(PARAM_KEYS)))
    fig,ax = plt.subplots(figsize=(10,5))
    for i in range(n):
        ax.plot(xs,norm_df[PARAM_KEYS].iloc[i].values,
                color=cmap(i),lw=1.4,alpha=0.85,label=df_pareto["label"].iloc[i])
    ax.set_xticks(xs); ax.set_xticklabels(PARAM_KEYS,rotation=20,ha="right",fontsize=9)
    ax.set_ylabel("Normalised value in search range [0–1]")
    ax.set_title("Phase-1 Parallel Coordinates — Pareto Structural Parameters")
    ax.axhline(0.05,color="red",linestyle=":",lw=0.8)
    ax.axhline(0.95,color="orange",linestyle=":",lw=0.8)
    ax.legend(fontsize=7,loc="upper right",ncol=2)
    ax.grid(True,axis="y",alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,"p1_param_parallel.png"),dpi=150)
    plt.close()


def plot_vrel_distributions(vrel_map: dict):
    """
    Overlay velocity distributions from all Pareto solutions.
    This is a key diagnostic: it shows Phase-2 HOW DIFFERENT the v_rel
    distributions are across Pareto solutions, which directly affects
    the quality of the curve fit in Phase 2.
    """
    fig, axes = plt.subplots(1,2,figsize=(13,5))

    # Left: overlaid histograms
    cmap = cm.get_cmap("plasma",max(len(vrel_map),1))
    for i,(label,vrel) in enumerate(vrel_map.items()):
        if vrel is None: continue
        axes[0].hist(vrel,bins=100,density=True,alpha=0.45,color=cmap(i),label=label)
    axes[0].axvline(0,color="k",lw=1,ls="--")
    axes[0].set_xlabel("v_rel  [m/s]")
    axes[0].set_ylabel("Probability density")
    axes[0].set_title("Front-damper v_rel distributions\nfor each Pareto solution (Phase-1)")
    axes[0].legend(fontsize=7); axes[0].grid(True)

    # Right: box-plot comparison (compresses spread info into one panel)
    data   = [v for v in vrel_map.values() if v is not None]
    labels = [k for k,v in vrel_map.items() if v is not None]
    if data:
        axes[1].boxplot(data,labels=labels,vert=True,patch_artist=True)
        axes[1].axhline(0,color="k",lw=0.8,ls="--")
        axes[1].set_ylabel("v_rel  [m/s]")
        axes[1].set_title("v_rel spread per Pareto solution\n"
                          "(wider spread → Phase-2 fit covers more of the F-v curve)")
        axes[1].grid(True,axis="y",alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,"p1_vrel_comparison.png"),dpi=150)
    plt.close()
    print("[PLOT] p1_vrel_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("="*65)
    print("  PHASE 1 (MULTI-OBJECTIVE NSGA-II) — Linear front damper")
    print(f"  Variables : K_f  C_f  K_2  K_3")
    print(f"  Objectives: RMS_z  RMS_x  RMS_y  (all minimised)")
    print(f"  Budget    : POP={POP_SIZE} × GEN={N_GEN} ≈ {POP_SIZE*(N_GEN+1)} ODE evals")
    print("="*65+"\n")

    problem   = Phase1Problem()
    hv_cb     = HVCallback()
    algorithm = NSGA2(
        pop_size=POP_SIZE,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9,eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    result = pymoo_minimize(
        problem, algorithm, ("n_gen",N_GEN),
        callback=hv_cb, seed=42, verbose=False,
    )

    # ── Pareto extraction ────────────────────────────────────────────────────
    df_pareto = extract_pareto(result)

    print("\n"+"="*65)
    print("  PARETO FRONT (Phase 1 — structural parameters, linear damper)")
    print("="*65)
    print(df_pareto[["label","K_f","C_f","K_2","K_3",
                      "rms_z","rms_x","rms_y","rms_total"]].to_string(index=False))

    # ── v_rel collection for ALL Pareto solutions ────────────────────────────
    print("\n  Collecting v_rel for all Pareto solutions (needed by Phase 2)...")
    vrel_map = collect_vrel_for_pareto(df_pareto)

    # ── Save outputs ─────────────────────────────────────────────────────────
    csv_path  = save_pareto_csv(df_pareto)
    json_path = save_run_json(df_pareto)
    sel_path  = save_selected_params_json(df_pareto)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_pareto_2d(df_pareto)
    plot_pareto_3d(df_pareto)
    plot_hypervolume(hv_cb.hv_history)
    plot_rms_bars(df_pareto)
    plot_param_parallel(df_pareto)
    plot_vrel_distributions(vrel_map)

    # ── Summary ──────────────────────────────────────────────────────────────
    best = df_pareto.iloc[0]
    print("\n"+"="*65)
    print("  SUMMARY")
    print("="*65)
    print(f"  Total ODE evaluations : {len(_eval_log)}")
    print(f"  Pareto front size     : {len(df_pareto)}")
    print(f"\n  Default Phase-2 input → '{best['label']}'")
    print(f"    K_f = {best['K_f']:.2f}  C_f = {best['C_f']:.2f}"
          f"  K_2 = {best['K_2']:.2f}  K_3 = {best['K_3']:.2f}")
    print(f"    RMS z={best['rms_z']:.4f}  x={best['rms_x']:.4f}"
          f"  y={best['rms_y']:.4f}  total={best['rms_total']:.4f} m/s²")
    print(f"\n  v_rel files saved for Phase 2:")
    for label,vrel in vrel_map.items():
        status = f"{len(vrel)} samples" if vrel is not None else "FAILED"
        print(f"    v_rel_front_{label}.npy  ({status})")
    print(f"\n  → Run  mo_phase2_nsga2.py  next.")
    print(f"    It will load  {sel_path}")
    print(f"    and  v_rel_front_{best['label']}.npy  by default.")
    print(f"    Change 'PHASE1_SOLUTION_LABEL' in Phase 2 to pick a different Pareto point.")
    print(f"\n  Outputs : {RESULTS_DIR}/")
