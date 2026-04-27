"""
OPTION 2 — PHASE 2: Asymmetric Damper Curve Fitting
====================================================
Reads from Phase 1:
  • phase1_best_params.json   → CF*, KF*, K2*, K3*
  • v_rel_front.npy           → empirical relative-velocity time-series

Finds [cs_minus, asym_ratio, gamma_c, gamma_r] such that the
two-stage asymmetric damper best matches the linear optimal damper
CF* over the actual road-induced velocity distribution.

Matching criterion  (industry-standard):
  Minimise  Σ_v  w(v) · [F_asym(v) − CF* · v]²

  where w(v) is the empirical KDE-based velocity probability density
  — velocities the damper actually sees are weighted more heavily.

After fitting, a full ODE validation run is executed and the RMS
degradation vs the Phase-1 linear optimum is reported.

Output
------
  phase2_asym_params.json      fitted asymmetric parameters
  plots/fv_curve_comparison.png
  plots/seat_comparison_p1_p2.png
  plots/vrel_histogram_fit.png
  plots/validation_rms_summary.png
"""

# ── imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import json, os, time
import matplotlib.pyplot as plt
from scipy.optimize   import minimize
from scipy.stats      import gaussian_kde
from dataclasses      import dataclass
from typing           import Dict, Callable, Tuple
from numpy.linalg     import solve as lin_solve
from scipy.integrate  import solve_ivp
from scipy.optimize   import least_squares

# ── paths — point at the Phase-1 output directory ────────────────────────────
PHASE1_DIR  = "Laden_results_ode_bay_opt2"       # same as Phase 1 RESULTS_DIR
RESULTS_DIR = PHASE1_DIR
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION CONSTANTS  (must match Phase 1)
# ══════════════════════════════════════════════════════════════════════════════
DT       = 0.001
FS       = 1000
T_IGNORE = 0.5
T_END    = 466.945
t_eval_full = np.arange(0.0, T_END + DT, DT)

STATE_NAMES = ["z_c", "th_c", "ph_c", "z_s", "th_s", "ph_s"]
(ZC, THC, PHC, ZS, THS, PHS) = range(6)

# ── damper breakpoints (physical / from characterisation rig) ────────────────
ALPHA_C = -0.05   # m/s  compression breakpoint
ALPHA_R =  0.13   # m/s  rebound    breakpoint

# ══════════════════════════════════════════════════════════════════════════════
# BASE VEHICLE CONFIG  (identical to Phase 1)
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
    # asymmetric params — will be overwritten during validation
    "cs_minus":   0.3, "asym_ratio": 3.0,
    "gamma_c":    0.12, "gamma_r":   0.09,
    "baum_omega": 10.0, "baum_zeta": 1.0,
}

# ══════════════════════════════════════════════════════════════════════════════
# TWO-STAGE ASYMMETRIC DAMPER  (numpy — used in Phase-2 fitting & validation)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class TwoStageAsymmetricDamper:
    cs_minus:   float          # Compression low-speed slope
    asym_ratio: float          # c_plus / cs_minus
    gamma_c:    float          # High-speed compression slope multiplier
    gamma_r:    float          # High-speed rebound slope multiplier
    alpha_c:    float = ALPHA_C
    alpha_r:    float = ALPHA_R

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

    def force_array(self, v: np.ndarray) -> np.ndarray:
        """Vectorised version for fitting."""
        c_plus = self.asym_ratio * self.cs_minus
        F = np.empty_like(v)
        # compression
        mask_cls = (v < 0) & (v >= self.alpha_c)
        mask_chs = (v < 0) & (v <  self.alpha_c)
        F[mask_cls] = self.cs_minus * v[mask_cls]
        F[mask_chs] = self.cs_minus * (self.alpha_c + self.gamma_c * (v[mask_chs] - self.alpha_c))
        # rebound
        mask_rls = (v >= 0) & (v <= self.alpha_r)
        mask_rhs = (v >= 0) & (v >  self.alpha_r)
        F[mask_rls] = c_plus * v[mask_rls]
        F[mask_rhs] = c_plus * (self.alpha_r + self.gamma_r * (v[mask_rhs] - self.alpha_r))
        return F


def asym_force_array(v: np.ndarray, cs_minus, asym_ratio, gamma_c, gamma_r,
                     alpha_c=ALPHA_C, alpha_r=ALPHA_R) -> np.ndarray:
    """Standalone vectorised helper for the optimiser."""
    d = TwoStageAsymmetricDamper(cs_minus, asym_ratio, gamma_c, gamma_r, alpha_c, alpha_r)
    return d.force_array(v)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE-2 CORE: FIT ASYMMETRIC CURVE TO MATCH CF*
# ══════════════════════════════════════════════════════════════════════════════
def fit_asymmetric_damper(v_rel_samples: np.ndarray, CF_star: float,
                          n_starts: int = 30) -> Dict:
    """
    Find [cs_minus, asym_ratio, gamma_c, gamma_r] that minimise the
    KDE-weighted mean-squared error between the asymmetric damper and
    the optimal linear damper  F_lin = CF_star * v  over the velocity
    distribution actually seen during the optimal simulation.

    Parameters
    ----------
    v_rel_samples : empirical v_rel time-series from Phase-1 optimal run
    CF_star       : optimal linear CF from Phase-1
    n_starts      : number of random restarts (avoids local minima)

    Returns
    -------
    dict with fitted parameters + metadata
    """
    print(f"\n=== Phase-2 Curve Fitting ===")
    print(f"  CF_star = {CF_star:.2f}  |  v_rel samples = {len(v_rel_samples)}")

    # ── 1. Build evaluation grid weighted by empirical PDF ──────────────────
    # Dense grid covering the 1st–99th percentile of observed velocities
    v_lo = np.percentile(v_rel_samples, 1.0)
    v_hi = np.percentile(v_rel_samples, 99.0)
    v_eval = np.linspace(v_lo, v_hi, 600)

    # KDE weights — higher weight where damper actually operates
    kde     = gaussian_kde(v_rel_samples, bw_method="silverman")
    weights = kde(v_eval)
    weights = weights / weights.sum()            # normalise to sum = 1

    # Target: linear damper force
    F_target = CF_star * v_eval

    # ── 2. Objective  (weighted MSE, normalised by CF*² for scale-free grad) ─
    def objective(params):
        cs_minus, asym_ratio, gamma_c, gamma_r = params
        F_asym = asym_force_array(v_eval, cs_minus, asym_ratio, gamma_c, gamma_r)
        residual = F_asym - F_target
        return float(np.dot(weights, residual**2)) / (CF_star**2 + 1e-12)

    # ── 3. Physical bounds ───────────────────────────────────────────────────
    # cs_minus  : positive, order-of-magnitude around CF*
    #   • at low speed, F_comp = cs_minus * v  →  cs_minus ≈ CF* is a natural anchor
    #   • allow [0.2·CF*, 2·CF*]
    # asym_ratio: >1 means rebound stiffer than compression (typical for trucks)
    # gamma_c/r : <1 means progressive damping at high speed (degressive)
    bounds_list = [
        (0.20 * CF_star,  2.00 * CF_star),   # cs_minus
        (1.0,             5.0),              # asym_ratio
        (0.05,            0.95),             # gamma_c
        (0.05,            0.95),             # gamma_r
    ]

    # ── 4. Multi-start L-BFGS-B ─────────────────────────────────────────────
    rng       = np.random.default_rng(seed=42)
    best_res  = None
    best_cost = np.inf

    for trial_idx in range(n_starts):
        # Random start uniformly within bounds
        x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds_list])
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds_list,
                       options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10})
        if res.fun < best_cost:
            best_cost = res.fun
            best_res  = res

    cs_minus_f, asym_ratio_f, gamma_c_f, gamma_r_f = best_res.x

    # ── 5. Goodness-of-fit metrics ───────────────────────────────────────────
    F_fitted  = asym_force_array(v_eval, cs_minus_f, asym_ratio_f, gamma_c_f, gamma_r_f)
    force_rms = float(np.sqrt(np.dot(weights, (F_fitted - F_target)**2)))
    # Energy dissipated per cycle (trapezoidal over the velocity axis, weighted by PDF)
    E_linear = float(np.trapz(F_target * v_eval * weights, v_eval))
    E_asym   = float(np.trapz(F_fitted * v_eval * weights, v_eval))
    energy_err_pct = abs(E_asym - E_linear) / (abs(E_linear) + 1e-12) * 100.0

    print(f"\n  Fitted parameters:")
    print(f"    cs_minus   = {cs_minus_f:.4f}  N/(m/s)")
    print(f"    asym_ratio = {asym_ratio_f:.4f}  (c+ / c-)")
    print(f"    gamma_c    = {gamma_c_f:.4f}  (high-speed compr. multiplier)")
    print(f"    gamma_r    = {gamma_r_f:.4f}  (high-speed reb.  multiplier)")
    print(f"\n  Goodness-of-fit:")
    print(f"    Weighted force RMSE = {force_rms:.2f} N")
    print(f"    Energy dissipation error = {energy_err_pct:.2f} %")
    print(f"    Optimisation cost  = {best_cost:.6e}")

    return {
        "cs_minus":        float(cs_minus_f),
        "asym_ratio":      float(asym_ratio_f),
        "gamma_c":         float(gamma_c_f),
        "gamma_r":         float(gamma_r_f),
        "alpha_c":         ALPHA_C,
        "alpha_r":         ALPHA_R,
        "weighted_force_rmse": force_rms,
        "energy_error_pct":    energy_err_pct,
        "optimizer_cost":  best_cost,
    }, v_eval, weights, F_target, F_fitted


# ══════════════════════════════════════════════════════════════════════════════
# ROAD / ODE INFRASTRUCTURE  (identical to Phase 1 — needed for validation)
# ══════════════════════════════════════════════════════════════════════════════
def load_track(csv_path):
    df   = pd.read_csv(csv_path, skiprows=2, header=None)
    t    = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
    z    = pd.to_numeric(df.iloc[:, 1], errors="coerce").values
    mask = np.isfinite(t) & np.isfinite(z)
    return t[mask].astype(float), z[mask].astype(float)

def make_linear_interp(x, y):
    x = np.asarray(x); y = np.asarray(y)
    def f(xq):
        xq   = np.asarray(xq)
        xq_c = np.clip(xq, x[0], x[-1])
        idx  = np.clip(np.searchsorted(x, xq_c) - 1, 0, len(x)-2)
        w    = (xq_c - x[idx]) / np.maximum(x[idx+1]-x[idx], 1e-12)
        return y[idx]*(1-w) + y[idx+1]*w
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
        return (float(0.5*(zr1L+zr1R)), float((zr1L-zr1R)/cfg["WT1"]),
                float(0.5*(zr2L+zr2R)), float((zr2L-zr2R)/cfg["WT2"]),
                float(0.5*(zr3L+zr3R)), float((zr3L-zr3R)/cfg["WT3"]))

    def axle_input_rates(self, t, cfg, dt=DT):
        p = self.axle_inputs(t+dt, cfg); m = self.axle_inputs(t-dt, cfg)
        return tuple((a-b)/(2*dt) for a, b in zip(p, m))

def build_road_signals(cfg):
    return RoadSignals(
        make_linear_interp(*load_track(cfg["axlefront_left_csv"])),
        make_linear_interp(*load_track(cfg["axlefront_right_csv"])),
        make_linear_interp(*load_track(cfg["axlerear1_left_csv"])),
        make_linear_interp(*load_track(cfg["axlerear1_right_csv"])),
        make_linear_interp(*load_track(cfg["axlerear2_left_csv"])),
        make_linear_interp(*load_track(cfg["axlerear2_right_csv"])),
    )

def geom_constraints(q, t, cfg, road):
    z_s, th_s, ph_s = q[ZS], q[THS], q[PHS]
    _, _, z2, ph2, z3, ph3 = road.axle_inputs(t, cfg)
    l2 = cfg["L12"]; l3 = cfg["L12"] + cfg["L23"]
    S2, S3  = cfg["S_tf2"], cfg["S_tf3"]
    sl2, sl3 = cfg["s1"], cfg["s2"]
    bL2, bL3 = cfg["beta_L2"], cfg["beta_L3"]
    g2 = z_s+l2*th_s+S2*ph_s - sl2*np.sin(bL2-th_s) - (z2+0.5*cfg["WT2"]*ph2)
    g3 = z_s+l3*th_s+S3*ph_s - sl3*np.sin(bL3-th_s) - (z3+0.5*cfg["WT3"]*ph3)
    G = np.zeros((2,6))
    G[0,ZS]=1; G[0,THS]=l2+sl2*np.cos(bL2-th_s); G[0,PHS]=S2
    G[1,ZS]=1; G[1,THS]=l3+sl3*np.cos(bL3-th_s); G[1,PHS]=S3
    return np.array([g2,g3]), G


def build_M_R_asym(q, v, t, cfg, road):
    """
    PHASE-2 VALIDATION VERSION
    ──────────────────────────
    Front damper uses the full TwoStageAsymmetricDamper with the
    fitted parameters stored in cfg.
    """
    z_c,th_c,ph_c,z_s,th_s,ph_s   = q
    dz_c,dth_c,dph_c,dz_s,dth_s,dph_s = v
    z1f,ph_f,z2,ph2,z3,ph3 = road.axle_inputs(t,cfg)
    dz1f,dph_f,dz2,dph2,dz3,dph3 = road.axle_input_rates(t,cfg)

    phi_NRS2 = (cfg["beta_L2"]*cfg["L_DL2"]-cfg["beta_R2"]*cfg["L_DR2"])/max(cfg["S_tf2"],1e-6)
    phi_NRS3 = (cfg["beta_L3"]*cfg["L_DL3"]-cfg["beta_R3"]*cfg["L_DR3"])/max(cfg["S_tf3"],1e-6)

    m_c,I_xxc,I_yyc = cfg["m_c"],cfg["I_xxc"],cfg["I_yyc"]
    m_s,I_sxx,I_syy,I_sxy = cfg["m_s"],cfg["I_sxx"],cfg["I_syy"],cfg["I_sxy"]
    S1,S2,S3 = cfg["S_f"],cfg["S_tf2"],cfg["S_tf3"]
    a,b,hs,g = cfg["a"],cfg["b"],cfg["hs"],cfg["g"]
    l_cfcg,l_crcg = cfg["l_cfcg"],cfg["l_crcg"]
    l_cf,l_cr,lf,hcp = cfg["l_cf"],cfg["l_cr"],cfg["lf"],cfg["hcp"]
    l2 = cfg["L12"]; l3 = cfg["L12"]+cfg["L23"]
    bL2,bR2,bL3,bR3 = cfg["beta_L2"],cfg["beta_R2"],cfg["beta_L3"],cfg["beta_R3"]
    L_DL2,L_DR2,L_DL3,L_DR3 = cfg["L_DL2"],cfg["L_DR2"],cfg["L_DL3"],cfg["L_DR3"]
    Kcfl,Kcfr,Kcrl,Kcrr = cfg["K_cfl"],cfg["K_cfr"],cfg["K_crl"],cfg["K_crr"]
    Ccfl,Ccfr,Ccrl,Ccrr = cfg["C_cfl"],cfg["C_cfr"],cfg["C_crl"],cfg["C_crr"]
    K_f,C_f,K_2,C_2,K_3,C_3 = cfg["K_f"],cfg["C_f"],cfg["K_2"],cfg["C_2"],cfg["K_3"],cfg["C_3"]

    # ── PHASE-2: ASYMMETRIC front damper ─────────────────────────────────────
    asym = TwoStageAsymmetricDamper(
        cs_minus   = cfg["cs_minus"],
        asym_ratio = cfg["asym_ratio"],
        gamma_c    = cfg["gamma_c"],
        gamma_r    = cfg["gamma_r"],
        alpha_c    = cfg.get("alpha_c", ALPHA_C),
        alpha_r    = cfg.get("alpha_r", ALPHA_R),
    )
    v_f  = dz_s - lf*dth_s - dz1f
    F_df = C_f * asym.force(v_f)   # scaled by C_f exactly as in your original
    # ─────────────────────────────────────────────────────────────────────────

    Csum = Ccfl+Ccfr+Ccrl+Ccrr; Ksum = Kcfl+Kcfr+Kcrl+Kcrr

    M = np.zeros((6,6))
    M[ZC,ZC]=m_c; M[THC,THC]=I_yyc; M[PHC,PHC]=I_xxc
    M[ZS,ZS]=m_s; M[THS,THS]=I_syy; M[THS,PHS]=I_sxy
    M[PHS,THS]=I_sxy; M[PHS,PHS]=I_sxx+m_s*hs**2

    R = np.zeros(6)
    R[ZC] = (
        + Csum*(dz_c-dz_s)+Ksum*(z_c-z_s)
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
        + K_f*(z_s-lf*th_s-z1f)+F_df
        + K_2*(z_s-z2-bL2*L_DL2-bR2*L_DR2+l2*th_s)+C_2*(dz_s-dz2+l2*dth_s)
        + K_3*(z_s-z3-bL3*L_DL3-bR3*L_DR3+l3*th_s)+C_3*(dz_s-dz3+l3*dth_s))
    R[THS] = (
        -(Ccfl*l_cfcg+Ccfr*l_cfcg-Ccrl*l_crcg-Ccrr*l_crcg)*dz_c
        -(-Ccfl*l_cfcg**2-Ccfr*l_cfcg**2-Ccrl*l_crcg**2-Ccrr*l_crcg**2)*dth_c
        -(-Ccfl*l_cf-Ccfr*l_cf-Ccrl*l_cr-Ccrr*l_cr)*dz_s
        -(Ccfl*l_cfcg*l_cf+Ccfr*l_cfcg*l_cf-Ccrl*l_crcg*l_cr-Ccrr*l_crcg*l_cr)*dth_s
        -(Kcfl*l_cf+Kcfr*l_cf+Kcrl*l_cr+Kcrr*l_cr)*z_c
        -(-Kcfl*l_cfcg*l_cf-Kcfr*l_cfcg*l_cf+Kcrl*l_crcg*l_cr+Kcrr*l_crcg*l_cr)*th_c
        -(-Kcfl*l_cf-Kcfr*l_cf-Kcrl*l_cr-Kcrr*l_cr)*z_s
        -(Kcfl*l_cf**2+Kcfr*l_cf**2+Kcrl*l_cr**2+Kcrr*l_cr**2)*th_s
        - lf*(K_f*(z_s-lf*th_s-z1f)+F_df)
        + l2*(K_2*(z_s-z2-bL2*L_DL2-bR2*L_DR2+l2*th_s)+C_2*(dz_s-dz2+l2*dth_s))
        + l3*(K_3*(z_s-z3-bL3*L_DL3-bR3*L_DR3+l3*th_s)+C_3*(dz_s-dz3+l3*dth_s)))

    k_tf=0.5*K_f*S1**2; K_r1=0.5*K_2*S2**2; K_r2=0.5*K_3*S3**2
    C_tf=0.5*C_f*S1**2; C_r1=0.5*C_2*S2**2; C_r2=0.5*C_3*S3**2
    R[PHS] = (m_s*g*hs*ph_s
              - k_tf*(ph_s-ph_f) - C_tf*(dph_s-dph_f)
              - K_r1*(ph_s-ph2-phi_NRS2) - C_r1*(dph_s-dph2)
              - K_r2*(ph_s-ph3-phi_NRS3) - C_r2*(dph_s-dph3))
    R[PHS] *= -1.0
    return M, R


def rhs_asym(t, x, cfg, road):
    q, v = x[:6], x[6:]
    M, R = build_M_R_asym(q, v, t, cfg, road)
    gq, G = geom_constraints(q, t, cfg, road)
    w, zeta = cfg["baum_omega"], cfg["baum_zeta"]
    gamma = w**2*gq + 2*zeta*w*(G@v)
    nc = G.shape[0]
    A  = np.zeros((6+nc, 6+nc)); b = np.zeros(6+nc)
    A[:6,:6]=M; A[:6,6:]=G.T; A[6:,:6]=G
    b[:6]=-R; b[6:]=-gamma
    xdot = np.zeros_like(x)
    xdot[:6]=v; xdot[6:]=lin_solve(A,b)[:6]
    return xdot


def static_equilibrium_state(cfg, road, rhs_fn):
    y0 = np.zeros(8)
    def F(y):
        q,lam = y[:6],y[6:]
        M,R   = build_M_R_asym(q, np.zeros(6), 0.0, cfg, road)
        gq,G  = geom_constraints(q, 0.0, cfg, road)
        return np.hstack([R+G.T@lam, 1e3*gq])
    lsq = least_squares(F, y0, method="trf", loss="soft_l1",
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=800)
    if lsq.success:
        return np.hstack([lsq.x[:6], np.zeros(6)])
    cfg_r = {**cfg, **{k: cfg[k]*20 for k in ["C_2","C_3","C_cfl","C_cfr","C_crl","C_crr"]}}
    sol_r = solve_ivp(lambda t,x: rhs_fn(t,x,cfg_r,road),
                      (0,3), np.zeros(12), method="Radau", rtol=1e-7, atol=1e-9)
    q_r   = sol_r.y[:6,-1]
    lsq2  = least_squares(F, np.hstack([q_r,np.zeros(2)]), method="trf", loss="soft_l1",
                          xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
    return np.hstack([lsq2.x[:6] if lsq2.success else q_r, np.zeros(6)])


def run_validation(cfg_val: Dict, t_eval: np.ndarray) -> Tuple[pd.DataFrame, float]:
    """Run full ODE with asymmetric damper and return (df, seat_rms)."""
    road = build_road_signals(cfg_val)
    x0   = static_equilibrium_state(cfg_val, road, rhs_asym)
    print(f"\n=== Validation run | T_end={t_eval[-1]:.1f} s")
    t0 = time.time()
    sol = solve_ivp(lambda t,x: rhs_asym(t,x,cfg_val,road),
                    (float(t_eval[0]), float(t_eval[-1])), x0,
                    t_eval=t_eval, method="Radau",
                    max_step=0.01, rtol=1e-6, atol=1e-8)
    print(f"=== success={sol.success}, nfev={sol.nfev}, wall={time.time()-t0:.1f} s")
    if sol.status != 0 or not np.all(np.isfinite(sol.y)):
        raise RuntimeError("Validation ODE failed")
    rows = []
    for i, t in enumerate(sol.t):
        x = sol.y[:, i]
        qdd = rhs_asym(t, x, cfg_val, road)[6:]
        row = {"t": t}
        for j, name in enumerate(STATE_NAMES):
            row[name] = x[j]; row[f"qd_{name}"]=x[j+6]; row[f"qdd_{name}"]=qdd[j]
        rows.append(row)
    df  = pd.DataFrame(rows)
    mask = df["t"] >= T_IGNORE
    rms  = float(np.sqrt(np.mean((df.loc[mask,"qdd_z_s"] + cfg_val["hcp"]*df.loc[mask,"qdd_ph_s"])**2)))
    return df, rms


# ══════════════════════════════════════════════════════════════════════════════
# PHASE-2 DIAGNOSTIC PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def plot_fv_comparison(v_eval, weights, F_target, F_fitted,
                       CF_star, fitted_params, save_dir):
    """F-v curve: linear target vs fitted asymmetric, weighted by velocity PDF."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # — left: F-v overlay
    ax = axes[0]
    ax.plot(v_eval, F_target, "b-",  lw=2.5, label=f"Linear (CF*={CF_star:.0f} N·s/m)")
    ax.plot(v_eval, F_fitted, "r--", lw=2.5,
            label=f"Asymmetric fit\n"
                  f"cs⁻={fitted_params['cs_minus']:.3f}, γ={fitted_params['asym_ratio']:.2f}\n"
                  f"γ_c={fitted_params['gamma_c']:.3f}, γ_r={fitted_params['gamma_r']:.3f}")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.axvline(ALPHA_C, color="grey", lw=0.8, ls=":", label=f"α_c={ALPHA_C}")
    ax.axvline(ALPHA_R, color="grey", lw=0.8, ls="-.", label=f"α_r={ALPHA_R}")
    ax.set_xlabel("Relative velocity  [m/s]")
    ax.set_ylabel("Damper force  [N]")
    ax.set_title("F-v Curve: Linear Target vs Asymmetric Fit")
    ax.legend(fontsize=8); ax.grid(True)

    # — right: weighted residual
    ax = axes[1]
    residual = F_fitted - F_target
    ax.fill_between(v_eval, residual, 0, where=residual >= 0,
                    color="tomato", alpha=0.6, label="Over")
    ax.fill_between(v_eval, residual, 0, where=residual <  0,
                    color="steelblue", alpha=0.6, label="Under")
    ax2 = ax.twinx()
    ax2.plot(v_eval, weights / weights.max(), "k-", lw=1, alpha=0.5, label="PDF (norm.)")
    ax2.set_ylabel("Normalised PDF"); ax2.set_ylim(0, 2)
    ax.set_xlabel("Relative velocity  [m/s]")
    ax.set_ylabel("Force residual  [N]")
    ax.set_title("Weighted Residual  (F_asym − F_linear)")
    ax.axhline(0, color="k", lw=0.8)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fv_curve_comparison.png"), dpi=150)
    plt.close()
    print(f"  Saved → fv_curve_comparison.png")


def plot_seat_p1_p2(df_p1, df_p2, rms_p1, rms_p2, cfg, save_dir):
    t = df_p1["t"]
    a1 = df_p1["qdd_z_s"] + cfg["hcp"]*df_p1["qdd_ph_s"]
    a2 = df_p2["qdd_z_s"] + cfg["hcp"]*df_p2["qdd_ph_s"]
    plt.figure(figsize=(10, 4))
    plt.plot(t, a1, "b-",  alpha=0.7, lw=0.8, label=f"Phase-1 linear   (RMS={rms_p1:.4f})")
    plt.plot(t, a2, "r--", alpha=0.7, lw=0.8, label=f"Phase-2 asymmetric (RMS={rms_p2:.4f})")
    plt.xlabel("Time [s]"); plt.ylabel("Seat accel [m/s²]")
    plt.title("Seat Acceleration — Phase 1 (linear) vs Phase 2 (asymmetric)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "seat_comparison_p1_p2.png"), dpi=150)
    plt.close()
    print(f"  Saved → seat_comparison_p1_p2.png")


def plot_rms_summary(rms_base, rms_p1, rms_p2, save_dir):
    labels = ["Baseline\n(original cfg)", "Phase-1\n(linear opt.)", "Phase-2\n(asym. fit)"]
    values = [rms_base, rms_p1, rms_p2]
    colours = ["#4c72b0", "#55a868", "#c44e52"]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=colours, width=0.5, edgecolor="k", linewidth=0.8)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.0005,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("Seat RMS acceleration  [m/s²]")
    plt.title("RMS Summary — Baseline / Phase-1 / Phase-2")
    plt.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "validation_rms_summary.png"), dpi=150)
    plt.close()
    print(f"  Saved → validation_rms_summary.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Load Phase-1 results ─────────────────────────────────────────────────
    p1_json = os.path.join(PHASE1_DIR, "phase1_best_params.json")
    vrel_npy = os.path.join(PHASE1_DIR, "v_rel_front.npy")

    if not os.path.exists(p1_json):
        raise FileNotFoundError(f"Phase-1 output not found: {p1_json}\n"
                                "Run option2_bay_phase1.py first.")
    if not os.path.exists(vrel_npy):
        raise FileNotFoundError(f"v_rel data not found: {vrel_npy}\n"
                                "Run option2_bay_phase1.py first.")

    with open(p1_json) as fh:
        p1_params = json.load(fh)

    v_rel_opt = np.load(vrel_npy)
    CF_star   = float(p1_params["C_f"])

    print("=" * 60)
    print("OPTION 2 — PHASE 2: Asymmetric damper curve fitting")
    print("=" * 60)
    print(f"  Loaded Phase-1 params: {p1_params}")
    print(f"  CF* = {CF_star:.2f} N·s/m")
    print(f"  v_rel samples: {len(v_rel_opt)}")

    # ── Phase-2: Fit asymmetric curve ────────────────────────────────────────
    fitted, v_eval, weights, F_target, F_fitted = fit_asymmetric_damper(
        v_rel_opt, CF_star, n_starts=30
    )

    # Save fitted parameters
    with open(os.path.join(RESULTS_DIR, "phase2_asym_params.json"), "w") as fh:
        json.dump(fitted, fh, indent=2)
    print(f"\nFitted params saved → {RESULTS_DIR}/phase2_asym_params.json")

    # ── Phase-2 Validation: run full ODE with asymmetric damper ─────────────
    print("\n=== Phase-2 Validation: running full ODE with asymmetric damper ===")
    cfg_val = {**CFG, **p1_params,           # KF*, K2*, K3* from Phase 1
               "cs_minus":   fitted["cs_minus"],
               "asym_ratio": fitted["asym_ratio"],
               "gamma_c":    fitted["gamma_c"],
               "gamma_r":    fitted["gamma_r"]}

    df_p2, rms_p2 = run_validation(cfg_val, t_eval_full)

    # Phase-1 linear result (re-run for fair comparison on same t_eval)
    # (if you still have df_opt from Phase 1 saved you can load instead)
    from option2_bay_phase1 import (run_one_case as run_linear,
                                     compute_seat_rms)
    df_p1, _ = run_linear(p1_params, CFG, t_eval_full, collect_vrel=False)
    rms_p1   = compute_seat_rms(df_p1, CFG)

    # Baseline
    base_params = {"K_f": CFG["K_f"], "C_f": CFG["C_f"],
                   "K_2": CFG["K_2"], "K_3": CFG["K_3"]}
    df_base, _ = run_linear(base_params, CFG, t_eval_full, collect_vrel=False)
    rms_base   = compute_seat_rms(df_base, CFG)

    # ── Degradation check ────────────────────────────────────────────────────
    degradation = (rms_p2 - rms_p1) / rms_p1 * 100.0
    print("\n" + "=" * 60)
    print("PHASE-2 FINAL RESULTS")
    print("=" * 60)
    print(f"  Baseline RMS          : {rms_base:.5f} m/s²")
    print(f"  Phase-1 (linear opt)  : {rms_p1:.5f} m/s²")
    print(f"  Phase-2 (asym fitted) : {rms_p2:.5f} m/s²")
    print(f"  Degradation P1→P2     : {degradation:+.2f} %")
    if abs(degradation) <= 5.0:
        print("  ✓ Within ±5 % industry tolerance — asymmetric fit accepted.")
    else:
        print("  ✗ Degradation > 5 % — consider optional Phase-3 joint refinement.")
        print("    Tip: tighten the cs_minus bound toward CF_star and re-run.")

    print(f"\n  Fitted asymmetric parameters:")
    for k in ["cs_minus", "asym_ratio", "gamma_c", "gamma_r"]:
        print(f"    {k:15s} = {fitted[k]:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_fv_comparison(v_eval, weights, F_target, F_fitted,
                       CF_star, fitted, PLOTS_DIR)
    plot_seat_p1_p2(df_p1, df_p2, rms_p1, rms_p2, CFG, PLOTS_DIR)
    plot_rms_summary(rms_base, rms_p1, rms_p2, PLOTS_DIR)

    print(f"\nAll Phase-2 plots saved in: {PLOTS_DIR}")
